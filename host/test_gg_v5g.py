#!/usr/bin/env python3
# test GG v5g : v5f + W2 chunked + residual final FFN -> x_after_ffn[64]
# = layer 0 transformer complete en RTL
# RX : 13 bytes ('GG' + tok 2 + 10 shifts)  TX : 67 bytes (shift + x_after_ffn[64])

import time, serial
import numpy as np
from infer_v4sim import load_model, MODEL_PATH as MODEL
from infer_v4sim import to_i8_shift, from_i8_shift, rmsnorm_q, matvec_q, silu_q, mul_q
from transformer_ops import sd_load_matrix_chunked, D, H, KH, HS
from test_sdram_diag import sd_load

PORT = "COM6"; BAUD = 1_000_000
def i8(b): return b - 256 if b >= 128 else b

def call_gg(ser, token, shifts):
    pkt = b'GG' + bytes([token & 0xFF, (token >> 8) & 0xFF]) + bytes([s & 0xFF for s in shifts])
    ser.write(pkt)
    resp = ser.read(67)
    if len(resp) != 67 or resp[:2] != b'GK':
        raise RuntimeError(f"GG v5g: len={len(resp)} magic={resp[:2]!r}")
    return np.frombuffer(resp[3:], dtype=np.int8), i8(resp[2])

def main():
    m = load_model(MODEL); cfg = m['cfg']
    n_rep = H // KH
    ser = serial.Serial(PORT, BAUD, timeout=20.0)
    time.sleep(0.5); ser.reset_input_buffer()

    tok_emb_i8, sh_emb = to_i8_shift(m['tok_emb'])
    rms_att_i8, sh_rms = to_i8_shift(m['rms_att'][0])
    wq_i8, sh_q = to_i8_shift(m['wq'][0])
    wk_i8, sh_k = to_i8_shift(m['wk'][0])
    wv_i8, sh_v = to_i8_shift(m['wv'][0])
    wo_i8, sh_o = to_i8_shift(m['wo'][0])
    rms_ffn_i8, sh_rms_ffn = to_i8_shift(m['rms_ffn'][0])
    w1_i8, sh_h1 = to_i8_shift(m['w1'][0])
    w3_i8, sh_h3 = to_i8_shift(m['w3'][0])
    w2_i8, sh_w2 = to_i8_shift(m['w2'][0])

    sd_load_matrix_chunked(ser, 0x000000, tok_emb_i8, cfg['vocab_size'], D)
    sd_load(ser, 0x010000, rms_att_i8.tobytes())
    sd_load(ser, 0x010100, wq_i8.reshape(-1).tobytes())
    sd_load(ser, 0x011100, wk_i8.reshape(-1).tobytes())
    sd_load(ser, 0x011900, wv_i8.reshape(-1).tobytes())
    sd_load(ser, 0x012100, wo_i8.reshape(-1).tobytes())
    sd_load(ser, 0x013100, rms_ffn_i8.tobytes())
    sd_load_matrix_chunked(ser, 0x013200, w1_i8, 172, 64)
    sd_load_matrix_chunked(ser, 0x016200, w3_i8, 172, 64)
    sd_load_matrix_chunked(ser, 0x019200, w2_i8, 64, 172)    # ADDR_W2_L0

    n_pass = 0
    for tok in [1, 100]:
        print(f"\n=== tok={tok} ===")
        # Ref complete : v4 (x_after_attn) + FFN complete
        x_emb_i8 = tok_emb_i8[tok]
        x_orig = from_i8_shift(x_emb_i8, sh_emb)
        xn_ref, sh_n = rmsnorm_q(x_emb_i8, sh_emb, m['rms_att'][0])
        V_ref_i8, sV_ref = matvec_q(m['wv'][0], xn_ref, sh_n)
        V_ref = from_i8_shift(V_ref_i8, sV_ref).reshape(KH, HS)
        attn_ref = np.repeat(V_ref, n_rep, axis=0).reshape(D)
        attn_ref_i8, sh_attn_ref = to_i8_shift(attn_ref)
        Wo_ref_i8, sh_wo_ref = matvec_q(m['wo'][0], attn_ref_i8, sh_attn_ref)
        x_after_attn = x_orig + from_i8_shift(Wo_ref_i8, sh_wo_ref)
        xafn_i8, sh_xafn = to_i8_shift(x_after_attn)
        xnf_ref_i8, sh_nf_ref = rmsnorm_q(xafn_i8, sh_xafn, m['rms_ffn'][0])
        # W1 + silu + W3 + multiply (en float ref)
        h1_ref = np.concatenate([
            from_i8_shift(*matvec_q(m['w1'][0][:64,:],   xnf_ref_i8, sh_nf_ref)),
            from_i8_shift(*matvec_q(m['w1'][0][64:128,:], xnf_ref_i8, sh_nf_ref)),
            from_i8_shift(*matvec_q(m['w1'][0][128:172,:],xnf_ref_i8, sh_nf_ref)),
        ])
        h3_ref = np.concatenate([
            from_i8_shift(*matvec_q(m['w3'][0][:64,:],   xnf_ref_i8, sh_nf_ref)),
            from_i8_shift(*matvec_q(m['w3'][0][64:128,:], xnf_ref_i8, sh_nf_ref)),
            from_i8_shift(*matvec_q(m['w3'][0][128:172,:],xnf_ref_i8, sh_nf_ref)),
        ])
        silu_h1 = h1_ref / (1.0 + np.exp(-h1_ref))
        h_gated = silu_h1 * h3_ref
        # W2 @ h_gated
        w2_out = m['w2'][0] @ h_gated
        x_after_ffn_ref = x_after_attn + w2_out

        x_fpga_i8, sh_fpga = call_gg(ser, tok, [sh_emb, sh_rms, sh_q, sh_k, sh_v, sh_o, sh_rms_ffn, sh_h1, sh_h3, sh_w2])
        x_fpga = from_i8_shift(x_fpga_i8, sh_fpga)

        cos = np.dot(x_fpga, x_after_ffn_ref) / (np.linalg.norm(x_fpga)*np.linalg.norm(x_after_ffn_ref) + 1e-9)
        diff = np.abs(x_fpga - x_after_ffn_ref).max()
        match = cos > 0.85
        flag = "OK" if match else "FAIL"
        print(f"  shift FPGA={sh_fpga}")
        print(f"  ref[:6]  = {x_after_ffn_ref[:6].round(3)}")
        print(f"  fpga[:6] = {x_fpga[:6].round(3)}")
        print(f"  cos={cos:.4f}  diff_max={diff:.3f}  {flag}")
        if match: n_pass += 1

    print(f"\n{n_pass}/2 PASS")
    ser.close()

if __name__ == "__main__":
    main()
