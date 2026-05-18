#!/usr/bin/env python3
# Test GG v5f : v5e + multiply elementwise -> h_gated[192] (= silu(W1*xn) * (W3*xn))
# TX : 'G' 'K' shift h_gated[192]   (195 bytes)

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
    resp = ser.read(195)
    if len(resp) != 195 or resp[:2] != b'GK':
        raise RuntimeError(f"GG v5f: len={len(resp)} magic={resp[:2]!r}")
    return np.frombuffer(resp[3:], dtype=np.int8), i8(resp[2])

def main():
    m = load_model(MODEL); cfg = m['cfg']
    n_rep = H // KH
    ser = serial.Serial(PORT, BAUD, timeout=15.0)
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

    sd_load_matrix_chunked(ser, 0x000000, tok_emb_i8, cfg['vocab_size'], D)
    sd_load(ser, 0x010000, rms_att_i8.tobytes())
    sd_load(ser, 0x010100, wq_i8.reshape(-1).tobytes())
    sd_load(ser, 0x011100, wk_i8.reshape(-1).tobytes())
    sd_load(ser, 0x011900, wv_i8.reshape(-1).tobytes())
    sd_load(ser, 0x012100, wo_i8.reshape(-1).tobytes())
    sd_load(ser, 0x013100, rms_ffn_i8.tobytes())
    sd_load_matrix_chunked(ser, 0x013200, w1_i8, 172, 64)
    sd_load_matrix_chunked(ser, 0x016200, w3_i8, 172, 64)

    n_pass = 0
    for tok in [1, 100]:
        print(f"\n=== tok={tok} ===")
        # Ref
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
        # W1, W3 par chunks (ref aussi chunked pour respecter le pipeline FPGA)
        h1_full = np.concatenate([
            from_i8_shift(*matvec_q(m['w1'][0][  0: 64,:], xnf_ref_i8, sh_nf_ref)),
            from_i8_shift(*matvec_q(m['w1'][0][ 64:128,:], xnf_ref_i8, sh_nf_ref)),
            from_i8_shift(*matvec_q(m['w1'][0][128:172,:], xnf_ref_i8, sh_nf_ref)),
        ])  # 172 floats
        h3_full = np.concatenate([
            from_i8_shift(*matvec_q(m['w3'][0][  0: 64,:], xnf_ref_i8, sh_nf_ref)),
            from_i8_shift(*matvec_q(m['w3'][0][ 64:128,:], xnf_ref_i8, sh_nf_ref)),
            from_i8_shift(*matvec_q(m['w3'][0][128:172,:], xnf_ref_i8, sh_nf_ref)),
        ])
        # silu(h1) (ref directement sur float)
        silu_h1 = h1_full / (1.0 + np.exp(-h1_full))
        h_gated_ref = silu_h1 * h3_full   # 172 floats

        h_gated_fpga, sh_hg_fpga = call_gg(ser, tok, [sh_emb, sh_rms, sh_q, sh_k, sh_v, sh_o, sh_rms_ffn, sh_h1, sh_h3])
        h_gated_fpga_172 = from_i8_shift(h_gated_fpga[:172], sh_hg_fpga)
        cos = np.dot(h_gated_fpga_172, h_gated_ref) / (np.linalg.norm(h_gated_fpga_172)*np.linalg.norm(h_gated_ref) + 1e-9)
        diff = np.abs(h_gated_fpga_172 - h_gated_ref).max()
        match = cos > 0.85
        flag = "OK" if match else "FAIL"
        print(f"  shift FPGA={sh_hg_fpga}")
        print(f"  ref[:6]  = {h_gated_ref[:6].round(3)}")
        print(f"  fpga[:6] = {h_gated_fpga_172[:6].round(3)}")
        print(f"  cos={cos:.4f}  diff_max={diff:.3f}  {flag}")
        if match: n_pass += 1

    print(f"\n{n_pass}/2 PASS")
    ser.close()

if __name__ == "__main__":
    main()
