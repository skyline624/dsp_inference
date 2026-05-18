#!/usr/bin/env python3
# Test GG v5a : v4 + rmsnorm_ffn -> x_norm_ffn[64]
# RX : 'G' 'G' tok_lo tok_hi sh_emb sh_rms sh_q sh_k sh_v sh_o sh_rms_ffn  (11 bytes)
# TX : 'G' 'K' shift_x x_norm_ffn[64]                                       (67 bytes)

import time, serial
import numpy as np
from infer_v4sim import load_model, MODEL_PATH as MODEL
from infer_v4sim import to_i8_shift, from_i8_shift, rmsnorm_q, matvec_q
from transformer_ops import sd_load_matrix_chunked, D, H, KH, HS
from test_sdram_diag import sd_load

PORT = "COM6"; BAUD = 1_000_000

def i8(b): return b - 256 if b >= 128 else b

def call_gg_v5a(ser, token, sh_emb, sh_rms, sh_q, sh_k, sh_v, sh_o, sh_rms_ffn):
    pkt = b'GG' + bytes([token & 0xFF, (token >> 8) & 0xFF,
                         sh_emb & 0xFF, sh_rms & 0xFF,
                         sh_q & 0xFF, sh_k & 0xFF, sh_v & 0xFF, sh_o & 0xFF,
                         sh_rms_ffn & 0xFF])
    ser.write(pkt)
    resp = ser.read(67)
    if len(resp) != 67 or resp[:2] != b'GK':
        raise RuntimeError(f"GG v5a: len={len(resp)} magic={resp[:2]!r}")
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

    sd_load_matrix_chunked(ser, 0x000000, tok_emb_i8, cfg['vocab_size'], D)
    sd_load(ser, 0x010000, rms_att_i8.tobytes())
    sd_load(ser, 0x010100, wq_i8.reshape(-1).tobytes())
    sd_load(ser, 0x011100, wk_i8.reshape(-1).tobytes())
    sd_load(ser, 0x011900, wv_i8.reshape(-1).tobytes())
    sd_load(ser, 0x012100, wo_i8.reshape(-1).tobytes())
    sd_load(ser, 0x013100, rms_ffn_i8.tobytes())     # ADDR_RMS_FFN_L0

    test_tokens = [1, 100, 403]
    n_pass = 0
    for tok in test_tokens:
        print(f"\n=== tok={tok} ===")
        # Reference Python : v4 (x_after_attn) puis rmsnorm_ffn
        x_emb_i8 = tok_emb_i8[tok]
        x_orig = from_i8_shift(x_emb_i8, sh_emb)
        xn_ref, sh_n = rmsnorm_q(x_emb_i8, sh_emb, m['rms_att'][0])
        V_ref_i8, sV_ref = matvec_q(m['wv'][0], xn_ref, sh_n)
        V_ref = from_i8_shift(V_ref_i8, sV_ref).reshape(KH, HS)
        attn_ref = np.repeat(V_ref, n_rep, axis=0).reshape(D)
        attn_ref_i8, sh_attn_ref = to_i8_shift(attn_ref)
        Wo_ref_i8, sh_wo_ref = matvec_q(m['wo'][0], attn_ref_i8, sh_attn_ref)
        Wo_ref = from_i8_shift(Wo_ref_i8, sh_wo_ref)
        x_after_attn = x_orig + Wo_ref
        # rmsnorm_ffn
        xafn_i8, sh_xafn = to_i8_shift(x_after_attn)
        xnf_ref_i8, sh_nf_ref = rmsnorm_q(xafn_i8, sh_xafn, m['rms_ffn'][0])
        xnf_ref = from_i8_shift(xnf_ref_i8, sh_nf_ref)

        xnf_fpga_i8, sh_xnf_fpga = call_gg_v5a(ser, tok, sh_emb, sh_rms, sh_q, sh_k, sh_v, sh_o, sh_rms_ffn)
        xnf_fpga = from_i8_shift(xnf_fpga_i8, sh_xnf_fpga)
        cos = np.dot(xnf_fpga, xnf_ref) / (np.linalg.norm(xnf_fpga)*np.linalg.norm(xnf_ref) + 1e-9)
        diff = np.abs(xnf_fpga - xnf_ref).max()
        match = cos > 0.95
        flag = "OK" if match else "FAIL"
        print(f"  shift FPGA={sh_xnf_fpga} ref={sh_nf_ref}")
        print(f"  cos={cos:.4f}  diff_max={diff:.3f}  {flag}")
        if not match:
            print(f"  ref[:6]  = {xnf_ref[:6].round(3)}")
            print(f"  fpga[:6] = {xnf_fpga[:6].round(3)}")
        if match: n_pass += 1

    print(f"\n{n_pass}/{len(test_tokens)} PASS")
    ser.close()

if __name__ == "__main__":
    main()
