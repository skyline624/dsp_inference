#!/usr/bin/env python3
# Test GG v5d : v5c + W3 chunked -> h3[192]
# RX : 13 bytes ('GG' + tok 2 + 9 shifts), TX : 197 bytes (sh0/1/2 + h3[192])

import time, serial
import numpy as np
from infer_v4sim import load_model, MODEL_PATH as MODEL
from infer_v4sim import to_i8_shift, from_i8_shift, rmsnorm_q, matvec_q
from transformer_ops import sd_load_matrix_chunked, D, H, KH, HS
from test_sdram_diag import sd_load

PORT = "COM6"; BAUD = 1_000_000

def i8(b): return b - 256 if b >= 128 else b

def call_gg_v5d(ser, token, shifts):
    pkt = b'GG' + bytes([token & 0xFF, (token >> 8) & 0xFF]) + bytes([s & 0xFF for s in shifts])
    ser.write(pkt)
    resp = ser.read(197)
    if len(resp) != 197 or resp[:2] != b'GK':
        raise RuntimeError(f"GG v5d: len={len(resp)} magic={resp[:2]!r}")
    return np.frombuffer(resp[5:], dtype=np.int8), i8(resp[2]), i8(resp[3]), i8(resp[4])

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
    sd_load_matrix_chunked(ser, 0x016200, w3_i8, 172, 64)   # ADDR_W3_L0

    test_tokens = [1, 100]
    n_pass = 0
    for tok in test_tokens:
        print(f"\n=== tok={tok} ===")
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
        h3_ch0_ref_i8, sh_ch0_ref = matvec_q(m['w3'][0][  0: 64, :], xnf_ref_i8, sh_nf_ref)
        h3_ch1_ref_i8, sh_ch1_ref = matvec_q(m['w3'][0][ 64:128, :], xnf_ref_i8, sh_nf_ref)
        h3_ch2_ref_i8, sh_ch2_ref = matvec_q(m['w3'][0][128:172, :], xnf_ref_i8, sh_nf_ref)

        h3_fpga, sh_ch0, sh_ch1, sh_ch2 = call_gg_v5d(ser, tok, [sh_emb, sh_rms, sh_q, sh_k, sh_v, sh_o, sh_rms_ffn, sh_h1, sh_h3])
        print(f"  sh FPGA  ch0={sh_ch0} ch1={sh_ch1} ch2={sh_ch2}")
        print(f"  sh ref   ch0={sh_ch0_ref} ch1={sh_ch1_ref} ch2={sh_ch2_ref}")
        def cmp_chunk(label, f, r):
            c = np.dot(f, r) / (np.linalg.norm(f)*np.linalg.norm(r) + 1e-9)
            d = np.abs(f - r).max()
            flag = "OK" if c > 0.90 else "FAIL"
            print(f"    {label}: cos={c:.4f}  diff_max={d:.3f}  {flag}")
            return c > 0.90
        ok = True
        ok &= cmp_chunk("ch0",     from_i8_shift(h3_fpga[0:64],    sh_ch0), from_i8_shift(h3_ch0_ref_i8, sh_ch0_ref))
        ok &= cmp_chunk("ch1",     from_i8_shift(h3_fpga[64:128],  sh_ch1), from_i8_shift(h3_ch1_ref_i8, sh_ch1_ref))
        ok &= cmp_chunk("ch2(44)", from_i8_shift(h3_fpga[128:172], sh_ch2), from_i8_shift(h3_ch2_ref_i8, sh_ch2_ref))
        if ok: n_pass += 1

    print(f"\n{n_pass}/{len(test_tokens)} PASS")
    ser.close()

if __name__ == "__main__":
    main()
