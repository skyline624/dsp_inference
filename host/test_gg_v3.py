#!/usr/bin/env python3
# Test GG v3 : embed + rmsnorm + Q/K/V + multi-head attention (T=1, pos=0)
# RX : 'G' 'G' tok_lo tok_hi sh_emb sh_rms sh_q sh_k sh_v   (9 bytes)
# TX : 'G' 'K' sh_attn attn_out[64]                          (67 bytes)

import time, serial
import numpy as np
from infer_v4sim import load_model, MODEL_PATH as MODEL
from infer_v4sim import to_i8_shift, from_i8_shift, rmsnorm_q, matvec_q
from transformer_ops import sd_load_matrix_chunked, D, H, KH, HS
from test_sdram_diag import sd_load

PORT = "COM6"; BAUD = 1_000_000

def i8(b): return b - 256 if b >= 128 else b

def call_gg_v3(ser, token, sh_emb, sh_rms, sh_q, sh_k, sh_v):
    pkt = b'GG' + bytes([token & 0xFF, (token >> 8) & 0xFF,
                         sh_emb & 0xFF, sh_rms & 0xFF,
                         sh_q & 0xFF, sh_k & 0xFF, sh_v & 0xFF])
    ser.write(pkt)
    resp = ser.read(67)
    if len(resp) != 67 or resp[:2] != b'GK':
        raise RuntimeError(f"GG v3 response: len={len(resp)} magic={resp[:2]!r}")
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

    print(f"Load weights : sh_emb={sh_emb} sh_rms={sh_rms} sh_q={sh_q} sh_k={sh_k} sh_v={sh_v}")
    sd_load_matrix_chunked(ser, 0x000000, tok_emb_i8, cfg['vocab_size'], D)
    sd_load(ser, 0x010000, rms_att_i8.tobytes())
    sd_load(ser, 0x010100, wq_i8.reshape(-1).tobytes())
    sd_load(ser, 0x011100, wk_i8.reshape(-1).tobytes())
    sd_load(ser, 0x011900, wv_i8.reshape(-1).tobytes())

    test_tokens = [1, 100, 403]
    n_pass = 0
    for tok in test_tokens:
        print(f"\n=== tok={tok} ===")
        # Reference Python : embed -> rmsnorm -> Q/K/V matvecs -> attn (T=1, softmax=1)
        x_emb_i8 = tok_emb_i8[tok]
        xn_ref, sh_n = rmsnorm_q(x_emb_i8, sh_emb, m['rms_att'][0])
        Q_ref_i8, sQ_ref = matvec_q(m['wq'][0], xn_ref, sh_n)
        # For T=1, attn_out = V_repeated (softmax=1)
        V_ref_i8, sV_ref = matvec_q(m['wv'][0], xn_ref, sh_n)
        V_ref = from_i8_shift(V_ref_i8, sV_ref).reshape(KH, HS)
        attn_ref = np.repeat(V_ref, n_rep, axis=0).reshape(D)

        # FPGA
        attn_fpga_i8, sh_attn = call_gg_v3(ser, tok, sh_emb, sh_rms, sh_q, sh_k, sh_v)
        attn_fpga = from_i8_shift(attn_fpga_i8, sh_attn)

        cos = np.dot(attn_fpga, attn_ref) / (np.linalg.norm(attn_fpga)*np.linalg.norm(attn_ref) + 1e-9)
        diff = np.abs(attn_fpga - attn_ref).max()
        match = cos > 0.95
        flag = "OK" if match else "FAIL"
        print(f"  FPGA sh_attn={sh_attn}, ref-equiv sh={sV_ref}")
        print(f"  cos={cos:.4f}  diff_max={diff:.3f}  {flag}")
        if not match:
            print(f"  ref[:6]  = {attn_ref[:6].round(3)}")
            print(f"  fpga[:6] = {attn_fpga[:6].round(3)}")
        if match: n_pass += 1

    print(f"\n{n_pass}/{len(test_tokens)} PASS")
    ser.close()

if __name__ == "__main__":
    main()
