#!/usr/bin/env python3
# test GG v1 : embed + rmsnorm + matmul Wq  -> Q[64]
# RX : 'G' 'G' tok_lo tok_hi sh_emb sh_rms_att sh_q   (7 bytes)
# TX : 'G' 'K' shift_q Q[64]                          (67 bytes)

import time, serial
import numpy as np
from infer_v4sim import load_model, MODEL_PATH as MODEL
from infer_v4sim import to_i8_shift, from_i8_shift, rmsnorm_q, matvec_q
from transformer_ops import sd_load_matrix_chunked, D
from test_sdram_diag import sd_load

PORT = "COM6"; BAUD = 1_000_000

def i8(b): return b - 256 if b >= 128 else b

def call_gg_v1(ser, token, sh_emb, sh_rms_att, sh_q):
    pkt = b'GG' + bytes([token & 0xFF, (token >> 8) & 0xFF,
                         sh_emb & 0xFF, sh_rms_att & 0xFF, sh_q & 0xFF])
    ser.write(pkt)
    resp = ser.read(67)
    if len(resp) != 67 or resp[:2] != b'GK':
        raise RuntimeError(f"GG v1 response: len={len(resp)} magic={resp[:2]!r}")
    return np.frombuffer(resp[3:], dtype=np.int8), i8(resp[2])

def main():
    m = load_model(MODEL); cfg = m['cfg']
    H, HS = cfg['n_heads'], cfg['head_size']
    print(f"Model: dim={cfg['dim']} H={H} HS={HS}\n")

    ser = serial.Serial(PORT, BAUD, timeout=15.0)
    time.sleep(0.5); ser.reset_input_buffer()

    # Charger tok_emb, rms_att[0], wq[0]
    tok_emb_i8, sh_emb = to_i8_shift(m['tok_emb'])
    rms_att_i8, sh_rms = to_i8_shift(m['rms_att'][0])
    wq_i8,      sh_q   = to_i8_shift(m['wq'][0])
    print(f"Load tok_emb (sh={sh_emb}), rms_att[0] (sh={sh_rms}), wq[0] (sh={sh_q})")
    sd_load_matrix_chunked(ser, 0x000000, tok_emb_i8, cfg['vocab_size'], D)
    sd_load(ser, 0x010000, rms_att_i8.tobytes())
    sd_load(ser, 0x010100, wq_i8.reshape(-1).tobytes())   # ADDR_WQ_L0

    test_tokens = [1, 100, 403]
    print(f"\n{'tok':>4s}  {'FPGA shift':10s} {'Ref shift':10s}  {'diff_max':10s}  {'match':6s}")
    n_pass = n_fail = 0
    for tok in test_tokens:
        # reference Python : embed[tok] -> rmsnorm -> matvec_q(wq)
        x_emb_i8 = tok_emb_i8[tok]
        xn_ref_i8, sh_n_ref = rmsnorm_q(x_emb_i8, sh_emb, m['rms_att'][0])
        Q_ref_i8, sh_Q_ref = matvec_q(m['wq'][0], xn_ref_i8, sh_n_ref)

        # FPGA
        Q_fpga_i8, sh_Q_fpga = call_gg_v1(ser, tok, sh_emb, sh_rms, sh_q)

        Q_ref  = from_i8_shift(Q_ref_i8,  sh_Q_ref)
        Q_fpga = from_i8_shift(Q_fpga_i8, sh_Q_fpga)
        cos = np.dot(Q_fpga, Q_ref) / (np.linalg.norm(Q_fpga) * np.linalg.norm(Q_ref) + 1e-9)
        diff_max = np.abs(Q_fpga - Q_ref).max()
        match = cos > 0.95
        flag  = "OK" if match else "FAIL"
        print(f"{tok:4d}  {sh_Q_fpga:+3d} {sh_Q_ref:+3d}  cos={cos:.4f}  diff_max={diff_max:.3f}  {flag}")
        if match: n_pass += 1
        else:
            n_fail += 1
            print(f"      ref[:5]  = {Q_ref[:5].round(3)}")
            print(f"      fpga[:5] = {Q_fpga[:5].round(3)}")

    print(f"\n{n_pass}/{len(test_tokens)} PASS")
    ser.close()

if __name__ == "__main__":
    main()
