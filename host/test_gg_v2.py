#!/usr/bin/env python3
# test GG v2 : embed + rmsnorm + matmuls Q, K, V
# RX : 'G' 'G' tok_lo tok_hi sh_emb sh_rms sh_q sh_k sh_v   (9 bytes)
# TX : 'G' 'K' sh_q sh_k sh_v Q[64] K[32] V[32]              (133 bytes)

import time, serial
import numpy as np
from infer_v4sim import load_model, MODEL_PATH as MODEL
from infer_v4sim import to_i8_shift, from_i8_shift, rmsnorm_q, matvec_q
from transformer_ops import sd_load_matrix_chunked, D, H, KH, HS
from test_sdram_diag import sd_load

PORT = "COM6"; BAUD = 1_000_000

def i8(b): return b - 256 if b >= 128 else b

def call_gg_v2(ser, token, sh_emb, sh_rms, sh_q, sh_k, sh_v):
    pkt = b'GG' + bytes([token & 0xFF, (token >> 8) & 0xFF,
                         sh_emb & 0xFF, sh_rms & 0xFF,
                         sh_q & 0xFF, sh_k & 0xFF, sh_v & 0xFF])
    ser.write(pkt)
    resp = ser.read(133)
    if len(resp) != 133 or resp[:2] != b'GK':
        raise RuntimeError(f"GG v2 response: len={len(resp)} magic={resp[:2]!r}")
    sh_q_out = i8(resp[2]); sh_k_out = i8(resp[3]); sh_v_out = i8(resp[4])
    Q = np.frombuffer(resp[5:5+64], dtype=np.int8)
    K = np.frombuffer(resp[5+64:5+64+32], dtype=np.int8)
    V = np.frombuffer(resp[5+64+32:], dtype=np.int8)
    return Q, sh_q_out, K, sh_k_out, V, sh_v_out

def cmp(label, fpga, ref):
    cos = np.dot(fpga, ref) / (np.linalg.norm(fpga)*np.linalg.norm(ref) + 1e-9)
    diff = np.abs(fpga - ref).max()
    flag = "OK" if cos > 0.95 else "FAIL"
    print(f"    {label:6s}: cos={cos:.4f}  diff_max={diff:.3f}  {flag}")
    return cos > 0.95

def main():
    m = load_model(MODEL); cfg = m['cfg']
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
    sd_load(ser, 0x011100, wk_i8.reshape(-1).tobytes())   # ADDR_WK_L0
    sd_load(ser, 0x011900, wv_i8.reshape(-1).tobytes())   # ADDR_WV_L0

    test_tokens = [1, 100, 403]
    n_pass = 0
    for tok in test_tokens:
        print(f"\n=== tok={tok} ===")
        # reference Python
        x_emb_i8 = tok_emb_i8[tok]
        xn_ref, sh_n = rmsnorm_q(x_emb_i8, sh_emb, m['rms_att'][0])
        Q_ref_i8, sQ_ref = matvec_q(m['wq'][0], xn_ref, sh_n)
        K_ref_i8, sK_ref = matvec_q(m['wk'][0], xn_ref, sh_n)
        V_ref_i8, sV_ref = matvec_q(m['wv'][0], xn_ref, sh_n)

        # FPGA
        Q, sQ, K, sK, V, sV = call_gg_v2(ser, tok, sh_emb, sh_rms, sh_q, sh_k, sh_v)
        print(f"  shifts FPGA  Q={sQ} K={sK} V={sV}")
        print(f"  shifts ref   Q={sQ_ref} K={sK_ref} V={sV_ref}")

        all_ok  = cmp("Q", from_i8_shift(Q, sQ),  from_i8_shift(Q_ref_i8, sQ_ref))
        all_ok &= cmp("K", from_i8_shift(K, sK),  from_i8_shift(K_ref_i8, sK_ref))
        all_ok &= cmp("V", from_i8_shift(V, sV),  from_i8_shift(V_ref_i8, sV_ref))
        if all_ok: n_pass += 1

    print(f"\n{n_pass}/{len(test_tokens)} PASS")
    ser.close()

if __name__ == "__main__":
    main()
