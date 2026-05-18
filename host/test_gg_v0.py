#!/usr/bin/env python3
# test GG v0 : embed lookup + rmsnorm layer 0 -> x_norm[64], TOUT en RTL.
# RX : 'G' 'G' tok_lo tok_hi sh_emb sh_rms_att   (6 bytes)
# TX : 'G' 'K' shift_out x_norm[64]              (67 bytes)
#
# compare with : tok_emb[tok] (charge en SDRAM par PC) then rmsnorm sur PC.

import time, serial
import numpy as np
from infer_v4sim import load_model, MODEL_PATH as MODEL
from infer_v4sim import to_i8_shift, from_i8_shift, rmsnorm_q
from transformer_ops import sd_load_matrix_chunked, D
from test_sdram_diag import sd_load

PORT = "COM6"; BAUD = 1_000_000

def i8(b): return b - 256 if b >= 128 else b

def call_gg_v0(ser, token, sh_emb, sh_rms_att):
    pkt = b'GG' + bytes([token & 0xFF, (token >> 8) & 0xFF,
                         sh_emb & 0xFF, sh_rms_att & 0xFF])
    ser.write(pkt)
    resp = ser.read(67)
    if len(resp) != 67 or resp[:2] != b'GK':
        raise RuntimeError(f"GG response: len={len(resp)} magic={resp[:2]!r}")
    return np.frombuffer(resp[3:], dtype=np.int8), i8(resp[2])

def main():
    m = load_model(MODEL); cfg = m['cfg']
    print(f"Model: dim={cfg['dim']}\n")

    ser = serial.Serial(PORT, BAUD, timeout=15.0)
    time.sleep(0.5); ser.reset_input_buffer()

    # 1. Charger tok_emb a ADDR_TOK_EMB=0x000000
    tok_emb_i8, sh_emb = to_i8_shift(m['tok_emb'])
    print(f"Load tok_emb (sh={sh_emb})...")
    sd_load_matrix_chunked(ser, 0x000000, tok_emb_i8, cfg['vocab_size'], D)

    # 2. Charger rms_att[0] a ADDR_RMS_ATT_L0=0x010000
    rms_att_i8, sh_rms_att = to_i8_shift(m['rms_att'][0])
    print(f"Load rms_att[0] (sh={sh_rms_att})...")
    sd_load(ser, 0x010000, rms_att_i8.tobytes())

    # 3. test sur plusieurs tokens, comparer with ref Python
    test_tokens = [1, 100, 403]
    print(f"\n{'tok':>4s}  {'FPGA shift':10s} {'Ref shift':10s}  {'diff_max':10s}  {'match':6s}")
    n_pass = n_fail = 0
    for tok in test_tokens:
        # reference Python
        x_emb_i8 = tok_emb_i8[tok]   # quantif identical a celle chargee en SDRAM
        xn_ref_i8, sh_n_ref = rmsnorm_q(x_emb_i8, sh_emb, m['rms_att'][0])

        # FPGA
        xn_fpga_i8, sh_n_fpga = call_gg_v0(ser, tok, sh_emb, sh_rms_att)

        # compare en float
        xn_ref   = from_i8_shift(xn_ref_i8,  sh_n_ref)
        xn_fpga  = from_i8_shift(xn_fpga_i8, sh_n_fpga)
        diff_max = np.abs(xn_fpga - xn_ref).max()
        rel      = diff_max / max(np.abs(xn_ref).max(), 1e-9) * 100
        match    = rel < 10  # tolerance 10% pour quantif int8
        flag     = "OK" if match else "FAIL"
        print(f"{tok:4d}  {sh_n_fpga:10d} {sh_n_ref:10d}  {diff_max:10.4f} ({rel:.1f}%)  {flag}")
        if match: n_pass += 1
        else:
            n_fail += 1
            print(f"      ref[:5]   = {xn_ref[:5].round(3)}")
            print(f"      fpga[:5]  = {xn_fpga[:5].round(3)}")

    print(f"\n{n_pass}/{len(test_tokens)} PASS")
    ser.close()

if __name__ == "__main__":
    main()
