#!/usr/bin/env python3
# test commande EE : embedding lookup.
# RX : 'E' 'E' tok_lo tok_hi   (4 bytes)
# TX : 'E' 'K' x[64]            (66 bytes)
#
# compare with PC tok_emb[token] after chargement SDRAM.

import time, serial
import numpy as np
from infer_v4sim import load_model, MODEL_PATH as MODEL
from v4_quant import to_i8_shift
from transformer_ops import sd_load_matrix_chunked, D
from test_sdram_diag import sd_load

PORT = "COM6"; BAUD = 1_000_000

def call_ee(ser, token):
    pkt = b'EE' + bytes([token & 0xFF, (token >> 8) & 0xFF])
    ser.write(pkt)
    resp = ser.read(66)
    if len(resp) != 66 or resp[:2] != b'EK':
        raise RuntimeError(f"EE response: len={len(resp)} magic={resp[:2]!r}")
    return np.frombuffer(resp[2:], dtype=np.int8)

def main():
    m = load_model(MODEL); cfg = m['cfg']
    print(f"Model vocab={cfg['vocab_size']}")

    ser = serial.Serial(PORT, BAUD, timeout=8.0)
    time.sleep(0.5); ser.reset_input_buffer()

    # Charger tok_emb a l'adresse hardcoded (0x000000)
    tok_emb_i8, sh_emb = to_i8_shift(m['tok_emb'])
    print(f"Load tok_emb [{cfg['vocab_size']}, {D}] @ 0x000000 (sh={sh_emb})...")
    sd_load_matrix_chunked(ser, 0x000000, tok_emb_i8, cfg['vocab_size'], D)
    print("  OK\n")

    # test plusieurs tokens
    test_tokens = [0, 1, 17, 100, 256, 403, 511]
    print(f"{'tok':>4s}  {'attendu[:5]':25s}  {'recu[:5]':25s}  {'match':6s}")
    n_pass = n_fail = 0
    for tok in test_tokens:
        expected = tok_emb_i8[tok]
        received = call_ee(ser, tok)
        match = np.array_equal(expected, received)
        flag = "OK" if match else "FAIL"
        print(f"{tok:4d}  {str(expected[:5].tolist()):25s}  {str(received[:5].tolist()):25s}  {flag}")
        if match: n_pass += 1
        else:
            n_fail += 1
            print(f"      diff indices: {np.where(expected != received)[0][:10].tolist()}")
    print(f"\nResultat: {n_pass}/{len(test_tokens)} PASS")
    ser.close()
    return 0 if n_fail == 0 else 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
