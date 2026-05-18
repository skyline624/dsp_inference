#!/usr/bin/env python3
# Debug : appelle les 8 chunks de la matmul tok_emb un par un pour see
# lequel echoue.

import time
import numpy as np
import serial

from infer_v4sim import load_model, MODEL_PATH as MODEL, to_i8_shift, from_i8_shift
from transformer_ops import sd_load_matrix_chunked, D
from test_sdram_diag import sd_load, call_fq

PORT = "COM6"; BAUD = 1_000_000

def main():
    m = load_model(MODEL); cfg = m['cfg']
    print(f"Vocab = {cfg['vocab_size']}, D = {D}")

    tok_emb_i8, sh_emb = to_i8_shift(m['tok_emb'])
    addr_emb = 0x100000  # base differente pour eviter conflits

    ser = serial.Serial(PORT, BAUD, timeout=5.0)
    time.sleep(0.5); ser.reset_input_buffer()

    # first FQ "test" pour verifier que le FPGA repond (sur un W simple)
    print("Test 1: petit FQ N=8 a addr 0x200000 pour verifier que FPGA est vivant...")
    W_small = np.random.default_rng(0).integers(-30, 30, (8, 64), dtype=np.int8)
    sd_load(ser, 0x200000, W_small.reshape(-1).tobytes())
    x_dummy = np.arange(64, dtype=np.int8)
    y_t, sy_t = call_fq(ser, 8, -3, -6, x_dummy, 0x200000)
    if y_t is None:
        print("  FAIL : FPGA ne repond meme pas a un FQ simple"); ser.close(); return
    print(f"  OK : y={y_t.tolist()} sh={sy_t}")

    print("\nTest 2: charge tok_emb 512x64 chunked a 0x100000")
    sd_load_matrix_chunked(ser, addr_emb, tok_emb_i8, cfg['vocab_size'], D)
    print(f"tok_emb loaded (sh={sh_emb}), 8 blocks of 4 KiB each")

    # Dummy input (x_norm constant pour test rapide)
    xn_i8 = np.arange(-32, 32, dtype=np.int8)   # 64 values distinctes
    sxn = -3

    print(f"\nx_norm dummy: {xn_i8[:8].tolist()}... shift={sxn}")
    print(f"\nCall FQ per chunk:")
    for n_idx in range(8):
        n_pos = n_idx * 64
        addr = addr_emb + n_pos * 64
        t0 = time.time()
        try:
            y_i8, sy = call_fq(ser, 64, sxn, sh_emb, xn_i8, addr)
            dt = time.time() - t0
            if y_i8 is None:
                print(f"  chunk {n_idx} (addr 0x{addr:06x}) : FAIL apres {dt:.2f}s (timeout)")
            else:
                print(f"  chunk {n_idx} (addr 0x{addr:06x}) : OK sh={sy:+d}  min={y_i8.min():4d} max={y_i8.max():4d}  ({dt:.2f}s)")
        except Exception as e:
            print(f"  chunk {n_idx} (addr 0x{addr:06x}) : EXCEPTION {e}")
            break

    ser.close()

if __name__ == "__main__":
    main()
