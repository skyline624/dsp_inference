#!/usr/bin/env python3
# Test : FFN block avec hidden=172 (= stories260K reel) via chunking.

import time, serial
import numpy as np
from transformer_ops import ffn_block_full, ffn_block_ref, setup_ffn_weights
from v4_quant import to_i8_shift, from_i8_shift

PORT = "COM6"; BAUD = 1_000_000
D = 64
HIDDEN = 172

def main():
    ser = serial.Serial(PORT, BAUD, timeout=15.0)
    time.sleep(0.5); ser.reset_input_buffer()
    print(f"=== FFN block hidden={HIDDEN} (chunked) ===\n")

    rng = np.random.default_rng(31415)
    x_real = rng.normal(0, 1, D).astype(np.float32)
    w_real, w_fpga = setup_ffn_weights(ser, rng, base_addr=0x300000, hidden=HIDDEN)
    print(f"Poids charges en SDRAM (hidden={HIDDEN}, chunking N=3, K=3)")

    x_out_ref = ffn_block_ref(x_real, w_real)
    print(f"REF : x_out[:6] = {x_out_ref[:6].round(3)}")

    x_i8, sx = to_i8_shift(x_real)
    x_out_i8, sx_out = ffn_block_full(ser, x_i8, sx, w_fpga, hidden=HIDDEN)
    x_out_fpga = from_i8_shift(x_out_i8, sx_out)
    print(f"FPGA: x_out[:6] = {x_out_fpga[:6].round(3)}\n")

    diff = np.abs(x_out_fpga - x_out_ref).max()
    rel  = diff / np.abs(x_out_ref).max() * 100
    print(f"diff max = {diff:.4f}  ({rel:.1f}%)")
    print("==> OK" if rel < 30 else "==> ERREUR")
    ser.close()

if __name__ == "__main__":
    main()
