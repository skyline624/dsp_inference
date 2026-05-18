#!/usr/bin/env python3
# test : couche transformer complete = attention block + FFN block.
# Verifie qu'une couche complete sur FPGA produit le same resultat (a la quantif pres)
# que la reference numpy float.

import time, serial
import numpy as np
from transformer_ops import (
    transformer_layer, transformer_layer_ref,
    setup_attn_weights, setup_ffn_weights,
)
from v4_quant import to_i8_shift, from_i8_shift

PORT = "COM6"; BAUD = 1_000_000
D = 64

def main():
    ser = serial.Serial(PORT, BAUD, timeout=8.0)
    time.sleep(0.5); ser.reset_input_buffer()
    print("=== Couche transformer complete (attn + ffn) ===\n")

    rng = np.random.default_rng(2024)
    x_real = rng.normal(0, 1, D).astype(np.float32)

    # Setup poids
    attn_w_real, attn_w_fpga = setup_attn_weights(ser, rng, base_addr=0x100000)
    ffn_w_real,  ffn_w_fpga  = setup_ffn_weights( ser, rng, base_addr=0x110000)
    print("Poids attn et ffn charges en SDRAM\n")

    # reference
    x_out_ref = transformer_layer_ref(x_real, attn_w_real, ffn_w_real)
    print(f"REF : x_out[:6] = {x_out_ref[:6].round(3)}")

    # FPGA
    x_i8, sx = to_i8_shift(x_real)
    x_out_i8, sx_out = transformer_layer(ser, x_i8, sx, attn_w_fpga, ffn_w_fpga)
    x_out_fpga = from_i8_shift(x_out_i8, sx_out)
    print(f"FPGA: x_out[:6] = {x_out_fpga[:6].round(3)}\n")

    diff_max = np.abs(x_out_fpga - x_out_ref).max()
    diff_rel = diff_max / np.abs(x_out_ref).max() * 100
    print(f"diff max final = {diff_max:.4f}  ({diff_rel:.1f}% du max signal)")
    print("==> OK" if diff_rel < 20 else "==> ERREUR trop grosse")

    ser.close()

if __name__ == "__main__":
    main()
