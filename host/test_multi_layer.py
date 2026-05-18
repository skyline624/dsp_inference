#!/usr/bin/env python3
# Test : 5 couches transformer enchainees (= architecture stories260K).
# Chaque couche a ses propres poids attn et ffn en SDRAM.

import time, serial
import numpy as np
from transformer_ops import (
    transformer_layer, transformer_layer_ref,
    setup_attn_weights, setup_ffn_weights,
)
from v4_quant import to_i8_shift, from_i8_shift

PORT = "COM6"; BAUD = 1_000_000
D = 64
N_LAYERS = 5

def main():
    ser = serial.Serial(PORT, BAUD, timeout=8.0)
    time.sleep(0.5); ser.reset_input_buffer()
    print(f"=== {N_LAYERS} couches transformer enchainees ===\n")

    rng = np.random.default_rng(777)
    x0_real = rng.normal(0, 1, D).astype(np.float32)

    # Setup poids des N_LAYERS couches
    layers_real = []
    layers_fpga = []
    # On reserve 0x10000 = 64KiB par couche pour avoir de la marge
    for L in range(N_LAYERS):
        base_attn = 0x200000 + L * 0x20000
        base_ffn  = base_attn + 0x10000
        aw_r, aw_f = setup_attn_weights(ser, rng, base_addr=base_attn)
        fw_r, fw_f = setup_ffn_weights( ser, rng, base_addr=base_ffn)
        layers_real.append((aw_r, fw_r))
        layers_fpga.append((aw_f, fw_f))
        print(f"  couche {L} chargee (attn @ {base_attn:06x}, ffn @ {base_ffn:06x})")
    print()

    # Reference float : enchaine les N couches
    x_ref = x0_real.copy()
    for L, (aw, fw) in enumerate(layers_real):
        x_ref = transformer_layer_ref(x_ref, aw, fw)
    print(f"REF : x_final[:6] = {x_ref[:6].round(3)}")

    # FPGA : enchaine les N couches
    x_i8, sx = to_i8_shift(x0_real)
    print("\nExecution FPGA layer par layer :")
    for L, (aw_f, fw_f) in enumerate(layers_fpga):
        x_i8, sx = transformer_layer(ser, x_i8, sx, aw_f, fw_f)
        x_now = from_i8_shift(x_i8, sx)
        x_ref_at_L = x0_real.copy()
        for k in range(L+1):
            x_ref_at_L = transformer_layer_ref(x_ref_at_L, *layers_real[k])
        diff = np.abs(x_now - x_ref_at_L).max()
        diff_rel = diff / max(np.abs(x_ref_at_L).max(), 1e-9) * 100
        print(f"  apres couche {L}: sx={sx:+d}  diff vs ref = {diff:.4f}  ({diff_rel:.1f}%)")

    x_final_fpga = from_i8_shift(x_i8, sx)
    print(f"\nFPGA: x_final[:6] = {x_final_fpga[:6].round(3)}")
    diff_max = np.abs(x_final_fpga - x_ref).max()
    diff_rel = diff_max / np.abs(x_ref).max() * 100
    print(f"\ndiff max apres {N_LAYERS} couches = {diff_max:.4f}  ({diff_rel:.1f}%)")
    print("==> OK" if diff_rel < 30 else "==> ERREUR trop grosse")

    ser.close()

if __name__ == "__main__":
    main()
