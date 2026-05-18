#!/usr/bin/env python3
# Genere la LUT silu(x) = x / (1 + exp(-x)) pour x in [-8, 8).
# 256 entrees, output en Q4.11 (int16 signed, scale 2^-11, max +-16).
# index i correspond a x = (i - 128) / 16.

import numpy as np
import os

OUT = os.path.join(os.path.dirname(__file__), "..", "src", "silu_lut.hex")

def silu(x):
    return x / (1.0 + np.exp(-x))

with open(OUT, "w") as f:
    f.write(f"// LUT silu(x) pour x in [-8, 8) step 1/16, Q4.11 (scale 2^-11)\n")
    f.write(f"// index i -> x = (i - 128) / 16,  valeur = round(silu(x) * 2048)\n")
    for i in range(256):
        x = (i - 128) / 16.0
        y = silu(x)
        y_q11 = int(round(y * 2048))
        # clamp en int16 signed
        y_q11 = max(-32768, min(32767, y_q11))
        # encoder en hex 16-bit (two's complement pour negatifs)
        if y_q11 < 0:
            y_q11 += 65536
        f.write(f"{y_q11:04x}\n")

print(f"LUT silu generee dans {OUT}")
print(f"  256 entrees Q4.11, x in [-8, 8) step 1/16")
print(f"  silu(0)={silu(0):.4f}  silu(8)={silu(8):.4f}  silu(-8)={silu(-8):.6f}")
print(f"  Q4.11 max representable = +/-16  (silu_max(8)={silu(8):.3f} OK)")
