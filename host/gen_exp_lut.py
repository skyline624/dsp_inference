#!/usr/bin/env python3
# Genere la LUT exp(x) pour x in [-8, 0] (= apres soustraction de max).
# 256 entrees Q15 : LUT[i] = round(exp(-8 + i/32) * 32768), i in 0..255.
# LUT[0]=exp(-8)~=0, LUT[255]=exp(-0.03125)~=0.969.

import numpy as np, os
OUT = os.path.join(os.path.dirname(__file__), "..", "src", "exp_lut.hex")
with open(OUT, "w") as f:
    f.write(f"// LUT exp(x) pour x in [-8, 0) step 1/32, Q15\n")
    f.write(f"// index = round((x + 8) * 32), x = (index - 256) / 32\n")
    for i in range(256):
        x = -8 + i / 32.0
        y = np.exp(x)
        y_q15 = min(32767, int(round(y * 32768)))
        f.write(f"{y_q15:04x}\n")
print(f"LUT exp generee : {OUT}")
print(f"  exp(-8)={np.exp(-8):.6f}  exp(-4)={np.exp(-4):.4f}  exp(0)={np.exp(0):.4f}")
