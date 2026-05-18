#!/usr/bin/env python3
# Genere la LUT 1/x pour x in [1, 2). 256 entrees Q15.
# LUT[i] = round(1/((256+i)/256) * 32768) = round(2^15 * 256/(256+i))
# output en [0.5, 1] (LUT[0]=1.0=32767 cap, LUT[255]~=0.502)
import numpy as np, os
OUT = os.path.join(os.path.dirname(__file__), "..", "src", "inv_lut.hex")
with open(OUT, "w") as f:
    f.write(f"// LUT 1/x pour x in [1, 2) step 1/256, Q15\n")
    f.write(f"// index i -> x = (256+i)/256, valeur = round(1/x * 32768)\n")
    for i in range(256):
        x = (256 + i) / 256.0
        y = 1.0 / x
        y_q15 = min(32767, int(round(y * 32768)))
        f.write(f"{y_q15:04x}\n")
print(f"LUT 1/x generee : {OUT}")
