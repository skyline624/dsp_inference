#!/usr/bin/env python3
# Test du module silu_op sur FPGA.
# Protocole : 'S' 'S' shift_x x[64]
#  -> 'S' 'K' shift_out lut_idx[1] silu_int[2 LE] out[64]
# 69 octets de reponse au total.

import time, sys
import numpy as np
import serial
from v4_quant import to_i8_shift, from_i8_shift

PORT = "COM6"
BAUD = 1_000_000
D    = 64

def i8(b): return b - 256 if b >= 128 else b

def silu_fpga(ser, x_i8, sx):
    pkt = b'SS' + bytes([sx & 0xFF]) + x_i8.tobytes()
    ser.write(pkt)
    resp = ser.read(70)
    if len(resp) != 70 or resp[:2] != b'SK':
        raise RuntimeError(f"SS reponse {len(resp)}/70, magic={resp[:2]!r}")
    so       = i8(resp[2])
    lut_idx  = resp[3]
    silu_int = resp[4] | (resp[5] << 8)
    if silu_int & 0x8000: silu_int -= 0x10000
    out_i8   = np.frombuffer(resp[6:], dtype=np.int8)
    return out_i8, so, lut_idx, silu_int

def expected_lut_idx(x_i8_last, sx):
    """Index LUT pour le DERNIER element (idx = D-1)."""
    x_real = int(x_i8_last) * (2.0 ** sx)
    idx = int(round(x_real * 16 + 128))   # plus simple, pas exact en RTL
    # En RTL on fait shift entier, pas multiplication. Calcul exact :
    shx_p4 = sx + 4
    xs = int(x_i8_last)
    if shx_p4 >= 0:
        x16 = xs << shx_p4
    else:
        x16 = xs >> (-shx_p4)   # arithmetique vers -inf
    idx_int = x16 + 128
    if idx_int < 0: idx_int = 0
    if idx_int > 255: idx_int = 255
    return idx_int

def expected_silu_int(idx):
    """Valeur LUT silu Q4.11 pour cet index."""
    x = (idx - 128) / 16.0
    y = x / (1.0 + np.exp(-x))
    return int(round(y * 2048))

def silu_ref(x_i8, sx):
    """Reference Python : applique silu element par element, meme conventions que RTL."""
    out = np.zeros(D, dtype=np.int8)
    shx_p4 = sx + 4
    for i in range(D):
        xs = int(x_i8[i])
        x16 = xs << shx_p4 if shx_p4 >= 0 else xs >> (-shx_p4)
        idx = max(0, min(255, x16 + 128))
        x = (idx - 128) / 16.0
        y = x / (1.0 + np.exp(-x))
        silu_int = int(round(y * 2048))
        # silu_int >> (11 + sx) avec arrondi + clip
        out_shift = 11 + sx
        if out_shift > 0:
            o = (silu_int + (1 << (out_shift - 1))) >> out_shift
        elif out_shift < 0:
            o = silu_int << (-out_shift)
        else:
            o = silu_int
        out[i] = max(-128, min(127, o))
    return out

def main():
    ser = serial.Serial(PORT, BAUD, timeout=3.0)
    time.sleep(0.3); ser.reset_input_buffer()
    print(f"Test silu_op sur FPGA (D={D})\n")

    test_cases = [
        ("x=0 partout, sx=-3",       np.zeros(D, dtype=np.int8), -3),
        ("x=64 (=8.0) sx=-3",        np.full(D, 64, dtype=np.int8), -3),
        ("x=-64 (=-8.0) sx=-3",      np.full(D, -64, dtype=np.int8), -3),
        ("x=[i-32] (rampe -2..2) sx=-4", np.arange(-32, 32, dtype=np.int8), -4),
        ("x=N(0,1)*4 quantif",       *to_i8_shift(np.random.default_rng(7).normal(0,4,D).astype(np.float32))),
    ]

    for name, x_i8, sx in test_cases:
        print(f"=== {name} ===")
        try:
            out_i8, so, lut_idx, silu_int = silu_fpga(ser, x_i8, sx)
        except Exception as e:
            print(f"  ECHEC : {e}"); continue
        ref_i8 = silu_ref(x_i8, sx)
        # debug du DERNIER element (idx=63)
        exp_idx = expected_lut_idx(x_i8[D-1], sx)
        exp_si  = expected_silu_int(exp_idx)
        ok_idx = lut_idx == exp_idx
        ok_si  = silu_int == exp_si
        # compare outputs (element-wise int8)
        same = np.array_equal(out_i8, ref_i8)
        diff = np.max(np.abs(out_i8.astype(int) - ref_i8.astype(int)))

        print(f"  shift_out   : fpga={so:+3d}  (=shift_x={sx:+3d})  {'OK' if so==sx else 'FAUX'}")
        print(f"  lut_idx[63] : fpga={lut_idx:3d}  attendu={exp_idx:3d}   {'OK' if ok_idx else 'FAUX'}")
        print(f"  silu_int[63]: fpga={silu_int:6d}  attendu={exp_si:6d}   {'OK' if ok_si else 'FAUX'}")
        # comparer real outputs
        ref_real  = from_i8_shift(ref_i8, sx)
        fpga_real = from_i8_shift(out_i8, so)
        err = np.abs(fpga_real - ref_real).max()
        print(f"  outputs     : same_int8={same}  max_diff_int={diff}  err_real_max={err:.4f}")
        print(f"  ref[:4]  : {ref_real[:4]}")
        print(f"  fpga[:4] : {fpga_real[:4]}")
        print()
    ser.close()

if __name__ == "__main__":
    main()
