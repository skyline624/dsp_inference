#!/usr/bin/env python3
# Test du module softmax_op sur FPGA (K=32).
# Protocole : 'X''X' shift_x x[32]                                  (35 oct)
# Response  : 'X''K' shift_out max[1] sum[3] p_sum[1] inv_sum[2] out[32]  (42 oct)

import time, sys
import numpy as np
import serial

PORT = "COM6"
BAUD = 1_000_000
K    = 32

def i8(b): return b - 256 if b >= 128 else b

def softmax_fpga(ser, x_i8, sx):
    pkt = b'XX' + bytes([sx & 0xFF]) + x_i8.tobytes()
    assert len(pkt) == 35, len(pkt)
    ser.write(pkt)
    resp = ser.read(42)
    if len(resp) != 42 or resp[:2] != b'XK':
        raise RuntimeError(f"XX reponse {len(resp)}/42, magic={resp[:2]!r}")
    so       = i8(resp[2])
    max_v    = i8(resp[3])
    sum_v    = resp[4] | (resp[5] << 8) | (resp[6] << 16)
    p_sum    = resp[7] & 0x1F
    inv_sum  = resp[8] | (resp[9] << 8)
    out_i8   = np.frombuffer(resp[10:], dtype=np.int8)
    return out_i8, so, max_v, sum_v, p_sum, inv_sum

def softmax_ref_fpga(x_i8, sx):
    """Reference Python imitant le RTL : meme LUTs, meme arithmetique."""
    # 1) max
    max_val = int(np.max(x_i8))
    # 2) exp via LUT
    sum_exp = 0
    exp_vals = []
    shx_p5 = sx + 5
    for i in range(K):
        diff = int(x_i8[i]) - max_val   # signed
        if shx_p5 >= 0:
            diff_sc = diff << shx_p5
        else:
            diff_sc = diff >> (-shx_p5)
        idx = diff_sc + 256
        if idx < 0: idx = 0
        if idx > 255: idx = 255
        # LUT[i] = exp(-8 + i/32) * 32768
        exp_q15 = min(32767, int(round(np.exp(-8 + idx / 32.0) * 32768)))
        exp_vals.append(exp_q15)
        sum_exp += exp_q15
    # 3) 1/sum via LUT
    p_sum = max(0, sum_exp.bit_length() - 1)  # bit position
    shift_inv = 8 - p_sum
    if shift_inv >= 0:
        sum_norm = sum_exp << shift_inv
    else:
        sum_norm = sum_exp >> (-shift_inv)
    inv_idx = sum_norm & 0xFF
    inv_q15 = min(32767, int(round(1.0 / ((256 + inv_idx) / 256.0) * 32768)))
    # 4) normalize
    norm_shift = 16 - shift_inv
    out = np.zeros(K, dtype=np.int8)
    for i in range(K):
        prod = exp_vals[i] * inv_q15
        if norm_shift > 0:
            sh = (prod + (1 << (norm_shift - 1))) >> norm_shift
        elif norm_shift < 0:
            sh = prod << (-norm_shift)
        else:
            sh = prod
        out[i] = max(-128, min(127, sh))
    return out, max_val, sum_exp, p_sum, inv_q15

def main():
    ser = serial.Serial(PORT, BAUD, timeout=3.0)
    time.sleep(0.3); ser.reset_input_buffer()
    print(f"Test softmax_op sur FPGA (K={K})\n")

    test_cases = [
        ("tous 0",                    np.zeros(K, dtype=np.int8), -3),
        ("un seul peak (x[5]=64)",   np.array([0]*5 + [64] + [0]*26, dtype=np.int8), -3),
        ("rampe x=[i-16]",            np.arange(-16, 16, dtype=np.int8), -3),
        ("aleatoire N(0,1)*8",        np.clip(np.round(np.random.default_rng(11).normal(0,1,K)*8), -128, 127).astype(np.int8), -3),
        ("2 pics opposes",            np.array([-50]*16 + [50]*16, dtype=np.int8), -3),
    ]

    for name, x_i8, sx in test_cases:
        print(f"=== {name} ===")
        try:
            out_i8, so, max_v, sum_v, p_sum, inv_sum = softmax_fpga(ser, x_i8, sx)
        except Exception as e:
            print(f"  ECHEC : {e}"); continue
        ref_i8, ref_max, ref_sum, ref_p, ref_inv = softmax_ref_fpga(x_i8, sx)

        ok_max = max_v == ref_max
        ok_sum = sum_v == ref_sum
        ok_p   = p_sum == ref_p
        ok_inv = inv_sum == ref_inv
        same   = np.array_equal(out_i8, ref_i8)

        print(f"  shift_out  : {so:+3d}  (attendu -7)  {'OK' if so==-7 else 'FAUX'}")
        print(f"  max        : fpga={max_v:+4d}  ref={ref_max:+4d}   {'OK' if ok_max else 'FAUX'}")
        print(f"  sum_exp    : fpga={sum_v:7d}  ref={ref_sum:7d}   {'OK' if ok_sum else 'FAUX'}")
        print(f"  p_sum      : fpga={p_sum:3d}   ref={ref_p:3d}    {'OK' if ok_p else 'FAUX'}")
        print(f"  inv_sum    : fpga={inv_sum:5d} ref={ref_inv:5d}  {'OK' if ok_inv else 'FAUX'}")
        print(f"  out (vs ref): same={same}")
        # verifier que la somme vraie ~ 1 (proba)
        sum_real_fpga = out_i8.astype(np.int32).sum() / 128.0
        print(f"    sum(softmax_fpga real) = {sum_real_fpga:.4f}  (devrait etre ~1.0)")
        print(f"    ref [:6] : {ref_i8[:6].tolist()}")
        print(f"    fpga[:6] : {out_i8[:6].tolist()}")
        print()
    ser.close()

if __name__ == "__main__":
    main()
