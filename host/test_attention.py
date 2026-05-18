#!/usr/bin/env python3
# test du module attention_head_op sur FPGA (HS=8, T_MAX=8).
# Protocole : 'A''A' sq sk sv T Q[8] K[T*8] V[T*8]
# Response  : 'A''K' shift_out score_last[2 LE signed] max[1 signed] sum[2 LE] inv_sum[2 LE] out[8]
# Total reponse = 2+1+2+1+2+2+8 = 18 oct

import time, sys
import numpy as np
import serial

PORT = "COM6"
BAUD = 1_000_000
HS   = 8

def i8(b):  return b - 256 if b >= 128 else b
def i16(b): v = b[0] | (b[1]<<8); return v - 65536 if v >= 32768 else v
def u16(b): return b[0] | (b[1]<<8)

def attn_fpga(ser, Q_i8, K_i8, V_i8, sq, sk, sv, T):
    assert Q_i8.shape == (HS,) and K_i8.shape == (T, HS) and V_i8.shape == (T, HS)
    pkt = b'AA' + bytes([sq & 0xFF, sk & 0xFF, sv & 0xFF, T])
    pkt += Q_i8.tobytes()
    pkt += K_i8.tobytes()
    pkt += V_i8.tobytes()
    ser.write(pkt)
    resp = ser.read(18)
    if len(resp) != 18 or resp[:2] != b'AK':
        raise RuntimeError(f"AA reponse {len(resp)}/18, magic={resp[:2]!r}")
    so       = i8(resp[2])
    score_last = i16(resp[3:5])
    max_score  = i8(resp[5])
    exp_sum    = u16(resp[6:8])
    inv_sum    = u16(resp[8:10])
    out_i8     = np.frombuffer(resp[10:], dtype=np.int8)
    return out_i8, so, score_last, max_score, exp_sum, inv_sum

def attn_ref_fpga(Q_i8, K_i8, V_i8, sq, sk, sv, T):
    """Reference Python imitant le RTL : meme arithm, meme LUTs."""
    # 1) scores = Q . K[t], stocke clamp signed 16-bit
    scores = np.zeros(T, dtype=np.int64)
    for t in range(T):
        for i in range(HS):
            scores[t] += int(Q_i8[i]) * int(K_i8[t, i])
    # clip a int16
    scores = np.clip(scores, -32768, 32767).astype(np.int16)
    # 2) max sur scores[15:8] (proxy int8)
    scores_i8_proxy = (scores >> 8).astype(np.int8)
    max_score = int(scores_i8_proxy.max())
    # 3) exp via LUT
    sum_exp = 0
    exp_vals = []
    for t in range(T):
        s_curr = int(scores_i8_proxy[t]) << 8     # i8 -> i16
        diff = s_curr - (max_score << 8)
        diff_sc = diff >> 10
        idx = diff_sc + 256
        idx = max(0, min(255, idx))
        ev = min(32767, int(round(np.exp(-8 + idx / 32.0) * 32768)))
        exp_vals.append(ev)
        sum_exp += ev
    # 4) 1/sum
    p_sum = max(0, sum_exp.bit_length() - 1)
    shift_inv = 8 - p_sum
    if shift_inv >= 0: sum_norm = sum_exp << shift_inv
    else:              sum_norm = sum_exp >> (-shift_inv)
    inv_idx = sum_norm & 0xFF
    inv_q15 = min(32767, int(round(1.0 / ((256 + inv_idx) / 256.0) * 32768)))
    # 5) attn[t] = exp[t] * inv_sum >> norm_shift, ~Q15
    norm_shift = 15 - shift_inv
    attn = []
    for ev in exp_vals:
        prod = ev * inv_q15
        if norm_shift > 0:
            v = (prod + (1 << (norm_shift - 1))) >> norm_shift
        elif norm_shift < 0:
            v = prod << (-norm_shift)
        else:
            v = prod
        attn.append(v & 0xFFFF)
    # 6) out[d] = sum_t attn[t] * V[t,d], shift par 8 + clip int8
    out = np.zeros(HS, dtype=np.int8)
    for d in range(HS):
        acc = 0
        for t in range(T):
            acc += attn[t] * int(V_i8[t, d])
        sh = (acc + 128) >> 8
        out[d] = max(-128, min(127, sh))
    return out, scores[T-1], max_score, sum_exp & 0xFFFF, inv_q15

def main():
    ser = serial.Serial(PORT, BAUD, timeout=3.0)
    time.sleep(0.3); ser.reset_input_buffer()
    print(f"Test attention_head_op sur FPGA (HS={HS})\n")

    T = 4

    # test cases
    cases = []
    # 1. Identite : Q nul -> attn uniforme -> out = mean(V)
    Q = np.zeros(HS, dtype=np.int8)
    K = np.random.default_rng(1).integers(-10, 10, (T, HS), dtype=np.int8)
    V = np.array([[1]*HS, [2]*HS, [3]*HS, [4]*HS], dtype=np.int8)
    cases.append(("Q=0 -> attn uniforme, out~mean(V)", Q, K, V))

    # 2. Q aligne with K[2] (forte attention sur t=2)
    K2 = np.zeros((T, HS), dtype=np.int8)
    K2[2] = np.array([60, 60, 60, 60, 60, 60, 60, 60], dtype=np.int8)
    Q2 = np.array([60, 60, 60, 60, 60, 60, 60, 60], dtype=np.int8)
    V2 = np.array([[10]*HS, [20]*HS, [99]*HS, [40]*HS], dtype=np.int8)
    cases.append(("Q ~ K[2] -> attn pic sur t=2, out~V[2]=99", Q2, K2, V2))

    # 3. Random
    rng = np.random.default_rng(42)
    Q3 = rng.integers(-30, 30, HS, dtype=np.int8)
    K3 = rng.integers(-30, 30, (T, HS), dtype=np.int8)
    V3 = rng.integers(-50, 50, (T, HS), dtype=np.int8)
    cases.append(("Random", Q3, K3, V3))

    for name, Q, K, V in cases:
        print(f"=== {name} ===")
        try:
            out_i8, so, score_last, max_s, sum_e, inv_s = attn_fpga(ser, Q, K, V, -3, -3, -3, T)
        except Exception as e:
            print(f"  ECHEC : {e}"); continue
        ref_i8, ref_score_last, ref_max, ref_sum, ref_inv = attn_ref_fpga(Q, K, V, -3, -3, -3, T)
        ok_sl  = score_last == ref_score_last
        ok_mx  = max_s == ref_max
        ok_se  = sum_e == ref_sum
        ok_iv  = inv_s == ref_inv
        same   = np.array_equal(out_i8, ref_i8)
        print(f"  score_last : fpga={score_last:+6d}  ref={ref_score_last:+6d}  {'OK' if ok_sl else 'FAUX'}")
        print(f"  max_score  : fpga={max_s:+4d}      ref={ref_max:+4d}      {'OK' if ok_mx else 'FAUX'}")
        print(f"  exp_sum    : fpga={sum_e:5d}     ref={ref_sum:5d}     {'OK' if ok_se else 'FAUX'}")
        print(f"  inv_sum    : fpga={inv_s:5d}     ref={ref_inv:5d}     {'OK' if ok_iv else 'FAUX'}")
        print(f"  out same   : {same}")
        print(f"    ref  : {ref_i8.tolist()}")
        print(f"    fpga : {out_i8.tolist()}")
        print()
    ser.close()

if __name__ == "__main__":
    main()
