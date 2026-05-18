#!/usr/bin/env python3
# test multi-head attention (H=8, KH=4 GQA, HS=8) sur FPGA.
# Protocole : 'M''M' sq sk sv T(1) Q[64] K[T*32] V[T*32]
# Reponse   : 'M''K' shift_out out[64]  (67 oct)

import time, sys
import numpy as np
import serial

PORT = "COM6"
BAUD = 1_000_000
H, KH, HS = 8, 4, 8
N_REP = H // KH

def i8(b): return b - 256 if b >= 128 else b

def multi_attn_fpga(ser, Q_i8, K_i8, V_i8, sq, sk, sv, T):
    # Q : [H*HS=64]
    # K, V : [T, KH, HS] flatten -> T*KH*HS = T*32
    assert Q_i8.shape == (H*HS,) and K_i8.shape == (T, KH, HS) and V_i8.shape == (T, KH, HS)
    pkt = b'MM' + bytes([sq & 0xFF, sk & 0xFF, sv & 0xFF, T])
    pkt += Q_i8.tobytes()
    pkt += K_i8.reshape(-1).tobytes()
    pkt += V_i8.reshape(-1).tobytes()
    ser.write(pkt)
    resp = ser.read(67)
    if len(resp) != 67 or resp[:2] != b'MK':
        raise RuntimeError(f"MM reponse {len(resp)}/67, magic={resp[:2]!r}")
    so = i8(resp[2])
    out = np.frombuffer(resp[3:], dtype=np.int8)
    return out, so

def single_head_ref(Q_h, K_h, V_h, T):
    """Calcule l'attention pour UNE seule tete (imitant attention_head_op)."""
    # scores = Q . K[t], clip int16, proxy >>8 pour max
    scores = np.zeros(T, dtype=np.int64)
    for t in range(T):
        for i in range(HS):
            scores[t] += int(Q_h[i]) * int(K_h[t, i])
    scores = np.clip(scores, -32768, 32767).astype(np.int16)
    scores_i8 = (scores >> 8).astype(np.int8)
    max_s = int(scores_i8.max())
    exp_vals = []
    sum_e = 0
    for t in range(T):
        diff16 = int(scores[t]) - (max_s << 8)
        diff_sc = diff16 >> 10   # arith shift
        idx = max(0, min(255, diff_sc + 256))
        ev = min(32767, int(round(np.exp(-8 + idx / 32.0) * 32768)))
        exp_vals.append(ev)
        sum_e += ev
    # 1/sum LUT
    p_sum = max(0, sum_e.bit_length() - 1)
    shift_inv = 8 - p_sum
    sum_norm = (sum_e << shift_inv) if shift_inv >= 0 else (sum_e >> (-shift_inv))
    inv_idx = sum_norm & 0xFF
    inv_q15 = min(32767, int(round(1.0 / ((256 + inv_idx) / 256.0) * 32768)))
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
    out = np.zeros(HS, dtype=np.int8)
    for d in range(HS):
        acc = 0
        for t in range(T):
            acc += attn[t] * int(V_h[t, d])
        sh = (acc + 128) >> 8
        out[d] = max(-128, min(127, sh))
    return out

def multi_attn_ref(Q_i8, K_i8, V_i8, T):
    """Reference Python multi-head GQA."""
    out = np.zeros(H * HS, dtype=np.int8)
    for h in range(H):
        kvh = h // N_REP
        Q_h = Q_i8[h*HS : (h+1)*HS]
        K_h = K_i8[:, kvh, :]   # [T, HS]
        V_h = V_i8[:, kvh, :]
        out_h = single_head_ref(Q_h, K_h, V_h, T)
        out[h*HS : (h+1)*HS] = out_h
    return out

def main():
    ser = serial.Serial(PORT, BAUD, timeout=3.0)
    time.sleep(0.3); ser.reset_input_buffer()
    print(f"Test multi-head attention (H={H}, KH={KH} GQA, HS={HS})\n")

    T = 4
    rng = np.random.default_rng(123)

    cases = []
    # test 1 : Q=0 -> attn uniforme partout, out = mean(V) per head
    Q1 = np.zeros(H*HS, dtype=np.int8)
    K1 = rng.integers(-20, 20, (T, KH, HS), dtype=np.int8)
    V1 = np.array([[[1]*HS, [2]*HS, [3]*HS, [4]*HS],  # t=0
                   [[5]*HS, [6]*HS, [7]*HS, [8]*HS],
                   [[9]*HS, [10]*HS, [11]*HS, [12]*HS],
                   [[13]*HS, [14]*HS, [15]*HS, [16]*HS]], dtype=np.int8)
    cases.append(("Q=0, V variable -> mean per kv-head", Q1, K1, V1))

    # test 2 : random tout
    Q2 = rng.integers(-40, 40, H*HS, dtype=np.int8)
    K2 = rng.integers(-40, 40, (T, KH, HS), dtype=np.int8)
    V2 = rng.integers(-60, 60, (T, KH, HS), dtype=np.int8)
    cases.append(("Random complet", Q2, K2, V2))

    for name, Q, K, V in cases:
        print(f"=== {name} ===")
        try:
            out_fpga, so = multi_attn_fpga(ser, Q, K, V, -3, -3, -3, T)
        except Exception as e:
            print(f"  ECHEC : {e}"); continue
        out_ref = multi_attn_ref(Q, K, V, T)
        print(f"  shift_out : {so:+3d}  (attendu -10 = -3 - 7)")
        same = np.array_equal(out_fpga, out_ref)
        if not same:
            diff = np.abs(out_fpga.astype(int) - out_ref.astype(int))
            print(f"  out same  : FAUX  max_diff_int8 = {diff.max()}")
            for h in range(H):
                rh = out_ref[h*HS:(h+1)*HS]
                fh = out_fpga[h*HS:(h+1)*HS]
                marker = "" if np.array_equal(rh, fh) else " <-- diff"
                print(f"    head {h} : ref={rh.tolist()}  fpga={fh.tolist()}{marker}")
        else:
            print(f"  out same  : TRUE")
            for h in [0, 3, 7]:
                rh = out_ref[h*HS:(h+1)*HS]
                print(f"    head {h} : {rh.tolist()}")
        print()
    ser.close()

if __name__ == "__main__":
    main()
