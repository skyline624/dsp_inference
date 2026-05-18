#!/usr/bin/env python3
# Test du module rmsnorm_op avec debug outputs.
# Protocole : 'N' 'N' shift_x shift_w x[64] w[64]
#  -> 'N' 'K' shift_out acc[3] p[1] shift_amt[1signed] raw_inv[2] apply_shift[1signed] out[64]
#
# 75 octets de reponse au total. On compare chaque intermediaire a la reference Python.

import time, struct, sys
import numpy as np
import serial
from v4_quant import to_i8_shift, from_i8_shift
from v4_ops   import rmsnorm_i8_ref

PORT = "COM6"
BAUD = 1_000_000

def i8(b): return b - 256 if b >= 128 else b

def rmsnorm_fpga(ser, x_i8, sx, w_i8, sw):
    pkt = b'NN' + bytes([sx & 0xFF, sw & 0xFF]) + x_i8.tobytes() + w_i8.tobytes()
    ser.write(pkt)
    resp = ser.read(75)
    if len(resp) != 75 or resp[:2] != b'NK':
        raise RuntimeError(f"NN reponse invalide : {len(resp)}/75, magic={resp[:2]!r}")
    so          = i8(resp[2])
    acc         = resp[3] | (resp[4] << 8) | (resp[5] << 16)
    p           = resp[6] & 0x1F
    shift_amt   = resp[7] & 0x3F
    if shift_amt & 0x20: shift_amt -= 64    # signed 6-bit
    raw_inv     = resp[8] | (resp[9] << 8)
    apply_shift = i8(resp[10])
    out_i8      = np.frombuffer(resp[11:], dtype=np.int8)
    return out_i8, so, acc, p, shift_amt, raw_inv, apply_shift

def expected_intermediates(x_i8, sx):
    """Calcule les intermediaires attendus pour comparer au FPGA."""
    acc = int((x_i8.astype(np.int64) ** 2).sum())
    p = int(np.log2(max(acc, 1)))
    shift_amt = 8 - p
    if shift_amt >= 0:
        acc_norm = acc << shift_amt
    else:
        acc_norm = acc >> (-shift_amt)
    index = acc_norm & 0xFF
    # LUT : 1/sqrt((256+i)/256) * 32768, capped a 32767
    lut_val = int(min(round((1.0 / np.sqrt((256 + index) / 256.0)) * 32768), 32767))
    raw_inv_pre = lut_val
    SQRT2_Q15 = 0xB505
    if shift_amt & 1:
        raw_inv = (raw_inv_pre * SQRT2_Q15) >> 15
    else:
        raw_inv = raw_inv_pre
    # apply_shift = (shift_amt >>> 1) - 16
    # Verilog >>> sur signed : ex. shift_amt=-3 -> -2 (arrondi vers -inf), -1 -> -1, 1 -> 0, etc.
    apply_shift = (shift_amt >> 1) - 16   # Python >> sur negatif arrondi vers -inf, idem Verilog $signed >>>
    return acc, p, shift_amt, raw_inv, apply_shift

def main():
    D = 64
    ser = serial.Serial(PORT, BAUD, timeout=3.0)
    time.sleep(0.3); ser.reset_input_buffer()
    print(f"Test rmsnorm_op avec debug (D={D})\n")

    test_cases = [
        ("tous x=1, w=1, shift=0",     np.ones(D, dtype=np.int8),  0, np.ones(D, dtype=np.int8), 0),
        ("tous x=32, w=1, shift=0",    np.full(D, 32, dtype=np.int8), 0, np.ones(D, dtype=np.int8), 0),
        ("tous x=64, w=1, shift=-3",   np.full(D, 64, dtype=np.int8), -3, np.ones(D, dtype=np.int8), 0),
    ]
    # Tests negatifs : confirmer sign-extension
    test_cases.append(("tous x=-1, w=1", np.full(D, -1, dtype=np.int8), 0, np.ones(D, dtype=np.int8), 0))
    test_cases.append(("tous x=-42, w=1", np.full(D, -42, dtype=np.int8), 0, np.ones(D, dtype=np.int8), 0))
    # Tests positionnels : detecter erreur d'adressage
    x_pos = np.zeros(D, dtype=np.int8); x_pos[0] = 100   # seul x[0] non-nul
    test_cases.append(("x[0]=100 seul, w=1", x_pos, 0, np.ones(D, dtype=np.int8), 0))
    x_pos = np.zeros(D, dtype=np.int8); x_pos[63] = 100  # seul x[63] non-nul
    test_cases.append(("x[63]=100 seul, w=1", x_pos, 0, np.ones(D, dtype=np.int8), 0))
    x_pos = np.arange(D, dtype=np.int8)                  # x[i] = i (0..63)
    test_cases.append(("x[i]=i (rampe), w=1", x_pos, 0, np.ones(D, dtype=np.int8), 0))
    rng = np.random.default_rng(42)
    # N(0,1) avec w=1 encode en shift bas (= encodage realiste LLM, max precision)
    x_n, sx_n = to_i8_shift(rng.normal(0,1,D).astype(np.float32))
    w_n, sw_n = to_i8_shift(np.ones(D, dtype=np.float32))   # = ([64]*64, -6)
    test_cases.append(("N(0,1), w=1 (encodage precis)", x_n, sx_n, w_n, sw_n))

    for name, x_i8, sx, w_i8, sw in test_cases:
        print(f"=== {name} ===")
        # attendu (Python)
        e_acc, e_p, e_sa, e_ri, e_as = expected_intermediates(x_i8, sx)
        ref_i8, ref_so = rmsnorm_i8_ref(x_i8, sx, w_i8, sw)
        ref_real = from_i8_shift(ref_i8, ref_so)
        # FPGA
        try:
            out_i8, so, acc, p, sa, ri, ap = rmsnorm_fpga(ser, x_i8, sx, w_i8, sw)
        except Exception as e:
            print(f"  ECHEC TRANSFERT : {e}"); continue
        out_real = from_i8_shift(out_i8, so)

        # comparer chaque intermediaire
        print(f"  acc         : attendu={e_acc:9d}  fpga={acc:9d}   {'OK' if acc==e_acc else 'FAUX'}")
        print(f"  p           : attendu={e_p:+5d}      fpga={p:+5d}      {'OK' if p==e_p else 'FAUX'}")
        print(f"  shift_amt   : attendu={e_sa:+5d}      fpga={sa:+5d}      {'OK' if sa==e_sa else 'FAUX'}")
        print(f"  raw_inv     : attendu={e_ri:5d}      fpga={ri:5d}      {'OK' if ri==e_ri else 'FAUX'}")
        print(f"  apply_shift : attendu={e_as:+5d}      fpga={ap:+5d}      {'OK' if ap==e_as else 'FAUX'}")
        print(f"  shift_out   : ref={ref_so:+3d}        fpga={so:+3d}        {'OK' if ref_so==so else 'FAUX'}")
        err = np.abs(out_real - ref_real)
        print(f"  out (real)  : err_max={err.max():.4f}  ref[:4]={ref_real[:4]}  fpga[:4]={out_real[:4]}")
        print()
    ser.close()

if __name__ == "__main__":
    main()
