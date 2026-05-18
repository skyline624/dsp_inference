#!/usr/bin/env python3
# Test du module rope_op sur FPGA (HS=8, HALF=4).
# Protocole : 'R''R' shift_x x[8] cos[4]i16LE sin[4]i16LE  (27 oct)
# Response  : 'R''K' shift_out dbg_new_real[4 LE] dbg_new_imag[4 LE] out[8]  (19 oct)

import time, sys
import numpy as np
import serial

PORT = "COM6"
BAUD = 1_000_000
HS   = 8
HALF = 4

def i8(b): return b - 256 if b >= 128 else b
def i32_le(buf):
    v = buf[0] | (buf[1]<<8) | (buf[2]<<16) | (buf[3]<<24)
    return v - (1<<32) if v >= (1<<31) else v

def to_q15(x_float):
    return int(max(-32768, min(32767, round(x_float * 32768))))

def rope_fpga(ser, x_i8, sx, cos_arr_f, sin_arr_f):
    cos_q15 = [to_q15(c) for c in cos_arr_f]
    sin_q15 = [to_q15(s) for s in sin_arr_f]
    cos_bytes = b''.join(int.to_bytes(c & 0xFFFF, 2, 'little') for c in cos_q15)
    sin_bytes = b''.join(int.to_bytes(s & 0xFFFF, 2, 'little') for s in sin_q15)
    pkt = b'RR' + bytes([sx & 0xFF]) + x_i8.tobytes() + cos_bytes + sin_bytes
    assert len(pkt) == 27, len(pkt)
    ser.write(pkt)
    resp = ser.read(19)
    if len(resp) != 19 or resp[:2] != b'RK':
        raise RuntimeError(f"RR reponse {len(resp)}/19, magic={resp[:2]!r}")
    so = i8(resp[2])
    dbg_real = i32_le(resp[3:7])
    dbg_imag = i32_le(resp[7:11])
    out_i8 = np.frombuffer(resp[11:], dtype=np.int8)
    return out_i8, so, dbg_real, dbg_imag

def rope_ref(x_i8, sx, cos_arr_f, sin_arr_f):
    """Reference Python : meme conventions int + arrondi Q15."""
    out = np.zeros(HS, dtype=np.int8)
    for i in range(HALF):
        xr = int(x_i8[2*i])
        xi = int(x_i8[2*i + 1])
        c_q = to_q15(cos_arr_f[i])
        s_q = to_q15(sin_arr_f[i])
        nr = xr * c_q - xi * s_q
        ni = xr * s_q + xi * c_q
        # arrondi (+16384) puis >> 15
        nr_r = (nr + 16384) >> 15
        ni_r = (ni + 16384) >> 15
        out[2*i]     = max(-128, min(127, nr_r))
        out[2*i + 1] = max(-128, min(127, ni_r))
    return out

def main():
    ser = serial.Serial(PORT, BAUD, timeout=3.0)
    time.sleep(0.3); ser.reset_input_buffer()
    print(f"Test rope_op sur FPGA (HS={HS})\n")

    test_cases = [
        # (nom, x_i8, sx, cos_floats, sin_floats)
        ("identite (theta=0)", np.array([10, 20, -30, 40, 50, -60, 70, -80], dtype=np.int8),
         -3, [1.0]*HALF, [0.0]*HALF),
        ("rotation pi/2 (cos=0, sin=1)", np.array([10, 20, -30, 40, 50, -60, 70, -80], dtype=np.int8),
         -3, [0.0]*HALF, [1.0]*HALF),
        ("rotation pi/4", np.array([100, 0, 0, 100, -100, 0, 0, -100], dtype=np.int8),
         -3, [np.sqrt(0.5)]*HALF, [np.sqrt(0.5)]*HALF),
        ("frequences LLM-like", np.array([20, 40, 60, -20, -40, -60, 80, -80], dtype=np.int8),
         -3, [np.cos(t) for t in [0.0, 0.3, 0.6, 0.9]],
              [np.sin(t) for t in [0.0, 0.3, 0.6, 0.9]]),
    ]

    for name, x_i8, sx, cos_f, sin_f in test_cases:
        print(f"=== {name} ===")
        try:
            out_i8, so, dbg_r, dbg_i = rope_fpga(ser, x_i8, sx, cos_f, sin_f)
        except Exception as e:
            print(f"  ECHEC : {e}"); continue
        ref_i8 = rope_ref(x_i8, sx, cos_f, sin_f)

        # Verifier le DERNIER pair (i=3) : dbg correspond a new_real_raw et new_imag_raw avant clip
        i = HALF - 1
        c_q = to_q15(cos_f[i]); s_q = to_q15(sin_f[i])
        xr = int(x_i8[2*i]); xi = int(x_i8[2*i + 1])
        exp_real = xr * c_q - xi * s_q
        exp_imag = xr * s_q + xi * c_q
        ok_real = dbg_r == exp_real
        ok_imag = dbg_i == exp_imag
        ok_out  = np.array_equal(out_i8, ref_i8)

        print(f"  shift_out         : {so:+3d}  (=shift_x={sx:+3d})  {'OK' if so==sx else 'FAUX'}")
        print(f"  new_real_raw[3]   : fpga={dbg_r:+10d}  attendu={exp_real:+10d}  {'OK' if ok_real else 'FAUX'}")
        print(f"  new_imag_raw[3]   : fpga={dbg_i:+10d}  attendu={exp_imag:+10d}  {'OK' if ok_imag else 'FAUX'}")
        print(f"  out (vs ref)      : same={ok_out}")
        print(f"    ref  : {ref_i8.tolist()}")
        print(f"    fpga : {out_i8.tolist()}")
        print()
    ser.close()

if __name__ == "__main__":
    main()
