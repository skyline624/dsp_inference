#!/usr/bin/env python3
# Test CS : chain rmsnorm + matmul + requantize + SiLU.
# Protocole RX comme CN : 'C''S' sx sw_rms sw_mm x[64] N addr_rms[3] addr_mm[3]
# Reponse : 'C''S' shift_silu y_int8[N]
#
# Reference Python : chain FN + FQ + (silu via SS) sur les memes data.

import time
import numpy as np
import serial
from test_sdram_diag import sd_load, call_fn, call_fq

PORT = "COM6"
BAUD = 1_000_000

def addr_bytes(a): return bytes([a & 0xFF, (a >> 8) & 0xFF, (a >> 16) & 0xFF])
def i8(b): return b - 256 if b >= 128 else b

def call_cs(ser, x_i8, sx, sw_rms, sw_mm, N, addr_rms, addr_mm):
    pkt = (b'CS' + bytes([sx & 0xFF, sw_rms & 0xFF, sw_mm & 0xFF])
           + x_i8.tobytes() + bytes([N])
           + addr_bytes(addr_rms) + addr_bytes(addr_mm))
    assert len(pkt) == 76, len(pkt)
    ser.write(pkt)
    resp = ser.read(3 + N)
    if len(resp) != 3 + N or resp[:2] != b'CS':
        raise RuntimeError(f"CS reponse {len(resp)}/{3+N} magic={resp[:2]!r}")
    so = i8(resp[2])
    y = np.frombuffer(resp[3:], dtype=np.int8)
    return y, so

def call_ss(ser, x_i8, sx, K):
    """Run SS (SiLU standalone) sur K elements. Returns (out, shift_out)."""
    # SS takes x[64] always
    full_x = np.zeros(64, dtype=np.int8); full_x[:K] = x_i8
    pkt = b'SS' + bytes([sx & 0xFF]) + full_x.tobytes()
    ser.write(pkt)
    resp = ser.read(70)
    assert resp[:2] == b'SK', resp[:2]
    so = i8(resp[2])
    return np.frombuffer(resp[6:6+K], dtype=np.int8), so

def main():
    ser = serial.Serial(PORT, BAUD, timeout=8.0)
    time.sleep(0.5); ser.reset_input_buffer()
    print("Test CS (chain rmsnorm + matmul + silu)\n")

    rng = np.random.default_rng(7)
    sx, sw_rms, sw_mm = -3, -6, -6
    N = 8

    x_i8 = rng.integers(-30, 30, 64, dtype=np.int8)
    rms_w = np.full(64, 64, dtype=np.int8)
    mm_W = rng.integers(-50, 50, (N, 64), dtype=np.int8)
    addr_rms, addr_mm = 0x000400, 0x000800

    print("Ref : FN -> FQ -> SS (3 UART calls)")
    sd_load(ser, addr_rms, rms_w.tobytes())
    sd_load(ser, addr_mm,  mm_W.reshape(-1).tobytes())
    yn_ref, sh_rms = call_fn(ser, x_i8, sx, sw_rms, addr_rms)
    yn_arr = np.frombuffer(yn_ref, dtype=np.int8)
    yf_ref, sh_mm = call_fq(ser, N, sh_rms, sw_mm, yn_arr, addr_mm)
    print(f"  after matmul shift = {sh_mm:+d}, y = {yf_ref.tolist()}")
    ys_ref, sh_silu = call_ss(ser, yf_ref, sh_mm, N)
    print(f"  after silu  shift = {sh_silu:+d}, y = {ys_ref.tolist()}\n")

    print("CS chained on FPGA (1 UART call)")
    y_cs, st_cs = call_cs(ser, x_i8, sx, sw_rms, sw_mm, N, addr_rms, addr_mm)
    print(f"  y_silu fpga = {y_cs.tolist()}  shift = {st_cs:+d}\n")

    match_y = np.array_equal(y_cs, ys_ref)
    match_s = st_cs == sh_silu
    print(f"shift match : {'OK' if match_s else 'FAUX'}")
    print(f"y     match : {'OK' if match_y else 'FAUX'}")
    if not match_y:
        diffs = (y_cs.astype(int) - ys_ref.astype(int)).tolist()
        print(f"  diffs : {diffs}")

    ser.close()

if __name__ == "__main__":
    main()
