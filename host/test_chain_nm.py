#!/usr/bin/env python3
# test CN : chain rmsnorm + matmul + requantize, tout sur FPGA.
# Protocole : 'C''N' sx sw_rms sw_mm x[64] N(1) addr_rms[3] addr_mm[3]   (76 oct)
# Reponse   : 'C''N' shift_total(1) y_int8[N]                            (3+N oct)
#
# reference Python : chaine FN then FQ sur les memes data. compare.

import time
import numpy as np
import serial
from test_sdram_diag import sd_load, call_fn, call_fq

PORT = "COM6"
BAUD = 1_000_000

def addr_bytes(a): return bytes([a & 0xFF, (a >> 8) & 0xFF, (a >> 16) & 0xFF])
def i8(b): return b - 256 if b >= 128 else b

def call_cn(ser, x_i8, sx, sw_rms, sw_mm, N, addr_rms, addr_mm):
    pkt = (b'CN' + bytes([sx & 0xFF, sw_rms & 0xFF, sw_mm & 0xFF])
           + x_i8.tobytes() + bytes([N])
           + addr_bytes(addr_rms) + addr_bytes(addr_mm))
    assert len(pkt) == 76, len(pkt)
    ser.write(pkt)
    resp = ser.read(3 + N)
    if len(resp) != 3 + N or resp[:2] != b'CN':
        raise RuntimeError(f"CN reponse {len(resp)}/{3+N} magic={resp[:2]!r}")
    so = i8(resp[2])
    y = np.frombuffer(resp[3:], dtype=np.int8)
    return y, so

def main():
    ser = serial.Serial(PORT, BAUD, timeout=8.0)
    time.sleep(0.5); ser.reset_input_buffer()
    print("Test CN (chain rmsnorm + matmul + requantize)\n")

    rng = np.random.default_rng(123)
    sx, sw_rms, sw_mm = -3, -6, -6
    N = 8

    # data
    x_i8   = rng.integers(-30, 30, 64, dtype=np.int8)
    rms_w  = np.full(64, 64, dtype=np.int8)        # representera 1.0 a shift_w=-6
    mm_W   = rng.integers(-50, 50, (N, 64), dtype=np.int8)

    addr_rms = 0x000400
    addr_mm  = 0x000800

    # reference Python : chaine FN then FQ via UART (2 commandes)
    print("Reference : chaine FN + FQ (2 UART calls)...")
    sd_load(ser, addr_rms, rms_w.tobytes())
    sd_load(ser, addr_mm,  mm_W.reshape(-1).tobytes())
    y_norm_ref, sh_rms_ref = call_fn(ser, x_i8, sx, sw_rms, addr_rms)
    y_norm_arr = np.frombuffer(y_norm_ref, dtype=np.int8)
    y_final_ref, st_ref = call_fq(ser, N, sh_rms_ref, sw_mm, y_norm_arr, addr_mm)
    print(f"  y_norm shift = {sh_rms_ref:+d}")
    print(f"  y_final ref  = {y_final_ref.tolist()}  shift_total = {st_ref:+d}\n")

    # test CN
    print("CN chained on FPGA (1 UART call)...")
    y_cn, st_cn = call_cn(ser, x_i8, sx, sw_rms, sw_mm, N, addr_rms, addr_mm)
    print(f"  y_final fpga = {y_cn.tolist()}  shift_total = {st_cn:+d}\n")

    # compare
    match_y = np.array_equal(y_cn, y_final_ref)
    match_s = st_cn == st_ref
    print(f"shift_total match : {'OK' if match_s else 'FAUX'}")
    print(f"y_int8     match : {'OK' if match_y else 'FAUX'}")
    if not match_y:
        diffs = (y_cn.astype(int) - y_final_ref.astype(int)).tolist()
        print(f"  diffs : {diffs}")

    ser.close()

if __name__ == "__main__":
    main()
