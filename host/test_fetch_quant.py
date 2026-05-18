#!/usr/bin/env python3
# test FQ : matmul DMA + requantize int8+shift.
# Protocole : 'F''Q' N(1) sx sw x[64] addr[3]                  (72 oct)
# Reponse   : 'F''Q' shift_total(1) y_int8[N]                  (3+N oct)

import time
import numpy as np
import serial

PORT = "COM6"
BAUD = 1_000_000
K = 64

def i8(b):         return b - 256 if b >= 128 else b
def addr_bytes(a): return bytes([a & 0xFF, (a >> 8) & 0xFF, (a >> 16) & 0xFF])
def n_bytes(n):    return bytes([n & 0xFF, (n >> 8) & 0xFF])

def sd_load(ser, addr, data):
    pkt = b'LL' + addr_bytes(addr) + n_bytes(len(data)) + bytes(data)
    ser.write(pkt)
    return ser.read(2) == b'LK'

def call_fq(ser, N, sx, sw, x_i8, addr_w):
    pkt = b'FQ' + bytes([N, sx & 0xFF, sw & 0xFF]) + x_i8.tobytes() + addr_bytes(addr_w)
    ser.write(pkt)
    resp = ser.read(3 + N)
    if len(resp) != 3 + N or resp[:2] != b'FQ':
        raise RuntimeError(f"FQ resp {len(resp)}/{3+N} magic={resp[:2]!r}")
    shift_total = i8(resp[2])
    y_int8 = np.frombuffer(resp[3:], dtype=np.int8)
    return y_int8, shift_total

def main():
    ser = serial.Serial(PORT, BAUD, timeout=10.0)
    time.sleep(0.3); ser.reset_input_buffer()
    print("Test FQ (matmul + requantize int8+shift)\n")

    rng = np.random.default_rng(123)
    sx, sw = -3, -6

    nok, ntotal = 0, 0
    for N in [4, 8, 12, 15]:
        ntotal += 1
        W = rng.integers(-50, 50, (N, K), dtype=np.int8)
        x = rng.integers(-50, 50, K, dtype=np.int8)
        # reference Python
        y_int32 = (W.astype(np.int64) @ x.astype(np.int64))
        max_abs = int(np.max(np.abs(y_int32)))
        if max_abs == 0:
            shift_used, shift_total_ref = 0, sx + sw
            y_int8_ref = np.zeros(N, dtype=np.int8)
        else:
            lb = max_abs.bit_length() - 1   # leading bit position
            shift_used = max(0, lb - 6)
            shift_total_ref = sx + sw + shift_used
            # requantize : (y + half) >> shift, clip i8
            rounding = (1 << (shift_used - 1)) if shift_used > 0 else 0
            shifted = (y_int32 + rounding) >> shift_used
            y_int8_ref = np.clip(shifted, -128, 127).astype(np.int8)

        # Load W in SDRAM
        sd_load(ser, 0x000300, W.reshape(-1).tobytes())
        # Call FQ
        y_fpga, st_fpga = call_fq(ser, N, sx, sw, x, 0x000300)
        ok_shift = (st_fpga == shift_total_ref)
        ok_y     = np.array_equal(y_fpga, y_int8_ref)
        # Verifie aussi la value reelle
        y_real_ref  = y_int8_ref.astype(np.int64) * (2 ** shift_total_ref)
        y_real_fpga = y_fpga.astype(np.int64) * (2 ** st_fpga)
        print(f"N={N:2d}  shift_used={shift_used}  shift_total : ref={shift_total_ref:+d} fpga={st_fpga:+d}  {'OK' if ok_shift else 'FAUX'}")
        print(f"        y_int8  : ref={y_int8_ref.tolist()}")
        print(f"        y_int8  : fpga={y_fpga.tolist()}  {'OK' if ok_y else 'FAUX'}")
        if not ok_y:
            print(f"        diffs   : {(y_fpga.astype(int) - y_int8_ref.astype(int)).tolist()}")
        if ok_y and ok_shift: nok += 1
    print(f"\n{nok}/{ntotal} configs OK")
    ser.close()

if __name__ == "__main__":
    main()
