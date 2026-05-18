#!/usr/bin/env python3
# Diagnostic SDRAM : compare reliability FN (1 fetch de 64 bytes) vs FQ (N fetches)
# vs LL+CC roundtrip (pas de fetcher).

import time, sys
import numpy as np
import serial

PORT = "COM6"
BAUD = 1_000_000

def addr_bytes(a): return bytes([a & 0xFF, (a >> 8) & 0xFF, (a >> 16) & 0xFF])
def n_bytes(n):    return bytes([n & 0xFF, (n >> 8) & 0xFF])
def i8(b):         return b - 256 if b >= 128 else b

def sd_load(ser, addr, data):
    pkt = b'LL' + addr_bytes(addr) + n_bytes(len(data)) + bytes(data)
    ser.write(pkt)
    return ser.read(2) == b'LK'

def sd_dump(ser, addr, n):
    pkt = b'CC' + addr_bytes(addr) + n_bytes(n)
    ser.write(pkt)
    resp = ser.read(2 + n)
    if resp[:2] != b'CK' or len(resp) != 2+n: return None
    return resp[2:]

def call_fn(ser, x_i8, sx, sw, addr):
    pkt = b'FN' + bytes([sx & 0xFF, sw & 0xFF]) + x_i8.tobytes() + addr_bytes(addr)
    ser.write(pkt)
    resp = ser.read(75)
    if resp[:2] != b'FK' or len(resp) != 75: return None, None
    return resp[11:], i8(resp[2])

def call_fq(ser, N, sx, sw, x_i8, addr):
    pkt = b'FQ' + bytes([N, sx & 0xFF, sw & 0xFF]) + x_i8.tobytes() + addr_bytes(addr)
    ser.write(pkt)
    resp = ser.read(3 + N)
    if resp[:2] != b'FQ' or len(resp) != 3+N: return None, None
    return np.frombuffer(resp[3:], dtype=np.int8), i8(resp[2])

def main():
    ser = serial.Serial(PORT, BAUD, timeout=5.0)
    time.sleep(0.3); ser.reset_input_buffer()
    print("Diagnostic SDRAM : 20 runs de chaque type\n")

    rng = np.random.default_rng(0)

    def sd_read_byte(addr):
        ser.write(b'BB' + addr_bytes(addr))
        r = ser.read(3)
        return r[2] if (len(r) == 3 and r[:2] == b'BK') else 0
    def sd_dump_slow(addr, n):
        return bytes(sd_read_byte(addr+i) for i in range(n))

    # test 1a : LL + CC (bulk dump)
    print("--- Test 1a: LL load + CC bulk dump (64 bytes) ---")
    nok = 0
    for run in range(20):
        data = rng.bytes(64)
        sd_load(ser, 0x100000, data)
        got = sd_dump(ser, 0x100000, 64)
        if got == data: nok += 1
    print(f"  {nok}/20 OK\n")

    # test 1b : LL + BB (byte-by-byte dump, robust)
    print("--- Test 1b: LL load + BB byte-par-byte dump (64 bytes) ---")
    nok = 0
    for run in range(20):
        data = rng.bytes(64)
        sd_load(ser, 0x100100, data)
        got = sd_dump_slow(0x100100, 64)
        if got == data: nok += 1
    print(f"  {nok}/20 OK\n")

    # test 1c : WW (byte-by-byte write, robust) + BB
    print("--- Test 1c: WW byte-par-byte + BB byte-par-byte (16 bytes) ---")
    nok = 0
    for run in range(20):
        data = rng.bytes(16)
        for i, b in enumerate(data):
            ser.write(b'WW' + addr_bytes(0x100300 + i) + bytes([b]))
            ser.read(2)  # ack
        got = sd_dump_slow(0x100300, 16)
        if got == data: nok += 1
    print(f"  {nok}/20 OK\n")

    # test 2 : FN (rmsnorm with fetch w)
    # On compare le shift_out qui est deterministe a partir de x
    print("--- Test 2: FN (fetch 64 bytes + rmsnorm) ---")
    x_i8 = rng.integers(-30, 30, 64, dtype=np.int8)
    w_i8 = rng.integers(60, 70, 64, dtype=np.int8)
    sd_load(ser, 0x100200, w_i8.tobytes())
    # Run NN classique pour reference
    pkt = b'NN' + bytes([-3 & 0xFF, -6 & 0xFF]) + x_i8.tobytes() + w_i8.tobytes()
    ser.write(pkt)
    ref = ser.read(75)
    ref_out = ref[11:]
    nok = 0
    for run in range(20):
        out, so = call_fn(ser, x_i8, -3, -6, 0x100200)
        if out is not None and out == ref_out: nok += 1
    print(f"  {nok}/20 OK\n")

    # test 3 : FQ N=8
    print("--- Test 3: FQ N=8 (fetch 8*64 bytes + matmul) ---")
    W = rng.integers(-50, 50, (8, 64), dtype=np.int8)
    x = rng.integers(-50, 50, 64, dtype=np.int8)
    y_int32 = (W.astype(np.int64) @ x.astype(np.int64))
    max_abs = int(np.max(np.abs(y_int32)))
    lb = max_abs.bit_length() - 1
    sh_used = max(0, lb - 6)
    rounding = (1 << (sh_used - 1)) if sh_used > 0 else 0
    ref_int8 = np.clip((y_int32 + rounding) >> sh_used, -128, 127).astype(np.int8)
    sd_load(ser, 0x100400, W.reshape(-1).tobytes())
    nok = 0
    for run in range(20):
        out, st = call_fq(ser, 8, -3, -6, x, 0x100400)
        if out is not None and np.array_equal(out, ref_int8): nok += 1
    print(f"  {nok}/20 OK\n")

    ser.close()

if __name__ == "__main__":
    main()
