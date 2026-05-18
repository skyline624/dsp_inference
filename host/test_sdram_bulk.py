#!/usr/bin/env python3
# Test bulk load/dump SDRAM (LL/CC commands).
# 'LL' addr[3 LE] N[2 LE] data[N] -> 'LK'
# 'CC' addr[3 LE] N[2 LE]         -> 'CK' data[N]

import time, sys
import numpy as np
import serial

PORT = "COM6"
BAUD = 1_000_000

def addr_bytes(addr):
    return bytes([addr & 0xFF, (addr >> 8) & 0xFF, (addr >> 16) & 0xFF])

def n_bytes(n):
    return bytes([n & 0xFF, (n >> 8) & 0xFF])

def sd_load(ser, addr, data):
    pkt = b'LL' + addr_bytes(addr) + n_bytes(len(data)) + bytes(data)
    ser.write(pkt)
    resp = ser.read(2)
    if resp != b'LK':
        raise RuntimeError(f"LL reponse invalide : {resp!r}")

def sd_dump(ser, addr, n):
    pkt = b'CC' + addr_bytes(addr) + n_bytes(n)
    ser.write(pkt)
    resp = ser.read(2 + n)
    if len(resp) != 2 + n or resp[:2] != b'CK':
        raise RuntimeError(f"CC reponse invalide : {len(resp)}/{2+n}  magic={resp[:2]!r}")
    return resp[2:]

def main():
    ser = serial.Serial(PORT, BAUD, timeout=60.0)
    time.sleep(0.5); ser.reset_input_buffer()
    print("Test SDRAM bulk load/dump\n")

    # Test 0 : minimal 16 bytes
    print("--- Test 0: 16 bytes ---")
    addr = 0x008000
    data = bytes([i+1 for i in range(16)])
    sd_load(ser, addr, data)
    got = sd_dump(ser, addr, 16)
    print(f"  sent={list(data)}")
    print(f"  got ={list(got)}")
    print(f"  match: {'OK' if got==data else 'FAUX'}")

    # Test 1 : 256-byte chunk avec pattern
    print("--- Test 1: 256 bytes pattern ---")
    addr = 0x010000
    data = bytes([(i * 13 + 7) & 0xFF for i in range(256)])
    t0 = time.time()
    sd_load(ser, addr, data)
    t1 = time.time()
    got = sd_dump(ser, addr, 256)
    t2 = time.time()
    ok = got == data
    print(f"  load 256B : {(t1-t0)*1000:.1f} ms")
    print(f"  dump 256B : {(t2-t1)*1000:.1f} ms")
    print(f"  match     : {'OK' if ok else 'FAUX'}")
    if not ok:
        ndiff = sum(1 for a, b in zip(data, got) if a != b)
        print(f"  diffs     : {ndiff}/256")

    # Test 2 : 4 KB chunk
    print("\n--- Test 2: 4 KB pattern ---")
    addr = 0x020000
    data = bytes([(i ^ (i >> 3)) & 0xFF for i in range(4096)])
    t0 = time.time()
    sd_load(ser, addr, data)
    t1 = time.time()
    got = sd_dump(ser, addr, 4096)
    t2 = time.time()
    ok = got == data
    print(f"  load 4 KB : {(t1-t0)*1000:.1f} ms  ({4096*8/(t1-t0)/1e6:.2f} Mbps)")
    print(f"  dump 4 KB : {(t2-t1)*1000:.1f} ms")
    print(f"  match     : {'OK' if ok else 'FAUX'}")

    # Test 3 : 32 KB (~taille embedding stories260K)
    print("\n--- Test 3: 32 KB (~ embedding stories260K) ---")
    addr = 0x100000
    rng = np.random.default_rng(7)
    data = bytes(rng.integers(0, 256, 32768, dtype=np.uint8).tolist())
    t0 = time.time()
    sd_load(ser, addr, data)
    t1 = time.time()
    got = sd_dump(ser, addr, 32768)
    t2 = time.time()
    ok = got == data
    print(f"  load 32 KB : {(t1-t0)*1000:.0f} ms")
    print(f"  dump 32 KB : {(t2-t1)*1000:.0f} ms")
    print(f"  match     : {'OK' if ok else 'FAUX'}")

    ser.close()

if __name__ == "__main__":
    main()
