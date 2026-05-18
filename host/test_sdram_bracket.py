#!/usr/bin/env python3
# Bracketing : trouver a quelle taille la dump CC start a echouer.

import time, sys
import serial

PORT = "COM6"
BAUD = 1_000_000

def addr_bytes(a): return bytes([a & 0xFF, (a >> 8) & 0xFF, (a >> 16) & 0xFF])
def n_bytes(n):    return bytes([n & 0xFF, (n >> 8) & 0xFF])

def sd_load(ser, addr, data):
    pkt = b'LL' + addr_bytes(addr) + n_bytes(len(data)) + bytes(data)
    ser.write(pkt)
    resp = ser.read(2)
    return resp == b'LK'

def sd_dump(ser, addr, n):
    pkt = b'CC' + addr_bytes(addr) + n_bytes(n)
    ser.write(pkt)
    resp = ser.read(2 + n)
    return resp

def main():
    ser = serial.Serial(PORT, BAUD, timeout=30.0)
    time.sleep(0.5); ser.reset_input_buffer()

    # test multiple tailles a base addr distincte
    sizes = [4096, 6000, 8000, 10000, 12000, 16000, 17000, 17500, 17800, 17900, 18000, 20000]
    addr = 0x200000
    pattern = bytes([(i * 13 + 7) & 0xFF for i in range(20000)])
    # Charge 20 KB une fois
    print("Load 20 KB...")
    if not sd_load(ser, addr, pattern):
        print("Load failed"); ser.close(); return
    print("Load OK\n")

    for n in sizes:
        t0 = time.time()
        resp = sd_dump(ser, addr, n)
        dt = time.time() - t0
        expected = 2 + n
        got = len(resp)
        if got == expected and resp[:2] == b'CK':
            data = resp[2:]
            ok = data == pattern[:n]
            ndiff = sum(1 for a,b in zip(data, pattern[:n]) if a != b) if not ok else 0
            print(f"  n={n:5d}  resp={got:6d}/{expected:6d}  dt={dt*1000:6.0f}ms  match={'OK' if ok else f'FAUX {ndiff} diffs'}")
        else:
            print(f"  n={n:5d}  resp={got:6d}/{expected:6d}  dt={dt*1000:6.0f}ms  STUCK ({got-2} data bytes recus)")
            # tail clean for next test
            time.sleep(0.2)
            ser.reset_input_buffer()

    ser.close()

if __name__ == "__main__":
    main()
