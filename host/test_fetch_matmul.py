#!/usr/bin/env python3
# test FM (matmul with W fetche from SDRAM).
# Protocole : 'F''M' N(1) sx sw x[64] addr[3]                    (72 oct envoyes)
# Reponse   : 'F''M' y[N*4 LE int32]                             (2+4N oct)
# K=64 hardcode.

import time, struct
import numpy as np
import serial

PORT = "COM6"
BAUD = 1_000_000
K = 64

def addr_bytes(a): return bytes([a & 0xFF, (a >> 8) & 0xFF, (a >> 16) & 0xFF])
def n_bytes(n):    return bytes([n & 0xFF, (n >> 8) & 0xFF])

def sd_load(ser, addr, data):
    pkt = b'LL' + addr_bytes(addr) + n_bytes(len(data)) + bytes(data)
    ser.write(pkt)
    return ser.read(2) == b'LK'

def call_fm(ser, N, sx, sw, x_i8, addr_w):
    pkt = b'FM' + bytes([N, sx & 0xFF, sw & 0xFF]) + x_i8.tobytes() + addr_bytes(addr_w)
    ser.write(pkt)
    resp = ser.read(2 + 4*N)
    if len(resp) != 2 + 4*N or resp[:2] != b'FM':
        raise RuntimeError(f"FM reponse {len(resp)}/{2+4*N} magic={resp[:2]!r}")
    y = np.frombuffer(resp[2:], dtype='<i4')
    return y

def main():
    ser = serial.Serial(PORT, BAUD, timeout=10.0)
    time.sleep(0.3); ser.reset_input_buffer()
    print("Test FM (matmul DMA SDRAM->BSRAM)\n")

    rng = np.random.default_rng(42)
    sx, sw = -3, -6

    # test plusieurs tailles N
    for N in [4, 8, 12, 15]:
        W = rng.integers(-50, 50, (N, K), dtype=np.int8)
        x = rng.integers(-50, 50, K, dtype=np.int8)
        y_ref = (W.astype(np.int64) @ x.astype(np.int64))
        addr_w = 0x000200
        sd_load(ser, addr_w, W.reshape(-1).tobytes())
        y_fpga = call_fm(ser, N, sx, sw, x, addr_w)
        same = np.array_equal(y_fpga, y_ref)
        print(f"N={N:2d}  match={'OK' if same else 'FAUX'}  y_ref[:3]={y_ref[:3].tolist()}  y_fpga[:3]={y_fpga[:3].tolist()}")
        if not same:
            print(f"     diffs: {(y_fpga - y_ref).tolist()}")

    ser.close()

if __name__ == "__main__":
    main()
