#!/usr/bin/env python3
# test fetcher DMA SDRAM->BSRAM : commande FN.
# 1. Charge w[64] in SDRAM via LL
# 2. sends FN : sx sw x[64] addr[3] -> FPGA fetche w from SDRAM, run rmsnorm
# 3. compare a la commande NN classique (w envoye en direct via UART)
# Si match -> le fetcher DMA marche.

import time
import numpy as np
import serial
from v4_quant import to_i8_shift, from_i8_shift
from v4_ops import rmsnorm_i8_ref

PORT = "COM6"
BAUD = 1_000_000
D = 64

def i8(b): return b - 256 if b >= 128 else b

def addr_bytes(a): return bytes([a & 0xFF, (a >> 8) & 0xFF, (a >> 16) & 0xFF])
def n_bytes(n):    return bytes([n & 0xFF, (n >> 8) & 0xFF])

def sd_load(ser, addr, data):
    pkt = b'LL' + addr_bytes(addr) + n_bytes(len(data)) + bytes(data)
    ser.write(pkt)
    return ser.read(2) == b'LK'

def call_nn(ser, x_i8, sx, w_i8, sw):
    """RMSNorm classique : w via UART."""
    pkt = b'NN' + bytes([sx & 0xFF, sw & 0xFF]) + x_i8.tobytes() + w_i8.tobytes()
    ser.write(pkt)
    resp = ser.read(75)
    if resp[:2] != b'NK': raise RuntimeError(f"NN: {resp[:2]!r}")
    so = i8(resp[2])
    return np.frombuffer(resp[11:], dtype=np.int8), so

def call_fn(ser, x_i8, sx, sw, addr):
    """RMSNorm avec w fetche depuis SDRAM[addr]."""
    pkt = b'FN' + bytes([sx & 0xFF, sw & 0xFF]) + x_i8.tobytes() + addr_bytes(addr)
    ser.write(pkt)
    resp = ser.read(75)
    if resp[:2] != b'FK': raise RuntimeError(f"FN: {resp[:2]!r} len={len(resp)}")
    so = i8(resp[2])
    return np.frombuffer(resp[11:], dtype=np.int8), so

def main():
    ser = serial.Serial(PORT, BAUD, timeout=5.0)
    time.sleep(0.3); ser.reset_input_buffer()
    print("Test FN : fetcher DMA SDRAM -> BSRAM\n")

    rng = np.random.default_rng(7)
    # reference: w aleatoire encodee int8+shift
    w_f = rng.normal(1, 0.2, D).astype(np.float32)
    w_i8, sw = to_i8_shift(w_f)
    # x aleatoire
    x_f = rng.normal(0, 1, D).astype(np.float32)
    x_i8, sx = to_i8_shift(x_f)

    # 1. Charge w in SDRAM a l'addresse 0x000800
    sdram_addr = 0x000800
    print(f"Charge w[{D}] (sw={sw:+d}) dans SDRAM[0x{sdram_addr:06X}]...")
    # bytes signed -> uint8 representation
    w_bytes = w_i8.tobytes()   # raw bytes ; SDRAM/FPGA traite comme signed quand recharge
    sd_load(ser, sdram_addr, w_bytes)
    print("  load OK")
    # Verif : relire w from SDRAM via CC, comparer
    pkt = b'CC' + addr_bytes(sdram_addr) + n_bytes(D)
    ser.write(pkt)
    cc_resp = ser.read(2 + D)
    cc_w = cc_resp[2:]
    if cc_w == w_bytes:
        print("  CC readback : OK (SDRAM contient bien w)\n")
    else:
        ndiff = sum(1 for a,b in zip(cc_w, w_bytes) if a != b)
        print(f"  CC readback : {ndiff} diffs ! Corruption SDRAM detectee\n")

    # 2. Run NN classique (reference comportement FPGA)
    print("Run NN classique (w via UART)...")
    nn_out, nn_so = call_nn(ser, x_i8, sx, w_i8, sw)
    print(f"  shift_out={nn_so:+3d}  out[:6]={nn_out[:6].tolist()}\n")

    # 3. Run FN (fetche w from SDRAM)
    print("Run FN (w fetche depuis SDRAM)...")
    fn_out, fn_so = call_fn(ser, x_i8, sx, sw, sdram_addr)
    print(f"  shift_out={fn_so:+3d}  out[:6]={fn_out[:6].tolist()}\n")

    # 4. compare
    same = np.array_equal(nn_out, fn_out) and nn_so == fn_so
    print(f"Match NN vs FN : {'OK' if same else 'FAUX'}")
    if not same:
        diffs = np.where(nn_out != fn_out)[0]
        print(f"  diffs aux positions {diffs.tolist()}")
        print(f"  nn  : {nn_out.tolist()}")
        print(f"  fn  : {fn_out.tolist()}")

    ser.close()

if __name__ == "__main__":
    main()
