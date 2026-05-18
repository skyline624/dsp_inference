#!/usr/bin/env python3
# test SDRAM write/read byte par byte sur le FPGA.
# Protocole :
#   'W''W' addr[3 LE] data(1)  -> 'W''K'              (write)
#   'B''B' addr[3 LE]          -> 'B''K' data(1)      (read)

import time, sys
import serial

PORT = "COM6"
BAUD = 1_000_000

def addr_bytes(addr):
    return bytes([addr & 0xFF, (addr >> 8) & 0xFF, (addr >> 16) & 0xFF])

def sd_write(ser, addr, data):
    pkt = b'WW' + addr_bytes(addr) + bytes([data & 0xFF])
    ser.write(pkt)
    resp = ser.read(2)
    if resp != b'WK':
        raise RuntimeError(f"WW reponse invalide : {resp!r}")

def sd_read(ser, addr):
    pkt = b'BB' + addr_bytes(addr)
    ser.write(pkt)
    resp = ser.read(3)
    if len(resp) != 3 or resp[:2] != b'BK':
        raise RuntimeError(f"BB reponse invalide : {len(resp)}/3 {resp!r}")
    return resp[2]

def main():
    ser = serial.Serial(PORT, BAUD, timeout=3.0)
    time.sleep(0.5); ser.reset_input_buffer()
    print(f"Test SDRAM write/read sur FPGA\n")

    # test 1 : ecrire + relire UN bytes
    print("--- Test 1: ecrire + relire 1 octet ---")
    for addr, data in [(0x000000, 0x42), (0x000010, 0xAB), (0x123456, 0x99)]:
        sd_write(ser, addr, data)
        r = sd_read(ser, addr)
        status = "OK" if r == data else f"FAUX (lu {r:#x})"
        print(f"  addr=0x{addr:06X}  ecrit=0x{data:02X}  relu=0x{r:02X}  {status}")

    # test 2 : pattern sur 32 bytes
    print("\n--- Test 2: pattern lineaire (32 octets) ---")
    base = 0x001000
    for i in range(32):
        sd_write(ser, base + i, (i * 7 + 11) & 0xFF)
    nfail = 0
    for i in range(32):
        exp = (i * 7 + 11) & 0xFF
        got = sd_read(ser, base + i)
        if got != exp:
            print(f"    [{i}] addr=0x{base+i:06X}  exp=0x{exp:02X} got=0x{got:02X}")
            nfail += 1
    print(f"  {32-nfail}/32 OK")

    # test 3 : persistence (relire un bytes ecrit au test 1)
    print("\n--- Test 3: persistence (relire test 1 apres test 2) ---")
    r = sd_read(ser, 0x000000)
    print(f"  addr=0x000000 attendu=0x42 lu=0x{r:02X}  {'OK' if r == 0x42 else 'FAUX'}")

    ser.close()

if __name__ == "__main__":
    main()
