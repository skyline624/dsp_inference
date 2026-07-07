#!/usr/bin/env python3
"""Read the SD bring-up harness output over UART.

The FPGA (sd_bringup) streams, repeatedly:  0xA5 0x5A <status> <512 data bytes>.
This script syncs on the 0xA5 0x5A marker, reads a frame, prints the status, a
hex dump of the first 64 bytes, and checks the FAT/MBR boot signature (0x55 0xAA
at offset 510) -- if present, the controller read a real SD card correctly.

Usage:  python read_sd.py [COMx]     (default COM6, 1 Mbaud)
"""

import sys
import serial

PORT = sys.argv[1] if len(sys.argv) > 1 else "COM6"
BAUD = 1_000_000


def read_exact(ser, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = ser.read(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return bytes(buf)


def sync(ser):
    # scan for 0xA5 0x5A
    prev = None
    while True:
        b = ser.read(1)
        if not b:
            return False
        if prev == 0xA5 and b[0] == 0x5A:
            return True
        prev = b[0]


def main():
    print(f"Ouverture {PORT} @ {BAUD} baud ... (Ctrl-C pour arreter)\n")
    ser = serial.Serial(PORT, BAUD, timeout=3.0)

    frames = 0
    while frames < 3:
        if not sync(ser):
            print("Pas de marqueur 0xA5 0x5A recu (timeout).")
            print(" -> FPGA flashe ? bon port ? carte SD inseree ? (LED[2] = erreur)")
            break
        hdr = read_exact(ser, 1)
        data = read_exact(ser, 512)
        if hdr is None or data is None:
            print("Trame incomplete (timeout).")
            break
        status = hdr[0]
        frames += 1
        print(f"=== Trame {frames} : status = 0x{status:02X} "
              f"({'OK' if status == 0x01 else 'ERREUR/pas de carte' if status == 0xEE else '?'}) ===")
        # hex dump of first 64 bytes
        for off in range(0, 64, 16):
            row = data[off:off+16]
            hexs = " ".join(f"{b:02X}" for b in row)
            asc = "".join(chr(b) if 32 <= b < 127 else "." for b in row)
            print(f"  {off:03X}: {hexs}  {asc}")
        sig = data[510] == 0x55 and data[511] == 0xAA
        nonzero = any(b not in (0x00, 0xFF) for b in data)
        print(f"  signature 0x55AA @510 : {'OUI (bloc FAT/MBR valide !)' if sig else 'non'}"
              f"   |  donnees non-triviales : {'oui' if nonzero else 'non'}")
        if status == 0x01 and (sig or nonzero):
            print("  >>> Le controleur SD lit la vraie carte correctement.\n")
        else:
            print()

    ser.close()


if __name__ == "__main__":
    main()
