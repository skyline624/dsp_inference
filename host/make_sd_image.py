#!/usr/bin/env python3
"""make_sd_image.py - build the raw SD image of the model for the SD bootloader.

Reuses infer_fpga.quantize_and_load_weights (so the SDRAM layout is IDENTICAL to
the PC-orchestrated flow) but captures the LL load packets into a byte image
instead of sending them over UART. The bootloader in top.v copies SD block b to
SDRAM byte b*512, so this image = exactly what the SDRAM must contain.

Output: host/model.img  (+ the SD_NBLK value to set in src/top.v before rebuild).
Write model.img raw to the SD card (block 0):
  Linux/mac : sudo dd if=model.img of=/dev/sdX bs=1M conv=fsync
  Windows   : balenaEtcher / Win32DiskImager / 'dd for Windows'
"""

import os
import infer_fpga as fp


class ImgSer:
    """Fake serial that turns LL load packets into a linear byte image."""
    def __init__(self):
        self.image = bytearray()

    def write(self, pkt):
        if len(pkt) >= 7 and pkt[:2] == b'LL':
            addr = pkt[2] | (pkt[3] << 8) | (pkt[4] << 16)
            ln = pkt[5] | (pkt[6] << 8)
            data = pkt[7:7 + ln]
            end = addr + ln
            if end > len(self.image):
                self.image.extend(b'\x00' * (end - len(self.image)))
            self.image[addr:end] = data

    def read(self, n):
        return b'LK' if n == 2 else b'\x00' * n

    def reset_input_buffer(self):
        pass


def main():
    print(f"Chargement modele : {fp.MODEL}")
    m = fp.load_model(fp.MODEL)
    print("Quantification + construction de l'image SDRAM...")
    s = ImgSer()
    fp.quantize_and_load_weights(s, m)     # fills s.image with the SDRAM layout

    img = bytes(s.image)
    if len(img) % 512:
        img += b'\x00' * (512 - len(img) % 512)
    nblk = len(img) // 512

    out = os.path.join(fp.HERE, "model.img")
    with open(out, "wb") as f:
        f.write(img)

    print(f"\nImage ecrite : {out}")
    print(f"  taille : {len(img)} octets  =  {nblk} blocs de 512")
    print(f"\n>>> 1. Dans src/top.v, mettre le parametre  SD_NBLK = 16'd{nblk}")
    print(f"       (ou dans build_sd_node.tcl si tu preferes), puis rebuild dsp_node_sd.fs")
    print(f">>> 2. Ecrire model.img brut sur la carte SD (bloc 0) :")
    print(f"       Windows : balenaEtcher (choisir 'Flash from file' -> model.img)")
    print(f"       Linux   : sudo dd if=model.img of=/dev/sdX bs=1M conv=fsync")


if __name__ == "__main__":
    main()
