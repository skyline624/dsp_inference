#!/usr/bin/env python3
"""infer_fpga_sd.py - inference on the FPGA when the model was loaded from SD.

The FPGA (dsp_node_sd.fs) loads the model SD -> SDRAM by itself at power-on. This
host therefore SKIPS the weight upload: it only recomputes the shifts/addresses
metadata (deterministic, via a null serial), waits for the card's boot to finish,
then runs the forward pass. Zero weight transfer over UART.
"""

import time
import numpy as np
import serial
import infer_fpga as fp


def addr_bytes(a): return bytes([a & 0xFF, (a >> 8) & 0xFF, (a >> 16) & 0xFF])


class NullSer:
    """No-op serial: quantize_and_load_weights runs for its metadata only."""
    def write(self, b): pass
    def read(self, n): return b'LK' if n == 2 else b'\x00' * n
    def reset_input_buffer(self): pass


def wait_boot(ser, tries=600):
    """Poll until the card responds (boot SD->SDRAM done, FSM ready)."""
    for _ in range(tries):
        ser.reset_input_buffer()
        ser.write(b'BB' + addr_bytes(0x000000))   # read SDRAM byte 0
        r = ser.read(3)
        if len(r) == 3 and r[:2] == b'BK':
            return True
        time.sleep(0.1)
    return False


def main():
    print(f"Modele : {fp.MODEL}")
    m = fp.load_model(fp.MODEL); cfg = m['cfg']
    vocab = fp.load_tok(fp.TOK, cfg['vocab_size'])
    print("Calcul des shifts/adresses (poids deja en SDRAM depuis la carte SD)...")
    w = fp.quantize_and_load_weights(NullSer(), m)      # metadata only, no UART upload

    ser = serial.Serial(fp.PORT, fp.BAUD, timeout=15.0)
    time.sleep(0.5); ser.reset_input_buffer()
    print("Attente du boot SD->SDRAM de la carte (le FPGA charge le modele tout seul)...")
    if not wait_boot(ser):
        print("La carte ne repond pas. Boot termine ? image SD ecrite ? dsp_node_sd.fs flashe ?")
        ser.close(); return
    print("Carte prete (boot SD termine).\n")

    L, KHh, HSh, S = cfg['n_layers'], cfg['n_kv_heads'], cfg['head_size'], cfg['seq_len']
    kv_caches = [{'K':  np.zeros((S, KHh, HSh), dtype=np.int8),
                  'sK': np.zeros(S, dtype=np.int32),
                  'V':  np.zeros((S, KHh, HSh), dtype=np.int8),
                  'sV': np.zeros(S, dtype=np.int32)} for _ in range(L)]
    freq_cis = (m['freq_cis_real'], m['freq_cis_imag'])

    print("Generation (greedy argmax, 17 tokens) :")
    tokens = [1]; text = b""; nxt = 1
    for pos in range(17):
        t0 = time.time()
        logits = fp.forward_fpga(ser, m, w, nxt, kv_caches, pos, freq_cis)
        prev = nxt; nxt = int(np.argmax(logits)); tokens.append(nxt)
        p = fp.decode(prev, nxt, vocab); text += p
        print(f"  pos={pos:2d}  tok={nxt:4d}  ({time.time()-t0:.2f}s)  {repr(p.decode('utf-8','replace'))}")

    print(f"\nTexte FPGA (autonome, poids depuis la carte SD) : {text.decode('utf-8','replace')!r}")
    print(f"Tokens : {tokens}")
    ser.close()


if __name__ == "__main__":
    main()
