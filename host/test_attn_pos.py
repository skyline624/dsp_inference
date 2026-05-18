#!/usr/bin/env python3
# Test : attention block avec rope (pos>0) et KV cache, sur plusieurs positions.

import time, serial
import numpy as np
from transformer_ops import (
    attention_block_full, attention_block_ref,
    setup_attn_weights, D, H, KH, HS,
)
from v4_quant import to_i8_shift, from_i8_shift

PORT = "COM6"; BAUD = 1_000_000
SEQ_LEN = 6

def make_freq_cis(seq_len, hs):
    """RoPE frequencies a la Llama2 stories260K."""
    theta = 10000.0
    freqs = 1.0 / (theta ** (np.arange(0, hs, 2).astype(np.float32) / hs))
    t = np.arange(seq_len).astype(np.float32)
    angles = np.outer(t, freqs)
    return np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)

def main():
    ser = serial.Serial(PORT, BAUD, timeout=15.0)
    time.sleep(0.5); ser.reset_input_buffer()
    print(f"=== Attention avec rope + KV cache, {SEQ_LEN} positions ===\n")

    rng = np.random.default_rng(9999)
    fr, fi = make_freq_cis(SEQ_LEN, HS)
    freq_cis = (fr, fi)

    # Une seule attention layer
    w_real, w_fpga = setup_attn_weights(ser, rng, base_addr=0x400000)
    print("Poids charges en SDRAM\n")

    # Generer une sequence d'inputs (un par position)
    xs = [rng.normal(0, 1, D).astype(np.float32) for _ in range(SEQ_LEN)]

    # Reference Python (KV cache numpy)
    kv_ref = {'K': np.zeros((SEQ_LEN, KH, HS), dtype=np.float32),
              'V': np.zeros((SEQ_LEN, KH, HS), dtype=np.float32)}
    outs_ref = []
    for pos in range(SEQ_LEN):
        out = attention_block_ref(xs[pos], w_real, pos, kv_ref, freq_cis)
        outs_ref.append(out)

    # FPGA (KV cache int8 + shifts)
    kv_fpga = {'K':  np.zeros((SEQ_LEN, KH, HS), dtype=np.int8),
               'sK': np.zeros(SEQ_LEN, dtype=np.int32),
               'V':  np.zeros((SEQ_LEN, KH, HS), dtype=np.int8),
               'sV': np.zeros(SEQ_LEN, dtype=np.int32)}
    outs_fpga = []
    for pos in range(SEQ_LEN):
        x_i8, sx = to_i8_shift(xs[pos])
        out_i8, sx_out = attention_block_full(ser, x_i8, sx, w_fpga, pos, kv_fpga, freq_cis)
        out_real = from_i8_shift(out_i8, sx_out)
        outs_fpga.append(out_real)
        diff = np.abs(out_real - outs_ref[pos]).max()
        rel  = diff / max(np.abs(outs_ref[pos]).max(), 1e-9) * 100
        print(f"  pos={pos}: ref[:3]={outs_ref[pos][:3].round(3)}  fpga[:3]={out_real[:3].round(3)}  diff={diff:.3f} ({rel:.1f}%)")

    max_rel = max(np.abs(f - r).max() / max(np.abs(r).max(), 1e-9) * 100
                  for f, r in zip(outs_fpga, outs_ref))
    print(f"\nMax erreur relative sur {SEQ_LEN} positions : {max_rel:.1f}%")
    print("==> OK" if max_rel < 30 else "==> ERREUR")

    ser.close()

if __name__ == "__main__":
    main()
