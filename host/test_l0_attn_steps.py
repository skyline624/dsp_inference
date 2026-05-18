#!/usr/bin/env python3
# Decortique chaque step de l'attention block layer 0, pos=0.
# Pour chaque step : compare la sortie FPGA vs v4sim Python.

import time
import numpy as np
import serial

from infer_v4sim import load_model, MODEL_PATH as MODEL
from infer_v4sim import (rmsnorm_q, matvec_q, apply_rope_q, to_i8_shift, from_i8_shift,
                         softmax_q)
from transformer_ops import (call_mm, call_rr, apply_rope_qk,
                              D, H, KH, HS)
from infer_fpga import quantize_and_load_weights
from test_sdram_diag import call_fn, call_fq

PORT = "COM6"; BAUD = 1_000_000

def cmp(label, fpga, ref):
    if isinstance(fpga, np.ndarray) and fpga.dtype == np.int8:
        fpga = fpga.astype(np.int32)
    if isinstance(ref, np.ndarray) and ref.dtype == np.int8:
        ref = ref.astype(np.int32)
    fpga = np.asarray(fpga, dtype=np.float64).ravel()
    ref  = np.asarray(ref,  dtype=np.float64).ravel()
    diff = np.abs(fpga - ref)
    rel  = diff.max() / max(np.abs(ref).max(), 1e-9) * 100
    cos  = np.dot(fpga, ref) / (np.linalg.norm(fpga)*np.linalg.norm(ref) + 1e-9)
    flag = " BAD" if cos < 0.95 else ""
    print(f"  {label:30s}: cos={cos:.4f}  diff_max={diff.max():.2f}  rel={rel:.1f}%{flag}")

def main():
    m = load_model(MODEL); cfg = m['cfg']
    L, HID = cfg['n_layers'], cfg['hidden_dim']
    print(f"Model: hidden={HID}")

    ser = serial.Serial(PORT, BAUD, timeout=15.0)
    time.sleep(0.5); ser.reset_input_buffer()
    print("Chargement poids FPGA...")
    w = quantize_and_load_weights(ser, m)
    attn = w['layers'][0]['attn']

    # Embedding
    x_real = m['tok_emb'][1].astype(np.float32).copy()
    x_i8, sx = to_i8_shift(x_real)
    print(f"\nx (token=1): shift={sx} range=[{x_i8.min()}, {x_i8.max()}]\n")

    # ─── 1. RMSNorm ─────────────────────────────────────────────────────
    print("STEP 1: RMSNorm")
    # v4sim ref
    xn_ref_i8, sh_n_ref = rmsnorm_q(x_i8, sx, m['rms_att'][0])
    # FPGA
    xn_b, sh_n = call_fn(ser, x_i8, sx, attn['sh_rms'], attn['addr_rms'])
    xn_fpga = np.frombuffer(xn_b, dtype=np.int8)
    print(f"  ref:  sh={sh_n_ref}  xn={xn_ref_i8[:8].tolist()}")
    print(f"  fpga: sh={sh_n}      xn={xn_fpga[:8].tolist()}")
    cmp("x_norm float", from_i8_shift(xn_fpga, sh_n), from_i8_shift(xn_ref_i8, sh_n_ref))

    # ─── 2. Q, K, V matmuls ─────────────────────────────────────────────
    print("\nSTEP 2: Q/K/V matmuls (using FPGA x_norm)")
    # Pour comparer juste le matmul on utilise le MEME input (x_norm FPGA)
    # ref = matvec_q sur xn_fpga
    Q_ref_i8, sQ_ref = matvec_q(m['wq'][0], xn_fpga, sh_n)
    K_ref_i8, sK_ref = matvec_q(m['wk'][0], xn_fpga, sh_n)
    V_ref_i8, sV_ref = matvec_q(m['wv'][0], xn_fpga, sh_n)
    # FPGA
    Q_fpga, sQ = call_fq(ser, H*HS,  sh_n, attn['sh_q'], xn_fpga, attn['addr_q'])
    K_fpga, sK = call_fq(ser, KH*HS, sh_n, attn['sh_k'], xn_fpga, attn['addr_k'])
    V_fpga, sV = call_fq(ser, KH*HS, sh_n, attn['sh_v'], xn_fpga, attn['addr_v'])
    cmp("Q float", from_i8_shift(Q_fpga, sQ), from_i8_shift(Q_ref_i8, sQ_ref))
    cmp("K float", from_i8_shift(K_fpga, sK), from_i8_shift(K_ref_i8, sK_ref))
    cmp("V float", from_i8_shift(V_fpga, sV), from_i8_shift(V_ref_i8, sV_ref))

    # ─── 3. RoPE ────────────────────────────────────────────────────────
    print("\nSTEP 3: RoPE (pos=0 = identity)")
    fr = m['freq_cis_real'][0]; fi = m['freq_cis_imag'][0]
    Q_h_ref = from_i8_shift(Q_fpga, sQ).reshape(H, HS).astype(np.float32)
    Q_rope_ref_i8, sQr_ref = apply_rope_q(*to_i8_shift(Q_h_ref), fr, fi)
    K_h_ref = from_i8_shift(K_fpga, sK).reshape(KH, HS).astype(np.float32)
    K_rope_ref_i8, sKr_ref = apply_rope_q(*to_i8_shift(K_h_ref), fr, fi)
    # FPGA
    Q_rope, sQr, K_rope, sKr = apply_rope_qk(ser, Q_fpga, sQ, K_fpga, sK, fr, fi)
    cmp("Q rope float", from_i8_shift(Q_rope, sQr), from_i8_shift(Q_rope_ref_i8, sQr_ref))
    cmp("K rope float", from_i8_shift(K_rope, sKr), from_i8_shift(K_rope_ref_i8, sKr_ref))

    # ─── 4. Attention (T=1) ──────────────────────────────────────────────
    print("\nSTEP 4: Multi-head attention (T=1)")
    # ref : pour T=1, softmax=1, attn_out = V_h reorganise
    n_rep = H // KH
    V_ref_2d = from_i8_shift(V_fpga, sV).reshape(KH, HS)
    V_ref_rep = np.repeat(V_ref_2d, n_rep, axis=0)
    attn_out_ref = V_ref_rep.reshape(D)
    # FPGA
    K_t = K_rope.reshape(1, KH, HS)
    V_t = V_fpga.reshape(1, KH, HS)
    # Re-align (comme dans attention_block_full)
    K_send_i8, sKsend = to_i8_shift(np.expand_dims(from_i8_shift(K_t, sKr), 0))
    V_send_i8, sVsend = to_i8_shift(np.expand_dims(from_i8_shift(V_t, sV), 0))
    K_send_i8 = K_send_i8.squeeze(0); V_send_i8 = V_send_i8.squeeze(0)
    attn_fpga_i8, sa = call_mm(ser, Q_rope, K_send_i8, V_send_i8, sQr, sKsend, sVsend, T=1)
    cmp("attn_out float", from_i8_shift(attn_fpga_i8, sa), attn_out_ref)

    # ─── 5. Wo matmul ───────────────────────────────────────────────────
    print("\nSTEP 5: Wo matmul")
    out_ref_i8, so_ref = matvec_q(m['wo'][0], attn_fpga_i8, sa)
    out_fpga, so = call_fq(ser, D, sa, attn['sh_o'], attn_fpga_i8, attn['addr_o'])
    cmp("Wo out float", from_i8_shift(out_fpga, so), from_i8_shift(out_ref_i8, so_ref))

    # ─── 6. Residual ────────────────────────────────────────────────────
    print("\nSTEP 6: Residual")
    x_after_ref = x_real + from_i8_shift(out_ref_i8, so_ref)
    x_after_fpga = x_real + from_i8_shift(out_fpga, so)
    cmp("x_after_attn", x_after_fpga, x_after_ref)

    ser.close()

if __name__ == "__main__":
    main()
