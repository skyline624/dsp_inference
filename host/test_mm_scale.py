#!/usr/bin/env python3
# Debug MM : pour T=1, attn_out = V (single head reorganise). Compare valeurs
# brutes int8 et shift retourne pour identifier d'ou vient le facteur d'echelle.

import time
import numpy as np
import serial

from infer_v4sim import load_model, MODEL_PATH as MODEL, to_i8_shift, from_i8_shift
from transformer_ops import call_mm, H, KH, HS, D

PORT = "COM6"; BAUD = 1_000_000

def main():
    m = load_model(MODEL)
    ser = serial.Serial(PORT, BAUD, timeout=10.0)
    time.sleep(0.5); ser.reset_input_buffer()

    rng = np.random.default_rng(42)

    # Test 1 : Q nul, V predictible. Pour T=1, attn = softmax(0) = 1, donc out = V (repeat).
    print("=== Test MM T=1 : V predictible, Q nul ===\n")
    print("Pour T=1, softmax(any single value) = 1.0, donc attn_out = V repeat selon GQA.\n")

    for trial, (Q_scale, V_scale) in enumerate([(0.5, 1.0), (1.0, 1.0), (0.1, 4.0), (2.0, 0.5)]):
        # Q small (impact peu)
        Q_real = rng.normal(0, Q_scale, H*HS).astype(np.float32)
        # V controle
        V_real = rng.normal(0, V_scale, (KH, HS)).astype(np.float32)
        Q_i8, sQ = to_i8_shift(Q_real)
        V_i8, sV = to_i8_shift(V_real)
        # K idem (impact peu pour T=1 softmax)
        K_real = rng.normal(0, 0.5, (KH, HS)).astype(np.float32)
        K_i8, sK = to_i8_shift(K_real)

        # ref : attn_out = repeat(V, n_rep, axis=0).reshape(D)
        n_rep = H // KH
        attn_ref = np.repeat(V_real, n_rep, axis=0).reshape(D)

        # FPGA MM
        K_t = K_i8.reshape(1, KH, HS)
        V_t = V_i8.reshape(1, KH, HS)
        attn_fpga_i8, sa = call_mm(ser, Q_i8, K_t, V_t, sQ, sK, sV, T=1)
        attn_fpga = from_i8_shift(attn_fpga_i8, sa)

        diff = np.abs(attn_fpga - attn_ref)
        cos  = np.dot(attn_fpga, attn_ref) / (np.linalg.norm(attn_fpga)*np.linalg.norm(attn_ref))
        ratio = np.mean(attn_fpga / np.where(np.abs(attn_ref)>1e-6, attn_ref, 1))
        print(f"--- Trial {trial+1} : Q_scale={Q_scale} V_scale={V_scale} ---")
        print(f"  sQ={sQ} sK={sK} sV={sV} -> FPGA sa={sa}")
        print(f"  V_ref[0,:4]      = {V_real[0,:4].round(3)}")
        print(f"  attn_ref[:4]     = {attn_ref[:4].round(3)}  (= V_ref[0,:4] car n_rep={n_rep})")
        print(f"  attn_fpga[:4]    = {attn_fpga[:4].round(3)}")
        print(f"  attn_fpga/ref    = {(attn_fpga[:4] / np.where(np.abs(attn_ref[:4])>1e-6, attn_ref[:4], 1)).round(3)}")
        print(f"  cos={cos:.4f}  diff_max={diff.max():.3f}  mean_ratio={ratio:.3f}")
        print(f"  log2(ratio)      = {np.log2(abs(ratio)) if abs(ratio)>0 else 'N/A':.3f}  <- si entier, juste un shift wrong")
        print()

    ser.close()

if __name__ == "__main__":
    main()
