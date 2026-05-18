#!/usr/bin/env python3
# Demo end-to-end : attention block stories260K orchestre par PC, calcul sur FPGA.
#
# Pipeline (dim=64, H=8, KH=4, HS=8, pos=0 pour simplicite) :
#   1. FN(x, rms_w_att) -> x_norm  (rmsnorm sur FPGA with poids fetched SDRAM)
#   2. FQ(x_norm, wq) -> Q[64]     (matmul Q sur FPGA)
#   3. FQ(x_norm, wk) -> K[32]     (matmul K)
#   4. FQ(x_norm, wv) -> V[32]     (matmul V)
#   5. rope Q et K sur PC          (pos=0)
#   6. MM(Q, K, V, T=1) -> attn_out[64]  (multi-head attention sur FPGA)
#   7. FQ(attn_out, wo) -> wo_out[64]    (matmul output)
#   8. PC : x_new = x + wo_out      (residual)
#
# reference : tout computes en numpy float, compare au resultat int8 chained.

import time
import numpy as np
import serial
from test_sdram_diag import sd_load, call_fn, call_fq
from v4_quant import to_i8_shift, from_i8_shift

PORT = "COM6"
BAUD = 1_000_000

D, H, KH, HS = 64, 8, 4, 8
N_REP = H // KH

def addr_bytes(a): return bytes([a & 0xFF, (a >> 8) & 0xFF, (a >> 16) & 0xFF])
def i8(b): return b - 256 if b >= 128 else b

def call_mm(ser, Q_i8, K_i8, V_i8, sq, sk, sv, T):
    """Multi-head attention sur FPGA."""
    pkt = b'MM' + bytes([sq & 0xFF, sk & 0xFF, sv & 0xFF, T])
    pkt += Q_i8.tobytes()
    pkt += K_i8.reshape(-1).tobytes()    # [T, KH, HS]
    pkt += V_i8.reshape(-1).tobytes()
    ser.write(pkt)
    resp = ser.read(67)
    if resp[:2] != b'MK':
        raise RuntimeError(f"MM: {resp[:2]!r}")
    so = i8(resp[2])
    return np.frombuffer(resp[3:], dtype=np.int8), so

def apply_rope_cpu(v_i8, shift, pos, head_size):
    """Applique rope sur les paires consecutives (pos=0 -> identite ici)."""
    # Pour pos=0 : cos=1, sin=0 partout -> rope = identite. Skip pour ce test.
    if pos == 0:
        return v_i8, shift
    # Implementation generale (pas utilisee ici)
    raise NotImplementedError

def main():
    ser = serial.Serial(PORT, BAUD, timeout=8.0)
    time.sleep(0.5); ser.reset_input_buffer()
    print("=== Attention block end-to-end (PC-orchestrated, FPGA-computed) ===\n")

    rng = np.random.default_rng(42)
    pos = 0

    # Poids du modele (random pour test)
    x_real = rng.normal(0, 1, D).astype(np.float32)
    rms_w_real = np.ones(D, dtype=np.float32)
    wq_real = rng.normal(0, 0.1, (H*HS, D)).astype(np.float32)    # [64, 64]
    wk_real = rng.normal(0, 0.1, (KH*HS, D)).astype(np.float32)   # [32, 64]
    wv_real = rng.normal(0, 0.1, (KH*HS, D)).astype(np.float32)
    wo_real = rng.normal(0, 0.1, (D, H*HS)).astype(np.float32)    # [64, 64]

    # --- reference float (numpy) ---
    def rmsnorm_f(x, w, eps=1e-5):
        return x * w / np.sqrt((x**2).mean() + eps)
    x_norm_ref = rmsnorm_f(x_real, rms_w_real)
    Q_ref = wq_real @ x_norm_ref
    K_ref = wk_real @ x_norm_ref
    V_ref = wv_real @ x_norm_ref
    # rope pos=0 = identite
    # attention (T=1, single position)
    Q_h = Q_ref.reshape(H, HS)
    K_h = K_ref.reshape(KH, HS)  # T=1, KH heads
    V_h = V_ref.reshape(KH, HS)
    attn_out_ref = np.zeros(H * HS, dtype=np.float32)
    for h in range(H):
        kvh = h // N_REP
        score = Q_h[h] @ K_h[kvh]  # scalar (T=1)
        attn_w = 1.0  # softmax of single value = 1
        attn_out_ref[h*HS:(h+1)*HS] = attn_w * V_h[kvh]
    wo_out_ref = wo_real @ attn_out_ref
    x_new_ref = x_real + wo_out_ref
    print(f"REF : x_new[:6] = {x_new_ref[:6].round(3)}\n")

    # --- FPGA pipeline ---
    # Quantize inputs
    x_i8, sx = to_i8_shift(x_real)
    rms_w_i8, sw_rms = to_i8_shift(rms_w_real)
    wq_i8, sw_q = to_i8_shift(wq_real)
    wk_i8, sw_k = to_i8_shift(wk_real)
    wv_i8, sw_v = to_i8_shift(wv_real)
    wo_i8, sw_o = to_i8_shift(wo_real)

    # Load poids in SDRAM
    addr_rms = 0x010000
    addr_wq  = 0x020000
    addr_wk  = 0x030000
    addr_wv  = 0x040000
    addr_wo  = 0x050000
    sd_load(ser, addr_rms, rms_w_i8.tobytes())
    sd_load(ser, addr_wq,  wq_i8.reshape(-1).tobytes())
    sd_load(ser, addr_wk,  wk_i8.reshape(-1).tobytes())
    sd_load(ser, addr_wv,  wv_i8.reshape(-1).tobytes())
    sd_load(ser, addr_wo,  wo_i8.reshape(-1).tobytes())
    print(f"Poids charges en SDRAM (sx={sx} sw_rms={sw_rms} sw_q={sw_q} sw_k={sw_k} sw_v={sw_v} sw_o={sw_o})")

    # Step 1: rmsnorm
    x_norm_b, sh_norm = call_fn(ser, x_i8, sx, sw_rms, addr_rms)
    x_norm_i8 = np.frombuffer(x_norm_b, dtype=np.int8)
    print(f"  x_norm shift={sh_norm}  diff from ref: {np.abs(from_i8_shift(x_norm_i8, sh_norm) - x_norm_ref).max():.4f}")

    # Step 2-4: Q, K, V matmuls
    Q_i8, sh_Q = call_fq(ser, 64, sh_norm, sw_q, x_norm_i8, addr_wq)
    K_i8, sh_K = call_fq(ser, 32, sh_norm, sw_k, x_norm_i8, addr_wk)
    V_i8, sh_V = call_fq(ser, 32, sh_norm, sw_v, x_norm_i8, addr_wv)
    print(f"  Q[64] sh={sh_Q}  diff max: {np.abs(from_i8_shift(Q_i8, sh_Q) - Q_ref).max():.4f}")
    print(f"  K[32] sh={sh_K}  diff max: {np.abs(from_i8_shift(K_i8, sh_K) - K_ref).max():.4f}")
    print(f"  V[32] sh={sh_V}  diff max: {np.abs(from_i8_shift(V_i8, sh_V) - V_ref).max():.4f}")

    # Step 5: rope (pos=0 -> identite)
    # Step 6: multi-head attention (T=1)
    # MM attend K shape [T, KH, HS] = [1, 4, 8] flatten = 32 bytes
    K_t = K_i8.reshape(1, KH, HS)
    V_t = V_i8.reshape(1, KH, HS)
    attn_out_i8, sh_attn = call_mm(ser, Q_i8, K_t, V_t, sh_Q, sh_K, sh_V, T=1)
    diff_attn = np.abs(from_i8_shift(attn_out_i8, sh_attn) - attn_out_ref).max()
    print(f"  attn_out[64] sh={sh_attn}  diff max: {diff_attn:.4f}")

    # Step 7: wo matmul
    wo_out_i8, sh_wo_out = call_fq(ser, 64, sh_attn, sw_o, attn_out_i8, addr_wo)
    diff_wo = np.abs(from_i8_shift(wo_out_i8, sh_wo_out) - wo_out_ref).max()
    print(f"  wo_out[64] sh={sh_wo_out}  diff max: {diff_wo:.4f}")

    # Step 8: residual
    x_new_real = x_real + from_i8_shift(wo_out_i8, sh_wo_out)
    diff_final = np.abs(x_new_real - x_new_ref).max()
    print(f"\nFPGA: x_new[:6] = {x_new_real[:6].round(3)}")
    print(f"diff max final = {diff_final:.4f}  (tolerance ~ quantif int8)")

    ser.close()

if __name__ == "__main__":
    main()
