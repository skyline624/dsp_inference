#!/usr/bin/env python3
# Demo end-to-end : FFN (SwiGLU) block, PC-orchestrated, FPGA-computed.
#
# Pipeline stories260K-like (D=64, hidden=64 pour rester in les limites RTL actuelles) :
#   1. x_norm = FN(x, rms_w_ffn)        rmsnorm sur FPGA
#   2. h1 = FQ(x_norm, W1)              matmul gate
#   3. h3 = FQ(x_norm, W3)              matmul up
#   4. h1_silu = SS(h1)                 silu sur FPGA
#   5. h_gated = h1_silu * h3            elementwise multiply (sur PC, todo RTL)
#   6. h_out = FQ(h_gated, W2)          matmul down
#   7. x_new = x + h_out                 residual
#
# Tout le compute lourd (rmsnorm, 3 matmul, silu) est sur FPGA. Le PC ne does que
# le multiply elementwise et le residu (= une op O(D) chacun, marginal).

import time
import numpy as np
import serial
from test_sdram_diag import sd_load, call_fn, call_fq
from v4_quant import to_i8_shift, from_i8_shift

PORT = "COM6"
BAUD = 1_000_000

D = 64
HIDDEN = 64   # toy : stories260K reel = 172 (necessite bump RTL)

def i8(b): return b - 256 if b >= 128 else b

def call_ss(ser, x_i8, sx, K=64):
    """SiLU standalone sur K elements (max 64)."""
    full_x = np.zeros(64, dtype=np.int8); full_x[:K] = x_i8[:K]
    pkt = b'SS' + bytes([sx & 0xFF]) + full_x.tobytes()
    ser.write(pkt)
    resp = ser.read(70)
    if resp[:2] != b'SK':
        raise RuntimeError(f"SS: magic {resp[:2]!r}")
    so = i8(resp[2])
    return np.frombuffer(resp[6:6+K], dtype=np.int8), so

def silu_f(x):
    return x / (1.0 + np.exp(-x))

def rmsnorm_f(x, w, eps=1e-5):
    return x * w / np.sqrt((x**2).mean() + eps)

def main():
    ser = serial.Serial(PORT, BAUD, timeout=8.0)
    time.sleep(0.5); ser.reset_input_buffer()
    print(f"=== FFN (SwiGLU) end-to-end (D={D}, hidden={HIDDEN}) ===\n")

    rng = np.random.default_rng(101)
    x_real     = rng.normal(0, 1, D).astype(np.float32)
    rms_w_real = np.ones(D, dtype=np.float32)
    W1_real    = rng.normal(0, 0.1, (HIDDEN, D)).astype(np.float32)
    W3_real    = rng.normal(0, 0.1, (HIDDEN, D)).astype(np.float32)
    W2_real    = rng.normal(0, 0.1, (D, HIDDEN)).astype(np.float32)

    # --- reference float ---
    x_norm_ref = rmsnorm_f(x_real, rms_w_real)
    h1_ref     = W1_real @ x_norm_ref
    h3_ref     = W3_real @ x_norm_ref
    h1_silu_ref = silu_f(h1_ref)
    h_gated_ref = h1_silu_ref * h3_ref
    h_out_ref  = W2_real @ h_gated_ref
    x_new_ref  = x_real + h_out_ref
    print(f"REF : x_new[:6] = {x_new_ref[:6].round(3)}\n")

    # --- FPGA pipeline ---
    x_i8,  sx     = to_i8_shift(x_real)
    rms_i8, sw_r  = to_i8_shift(rms_w_real)
    W1_i8, sw_1   = to_i8_shift(W1_real)
    W3_i8, sw_3   = to_i8_shift(W3_real)
    W2_i8, sw_2   = to_i8_shift(W2_real)

    addr_rms = 0x060000
    addr_W1  = 0x070000
    addr_W3  = 0x080000
    addr_W2  = 0x090000
    sd_load(ser, addr_rms, rms_i8.tobytes())
    sd_load(ser, addr_W1,  W1_i8.reshape(-1).tobytes())
    sd_load(ser, addr_W3,  W3_i8.reshape(-1).tobytes())
    sd_load(ser, addr_W2,  W2_i8.reshape(-1).tobytes())
    print(f"Poids charges (sx={sx} sw_rms={sw_r} sw_W1={sw_1} sw_W3={sw_3} sw_W2={sw_2})")

    # 1. rmsnorm
    xn_b, sh_norm = call_fn(ser, x_i8, sx, sw_r, addr_rms)
    xn_i8 = np.frombuffer(xn_b, dtype=np.int8)
    print(f"  x_norm   sh={sh_norm:+d}  diff vs ref: {np.abs(from_i8_shift(xn_i8, sh_norm) - x_norm_ref).max():.4f}")

    # 2-3. Q et K matmul (gate et up)
    h1_i8, sh_h1 = call_fq(ser, HIDDEN, sh_norm, sw_1, xn_i8, addr_W1)
    h3_i8, sh_h3 = call_fq(ser, HIDDEN, sh_norm, sw_3, xn_i8, addr_W3)
    print(f"  h1[gate] sh={sh_h1:+d}  diff vs ref: {np.abs(from_i8_shift(h1_i8, sh_h1) - h1_ref).max():.4f}")
    print(f"  h3[up]   sh={sh_h3:+d}  diff vs ref: {np.abs(from_i8_shift(h3_i8, sh_h3) - h3_ref).max():.4f}")

    # 4. silu sur h1
    h1s_i8, sh_h1s = call_ss(ser, h1_i8, sh_h1, K=HIDDEN)
    print(f"  silu(h1) sh={sh_h1s:+d}  diff vs ref: {np.abs(from_i8_shift(h1s_i8, sh_h1s) - h1_silu_ref).max():.4f}")

    # 5. elementwise multiply (PC -- todo RTL pour autonomie)
    h_gated_real = from_i8_shift(h1s_i8, sh_h1s) * from_i8_shift(h3_i8, sh_h3)
    h_gated_i8, sh_g = to_i8_shift(h_gated_real)
    print(f"  h_gated  sh={sh_g:+d}  diff vs ref: {np.abs(from_i8_shift(h_gated_i8, sh_g) - h_gated_ref).max():.4f}")

    # 6. W2 matmul (down)
    h_out_i8, sh_ho = call_fq(ser, D, sh_g, sw_2, h_gated_i8, addr_W2)
    diff_ho = np.abs(from_i8_shift(h_out_i8, sh_ho) - h_out_ref).max()
    print(f"  h_out    sh={sh_ho:+d}  diff vs ref: {diff_ho:.4f}")

    # 7. residual (PC)
    x_new_real = x_real + from_i8_shift(h_out_i8, sh_ho)
    diff_final = np.abs(x_new_real - x_new_ref).max()
    print(f"\nFPGA: x_new[:6] = {x_new_real[:6].round(3)}")
    print(f"diff max final = {diff_final:.4f}  (tolerance ~ quantif int8)")

    ser.close()

if __name__ == "__main__":
    main()
