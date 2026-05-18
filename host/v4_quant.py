#!/usr/bin/env python3
# =============================================================================
# v4_quant.py  --  Quantization helpers: int8 + power-of-2 scale.
#
# v4 convention:
#   An activation tensor = (int8 array, shift)
#   real_value = int8_data * 2**shift
#
# Advantage: conversions are pure shifts (free in hardware).
# Shift choice: pick the smallest shift such that max(|real|) < 128*2^shift,
# i.e. shift = ceil(log2(max_abs / 127)).
#
# KNOWN LIMITATION: a power-of-2 scale loses up to 1 bit of precision compared
# to an arbitrary fractional scale. For stories260K (35 chained matmuls) the
# cumulative error reaches ~30-40% on the worst significant element. If v4.5
# produces text different from v3e, we'd need to switch to a 16-bit Q-format
# scale (adds a small DSP multiplier, but simple).
# =============================================================================

import numpy as np

# ─── Conversions de base ──────────────────────────────────────────────────
def to_i8_shift(x_float):
    """float -> (int8, shift) tel que real ~= int8 * 2**shift."""
    max_abs = float(np.max(np.abs(x_float)))
    if max_abs == 0.0:
        return np.zeros_like(x_float, dtype=np.int8), 0
    # le plus petit shift tel que max_abs / 2^shift <= 127
    shift = int(np.ceil(np.log2(max_abs / 127.0)))
    int8 = np.clip(np.round(x_float / (2.0 ** shift)), -128, 127).astype(np.int8)
    return int8, shift

def from_i8_shift(x_i8, shift):
    """(int8, shift) -> float64."""
    return x_i8.astype(np.float64) * (2.0 ** shift)

# ─── Re-quantification d'un int32 (after matmul) en int8+shift ───────────
def requantize_i32(y_i32, shift_in):
    """int32 (avec shift_in) -> (int8, shift_out)."""
    max_abs = int(np.max(np.abs(y_i32)))
    if max_abs == 0:
        return np.zeros_like(y_i32, dtype=np.int8), shift_in
    # nombre de bits a decaler pour rentrer in int8 signe
    add_shift = max(0, int(np.ceil(np.log2(max_abs / 127.0))))
    # arithmetic right shift (with arrondi to le plus proche)
    half = 1 << (add_shift - 1) if add_shift > 0 else 0
    y_shifted = (y_i32 + half) >> add_shift if add_shift > 0 else y_i32
    y_i8 = np.clip(y_shifted, -128, 127).astype(np.int8)
    return y_i8, shift_in + add_shift

# ─── Tests de bon fonctionnement ─────────────────────────────────────────
def _err(a, b):
    """Renvoie (err_abs_moy, err_abs_max, err_rel_significative).
    L'erreur relative n'est calculee que sur les valeurs > 1% du max
    (sinon les valeurs proches de zero donnent des erreurs relatives explosives
    qui ne refletent pas la qualite numerique reelle)."""
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    err = np.abs(a - b)
    max_a = max(np.max(np.abs(a)), 1e-12)
    mask = np.abs(a) > 0.01 * max_a
    if mask.any():
        rel = err[mask] / np.abs(a[mask])
        rel_max = float(rel.max())
    else:
        rel_max = 0.0
    return float(err.mean()), float(err.max()), rel_max

def test_roundtrip():
    rng = np.random.default_rng(42)
    # plusieurs cas with des plages de values differentes
    cases = [
        ("petites valeurs [-0.1, 0.1]",     rng.uniform(-0.1, 0.1, 1024)),
        ("normales N(0,1)",                  rng.normal(0, 1, 1024)),
        ("grandes [-1000, 1000]",            rng.uniform(-1000, 1000, 1024)),
        ("mixte sparse",                     rng.normal(0, 0.01, 1024) + rng.choice([0, 10], 1024, p=[0.99,0.01])),
    ]
    print("Test 1 : round-trip float -> int8+shift -> float")
    print(f"{'cas':36s}  {'shift':>5s}  {'err_abs_moy':>12s}  {'err_rel_max':>12s}")
    for name, x in cases:
        x_i8, shift = to_i8_shift(x)
        x_rec = from_i8_shift(x_i8, shift)
        em, _eM, rM = _err(x, x_rec)
        print(f"  {name:34s}  {shift:5d}  {em:12.5f}  {rM*100:11.3f}%")

def test_matmul_requantize():
    """Simule une matmul puis re-quantification."""
    rng = np.random.default_rng(1)
    K = 64
    print("\nTest 2 : matmul int8 + re-quantification en int8")
    print(f"{'cas':30s}  {'shift_y':>7s}  {'err_abs_moy':>12s}  {'err_rel_max':>12s}")
    for trial in range(3):
        x = rng.normal(0, 1, K).astype(np.float32)
        W = rng.normal(0, 1/np.sqrt(K), (8, K)).astype(np.float32)
        y_ref = W @ x
        x_i8, sx = to_i8_shift(x)
        W_i8, sW = to_i8_shift(W)
        y_i32 = W_i8.astype(np.int64) @ x_i8.astype(np.int64)
        y_i8, shift_out = requantize_i32(y_i32, sx + sW)
        y_rec = from_i8_shift(y_i8, shift_out)
        em, _eM, rM = _err(y_ref, y_rec)
        print(f"  trial {trial+1} (K={K}, N=8){'':14s}  {shift_out:7d}  {em:12.5f}  {rM*100:11.3f}%")

def test_chain():
    """Plusieurs ops successives (simule pipeline)."""
    rng = np.random.default_rng(2)
    K = 64
    x = rng.normal(0, 1, K).astype(np.float32)
    W1 = rng.normal(0, 1/np.sqrt(K), (K, K)).astype(np.float32)
    W2 = rng.normal(0, 1/np.sqrt(K), (K, K)).astype(np.float32)
    W3 = rng.normal(0, 1/np.sqrt(K), (K, K)).astype(np.float32)

    # ref float
    y_ref = W3 @ (W2 @ (W1 @ x))

    # chaine int8 + re-quantif a each step
    x_i8, sx = to_i8_shift(x)
    W1_i8, sW1 = to_i8_shift(W1)
    y1_i32 = W1_i8.astype(np.int64) @ x_i8.astype(np.int64)
    y1_i8, s1 = requantize_i32(y1_i32, sx + sW1)

    W2_i8, sW2 = to_i8_shift(W2)
    y2_i32 = W2_i8.astype(np.int64) @ y1_i8.astype(np.int64)
    y2_i8, s2 = requantize_i32(y2_i32, s1 + sW2)

    W3_i8, sW3 = to_i8_shift(W3)
    y3_i32 = W3_i8.astype(np.int64) @ y2_i8.astype(np.int64)
    y3_i8, s3 = requantize_i32(y3_i32, s2 + sW3)

    y_rec = from_i8_shift(y3_i8, s3)
    em, eM, rM = _err(y_ref, y_rec)
    print(f"\nTest 3 : 3 matmuls successives (simule 3 etages d'un transformer)")
    print(f"  shifts intermediaires : {sx} -> {s1} -> {s2} -> {s3}")
    print(f"  err_abs_moy = {em:.5f}   err_rel_max (>1%) = {rM*100:.3f}%")
    if rM < 0.20:
        print("  ==> precision acceptable (<20% rel sur les valeurs significatives)")
    else:
        print("  ==> precision DEGRADEE")

if __name__ == "__main__":
    test_roundtrip()
    test_matmul_requantize()
    test_chain()
