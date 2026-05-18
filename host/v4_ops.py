#!/usr/bin/env python3
# =============================================================================
# v4_ops.py  --  References Python (int8 + shift) des operateurs non-matmul.
# Sert de modele pour les modules Verilog correspondants.
# =============================================================================

import numpy as np
from v4_quant import to_i8_shift, from_i8_shift

# ─── RMSNorm ──────────────────────────────────────────────────────────────
# Formule : out = w * x / sqrt(mean(x^2) + eps)
#
# Algorithme en arithmetique entiere :
#   1. acc = somme des x[i]^2 sur les D elements (int24 environ)
#   2. mean_int = acc / D  (decalage de log2(D), ici 6 bits pour D=64)
#   3. inv_rms = 1 / sqrt(mean_int * 2^(2*shift_x) + eps)
#   4. out[i] = x[i] * w[i] * inv_rms  -> re-quantifie en int8 + new shift
#
# Pour le hardware : LUT 1/sqrt(x) de 256 entrees after normalisation par le
# bit de tete (ramener mean in [1, 4) ou similaire).

def rmsnorm_i8_ref(x_i8, shift_x, w_i8, shift_w, eps=1e-5):
    """RMSNorm sur (x_i8, shift_x) avec poids (w_i8, shift_w).
    Renvoie (out_i8, shift_out) representant w * x / sqrt(mean(x^2)+eps)."""
    D = len(x_i8)

    # 1. somme des carres en int. x[i] real = x_i8[i] * 2^shift_x
    #    x[i]^2 real = x_i8[i]^2 * 2^(2*shift_x)
    sq = (x_i8.astype(np.int64)) ** 2          # int24 par value (max 127^2*64 = 1.03M)
    acc = int(sq.sum())                         # int24

    # 2. mean = acc / D, en value reelle
    mean_real = float(acc) * (2.0 ** (2 * shift_x)) / D + eps

    # 3. 1 / sqrt(mean)
    inv_rms = 1.0 / np.sqrt(mean_real)

    # 4. out_real[i] = x[i] * w[i] * inv_rms
    #    = (x_i8[i] * 2^shift_x) * (w_i8[i] * 2^shift_w) * inv_rms
    out_real = x_i8.astype(np.float64) * w_i8.astype(np.float64) \
               * (2.0 ** (shift_x + shift_w)) * inv_rms

    return to_i8_shift(out_real)


def silu_i8_ref(x_i8, shift_x):
    """SiLU(x) = x * sigmoid(x). Reference pour le module FPGA (qui utilisera
    une LUT 256 entrees pour sigmoid)."""
    x_real = from_i8_shift(x_i8, shift_x)
    out = x_real / (1.0 + np.exp(-x_real))
    return to_i8_shift(out)


def softmax_i8_ref(x_i8, shift_x):
    """Softmax. FPGA : recherche max, exp via LUT, somme, division."""
    x_real = from_i8_shift(x_i8, shift_x)
    x_centered = x_real - x_real.max()
    e = np.exp(x_centered)
    p = e / e.sum()
    return to_i8_shift(p)


def silu_mult_i8(gate_i8, shift_g, up_i8, shift_u):
    """h = SiLU(gate) * up   (operation FFN typique)."""
    gate_real = from_i8_shift(gate_i8, shift_g)
    up_real   = from_i8_shift(up_i8,   shift_u)
    h = gate_real * (1.0 / (1.0 + np.exp(-gate_real))) * up_real
    return to_i8_shift(h)


# ─── Tests ────────────────────────────────────────────────────────────────
def _err(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    err = np.abs(a - b)
    max_a = max(np.max(np.abs(a)), 1e-12)
    mask = np.abs(a) > 0.01 * max_a
    rel_max = float((err[mask] / np.abs(a[mask])).max()) if mask.any() else 0.0
    return float(err.mean()), rel_max

def test_rmsnorm():
    rng = np.random.default_rng(42)
    D = 64
    print("Test RMSNorm : reference float vs int8+shift")
    print(f"{'cas':30s}  {'err_abs_moy':>12s}  {'err_rel_max':>12s}")
    for trial in range(4):
        x = rng.normal(0, 1, D).astype(np.float32)
        w = np.ones(D, dtype=np.float32)  # weight neutre pour 1er test
        # reference float
        ref = w * x / np.sqrt(np.mean(x**2) + 1e-5)
        # int8+shift
        x_i8, sx = to_i8_shift(x); w_i8, sw = to_i8_shift(w)
        out_i8, so = rmsnorm_i8_ref(x_i8, sx, w_i8, sw)
        out_rec = from_i8_shift(out_i8, so)
        em, rM = _err(ref, out_rec)
        print(f"  trial {trial+1} (D={D}, w=1)            {em:12.5f}  {rM*100:11.3f}%")

def test_silu():
    rng = np.random.default_rng(1)
    print("\nTest SiLU : reference float vs int8+shift")
    print(f"{'cas':30s}  {'err_abs_moy':>12s}  {'err_rel_max':>12s}")
    for trial in range(3):
        x = rng.normal(0, 2, 172).astype(np.float32)
        ref = x / (1.0 + np.exp(-x))
        x_i8, sx = to_i8_shift(x)
        out_i8, so = silu_i8_ref(x_i8, sx)
        out_rec = from_i8_shift(out_i8, so)
        em, rM = _err(ref, out_rec)
        print(f"  trial {trial+1} (D=172)                {em:12.5f}  {rM*100:11.3f}%")

def test_softmax():
    rng = np.random.default_rng(2)
    print("\nTest softmax : reference float vs int8+shift")
    print(f"{'cas':30s}  {'err_abs_moy':>12s}  {'err_rel_max':>12s}")
    for D in [8, 32, 64, 512]:
        x = rng.normal(0, 2, D).astype(np.float32)
        ex = np.exp(x - x.max()); ref = ex / ex.sum()
        x_i8, sx = to_i8_shift(x)
        out_i8, so = softmax_i8_ref(x_i8, sx)
        out_rec = from_i8_shift(out_i8, so)
        em, rM = _err(ref, out_rec)
        print(f"  D={D:3d}                          {em:12.5f}  {rM*100:11.3f}%")

if __name__ == "__main__":
    test_rmsnorm()
    test_silu()
    test_softmax()
