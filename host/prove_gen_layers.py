#!/usr/bin/env python3
# =============================================================================
# prove_gen_layers.py - G2 proof for gen_seq step D (5-layer loop).
#
# Step D wraps the validated single layer (attn pos=0 + chunked FFN, step C) in a
# loop over NL=5 layers. The numerical risk is ACCUMULATION : each layer's int8
# residual re-quantizes x, and 5 layers of that could drift past the gate.
#
# This proves, BEFORE the RTL, that the full 5-layer quantized forward (the exact
# dataflow the RTL runs : per-layer KV at pos=0, chunked FFN 64+64+44, requantizing
# residuals) stays within the step-D gate tolerance (~0.60) of the float truth.
# It is also the executable spec for test_gen_D's reference (x after 5 layers, no
# lm_head).
#
# pos=0 note : with a single cached position T=1, softmax is 1.0, so attention is
# exactly the GQA-expanded V (Q/K don't affect the result) - same as step B/C.
# =============================================================================
import numpy as np
from infer_v4sim import (to_i8_shift, from_i8_shift, requantize_i32,
                         rmsnorm_q, silu_q, mul_q, matvec_q, add_q)

D, H, KH, HS, HID = 64, 8, 4, 8, 172
NREP = H // KH
NL = 5
CHUNKS = [(0, 64), (64, 128), (128, 172)]
EPS = 1e-5


def gqa_expand(V_i8):
    Vh = V_i8.reshape(KH, HS)
    return np.concatenate([Vh[h // NREP] for h in range(H)]).astype(np.int8)   # [H*HS=64]


def layer_q(x_i8, sx, w):
    # ---- attention (pos=0 : attn = GQA-expand(V)) ----
    xn_i8, sxn = rmsnorm_q(x_i8, sx, w['rms_att'])
    V_i8, sV = matvec_q(w['Wv'], xn_i8, sxn)
    attn_i8 = gqa_expand(V_i8)                          # shift sV
    out_i8, so = matvec_q(w['Wo'], attn_i8, sV)
    x_i8, sx = add_q(x_i8, sx, out_i8, so)             # attention residual (requantize)
    # ---- FFN (chunked 64+64+44, per-chunk output shift) ----
    xn_i8, sxn = rmsnorm_q(x_i8, sx, w['rms_ffn'])
    W1_i8, sw1 = to_i8_shift(w['W1']); W3_i8, sw3 = to_i8_shift(w['W3']); W2_i8, sw2 = to_i8_shift(w['W2'])
    partials = []
    for (a, b) in CHUNKS:
        g = W1_i8[a:b].astype(np.int64) @ xn_i8.astype(np.int64)
        H1_i8, s1 = requantize_i32(g, sxn + sw1)
        u = W3_i8[a:b].astype(np.int64) @ xn_i8.astype(np.int64)
        H3_i8, s3 = requantize_i32(u, sxn + sw3)
        SG_i8, ssg = silu_q(H1_i8, s1)
        HG_i8, sh = mul_q(SG_i8, ssg, H3_i8, s3)
        p = W2_i8[:, a:b].astype(np.int64) @ HG_i8.astype(np.int64)
        P_i8, sp = requantize_i32(p, sh + sw2)
        partials.append((P_i8, sp))
    sref = max(sp for _, sp in partials)
    acc = np.zeros(D, np.int64)
    for P_i8, sp in partials:
        acc += P_i8.astype(np.int64) >> (sref - sp)
    o_i8, so = requantize_i32(acc, sref)
    return add_q(x_i8, sx, o_i8, so)                   # FFN residual (requantize)


def layer_float(x, w):
    xn = x * w['rms_att'] / np.sqrt(np.mean(x ** 2) + EPS)
    V = w['Wv'] @ xn
    attn = np.concatenate([V.reshape(KH, HS)[h // NREP] for h in range(H)])
    x = x + w['Wo'] @ attn
    xn2 = x * w['rms_ffn'] / np.sqrt(np.mean(x ** 2) + EPS)
    g = w['W1'] @ xn2; u = w['W3'] @ xn2
    sg = g / (1.0 + np.exp(-g))
    return x + w['W2'] @ (sg * u)


def relerr(a, b):
    ma = float(np.max(np.abs(b))) or 1.0
    return float(np.max(np.abs(a - b))) / ma


def main():
    worst = 0.0
    for seed in range(20):
        rng = np.random.default_rng(seed)
        layers = []
        for _ in range(NL):
            layers.append({
                'rms_att': np.ones(D), 'rms_ffn': np.ones(D),
                'Wv': rng.normal(0, 0.1, (KH * HS, D)),
                'Wo': rng.normal(0, 0.1, (D, H * HS)),
                'W1': rng.normal(0, 0.1, (HID, D)),
                'W3': rng.normal(0, 0.1, (HID, D)),
                'W2': rng.normal(0, 0.1, (D, HID)),
            })
        x0 = rng.standard_normal(D)

        # quantized forward
        x_i8, sx = to_i8_shift(x0)
        for l in range(NL):
            x_i8, sx = layer_q(x_i8, sx, layers[l])
        xq = from_i8_shift(x_i8, sx)
        # float forward
        xf = x0.copy()
        for l in range(NL):
            xf = layer_float(xf, layers[l])

        e = relerr(xq, xf)
        worst = max(worst, e)
        if seed < 4:
            print(f"seed {seed}: 5-layer quant-vs-float = {e:.4f}   |x|max={np.max(np.abs(xf)):.2f}")

    print(f"\nWORST over 20 seeds : 5-layer quant-vs-float = {worst:.4f}")
    gate = 0.60
    print(f"GATE (quant-vs-float < {gate}) : {'PASS' if worst < gate else 'FAIL'}")
    return 0 if worst < gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
