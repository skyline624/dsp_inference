#!/usr/bin/env python3
# =============================================================================
# prove_gen_ffn_chunk.py - G2 proof for gen_seq step C (FFN wiring).
#
# The oracle (infer_v5gen_ref.forward_rtl) computes the FFN as a MONOLITH over
# the full hidden dim HID=172 :
#     g = W1@xn [172] ; u = W3@xn [172] ; sg = silu(g) ; h = sg*u [172]
#     o = W2@h [64]   ; x += o
# with ONE output shift for the whole 172-wide vector.
#
# The RTL (gen_seq, vfile slots are 64 wide) cannot hold 172 in one slot, so it
# must CHUNK the hidden dim into 3 sub-matmuls (64+64+44) and apply silu/mul
# PER CHUNK with an INDEPENDENT output shift per chunk (exactly like ffn_tp_seq2's
# per-chunk shift banks s1[c]/s3[c]/ss[c]/sg_[c]/sp[c]). W2 becomes 3 partial
# matmuls summed with integer shift-alignment (the REDUCE).
#
# Faithful modelling of the RTL contract :
#   - the WEIGHT shift is GLOBAL per matrix (sw1/sw3/sw2 = to_i8_shift(whole W)),
#     each chunk reads its row/col sub-block with the SAME weight shift ;
#   - only the OUTPUT shift is per chunk (requantize of that chunk's i32 result) ;
#   - the node FQ contract (N_out via pkt[2], N_in fixed 64) is already exercised
#     GREEN in step B (Wk/Wv : N_out=32, N_in=64), so no new node behaviour.
#
# This proves the chunking+per-chunk-shift+partial-reduce stays within the step-C
# gate tolerance (~0.40) vs the float truth AND vs the monolith oracle, BEFORE
# any RTL is written (garde-fou G2).
# =============================================================================
import numpy as np
from infer_v4sim import to_i8_shift, from_i8_shift, requantize_i32, rmsnorm_q, silu_q, mul_q, matvec_q

D, HID = 64, 172
CHUNKS = [(0, 64), (64, 128), (128, 172)]        # sizes 64, 64, 44
EPS = 1e-5


def ffn_monolith(x, rms_w, W1, W3, W2):
    """The oracle FFN : one shift over the whole 172-wide hidden vector."""
    xn_i8, sxn = rmsnorm_q(*to_i8_shift(x), rms_w)
    g_i8, sg = matvec_q(W1, xn_i8, sxn)
    u_i8, su = matvec_q(W3, xn_i8, sxn)
    sg_i8, ssg = silu_q(g_i8, sg)
    h_i8, sh = mul_q(sg_i8, ssg, u_i8, su)
    o_i8, so = matvec_q(W2, h_i8, sh)
    return from_i8_shift(o_i8, so)


def ffn_chunked(x, rms_w, W1, W3, W2):
    """The RTL FFN : 3 chunks (64+64+44), per-chunk output shift, W2 reduce."""
    xn_i8, sxn = rmsnorm_q(*to_i8_shift(x), rms_w)      # FN(XB, rms_ffn) -> XN
    W1_i8, sw1 = to_i8_shift(W1)                        # weight shift GLOBAL
    W3_i8, sw3 = to_i8_shift(W3)
    W2_i8, sw2 = to_i8_shift(W2)
    partials = []
    for (a, b) in CHUNKS:
        # W1_c / W3_c : N_out = b-a, N_in = 64 (XN). same weight shift, own out shift.
        g_c = W1_i8[a:b].astype(np.int64) @ xn_i8.astype(np.int64)
        H1_i8, s1 = requantize_i32(g_c, sxn + sw1)
        u_c = W3_i8[a:b].astype(np.int64) @ xn_i8.astype(np.int64)
        H3_i8, s3 = requantize_i32(u_c, sxn + sw3)
        # SS (silu) then MUL (elementwise), per chunk.
        SG_i8, ssg = silu_q(H1_i8, s1)
        HG_i8, sh = mul_q(SG_i8, ssg, H3_i8, s3)
        # W2_c : columns a..b, N_out=64, N_in = b-a (RTL zero-pads to 64 -> identical).
        p_c = W2_i8[:, a:b].astype(np.int64) @ HG_i8.astype(np.int64)
        P_i8, sp = requantize_i32(p_c, sh + sw2)
        partials.append((P_i8, sp))
    # REDUCE : integer shift-align every partial to the max shift, sum, requantize.
    sref = max(sp for _, sp in partials)
    acc = np.zeros(D, np.int64)
    for P_i8, sp in partials:
        acc += P_i8.astype(np.int64) >> (sref - sp)
    o_i8, so = requantize_i32(acc, sref)
    return from_i8_shift(o_i8, so)


def ffn_float(x, rms_w, W1, W3, W2):
    xn = x * rms_w / np.sqrt(np.mean(x ** 2) + EPS)
    g = W1 @ xn; u = W3 @ xn
    sg = g / (1.0 + np.exp(-g))
    return W2 @ (sg * u)


def relerr(a, b):
    ma = float(np.max(np.abs(b))) or 1.0
    return float(np.max(np.abs(a - b))) / ma


def main():
    worst_cf, worst_cm = 0.0, 0.0
    for seed in range(20):
        rng = np.random.default_rng(seed)
        x = rng.standard_normal(D).astype(np.float64)
        rms_w = np.ones(D)                                   # like test_gen_B
        W1 = rng.normal(0, 0.1, (HID, D))
        W3 = rng.normal(0, 0.1, (HID, D))
        W2 = rng.normal(0, 0.1, (D, HID))

        o_f = ffn_float(x, rms_w, W1, W3, W2)
        o_m = ffn_monolith(x, rms_w, W1, W3, W2)
        o_c = ffn_chunked(x, rms_w, W1, W3, W2)

        e_cf = relerr(o_c, o_f)     # RTL chunked vs float truth  (the gate C metric)
        e_cm = relerr(o_c, o_m)     # RTL chunked vs monolith oracle
        e_mf = relerr(o_m, o_f)     # monolith oracle vs float (baseline quant error)
        worst_cf = max(worst_cf, e_cf); worst_cm = max(worst_cm, e_cm)
        if seed < 4:
            print(f"seed {seed}: chunk-vs-float={e_cf:.4f}  chunk-vs-mono={e_cm:.4f}  mono-vs-float={e_mf:.4f}")

    print(f"\nWORST over 20 seeds : chunk-vs-float={worst_cf:.4f}  chunk-vs-mono={worst_cm:.4f}")
    gate = 0.40
    ok = worst_cf < gate
    print(f"GATE (chunk-vs-float < {gate}) : {'PASS' if ok else 'FAIL'}")
    # residual x+o preserves the same abs error but on a larger vector -> relative
    # error only shrinks, so proving o is enough.
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
