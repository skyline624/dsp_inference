#!/usr/bin/env python3
# =============================================================================
# prove_rope_decomp.py - G2 proof for Phase 5b (causal attention RTL).
#
# The oracle infer_v5gen_ref.forward_rtl ropes with apply_rope_q (infer_v4sim) :
# a FLOAT rope followed by to_i8_shift -> a NEW output shift, all heads at once.
# The RTL does NOT do that. The node's RR primitive is rope_op, whose exact
# fixed-point behaviour is pinned by host/test_rope.py::rope_ref :
#   * cos/sin quantized to Q15,
#   * integer products, round with (+1<<14) >> 15, clip to int8,
#   * SHIFT PRESERVED : shift_out == shift_x (no requantize).
# and it ropes ONE head (HS=8) per call.
#
# Consequence for the sequencer : call RR per head; every head comes back at the
# SAME shift (the input shift), so the heads concatenate directly - there is NO
# max-shift re-align step for rope (unlike the KV cache across positions).
#
# This hardware rope differs numerically from apply_rope_q, so per garde-fou G2
# it must be proven to leave the generated text bit-identical BEFORE any RTL.
#
# Expected text (the oracle) :
#   'Once upon a time, there was a little girl named Lily. She lo'
# =============================================================================
import os
import numpy as np

import infer_v4sim as S
from infer_v4sim import apply_rope_q
import infer_v5gen_ref as ORACLE

HERE = os.path.dirname(os.path.abspath(__file__))
S.MODEL_PATH = os.path.join(HERE, "models", "stories260K.bin")
S.TOK_PATH   = os.path.join(HERE, "models", "tok512.bin")

EXPECTED = 'Once upon a time, there was a little girl named Lily. She lo'
HS, HALF = 8, 4


def to_q15(v):
    return int(max(-32768, min(32767, round(v * 32768.0))))


def rope_op_hw(head_i8, sx, fr, fi):
    """Faithful model of the node's rope_op primitive (== test_rope.py::rope_ref).
       Q15 cos/sin, integer product, round (+1<<14) >> 15, clip int8, shift kept."""
    out = np.zeros(HS, dtype=np.int8)
    for i in range(HALF):
        xr = int(head_i8[2 * i]); xi = int(head_i8[2 * i + 1])
        cq = to_q15(fr[i]); sq = to_q15(fi[i])
        nr = (xr * cq - xi * sq + 16384) >> 15   # arithmetic (floor) shift, matches signed >>>
        ni = (xr * sq + xi * cq + 16384) >> 15
        out[2 * i]     = max(-128, min(127, nr))
        out[2 * i + 1] = max(-128, min(127, ni))
    return out, sx                                # shift_out == shift_x


def rope_hw_allheads(x_i8, sx, fr, fi):
    """Sequencer rope path : RR per head, all heads keep shift sx -> plain concat,
       NO max-shift re-align. Drop-in signature for apply_rope_q."""
    nh, _ = x_i8.shape
    out = np.stack([rope_op_hw(x_i8[h], sx, fr, fi)[0] for h in range(nh)])
    return out, sx


def run():
    ORACLE.apply_rope_q = rope_hw_allheads       # monkeypatch forward_rtl's rope
    try:
        return ORACLE.generate()
    finally:
        ORACLE.apply_rope_q = apply_rope_q        # restore


if __name__ == "__main__":
    print("G2 proof : RTL rope = node rope_op primitive (Q15, shift-preserving, per head)")
    print(f"  expected : {EXPECTED!r}\n")
    txt, toks = run()
    ok = (txt == EXPECTED)
    print(f"  hardware rope : {'OK  ' if ok else 'MISMATCH'} : {txt!r}")
    if not ok:
        print(f"  tokens : {toks}")
    print()
    print("PROOF PASS - the node's rope_op is text-safe; sequencer concatenates heads "
          "at the shared shift (no re-align)." if ok else
          "PROOF FAIL - hardware rope changes the text; reconcile before writing RTL.")
