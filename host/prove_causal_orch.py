#!/usr/bin/env python3
# =============================================================================
# prove_causal_orch.py - G2 proof for the causal-attention SEQUENCER.
#
# Models the exact byte-level orchestration the RTL sequencer will perform for
# ONE causal attention block, using ONLY the node primitives (FN, FQ, RR, MM)
# and integer KV re-align -- no float shortcuts. Proven against the oracle text
# so the RTL has a byte-exact spec before a line of Verilog is written (G2).
#
# Sequencer steps (mirror of transformer_ops.attention_block_full, but with the
# oracle's INTEGER KV re-align and per-head shift-preserving rope):
#   1. FN(x)                          -> xn, shift preserved-ish (FN requantizes)
#   2. FQ(Wq) FQ(Wk) FQ(Wv)           -> Q[64], K[32], V[32] each with a shift
#   3. RR per head on Q (8 heads) and K (4 heads)  [rope_op: shift preserved]
#      -> heads concatenate at the shared input shift (proven in prove_rope_decomp)
#   4. KV-cache[pos] = (K_roped_i8, sK), (V_i8, sV)
#   5. re-align K[0..pos],V[0..pos] to sKref=max(sK), sVref=max(sV) by integer
#      right-shift; pack as [t][kvh][hs]  (MM layout, stride 32)
#   6. MM(Q, Kpack, Vpack, sQ, sKref, sVref, T=pos+1) -> attn[64], sA
#   7. FQ(Wo, attn) -> out ; residual x + out
#
# The primitives are modeled by their exact fixed-point behaviour (the same
# references the isolated node gates test_rr_node / test_ffn use), so a PASS here
# means the orchestration is numerically what the RTL must reproduce.
# =============================================================================
import os
import numpy as np

import infer_v4sim as S
from infer_v4sim import (to_i8_shift, from_i8_shift, rmsnorm_q, matvec_q,
                         silu_q, mul_q, softmax_q)

HERE = os.path.dirname(os.path.abspath(__file__))
S.MODEL_PATH = os.path.join(HERE, "models", "stories260K.bin")
S.TOK_PATH   = os.path.join(HERE, "models", "tok512.bin")

EXPECTED = 'Once upon a time, there was a little girl named Lily. She lo'
HS, HALF = 8, 4


def to_q15(v):
    return int(max(-32768, min(32767, round(v * 32768.0))))


def rr_head(head_i8, fr, fi):
    """node rope_op : Q15, round(+16384)>>15, clip int8, shift preserved."""
    out = np.zeros(HS, dtype=np.int8)
    for i in range(HALF):
        xr = int(head_i8[2*i]); xi = int(head_i8[2*i+1])
        cq = to_q15(fr[i]); sq = to_q15(fi[i])
        out[2*i]   = max(-128, min(127, (xr*cq - xi*sq + 16384) >> 15))
        out[2*i+1] = max(-128, min(127, (xr*sq + xi*cq + 16384) >> 15))
    return out


def rope_seq(x_i8, sx, nh, fr, fi):
    """RR per head; all heads keep shift sx -> plain concat (no re-align)."""
    return np.concatenate([rr_head(x_i8[h*HS:(h+1)*HS], fr, fi) for h in range(nh)]), sx


def mm_node(Q_i8, sQ, Kpack_i8, sK, Vpack_i8, sV, T, H, KH):
    """Faithful model of the node MM: per query-head h, GQA kv-head = h//n_rep,
       scores = Q_h . K[t,kvh] / sqrt(HS), softmax over t, out = sum_t p_t V[t,kvh].
       Kpack/Vpack are [T, KH, HS] flattened. Uses the same softmax_q as the node."""
    n_rep = H // KH
    Qf = from_i8_shift(Q_i8.astype(np.int32), sQ).reshape(H, HS)
    Kf = from_i8_shift(Kpack_i8.astype(np.int32), sK).reshape(T, KH, HS)
    Vf = from_i8_shift(Vpack_i8.astype(np.int32), sV).reshape(T, KH, HS)
    out = np.zeros((H, HS), dtype=np.float64)
    for h in range(H):
        kvh = h // n_rep
        scores = np.array([Qf[h] @ Kf[t, kvh] for t in range(T)]) / np.sqrt(HS)
        p_i8, sp = softmax_q(*to_i8_shift(scores))
        p = from_i8_shift(p_i8, sp)
        out[h] = sum(p[t] * Vf[t, kvh] for t in range(T))
    return to_i8_shift(out.reshape(-1))


def forward_seq(m, token, kv, pos):
    cfg = m['cfg']
    H, KH, HSd, D = cfg['n_heads'], cfg['n_kv_heads'], cfg['head_size'], cfg['dim']
    x = m['tok_emb'][token].astype(np.float32).copy()
    for l in range(cfg['n_layers']):
        # 1. FN (rmsnorm requantizes -> its own output shift)
        xn_i8, sxn = rmsnorm_q(*to_i8_shift(x), m['rms_att'][l])
        # 2. FQ Wq/Wk/Wv
        Q_i8, sQ = matvec_q(m['wq'][l], xn_i8, sxn)
        K_i8, sK = matvec_q(m['wk'][l], xn_i8, sxn)
        V_i8, sV = matvec_q(m['wv'][l], xn_i8, sxn)
        # 3. RR per head (shift preserved -> concat)
        fr = m['freq_cis_real'][pos]; fi = m['freq_cis_imag'][pos]
        Qr_i8, sQ = rope_seq(Q_i8, sQ, H, fr, fi)
        Kr_i8, sK = rope_seq(K_i8, sK, KH, fr, fi)
        # 4. KV cache
        kv[l]['K'][pos] = Kr_i8.reshape(KH, HSd); kv[l]['sK'][pos] = sK
        kv[l]['V'][pos] = V_i8.reshape(KH, HSd); kv[l]['sV'][pos] = sV
        # 5. integer re-align to max shift, pack [t][kvh][hs]
        T = pos + 1
        sKref = max(kv[l]['sK'][p] for p in range(T))
        sVref = max(kv[l]['sV'][p] for p in range(T))
        Kpack = np.concatenate([
            np.clip(kv[l]['K'][p].astype(np.int32) >> (sKref - kv[l]['sK'][p]), -128, 127).reshape(-1)
            for p in range(T)]).astype(np.int8)
        Vpack = np.concatenate([
            np.clip(kv[l]['V'][p].astype(np.int32) >> (sVref - kv[l]['sV'][p]), -128, 127).reshape(-1)
            for p in range(T)]).astype(np.int8)
        # 6. MM
        attn_i8, sA = mm_node(Qr_i8, sQ, Kpack, sKref, Vpack, sVref, T, H, KH)
        # 7. FQ Wo + residual
        out_i8, so = matvec_q(m['wo'][l], attn_i8, sA)
        x = x + from_i8_shift(out_i8, so)
        # ---- FFN (unchanged, already validated) ----
        xn_i8, sxn = rmsnorm_q(*to_i8_shift(x), m['rms_ffn'][l])
        g_i8, sg = matvec_q(m['w1'][l], xn_i8, sxn)
        u_i8, su = matvec_q(m['w3'][l], xn_i8, sxn)
        sg_i8, ssg = silu_q(g_i8, sg)
        h_i8, sh = mul_q(sg_i8, ssg, u_i8, su)
        o_i8, so = matvec_q(m['w2'][l], h_i8, sh)
        x = x + from_i8_shift(o_i8, so)
    xn_i8, sxn = rmsnorm_q(*to_i8_shift(x), m['rms_final'])
    lg_i8, sl = matvec_q(m['tok_emb'], xn_i8, sxn)
    return from_i8_shift(lg_i8, sl)


def generate(n=17):
    m = S.load_model(S.MODEL_PATH); cfg = m['cfg']
    vocab = S.load_tok(S.TOK_PATH, cfg['vocab_size'])
    L, KH, HSd = cfg['n_layers'], cfg['n_kv_heads'], cfg['head_size']; Sq = 32
    kv = [{'K': np.zeros((Sq, KH, HSd), np.int8), 'sK': np.zeros(Sq, np.int32),
           'V': np.zeros((Sq, KH, HSd), np.int8), 'sV': np.zeros(Sq, np.int32)} for _ in range(L)]
    tokens = [1]; text = b""; nxt = 1
    for pos in range(n):
        lg = forward_seq(m, nxt, kv, pos); prev = nxt; nxt = int(np.argmax(lg))
        tokens.append(nxt); text += S.decode(prev, nxt, vocab)
    return text.decode('utf-8', 'replace'), tokens


if __name__ == "__main__":
    print("G2 proof : causal-attention sequencer orchestration (node primitives + int KV)")
    print(f"  expected : {EXPECTED!r}\n")
    txt, toks = generate()
    ok = (txt == EXPECTED)
    print(f"  seq orch : {'OK  ' if ok else 'MISMATCH'} : {txt!r}")
    if not ok:
        print(f"  tokens : {toks}")
    print("\nPROOF PASS - the causal sequencer orchestration is text-exact; "
          "this is the byte-level spec the RTL must reproduce." if ok else
          "PROOF FAIL - reconcile the orchestration before writing RTL.")
