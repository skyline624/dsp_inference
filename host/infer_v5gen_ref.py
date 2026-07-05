#!/usr/bin/env python3
# =============================================================================
# infer_v5gen_ref.py - RTL-FRIENDLY reference for the Phase-5 generation head.
#
# Same greedy generation as infer_v4sim.py, but with two simplifications proven
# (below) to leave the generated text bit-identical, and chosen because they map
# to integer-only RTL:
#
#   1. KV-cache stored as (int8, per-position shift). At each MM, every cached
#      position is right-shifted (integer) to a common reference shift = max of
#      the T shifts. No float dequant/requant -> a hardware sequencer can do it
#      with plain arithmetic right-shifts.
#   2. Everything else identical to infer_v4sim (rmsnorm, matmul, silu, rope,
#      softmax all in int8+pow2-shift).
#
# Verified: produces exactly
#   'Once upon a time, there was a little girl named Lily. She lo'
# with the same 17 tokens as infer_v4sim.py -> this is the oracle the RTL
# generation sequencer (ffn/attn/lm_head + argmax + KV loop) must match.
# =============================================================================
import os
import numpy as np

import infer_v4sim as S
from infer_v4sim import (to_i8_shift, from_i8_shift, rmsnorm_q, matvec_q,
                         silu_q, mul_q, apply_rope_q, softmax_q)

HERE = os.path.dirname(os.path.abspath(__file__))
S.MODEL_PATH = os.path.join(HERE, "models", "stories260K.bin")
S.TOK_PATH   = os.path.join(HERE, "models", "tok512.bin")


def forward_rtl(m, token, kv, pos):
    cfg = m['cfg']
    H, KH, HS, D = cfg['n_heads'], cfg['n_kv_heads'], cfg['head_size'], cfg['dim']
    n_rep = H // KH
    x = m['tok_emb'][token].astype(np.float32).copy()
    for l in range(cfg['n_layers']):
        # ---- attention ----
        xn_i8, sxn = rmsnorm_q(*to_i8_shift(x), m['rms_att'][l])
        Q_i8, sQ = matvec_q(m['wq'][l], xn_i8, sxn)
        K_i8, sK = matvec_q(m['wk'][l], xn_i8, sxn)
        V_i8, sV = matvec_q(m['wv'][l], xn_i8, sxn)
        Q = from_i8_shift(Q_i8, sQ).reshape(H, HS).astype(np.float32)
        K = from_i8_shift(K_i8, sK).reshape(KH, HS).astype(np.float32)
        V = from_i8_shift(V_i8, sV).reshape(KH, HS).astype(np.float32)
        fr = m['freq_cis_real'][pos]; fi = m['freq_cis_imag'][pos]
        Q_i8, sQ = apply_rope_q(*to_i8_shift(Q), fr, fi)
        Kr_i8, sKr = apply_rope_q(*to_i8_shift(K), fr, fi)
        Vv_i8, sVv = to_i8_shift(V)
        kv[l]['K'][pos] = Kr_i8.reshape(KH, HS); kv[l]['sK'][pos] = sKr
        kv[l]['V'][pos] = Vv_i8.reshape(KH, HS); kv[l]['sV'][pos] = sVv

        T = pos + 1
        Q_f = from_i8_shift(Q_i8, sQ)
        # integer KV re-align : right-shift every position to the max shift
        sKref = max(kv[l]['sK'][p] for p in range(T))
        sVref = max(kv[l]['sV'][p] for p in range(T))
        Ks = np.stack([kv[l]['K'][p].astype(np.int32) >> (sKref - kv[l]['sK'][p]) for p in range(T)])
        Vs = np.stack([kv[l]['V'][p].astype(np.int32) >> (sVref - kv[l]['sV'][p]) for p in range(T)])
        Ks_q = np.repeat(from_i8_shift(Ks, sKref), n_rep, axis=1)
        Vs_q = np.repeat(from_i8_shift(Vs, sVref), n_rep, axis=1)
        scores = np.einsum('hd,thd->ht', Q_f, Ks_q) / np.sqrt(HS)
        attn_i8, sa = softmax_q(*to_i8_shift(scores), axis=-1)
        out = np.einsum('ht,thd->hd', from_i8_shift(attn_i8, sa), Vs_q).reshape(D)
        out_i8, so = matvec_q(m['wo'][l], *to_i8_shift(out))
        x = x + from_i8_shift(out_i8, so)

        # ---- ffn ----
        xn_i8, sxn = rmsnorm_q(*to_i8_shift(x), m['rms_ffn'][l])
        g_i8, sg = matvec_q(m['w1'][l], xn_i8, sxn)
        u_i8, su = matvec_q(m['w3'][l], xn_i8, sxn)
        sg_i8, ssg = silu_q(g_i8, sg)
        h_i8, sh = mul_q(sg_i8, ssg, u_i8, su)
        o_i8, so = matvec_q(m['w2'][l], h_i8, sh)
        x = x + from_i8_shift(o_i8, so)

    # ---- lm_head ----
    xn_i8, sxn = rmsnorm_q(*to_i8_shift(x), m['rms_final'])
    lg_i8, sl = matvec_q(m['tok_emb'], xn_i8, sxn)
    return from_i8_shift(lg_i8, sl)


def generate(n_tokens=17):
    m = S.load_model(S.MODEL_PATH); cfg = m['cfg']
    vocab = S.load_tok(S.TOK_PATH, cfg['vocab_size'])
    L, KH, HS = cfg['n_layers'], cfg['n_kv_heads'], cfg['head_size']; Sq = 32
    kv = [{'K': np.zeros((Sq, KH, HS), np.int8), 'sK': np.zeros(Sq, np.int32),
           'V': np.zeros((Sq, KH, HS), np.int8), 'sV': np.zeros(Sq, np.int32)} for _ in range(L)]
    tokens = [1]; text = b""; nxt = 1
    for pos in range(n_tokens):
        lg = forward_rtl(m, nxt, kv, pos); prev = nxt; nxt = int(np.argmax(lg))
        tokens.append(nxt); text += S.decode(prev, nxt, vocab)
    return text.decode('utf-8', 'replace'), tokens


if __name__ == "__main__":
    txt, toks = generate()
    print("RTL-friendly generation reference:")
    print(f"  text   : {txt!r}")
    print(f"  tokens : {toks}")
    expected = 'Once upon a time, there was a little girl named Lily. She lo'
    print(f"  {'OK' if txt == expected else 'MISMATCH'} vs infer_v4sim baseline")
