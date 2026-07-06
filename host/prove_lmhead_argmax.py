#!/usr/bin/env python3
# =============================================================================
# prove_lmhead_argmax.py - G2 proof for the lm_head + argmax RTL (Phase 5c).
#
# The generation head after the 5 layers : final RMSNorm, then project x[64] to
# logits[vocab=512] via tok_emb (shared classifier), then argmax -> next token.
# The RTL does this with node primitives + integer re-align, no float :
#   1. FN(x, rms_final)                         -> xn[64]
#   2. 8 x FQ(N=64, tok_emb chunk c)            -> logits chunk c [64], shift sy_c
#   3. re-align : right-shift each chunk to sref=max(sy_c)  (integer)
#   4. argmax over the 512 aligned int logits   -> token (plain > comparison)
#
# Proven : with the full integer causal forward (prove_causal_orch primitives),
# this integer lm_head+argmax yields exactly the oracle text + 17 tokens. So the
# RTL argmax is a plain running-max over a single common shift (no cross-shift
# compare), and the chunk matmul is the FQ the node already does.
# =============================================================================
import os
import numpy as np

import infer_v4sim as S
from infer_v4sim import to_i8_shift, from_i8_shift, rmsnorm_q, matvec_q, silu_q, mul_q
import prove_causal_orch as PC

HERE = os.path.dirname(os.path.abspath(__file__))
S.MODEL_PATH = os.path.join(HERE, "models", "stories260K.bin")
S.TOK_PATH   = os.path.join(HERE, "models", "tok512.bin")
EXPECTED = 'Once upon a time, there was a little girl named Lily. She lo'


def lm_head_argmax_int(m, x):
    V = m['cfg']['vocab_size']
    xn_i8, sxn = rmsnorm_q(*to_i8_shift(x), m['rms_final'])
    W = m['tok_emb']                     # [vocab, 64], shared classifier
    ci8, csh = [], []
    for c in range(V // 64):
        y_i8, sy = matvec_q(W[c*64:(c+1)*64], xn_i8, sxn)
        ci8.append(y_i8.astype(np.int32)); csh.append(sy)
    sref = max(csh)
    logits = np.concatenate([ci8[c] >> (sref - csh[c]) for c in range(V // 64)])
    return int(np.argmax(logits))


def forward_full(m, token, kv, pos):
    cfg = m['cfg']; H, KH, HS, D = cfg['n_heads'], cfg['n_kv_heads'], cfg['head_size'], cfg['dim']
    x = m['tok_emb'][token].astype(np.float32).copy()
    for l in range(cfg['n_layers']):
        xn_i8, sxn = rmsnorm_q(*to_i8_shift(x), m['rms_att'][l])
        Q_i8, sQ = matvec_q(m['wq'][l], xn_i8, sxn)
        K_i8, sK = matvec_q(m['wk'][l], xn_i8, sxn)
        Vv_i8, sV = matvec_q(m['wv'][l], xn_i8, sxn)
        fr = m['freq_cis_real'][pos]; fi = m['freq_cis_imag'][pos]
        Qr, sQ = PC.rope_seq(Q_i8, sQ, H, fr, fi)
        Kr, sK = PC.rope_seq(K_i8, sK, KH, fr, fi)
        kv[l]['K'][pos] = Kr.reshape(KH, HS); kv[l]['sK'][pos] = sK
        kv[l]['V'][pos] = Vv_i8.reshape(KH, HS); kv[l]['sV'][pos] = sV
        T = pos + 1
        sKref = max(kv[l]['sK'][p] for p in range(T)); sVref = max(kv[l]['sV'][p] for p in range(T))
        Kp = np.concatenate([np.clip(kv[l]['K'][p].astype(np.int32) >> (sKref - kv[l]['sK'][p]), -128, 127).reshape(-1) for p in range(T)]).astype(np.int8)
        Vp = np.concatenate([np.clip(kv[l]['V'][p].astype(np.int32) >> (sVref - kv[l]['sV'][p]), -128, 127).reshape(-1) for p in range(T)]).astype(np.int8)
        attn_i8, sA = PC.mm_node(Qr, sQ, Kp, sKref, Vp, sVref, T, H, KH)
        out_i8, so = matvec_q(m['wo'][l], attn_i8, sA); x = x + from_i8_shift(out_i8, so)
        xn_i8, sxn = rmsnorm_q(*to_i8_shift(x), m['rms_ffn'][l])
        g_i8, sg = matvec_q(m['w1'][l], xn_i8, sxn); u_i8, su = matvec_q(m['w3'][l], xn_i8, sxn)
        sg_i8, ssg = silu_q(g_i8, sg); h_i8, sh = mul_q(sg_i8, ssg, u_i8, su)
        o_i8, so = matvec_q(m['w2'][l], h_i8, sh); x = x + from_i8_shift(o_i8, so)
    return lm_head_argmax_int(m, x)


def generate(n=17):
    m = S.load_model(S.MODEL_PATH); cfg = m['cfg']
    vocab = S.load_tok(S.TOK_PATH, cfg['vocab_size'])
    L, KH, HS = cfg['n_layers'], cfg['n_kv_heads'], cfg['head_size']; Sq = 32
    kv = [{'K': np.zeros((Sq, KH, HS), np.int8), 'sK': np.zeros(Sq, np.int32),
           'V': np.zeros((Sq, KH, HS), np.int8), 'sV': np.zeros(Sq, np.int32)} for _ in range(L)]
    tokens = [1]; text = b""; nxt = 1
    for pos in range(n):
        nxt = forward_full(m, nxt, kv, pos); prev = tokens[-1]
        tokens.append(nxt); text += S.decode(prev, nxt, vocab)
    return text.decode('utf-8', 'replace'), tokens


if __name__ == "__main__":
    print("G2 proof : lm_head (rms_final + 8 FQ chunks) + integer argmax")
    print(f"  expected : {EXPECTED!r}\n")
    txt, toks = generate()
    ok = (txt == EXPECTED)
    print(f"  int lm_head+argmax : {'OK  ' if ok else 'MISMATCH'} : {txt!r}")
    print("\nPROOF PASS - argmax is a plain running-max at a single common shift; "
          "the RTL head reuses FN + chunked FQ + integer re-align." if ok else
          "PROOF FAIL - reconcile before writing RTL.")
