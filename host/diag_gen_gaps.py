#!/usr/bin/env python3
# diagnostic : per-token top-2 logit gap of the FLOAT oracle (infer_v5gen_ref).
# A tiny gap at a token means the argmax is a near-tie -> the node's LUT-based
# forward (rmsnorm/silu/softmax) can flip it vs the float oracle. Classifies the
# token-9 divergence seen in test_gen_F (RTL 9/17 exact then diverges).
import numpy as np
import infer_v4sim as S
from infer_v4sim import (to_i8_shift, from_i8_shift, rmsnorm_q, matvec_q, silu_q, mul_q)
import infer_v5gen_ref as R
import prove_causal_orch as PC


def logits_int(m, x):
    V = m['cfg']['vocab_size']
    xn_i8, sxn = rmsnorm_q(*to_i8_shift(x), m['rms_final'])
    W = m['tok_emb']
    ci8, csh = [], []
    for c in range(V // 64):
        y_i8, sy = matvec_q(W[c*64:(c+1)*64], xn_i8, sxn)
        ci8.append(y_i8.astype(np.int32)); csh.append(sy)
    sref = max(csh)
    return np.concatenate([ci8[c] >> (sref - csh[c]) for c in range(V // 64)])


def forward_x(m, token, kv, pos):
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
    return x


def main():
    m = S.load_model(S.MODEL_PATH); cfg = m['cfg']
    L, KH, HS = cfg['n_layers'], cfg['n_kv_heads'], cfg['head_size']; Sq = 32
    kv = [{'K': np.zeros((Sq, KH, HS), np.int8), 'sK': np.zeros(Sq, np.int32),
           'V': np.zeros((Sq, KH, HS), np.int8), 'sV': np.zeros(Sq, np.int32)} for _ in range(L)]
    rtl = [403, 407, 261, 378, 432, 383, 286, 261, 376, 268, 414, 421, 303, 428, 415, 412, 264]
    nxt = 1
    print(f"{'pos':>3} {'oracle':>6} {'rtl':>5} {'top1':>6} {'top2':>6} {'gap':>5} {'rtl_rank':>8} {'rtl_logit':>9}")
    for pos in range(17):
        x = forward_x(m, nxt, kv, pos)
        lg = logits_int(m, x)
        order = np.argsort(lg)[::-1]
        top1, top2 = int(order[0]), int(order[1])
        tok = top1
        rtl_tok = rtl[pos]
        rank = int(np.where(order == rtl_tok)[0][0])
        print(f"{pos:>3} {tok:>6} {rtl_tok:>5} {int(lg[top1]):>6} {int(lg[top2]):>6} "
              f"{int(lg[top1]-lg[top2]):>5} {rank:>8} {int(lg[rtl_tok]):>9}"
              f"{'  <-- DIVERGE' if tok != rtl_tok else ''}")
        nxt = tok      # follow the ORACLE path (so pos>9 gaps are the oracle's, not the RTL's)


if __name__ == "__main__":
    main()
