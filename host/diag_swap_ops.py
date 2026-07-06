#!/usr/bin/env python3
# diagnostic : swap individual ops between FLOAT (oracle) and NODE-LUT (hardware)
# in the 17-token forward, to find which LUT drives the token-9 divergence
# (oracle 298 vs RTL 268). rmsnorm is used 11x/token, silu/softmax 5x each.
import numpy as np
import infer_v4sim as S
from infer_v4sim import (to_i8_shift, from_i8_shift, rmsnorm_q, matvec_q, silu_q, mul_q)
import infer_v5gen_ref as R
import prove_causal_orch as PC
import os

RTL = [403, 407, 261, 378, 432, 383, 286, 261, 376, 268, 414, 421, 303, 428, 415, 412, 264]
ORC = [403, 407, 261, 378, 432, 383, 286, 261, 376, 298, 315, 421, 395, 317, 426, 338, 401]

# ---- node rmsnorm replica (bit-exact, from step E / rmsnorm_op.v) ----
def load_lut(p):
    l = []
    for line in open(p):
        s = line.strip()
        if s and not s.startswith("//"): l.append(int(s, 16))
    return l
LUT = load_lut(os.path.join(os.path.dirname(__file__), "..", "src", "rsqrt_lut.hex"))

def node_rmsnorm(x_i8, w_i8, shift_w):
    D = len(x_i8)
    acc = int(sum(int(xi)*int(xi) for xi in x_i8)) & 0xFFFFFF
    p = 0
    for ii in range(24):
        if (acc >> ii) & 1: p = ii
    sa = 8 - p
    acc_norm = ((acc << sa) & 0xFFFFFFFF) if sa >= 0 else (acc >> (-sa))
    raw = LUT[acc_norm & 0xFF]
    if sa & 1: raw = ((raw * 0xB505) >> 15) & 0xFFFF
    apsh = (sa >> 1) - 16
    out = []
    for i in range(D):
        prod = int(x_i8[i]) * int(w_i8[i]) * raw
        if apsh >= 0: sh = prod << apsh
        else:
            na = -apsh; rnd = 0 if na == 1 else (1 << (na-1)); sh = (prod + rnd) >> na
        out.append(max(-128, min(127, sh)))
    return np.array(out, np.int8), shift_w

def rms(x, weight, use_node):
    x_i8, sx = to_i8_shift(x)
    if not use_node:
        return rmsnorm_q(x_i8, sx, weight)
    w_i8, sw = to_i8_shift(weight)
    return node_rmsnorm(x_i8, w_i8, sw)

# ---- node silu replica (bit-exact, from silu_op.v) ----
def load_lut16s(p):
    l = []
    for line in open(p):
        s = line.strip()
        if s and not s.startswith("//"):
            v = int(s, 16)
            if v >= 0x8000: v -= 0x10000
            l.append(v)
    return l
SILU = load_lut16s(os.path.join(os.path.dirname(__file__), "..", "src", "silu_lut.hex"))

def node_silu(x_i8, shift_x):
    shx_p4 = shift_x + 4
    out_shift = 11 + shift_x
    out = []
    for xi in x_i8:
        xs = int(xi)
        x16 = (xs << shx_p4) if shx_p4 >= 0 else (xs >> (-shx_p4))
        x16_17 = x16 & 0x1FFFF
        if x16_17 >= 0x10000: x16_17 -= 0x20000
        isg = x16_17 + 128
        li = 0 if isg < 0 else (255 if isg > 255 else isg)
        sv = SILU[li]
        if out_shift > 0:
            shifted = (sv + (1 << (out_shift-1))) >> out_shift
        elif out_shift < 0:
            shifted = sv << (-out_shift)
        else:
            shifted = sv
        out.append(max(-128, min(127, shifted)))
    return np.array(out, np.int8), shift_x

def silu(g_i8, sg, use_node):
    return node_silu(g_i8, sg) if use_node else silu_q(g_i8, sg)

# ---- node attention replica (bit-exact, from attention_head_op.v) ----
def load_lut16u(p):
    l = []
    for line in open(p):
        s = line.strip()
        if s and not s.startswith("//"): l.append(int(s, 16))
    return l
EXP = load_lut16u(os.path.join(os.path.dirname(__file__), "..", "src", "exp_lut.hex"))
INV = load_lut16u(os.path.join(os.path.dirname(__file__), "..", "src", "inv_lut.hex"))
HSN = 8

def node_mm(Q_i8, sQ, Kpack_i8, sK, Vpack_i8, sV, T, H, KH):
    n_rep = H // KH
    Q = np.asarray(Q_i8, np.int64).reshape(H, HSN)
    K = np.asarray(Kpack_i8, np.int64).reshape(T, KH, HSN)
    V = np.asarray(Vpack_i8, np.int64).reshape(T, KH, HSN)
    out = np.zeros((H, HSN), np.int64)
    for h in range(H):
        kvh = h // n_rep
        scores = [max(-32768, min(32767, int(sum(int(Q[h][i])*int(K[t][kvh][i]) for i in range(HSN))))) for t in range(T)]
        maxs = -128
        for t in range(T):
            hi = scores[t] >> 8                 # top byte, signed
            if hi > maxs: maxs = hi
        ev = []; ssum = 0
        for t in range(T):
            diff = scores[t] - (maxs << 8)
            eidx = (diff >> 10) + 256
            eidx = 0 if eidx < 0 else (255 if eidx > 255 else eidx)
            e = EXP[eidx]; ev.append(e); ssum += e
        psum = 0
        for ii in range(24):
            if (ssum >> ii) & 1: psum = ii
        shift_inv = 8 - psum
        sum_norm = (ssum << shift_inv) if shift_inv >= 0 else (ssum >> (-shift_inv))
        inv_sum = INV[sum_norm & 0xFF]
        norm_shift = 15 - shift_inv
        att = []
        for t in range(T):
            prod = ev[t] * inv_sum
            if norm_shift > 0: a = (prod + (1 << (norm_shift-1))) >> norm_shift
            elif norm_shift < 0: a = prod << (-norm_shift)
            else: a = prod
            att.append(a & 0xFFFF)              # attn_norm is 16-bit
        for d in range(HSN):
            acc = sum(att[t] * int(V[t][kvh][d]) for t in range(T))
            out[h][d] = max(-128, min(127, (acc + 128) >> 8))
    return out.reshape(-1).astype(np.int8), sV


def forward_x(m, token, kv, pos, node_rms=False, node_silu=False, node_attn=False):
    cfg = m['cfg']; H, KH, HS, D = cfg['n_heads'], cfg['n_kv_heads'], cfg['head_size'], cfg['dim']
    x = m['tok_emb'][token].astype(np.float32).copy()
    for l in range(cfg['n_layers']):
        xn_i8, sxn = rms(x, m['rms_att'][l], node_rms)
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
        mm = node_mm if node_attn else PC.mm_node
        attn_i8, sA = mm(Qr, sQ, Kp, sKref, Vp, sVref, T, H, KH)
        out_i8, so = matvec_q(m['wo'][l], attn_i8, sA); x = x + from_i8_shift(out_i8, so)
        xn_i8, sxn = rms(x, m['rms_ffn'][l], node_rms)
        g_i8, sg = matvec_q(m['w1'][l], xn_i8, sxn); u_i8, su = matvec_q(m['w3'][l], xn_i8, sxn)
        sg_i8, ssg = silu(g_i8, sg, node_silu); h_i8, sh = mul_q(sg_i8, ssg, u_i8, su)
        o_i8, so = matvec_q(m['w2'][l], h_i8, sh); x = x + from_i8_shift(o_i8, so)
    return x

def logits_int(m, x, node_rms=False):
    V = m['cfg']['vocab_size']
    xn_i8, sxn = rms(x, m['rms_final'], node_rms)
    W = m['tok_emb']; ci8, csh = [], []
    for c in range(V // 64):
        y_i8, sy = matvec_q(W[c*64:(c+1)*64], xn_i8, sxn)
        ci8.append(y_i8.astype(np.int32)); csh.append(sy)
    sref = max(csh)
    return np.concatenate([ci8[c] >> (sref - csh[c]) for c in range(V // 64)])


def gen(m, node_rms=False, node_silu=False, node_attn=False):
    cfg = m['cfg']; L, KH, HS = cfg['n_layers'], cfg['n_kv_heads'], cfg['head_size']; Sq = 32
    kv = [{'K': np.zeros((Sq, KH, HS), np.int8), 'sK': np.zeros(Sq, np.int32),
           'V': np.zeros((Sq, KH, HS), np.int8), 'sV': np.zeros(Sq, np.int32)} for _ in range(L)]
    toks = []; nxt = 1
    for pos in range(17):
        x = forward_x(m, nxt, kv, pos, node_rms, node_silu, node_attn)
        lg = logits_int(m, x, node_rms)
        nxt = int(np.argmax(lg)); toks.append(nxt)
    return toks


def firstdiff(a, b):
    for i in range(len(a)):
        if a[i] != b[i]: return i
    return -1


def main():
    m = S.load_model(S.MODEL_PATH)
    print("RTL (hardware)          :", RTL)
    print("ORACLE (all float)      :", ORC)
    for name, kw in [("all-float", {}),
                     ("node-attn", dict(node_attn=True)),
                     ("node-rms+silu+attn", dict(node_rms=True, node_silu=True, node_attn=True))]:
        t = gen(m, **kw)
        dR = firstdiff(t, RTL); dO = firstdiff(t, ORC)
        print(f"{name:18s}: {t}")
        print(f"{'':18s}  1st diff vs RTL @ pos {dR} ; vs ORACLE @ pos {dO}")


if __name__ == "__main__":
    main()
