#!/usr/bin/env python3
# =============================================================================
# prove_gen_node.py - bit-exact Python model of the gen_seq RTL forward.
#
# The float oracle (infer_v5gen_ref) diverges from the hardware at token 9 (a
# gap-3 near-tie) because the node's attention_head_op computes a SIMPLIFIED
# softmax (no /sqrt(HS), integer score>>10, LUT exp/inv) - see host/diag_swap_ops.
# This is a NODE property, not an RTL bug.
#
# So the correct reference for the RTL is a NODE-FAITHFUL forward : every op is a
# bit-exact replica of the node RTL (rmsnorm_op, silu_op, attention_head_op,
# vec_alu2), x kept as int8+shift throughout with requantizing residuals, FFN
# chunked 3x64 with global weight shifts. If this reproduces the RTL's 17 tokens
# EXACTLY, the sequencer is proven correct (F gate = RTL == node-faithful ref).
# =============================================================================
import json
import os
import numpy as np
import infer_v4sim as S
import infer_v5gen_ref as R

HERE = os.path.dirname(os.path.abspath(__file__))
D, H, KH, HS, HID, NL, VOCAB = 64, 8, 4, 8, 172, 5, 512
NREP = H // KH
RTL_TOKENS = [403, 407, 261, 378, 432, 383, 286, 261, 376, 268, 414, 421, 303, 428, 415, 412, 264]


def load_lut(p, signed=False):
    l = []
    for line in open(p):
        s = line.strip()
        if s and not s.startswith("//"):
            v = int(s, 16)
            if signed and v >= 0x8000: v -= 0x10000
            l.append(v)
    return l
RSQRT = load_lut(os.path.join(HERE, "..", "src", "rsqrt_lut.hex"))
SILU  = load_lut(os.path.join(HERE, "..", "src", "silu_lut.hex"), signed=True)
EXP   = load_lut(os.path.join(HERE, "..", "src", "exp_lut.hex"))
INV   = load_lut(os.path.join(HERE, "..", "src", "inv_lut.hex"))


# ---- ops (bit-exact replicas of the node RTL) ----
def requant(y, sin):                                  # requantize_i32
    m = max((abs(int(v)) for v in y), default=0)
    if m == 0: return [0]*len(y), sin
    add = max(0, (m.bit_length() - 1) - 6) if (m.bit_length()-1) > 6 else 0
    if add == 0: return [max(-128, min(127, int(v))) for v in y], sin
    half = 1 << (add-1)
    return [max(-128, min(127, (int(v)+half) >> add)) for v in y], sin+add

def node_rmsnorm(x_i8, w_i8, shift_w):                # rmsnorm_op.v
    acc = int(sum(int(xi)*int(xi) for xi in x_i8)) & 0xFFFFFF
    p = 0
    for ii in range(24):
        if (acc >> ii) & 1: p = ii
    sa = 8 - p
    acc_norm = ((acc << sa) & 0xFFFFFFFF) if sa >= 0 else (acc >> (-sa))
    raw = RSQRT[acc_norm & 0xFF]
    if sa & 1: raw = ((raw * 0xB505) >> 15) & 0xFFFF
    apsh = (sa >> 1) - 16
    out = []
    for i in range(len(x_i8)):
        prod = int(x_i8[i]) * int(w_i8[i]) * raw
        sh = (prod << apsh) if apsh >= 0 else ((prod + (0 if -apsh == 1 else (1 << (-apsh-1)))) >> (-apsh))
        out.append(max(-128, min(127, sh)))
    return out, shift_w

def node_fq(x_i8, sx, W_i8, sw, nout):                # node FQ (matmul + requant)
    y = [int(sum(int(W_i8[r][k])*int(x_i8[k]) for k in range(len(x_i8)))) for r in range(nout)]
    return requant(y, sx + sw)

def node_silu(x_i8, shift_x):                         # silu_op.v
    shx = shift_x + 4; osh = 11 + shift_x; out = []
    for xi in x_i8:
        x16 = (int(xi) << shx) if shx >= 0 else (int(xi) >> (-shx))
        x16 &= 0x1FFFF
        if x16 >= 0x10000: x16 -= 0x20000
        isg = x16 + 128
        li = 0 if isg < 0 else (255 if isg > 255 else isg)
        sv = SILU[li]
        sh = ((sv + (1 << (osh-1))) >> osh) if osh > 0 else (sv << (-osh) if osh < 0 else sv)
        out.append(max(-128, min(127, sh)))
    return out, shift_x

def node_valu(a_i8, sa, b_i8, sb, op):                # vec_alu2.v (0=MUL, 1=ADD)
    n = len(a_i8)
    if op == 0:
        base = sa + sb; acc = [int(a_i8[i])*int(b_i8[i]) for i in range(n)]
    else:
        base = min(sa, sb); da = sa - base; db = sb - base
        acc = [(int(a_i8[i]) << da) + (int(b_i8[i]) << db) for i in range(n)]
    m = max((abs(v) for v in acc), default=0)
    add = ((m.bit_length()-1) - 6) if m and (m.bit_length()-1) > 6 else 0
    out = []
    for v in acc:
        r = v + ((1 << (add-1)) if add > 0 else 0)
        out.append(max(-128, min(127, r >> add)))
    return out, base + add

def rr_head(head, cq, sq):                            # rope_op.v (Q15)
    out = [0]*HS
    for i in range(HS//2):
        xr = int(head[2*i]); xi = int(head[2*i+1])
        out[2*i]   = max(-128, min(127, (xr*cq[i] - xi*sq[i] + 16384) >> 15))
        out[2*i+1] = max(-128, min(127, (xr*sq[i] + xi*cq[i] + 16384) >> 15))
    return out

def node_rope(x_i8, nh, cq, sq):
    out = []
    for h in range(nh): out += rr_head(x_i8[h*HS:(h+1)*HS], cq, sq)
    return out

def clip8(v): return max(-128, min(127, v))

def node_mm(Q_i8, Kp, Vp, T):                         # attention_head_op.v (all heads)
    out = [0]*(H*HS)
    for h in range(H):
        kvh = h // NREP
        Qh = Q_i8[h*HS:(h+1)*HS]
        scores = []
        for t in range(T):
            dot = int(sum(int(Qh[i])*int(Kp[t][kvh*HS+i]) for i in range(HS)))
            scores.append(max(-32768, min(32767, dot)))
        mx = -128
        for t in range(T):
            hi = scores[t] >> 8
            if hi > mx: mx = hi
        ev = []; ssum = 0
        for t in range(T):
            eidx = ((scores[t] - (mx << 8)) >> 10) + 256
            eidx = 0 if eidx < 0 else (255 if eidx > 255 else eidx)
            ev.append(EXP[eidx]); ssum += EXP[eidx]
        psum = 0
        for ii in range(24):
            if (ssum >> ii) & 1: psum = ii
        shinv = 8 - psum
        snorm = (ssum << shinv) if shinv >= 0 else (ssum >> (-shinv))
        invs = INV[snorm & 0xFF]
        nsh = 15 - shinv
        att = []
        for t in range(T):
            prod = ev[t] * invs
            a = ((prod + (1 << (nsh-1))) >> nsh) if nsh > 0 else (prod << (-nsh) if nsh < 0 else prod)
            att.append(a & 0xFFFF)
        for d in range(HS):
            acc = sum(att[t] * int(Vp[t][kvh*HS+d]) for t in range(T))
            out[h*HS+d] = clip8((acc + 128) >> 8)
    return out


def to_i8_shift_list(vec):
    a = np.asarray(vec, np.float64); w, s = S.to_i8_shift(a)
    return w.astype(int).tolist(), int(s)


def gen_node_tokens(dump, n_tok=17):
    """Bit-exact node-faithful forward -> the 17 tokens the gen_seq RTL produces."""
    Lw = dump['layers']
    TMAX = 32

    # KV cache : per (layer, pos) roped-K, V (int8) + shifts
    Kc = [[None]*TMAX for _ in range(NL)]; Ksh = [[0]*TMAX for _ in range(NL)]
    Vc = [[None]*TMAX for _ in range(NL)]; Vsh = [[0]*TMAX for _ in range(NL)]

    toks = []; cur = 1
    for pos in range(n_tok):
        cq = dump['cos'][pos]; sq = dump['sin'][pos]
        # EMBED : x = tok_emb[cur] (row), shift = sw_emb
        x_i8 = [dump['tok_emb'][cur][k] for k in range(D)]; sxb = dump['tok_emb_s']
        for l in range(NL):
            w = Lw[l]
            # attention
            xn, sxn = node_rmsnorm(x_i8, w['rms_att'], w['rms_att_s'])
            Q, sQ = node_fq(xn, sxn, w['wq'], w['wq_s'], H*HS)
            Kk, sK = node_fq(xn, sxn, w['wk'], w['wk_s'], KH*HS)
            Vv, sV = node_fq(xn, sxn, w['wv'], w['wv_s'], KH*HS)
            Qr = node_rope(Q, H, cq, sq); Kr = node_rope(Kk, KH, cq, sq)
            Kc[l][pos] = Kr; Ksh[l][pos] = sK; Vc[l][pos] = Vv; Vsh[l][pos] = sV
            T = pos + 1
            sKref = max(Ksh[l][p] for p in range(T)); sVref = max(Vsh[l][p] for p in range(T))
            Kp = [[clip8(int(Kc[l][t][j]) >> (sKref - Ksh[l][t])) for j in range(KH*HS)] for t in range(T)]
            Vp = [[clip8(int(Vc[l][t][j]) >> (sVref - Vsh[l][t])) for j in range(KH*HS)] for t in range(T)]
            attn = node_mm(Qr, Kp, Vp, T); sA = sVref
            outv, so = node_fq(attn, sA, w['wo'], w['wo_s'], D)
            x_i8, sxb = node_valu(x_i8, sxb, outv, so, 1)               # attn residual
            # ffn (chunked 3x64, global weight shifts)
            xn, sxn = node_rmsnorm(x_i8, w['rms_ffn'], w['rms_ffn_s'])
            W1 = w['w1'] + [[0]*D for _ in range(192-HID)]
            W3 = w['w3'] + [[0]*D for _ in range(192-HID)]
            OUTV = None; sov = 0
            for c in range(3):
                H1, s1 = node_fq(xn, sxn, W1[c*64:c*64+64], w['w1_s'], 64)
                H3, s3 = node_fq(xn, sxn, W3[c*64:c*64+64], w['w3_s'], 64)
                SG, ssg = node_silu(H1, s1)
                HG, sh = node_valu(SG, ssg, H3, s3, 0)                  # MUL
                W2c = [[(w['w2'][r][c*64+k] if c*64+k < HID else 0) for k in range(64)] for r in range(D)]
                P, sp = node_fq(HG, sh, W2c, w['w2_s'], D)
                if c == 0: OUTV, sov = P, sp
                else: OUTV, sov = node_valu(OUTV, sov, P, sp, 1)        # reduce
            x_i8, sxb = node_valu(x_i8, sxb, OUTV, sov, 1)             # ffn residual
        # lm_head
        xn, sxn = node_rmsnorm(x_i8, dump['rms_final'], dump['rms_final_s'])
        csh = []; logits = []
        for c in range(VOCAB // 64):
            chunk = [dump['tok_emb'][c*64+r] for r in range(64)]
            y, sy = node_fq(xn, sxn, chunk, dump['tok_emb_s'], 64)
            logits += y; csh.append(sy)
        sref = max(csh)
        aligned = [logits[i] >> (sref - csh[i//64]) for i in range(VOCAB)]
        cur = max(range(VOCAB), key=lambda i: aligned[i])
        toks.append(cur)
    return toks


def main():
    dump = json.load(open(os.path.join(HERE, "gen_model_dump.json")))
    toks = gen_node_tokens(dump)
    print("node-faithful:", toks)
    print("RTL          :", RTL_TOKENS)
    ok = toks == RTL_TOKENS
    print("MATCH 17/17  :", "PASS" if ok else "FAIL")
    if not ok:
        for i in range(17):
            if toks[i] != RTL_TOKENS[i]:
                print(f"  first diff @ pos {i}: node={toks[i]} rtl={RTL_TOKENS[i]}"); break
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
