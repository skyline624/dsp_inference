#!/usr/bin/env python3
# =============================================================================
# infer_v4sim.py
# Simule v4 ENTIEREMENT en Python avec la contrainte int8 + power-of-2 shift
# partout (matmuls et activations). Si le texte genere est lisible, la
# convention numerique est suffisante pour passer au RTL.
#
# Pas de FPGA dans ce script - tout calcule en numpy. But : valider la
# precision avant d'investir dans 2000 lignes de Verilog.
# =============================================================================

import os, re, struct, sys
import numpy as np

HERE       = os.path.dirname(os.path.abspath(__file__))
DSPC_HOST  = os.path.join(HERE, "..", "..", "dsp_coproc", "host")
MODEL_PATH = os.path.join(DSPC_HOST, "models", "stories260K.bin")
TOK_PATH   = os.path.join(DSPC_HOST, "models", "tok512.bin")

# ─── Quantification power-of-2 ────────────────────────────────────────────
def to_i8_shift(x):
    """float -> (int8, shift)."""
    m = float(np.max(np.abs(x)))
    if m == 0: return np.zeros_like(x, dtype=np.int8), 0
    s = int(np.ceil(np.log2(m / 127.0)))
    q = np.clip(np.round(x / (2.0 ** s)), -128, 127).astype(np.int8)
    return q, s

def from_i8_shift(x_i8, s):
    return x_i8.astype(np.float64) * (2.0 ** s)

def requantize_i32(y, shift_in):
    m = int(np.max(np.abs(y)))
    if m == 0: return np.zeros_like(y, dtype=np.int8), shift_in
    add = max(0, int(np.ceil(np.log2(m / 127.0))))
    if add == 0:
        y8 = np.clip(y, -128, 127).astype(np.int8)
    else:
        half = 1 << (add - 1)
        y8 = np.clip((y + half) >> add, -128, 127).astype(np.int8)
    return y8, shift_in + add

# ─── Operateurs en int8+shift ────────────────────────────────────────────
def matvec_q(W, x_i8, sx):
    """y = W . x (W float, x deja quantifie int8+sx). Renvoie (y_i8, sy)."""
    W_i8, sw = to_i8_shift(W)
    y_i32 = W_i8.astype(np.int64) @ x_i8.astype(np.int64)
    return requantize_i32(y_i32, sx + sw)

def rmsnorm_q(x_i8, sx, weight, eps=1e-5):
    """RMSNorm en int : sortie (y_i8, sy)."""
    # passe en float pour clarte (un RTL utiliserait LUT 1/sqrt)
    x = from_i8_shift(x_i8, sx)
    y = x * weight / np.sqrt(np.mean(x**2) + eps)
    return to_i8_shift(y)

def silu_q(x_i8, sx):
    x = from_i8_shift(x_i8, sx)
    y = x / (1.0 + np.exp(-x))
    return to_i8_shift(y)

def softmax_q(x_i8, sx, axis=-1):
    x = from_i8_shift(x_i8, sx)
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x); s = e / e.sum(axis=axis, keepdims=True)
    return to_i8_shift(s)

def add_q(a_i8, sa, b_i8, sb):
    """Addition de deux tenseurs quantifies."""
    a = from_i8_shift(a_i8, sa); b = from_i8_shift(b_i8, sb)
    return to_i8_shift(a + b)

def mul_q(a_i8, sa, b_i8, sb):
    """Multiplication element par element."""
    a = from_i8_shift(a_i8, sa); b = from_i8_shift(b_i8, sb)
    return to_i8_shift(a * b)

def apply_rope_q(x_i8, sx, fr, fi):
    """RoPE : sortie quantifiee."""
    x = from_i8_shift(x_i8, sx)
    h, hs = x.shape
    x = x.reshape(h, hs // 2, 2)
    r = x[..., 0]; i = x[..., 1]
    nr = r * fr - i * fi
    ni = r * fi + i * fr
    out = np.stack([nr, ni], axis=-1).reshape(h, hs)
    return to_i8_shift(out)

# ─── Loader (reuse v3e logic) ─────────────────────────────────────────────
def load_model(path):
    with open(path,'rb') as f: header = f.read(28)
    fields = struct.unpack('<7i', header)
    keys = ['dim','hidden_dim','n_layers','n_heads','n_kv_heads','vocab_size','seq_len']
    cfg = dict(zip(keys, fields))
    cfg['shared_classifier'] = cfg['vocab_size'] > 0
    cfg['vocab_size'] = abs(cfg['vocab_size'])
    cfg['head_size'] = cfg['dim'] // cfg['n_heads']
    raw = np.fromfile(path, dtype=np.float32, offset=28)
    pos=[0]
    def t(*s):
        n=int(np.prod(s)); v=raw[pos[0]:pos[0]+n].reshape(s).copy(); pos[0]+=n; return v
    L,D,H,KH,HS,HID,V,S = (cfg['n_layers'],cfg['dim'],cfg['n_heads'],
        cfg['n_kv_heads'],cfg['head_size'],cfg['hidden_dim'],cfg['vocab_size'],cfg['seq_len'])
    return {'cfg':cfg, 'tok_emb':t(V,D), 'rms_att':t(L,D),
            'wq':t(L,H*HS,D), 'wk':t(L,KH*HS,D), 'wv':t(L,KH*HS,D), 'wo':t(L,D,H*HS),
            'rms_ffn':t(L,D), 'w1':t(L,HID,D), 'w2':t(L,D,HID), 'w3':t(L,HID,D),
            'rms_final':t(D),
            'freq_cis_real':t(S,HS//2), 'freq_cis_imag':t(S,HS//2)}

def load_tok(p, v):
    out=[]
    with open(p,'rb') as f:
        struct.unpack('<i', f.read(4))
        for _ in range(v):
            struct.unpack('<f', f.read(4)); n=struct.unpack('<i', f.read(4))[0]
            out.append(f.read(n))
    return out

BT = re.compile(rb'<0x([0-9A-Fa-f]{2})>')
def decode(prev, t, vocab):
    p = vocab[t]
    if prev==1 and p.startswith(b' '): p=p[1:]
    m=BT.match(p)
    if m: p=bytes([int(m.group(1),16)])
    return p

# ─── Forward avec QUANTIFICATION POWER-OF-2 PARTOUT ──────────────────────
def forward(m, token, kv_caches, pos):
    cfg = m['cfg']
    H,KH,HS,D = cfg['n_heads'],cfg['n_kv_heads'],cfg['head_size'],cfg['dim']
    n_rep = H // KH

    # embedding (reste en float, on quantifie apres)
    x = m['tok_emb'][token].astype(np.float32).copy()

    for l in range(cfg['n_layers']):
        # ----- ATTN -----
        # RMSNorm puis quantifie pour les matmuls
        x_norm_i8, sxn = rmsnorm_q(*to_i8_shift(x), m['rms_att'][l])

        Q_i8, sQ = matvec_q(m['wq'][l], x_norm_i8, sxn)
        K_i8, sK = matvec_q(m['wk'][l], x_norm_i8, sxn)
        V_i8, sV = matvec_q(m['wv'][l], x_norm_i8, sxn)

        Q = from_i8_shift(Q_i8, sQ).reshape(H, HS).astype(np.float32)
        K = from_i8_shift(K_i8, sK).reshape(KH, HS).astype(np.float32)
        V = from_i8_shift(V_i8, sV).reshape(KH, HS).astype(np.float32)

        fr = m['freq_cis_real'][pos]; fi = m['freq_cis_imag'][pos]
        Q_i8, sQ = apply_rope_q(*to_i8_shift(Q), fr, fi)
        K_i8, sK = apply_rope_q(*to_i8_shift(K), fr, fi)
        V_i8_2d, sV2 = to_i8_shift(V)        # re-quantifie V en 2D (KH, HS)

        kv_caches[l]['K'][pos] = K_i8; kv_caches[l]['sK'][pos] = sK
        kv_caches[l]['V'][pos] = V_i8_2d; kv_caches[l]['sV'][pos] = sV2

        # Attention en float (avec K/V dequantifies depuis cache)
        Q_f = from_i8_shift(Q_i8, sQ)
        Ks = np.array([from_i8_shift(kv_caches[l]['K'][p], kv_caches[l]['sK'][p])
                       for p in range(pos+1)])     # (t, KH, HS)
        Vs = np.array([from_i8_shift(kv_caches[l]['V'][p], kv_caches[l]['sV'][p])
                       for p in range(pos+1)])
        Ks_q = np.repeat(Ks, n_rep, axis=1)
        Vs_q = np.repeat(Vs, n_rep, axis=1)

        scores = np.einsum('hd,thd->ht', Q_f, Ks_q) / np.sqrt(HS)
        attn_i8, sa = softmax_q(*to_i8_shift(scores), axis=-1)
        attn = from_i8_shift(attn_i8, sa)
        out = np.einsum('ht,thd->hd', attn, Vs_q).reshape(D)

        # Wo + residu
        out_i8, so = matvec_q(m['wo'][l], *to_i8_shift(out))
        x = x + from_i8_shift(out_i8, so)

        # ----- FFN -----
        x_norm_i8, sxn = rmsnorm_q(*to_i8_shift(x), m['rms_ffn'][l])
        gate_i8, sg = matvec_q(m['w1'][l], x_norm_i8, sxn)
        up_i8,   su = matvec_q(m['w3'][l], x_norm_i8, sxn)
        # h = silu(gate) * up
        silu_g_i8, ssg = silu_q(gate_i8, sg)
        h_i8, sh = mul_q(silu_g_i8, ssg, up_i8, su)
        out_i8, so = matvec_q(m['w2'][l], h_i8, sh)
        x = x + from_i8_shift(out_i8, so)

    # final
    x_norm_i8, sxn = rmsnorm_q(*to_i8_shift(x), m['rms_final'])
    logits_i8, sl = matvec_q(m['tok_emb'], x_norm_i8, sxn)
    return from_i8_shift(logits_i8, sl)

def main():
    print(f"Chargement modele : {MODEL_PATH}")
    m = load_model(MODEL_PATH); cfg = m['cfg']
    print(f"  dim={cfg['dim']} layers={cfg['n_layers']} heads={cfg['n_heads']}/{cfg['n_kv_heads']} vocab={cfg['vocab_size']}")
    vocab = load_tok(TOK_PATH, cfg['vocab_size'])

    print("Simulation v4 (int8 + power-of-2 shift PARTOUT, sampling argmax)")
    print("Reference v3e attendue : 'Once upon a time, there was a girl named Lily. She loved'\n")

    L, KH, HS, S = cfg['n_layers'], cfg['n_kv_heads'], cfg['head_size'], 32
    kv_caches = [{
        'K': np.zeros((S, KH, HS), dtype=np.int8),
        'sK': np.zeros(S, dtype=np.int32),
        'V': np.zeros((S, KH, HS), dtype=np.int8),
        'sV': np.zeros(S, dtype=np.int32),
    } for _ in range(L)]

    tokens = [1]; text = b""
    nxt = 1
    for pos in range(17):
        logits = forward(m, nxt, kv_caches, pos)
        prev = nxt
        nxt = int(np.argmax(logits))     # argmax = greedy deterministe
        tokens.append(nxt)
        p = decode(prev, nxt, vocab); text += p
        print(f"  pos={pos:2d} -> tok={nxt:4d}  {repr(p.decode('utf-8', errors='replace'))}")

    print(f"\nTexte v4sim : {text.decode('utf-8', errors='replace')!r}")
    print(f"Tokens     : {tokens}")

if __name__ == "__main__":
    main()
