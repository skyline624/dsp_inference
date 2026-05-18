#!/usr/bin/env python3
# Operations transformer reutilisables : attention block, FFN block, lm_head.
# Tout le compute lourd va sur FPGA via UART (FN/FQ/SS/MM/RR).
# Le chunking permet de gerer hidden=172 et vocab=512 sans bumper le RTL.

import numpy as np
from test_sdram_diag import sd_load, call_fn, call_fq
from v4_quant import to_i8_shift, from_i8_shift

D, H, KH, HS = 64, 8, 4, 8
N_REP = H // KH
FQ_MAX_N = 64    # limite RTL : sortie obuf
FQ_K     = 64    # limite RTL : xbuf

def i8(b): return b - 256 if b >= 128 else b
def addr_bytes(a): return bytes([a & 0xFF, (a >> 8) & 0xFF, (a >> 16) & 0xFF])

# ─── Primitives FPGA ──────────────────────────────────────────────────────
def call_mm_fpga_raw(ser, Q_i8, K_i8, V_i8, sq, sk, sv, T=1):
    """Wrapper bas niveau du MM hardware. **3 BUGS RTL CONNUS** :
       1. cur_shift_out n'inclut pas op_mh -> sa retourne stale (typ -5)
       2. attention_head_op T_MAX=8 hardcode -> garbage si T>8
       3. T sur 4 bits dans top.v -> T=16 wrap a 0 -> hang FSM
       A utiliser uniquement T<=8 et avec sV override."""
    pkt = b'MM' + bytes([sq & 0xFF, sk & 0xFF, sv & 0xFF, T])
    pkt += Q_i8.tobytes()
    pkt += K_i8.reshape(-1).tobytes()
    pkt += V_i8.reshape(-1).tobytes()
    ser.write(pkt)
    resp = ser.read(67)
    if resp[:2] != b'MK':
        raise RuntimeError(f"MM: magic {resp[:2]!r}")
    return np.frombuffer(resp[3:], dtype=np.int8), sv

def call_mm_pc(ser, Q_i8, K_i8, V_i8, sq, sk, sv, T=1):
    """Implementation PC du multi-head attention. Recoit Q[64], K[T,KH,HS], V[T,KH,HS] int8+shift,
       retourne attn_out[64] int8+shift, equivalent au call_mm_fpga_raw mais sans
       les bugs RTL T_MAX/wrap/shift. Calcul en float pour simplicite.
       ~2000 ops/token, negligeable vs les ~30000 ops de matmul deja sur FPGA."""
    Q = from_i8_shift(Q_i8.astype(np.int32), sq).reshape(H, HS)
    K = from_i8_shift(K_i8.astype(np.int32), sk).reshape(T, KH, HS)
    V = from_i8_shift(V_i8.astype(np.int32), sv).reshape(T, KH, HS)
    n_rep = H // KH
    # repeat KH heads -> H heads pour GQA
    K_rep = np.repeat(K, n_rep, axis=1)   # [T, H, HS]
    V_rep = np.repeat(V, n_rep, axis=1)
    # scores[h, t] = Q[h] @ K[t, h] / sqrt(HS)
    scores = np.einsum('hd,thd->ht', Q, K_rep) / np.sqrt(HS)
    scores -= scores.max(axis=-1, keepdims=True)
    expv = np.exp(scores); attn_w = expv / expv.sum(axis=-1, keepdims=True)
    out = np.einsum('ht,thd->hd', attn_w, V_rep).reshape(D)
    return to_i8_shift(out)

# Par defaut : utilise FPGA MM (apres fix RTL 2026-05-18 : T_MAX=32, shift fixe).
# Fallback PC si bug : call_mm = call_mm_pc.
call_mm = call_mm_fpga_raw

def call_ss(ser, x_i8, sx, K=64):
    """SiLU sur K elements (max 64)."""
    full_x = np.zeros(64, dtype=np.int8); full_x[:K] = x_i8[:K]
    pkt = b'SS' + bytes([sx & 0xFF]) + full_x.tobytes()
    ser.write(pkt)
    resp = ser.read(70)
    if resp[:2] != b'SK':
        raise RuntimeError(f"SS: magic {resp[:2]!r}")
    so = i8(resp[2])
    return np.frombuffer(resp[6:6+K], dtype=np.int8), so

def to_q15(x): return int(max(-32768, min(32767, round(x * 32768))))

def call_rr(ser, x_i8, sx, cos_f, sin_f):
    """RoPE sur un head (HS=8)."""
    cos_b = b''.join(int.to_bytes(to_q15(c) & 0xFFFF, 2, 'little') for c in cos_f)
    sin_b = b''.join(int.to_bytes(to_q15(s) & 0xFFFF, 2, 'little') for s in sin_f)
    pkt = b'RR' + bytes([sx & 0xFF]) + x_i8.tobytes() + cos_b + sin_b
    ser.write(pkt)
    resp = ser.read(19)
    if resp[:2] != b'RK':
        raise RuntimeError(f"RR: magic {resp[:2]!r}")
    so = i8(resp[2])
    return np.frombuffer(resp[11:], dtype=np.int8), so

# ─── Matmul avec chunking pour contourner limites RTL ─────────────────────
def matvec_chunked_N(ser, x_i8, sx, sw, addr_W, N_total, K=FQ_K):
    """W [N_total, K] @ x [K] -> y [N_total], avec N_total > FQ_MAX_N.
       Decoupe en sous-matmul de N=FQ_MAX_N max, ré-aligne shifts en float, requantize.
       Chaque chunk de W est stocke en SDRAM a addr_W + chunk_offset * K."""
    assert K == FQ_K
    chunks = []
    shifts = []
    pos = 0
    while pos < N_total:
        n = min(FQ_MAX_N, N_total - pos)
        # Si n < FQ_MAX_N on doit appeler FQ avec N=n. FQ supporte N variable.
        y_i8, sy = call_fq(ser, n, sx, sw, x_i8, addr_W + pos * K)
        chunks.append(y_i8)
        shifts.append(sy)
        pos += n
    # Re-aligner : dequantize chaque chunk en float, concatenate, requantize
    y_real = np.concatenate([from_i8_shift(c, s) for c, s in zip(chunks, shifts)])
    return to_i8_shift(y_real)

def matvec_chunked_K(ser, x_i8, sx, sw, addr_W, N, K_total):
    """W [N, K_total] @ x [K_total] -> y [N], avec K_total > FQ_K.
       Decoupe l'INPUT en chunks de FQ_K. Chaque chunk de W de taille [N, FQ_K]
       est stocke en SDRAM a addr_W + chunk_offset * N.
       Important : on dequantize chaque partial result en float, somme,
       puis requantize a la fin."""
    assert N <= FQ_MAX_N
    y_acc = np.zeros(N, dtype=np.float64)
    k_pos = 0
    chunk_idx = 0
    while k_pos < K_total:
        kc = min(FQ_K, K_total - k_pos)
        # Si kc < FQ_K il faut padder x avec des zeros (les W correspondants
        # sont supposes padder a 0 lors du sd_load)
        x_chunk = np.zeros(FQ_K, dtype=np.int8)
        x_chunk[:kc] = x_i8[k_pos:k_pos+kc]
        addr_chunk = addr_W + chunk_idx * N * FQ_K
        y_i8, sy = call_fq(ser, N, sx, sw, x_chunk, addr_chunk)
        y_acc += from_i8_shift(y_i8, sy)
        k_pos += kc
        chunk_idx += 1
    return to_i8_shift(y_acc)

def matvec_chunked_NK(ser, x_i8, sx, sw, addr_W, N_total, K_total):
    """Cas general : N>64 et K>64. Decoupe sur les deux axes."""
    chunks = []
    n_pos = 0
    n_chunk_idx = 0
    n_chunks_per_row = (N_total + FQ_MAX_N - 1) // FQ_MAX_N
    while n_pos < N_total:
        n = min(FQ_MAX_N, N_total - n_pos)
        # Pour ce slice de N, accumuler sur les chunks K
        y_acc = np.zeros(n, dtype=np.float64)
        k_pos = 0
        k_chunk_idx = 0
        while k_pos < K_total:
            kc = min(FQ_K, K_total - k_pos)
            x_chunk = np.zeros(FQ_K, dtype=np.int8)
            x_chunk[:kc] = x_i8[k_pos:k_pos+kc]
            # Layout : on a (N_chunks * K_chunks) blocks de [n_max, FQ_K].
            # Index : k_chunk_idx * n_chunks_per_row + n_chunk_idx
            block_idx = k_chunk_idx * n_chunks_per_row + n_chunk_idx
            addr_chunk = addr_W + block_idx * FQ_MAX_N * FQ_K
            y_i8, sy = call_fq(ser, n, sx, sw, x_chunk, addr_chunk)
            y_acc += from_i8_shift(y_i8, sy)
            k_pos += kc
            k_chunk_idx += 1
        chunks.append(y_acc)
        n_pos += n
        n_chunk_idx += 1
    y_real = np.concatenate(chunks)
    return to_i8_shift(y_real)

# ─── Helper : load une matrice en chunks dans la SDRAM ────────────────────
def sd_load_matrix_chunked(ser, addr, W_i8, N, K):
    """Charge W [N, K] en SDRAM, decoupee en chunks [min(FQ_MAX_N, N), FQ_K].
       Si N>64 ou K>64, on padde a 64 pour chaque chunk.
       Layout :
         - Pour chaque k_chunk (de gauche a droite dans K), pour chaque n_chunk (de haut en bas),
           on stocke un bloc [FQ_MAX_N, FQ_K] (paddé avec des 0)
         - Index dans la SDRAM : (k_chunk_idx * n_chunks_per_row + n_chunk_idx) * FQ_MAX_N * FQ_K
    """
    n_chunks_per_row = (N + FQ_MAX_N - 1) // FQ_MAX_N
    k_chunks         = (K + FQ_K     - 1) // FQ_K
    for k_idx in range(k_chunks):
        k0 = k_idx * FQ_K
        k1 = min(k0 + FQ_K, K)
        for n_idx in range(n_chunks_per_row):
            n0 = n_idx * FQ_MAX_N
            n1 = min(n0 + FQ_MAX_N, N)
            block = np.zeros((FQ_MAX_N, FQ_K), dtype=np.int8)
            # Pour le matmul FQ : W[n, k] avec input x[k]. On range tel que
            # le row-major standard.
            block[:n1-n0, :k1-k0] = W_i8[n0:n1, k0:k1]
            block_idx = k_idx * n_chunks_per_row + n_idx
            block_addr = addr + block_idx * FQ_MAX_N * FQ_K
            sd_load(ser, block_addr, block.tobytes())
    return n_chunks_per_row * k_chunks * FQ_MAX_N * FQ_K  # taille totale utilisee

# ─── Blocks transformer (avec KV cache et rope) ───────────────────────────
def apply_rope_qk(ser, Q_i8, sQ, K_i8, sK, cos_f, sin_f):
    """Applique rope a chaque head de Q [H,HS] et K [KH,HS].
       cos_f/sin_f shape [HS//2]."""
    Q_out = np.zeros(H*HS, dtype=np.int8)
    K_out = np.zeros(KH*HS, dtype=np.int8)
    # Tous les heads partagent les memes cos/sin. Mais RR fait UN head et
    # renvoie un shift par head. Pour cohérence on dequantize/concat/requantize.
    Q_floats = []
    for h in range(H):
        seg = Q_i8[h*HS:(h+1)*HS]
        out_i8, so = call_rr(ser, seg, sQ, cos_f, sin_f)
        Q_floats.append(from_i8_shift(out_i8, so))
    Q_concat = np.concatenate(Q_floats)
    Q_out_i8, sQ_out = to_i8_shift(Q_concat)
    K_floats = []
    for h in range(KH):
        seg = K_i8[h*HS:(h+1)*HS]
        out_i8, so = call_rr(ser, seg, sK, cos_f, sin_f)
        K_floats.append(from_i8_shift(out_i8, so))
    K_concat = np.concatenate(K_floats)
    K_out_i8, sK_out = to_i8_shift(K_concat)
    return Q_out_i8, sQ_out, K_out_i8, sK_out

def attention_block_full(ser, x_i8, sx, weights, pos, kv_cache, freq_cis):
    """Attention block complet : rmsnorm + Q/K/V + rope + KV cache + multi-head + Wo + residu.
       kv_cache = dict avec K[seq_len, KH, HS], sK[seq_len], V idem.
       freq_cis = (real[seq_len, HS//2], imag[seq_len, HS//2])."""
    fr_all, fi_all = freq_cis
    # rmsnorm
    xn_b, sh_n = call_fn(ser, x_i8, sx, weights['sh_rms'], weights['addr_rms'])
    xn = np.frombuffer(xn_b, dtype=np.int8)
    # Q, K, V (toutes les dim sont <= 64 ici donc pas de chunking)
    Q, shQ = call_fq(ser, H*HS,  sh_n, weights['sh_q'], xn, weights['addr_q'])
    K, shK = call_fq(ser, KH*HS, sh_n, weights['sh_k'], xn, weights['addr_k'])
    V, shV = call_fq(ser, KH*HS, sh_n, weights['sh_v'], xn, weights['addr_v'])
    # rope sur Q et K
    Q, shQ, K, shK = apply_rope_qk(ser, Q, shQ, K, shK, fr_all[pos], fi_all[pos])
    # Stocker K et V dans le cache (a pos)
    kv_cache['K'][pos]  = K.reshape(KH, HS)
    kv_cache['sK'][pos] = shK
    kv_cache['V'][pos]  = V.reshape(KH, HS)
    kv_cache['sV'][pos] = shV
    # Multi-head attention : on doit envoyer K[0..pos] et V[0..pos]
    # On re-aligne le KV cache en float (chaque position a son propre shift).
    T = pos + 1
    K_floats = np.stack([from_i8_shift(kv_cache['K'][p].astype(np.int32), kv_cache['sK'][p]) for p in range(T)])
    V_floats = np.stack([from_i8_shift(kv_cache['V'][p].astype(np.int32), kv_cache['sV'][p]) for p in range(T)])
    K_send_i8, shK_send = to_i8_shift(K_floats)
    V_send_i8, shV_send = to_i8_shift(V_floats)
    # MM : Q[64], K[T*32], V[T*32] -> attn_out[64]
    # IMPORTANT : MM attend exactement T positions. Si T > MM_MAX (verifier), il faut chunker.
    # Pour stories260K seq_len max = 32, on prend ca comme limite.
    attn, sh_a = call_mm(ser, Q, K_send_i8, V_send_i8, shQ, shK_send, shV_send, T=T)
    # Wo
    out_i8, sh_o = call_fq(ser, D, sh_a, weights['sh_o'], attn, weights['addr_o'])
    # residu
    x_real    = from_i8_shift(x_i8, sx)
    out_real  = from_i8_shift(out_i8, sh_o)
    new_i8, new_sh = to_i8_shift(x_real + out_real)
    return new_i8, new_sh

def ffn_block_full(ser, x_i8, sx, weights, hidden):
    """FFN SwiGLU complet avec chunking pour hidden > 64."""
    # rmsnorm (D=64, pas de chunking)
    xn_b, sh_n = call_fn(ser, x_i8, sx, weights['sh_rms'], weights['addr_rms'])
    xn = np.frombuffer(xn_b, dtype=np.int8)
    # W1 et W3 : [hidden, 64] -> output de taille hidden (>64 -> chunking N)
    if hidden <= FQ_MAX_N:
        h1, shH1 = call_fq(ser, hidden, sh_n, weights['sh_w1'], xn, weights['addr_w1'])
        h3, shH3 = call_fq(ser, hidden, sh_n, weights['sh_w3'], xn, weights['addr_w3'])
    else:
        h1, shH1 = matvec_chunked_N(ser, xn, sh_n, weights['sh_w1'], weights['addr_w1'], hidden)
        h3, shH3 = matvec_chunked_N(ser, xn, sh_n, weights['sh_w3'], weights['addr_w3'], hidden)
    # silu(h1) (chunks de 64)
    h1s_floats = []
    shH1s = None
    for k0 in range(0, hidden, 64):
        k1 = min(k0 + 64, hidden)
        seg = h1[k0:k1]
        out_i8, so = call_ss(ser, seg, shH1, K=k1-k0)
        h1s_floats.append(from_i8_shift(out_i8, so))
    h1s_real = np.concatenate(h1s_floats)
    h1s_i8, shH1s = to_i8_shift(h1s_real)
    # multiply h1s * h3 (PC)
    hg_real = h1s_real * from_i8_shift(h3, shH3)
    hg_i8, sh_g = to_i8_shift(hg_real)
    # W2 : [64, hidden] -> output D=64 avec input hidden>64 -> chunking K
    if hidden <= FQ_K:
        out_i8, sh_o = call_fq(ser, D, sh_g, weights['sh_w2'], hg_i8, weights['addr_w2'])
    else:
        out_i8, sh_o = matvec_chunked_K(ser, hg_i8, sh_g, weights['sh_w2'], weights['addr_w2'], D, hidden)
    # residu
    x_real    = from_i8_shift(x_i8, sx)
    out_real  = from_i8_shift(out_i8, sh_o)
    new_i8, new_sh = to_i8_shift(x_real + out_real)
    return new_i8, new_sh

def lm_head(ser, x_i8, sx, weights, vocab):
    """Final RMSNorm + projection vers logits[vocab], chunkee si vocab > 64."""
    xn_b, sh_n = call_fn(ser, x_i8, sx, weights['sh_rms'], weights['addr_rms'])
    xn = np.frombuffer(xn_b, dtype=np.int8)
    # tok_emb shape [vocab, D=64], reutilisee comme lm_head (shared weights)
    logits_i8, sh_lo = matvec_chunked_N(ser, xn, sh_n, weights['sh_emb'], weights['addr_emb'], vocab)
    return from_i8_shift(logits_i8, sh_lo)

# ─── References numpy float (pour test) ──────────────────────────────────
def rmsnorm_f(x, w, eps=1e-5):
    return x * w / np.sqrt((x**2).mean() + eps)

def silu_f(x):
    return x / (1.0 + np.exp(-x))

def rope_ref(v, cos_f, sin_f):
    """v shape [n_heads, HS]. Retourne shape identique."""
    nh, hs = v.shape
    out = np.zeros_like(v)
    for h in range(nh):
        for i in range(hs // 2):
            r = v[h, 2*i]; im = v[h, 2*i+1]
            out[h, 2*i]   = r * cos_f[i] - im * sin_f[i]
            out[h, 2*i+1] = r * sin_f[i] + im * cos_f[i]
    return out

def attention_block_ref(x, w, pos, kv_cache_ref, freq_cis):
    fr_all, fi_all = freq_cis
    xn = rmsnorm_f(x, w['rms_w'])
    Q = (w['W_q'] @ xn).reshape(H, HS)
    K = (w['W_k'] @ xn).reshape(KH, HS)
    V = (w['W_v'] @ xn).reshape(KH, HS)
    Q = rope_ref(Q, fr_all[pos], fi_all[pos])
    K = rope_ref(K, fr_all[pos], fi_all[pos])
    kv_cache_ref['K'][pos] = K
    kv_cache_ref['V'][pos] = V
    T = pos + 1
    Ks = kv_cache_ref['K'][:T]   # [T, KH, HS]
    Vs = kv_cache_ref['V'][:T]
    # repeat KH heads -> H heads
    Ks_rep = np.repeat(Ks, N_REP, axis=1)
    Vs_rep = np.repeat(Vs, N_REP, axis=1)
    scores = np.einsum('hd,thd->ht', Q, Ks_rep) / np.sqrt(HS)
    scores -= scores.max(axis=-1, keepdims=True)
    expv = np.exp(scores); attn = expv / expv.sum(axis=-1, keepdims=True)
    out = np.einsum('ht,thd->hd', attn, Vs_rep).reshape(D)
    out = w['W_o'] @ out
    return x + out

def ffn_block_ref(x, w):
    xn = rmsnorm_f(x, w['rms_w'])
    h1 = w['W1'] @ xn
    h3 = w['W3'] @ xn
    hg = silu_f(h1) * h3
    out = w['W2'] @ hg
    return x + out

# ─── Setup poids (pour tests random) ─────────────────────────────────────
def setup_attn_weights(ser, rng, base_addr):
    """Genere des poids random pour attention, quantize, charge en SDRAM."""
    rms_w = np.ones(D, dtype=np.float32)
    W_q = rng.normal(0, 0.1, (H*HS,  D)).astype(np.float32)
    W_k = rng.normal(0, 0.1, (KH*HS, D)).astype(np.float32)
    W_v = rng.normal(0, 0.1, (KH*HS, D)).astype(np.float32)
    W_o = rng.normal(0, 0.1, (D, H*HS)).astype(np.float32)
    rms_i8, sh_r = to_i8_shift(rms_w)
    Wq_i8,  sh_q = to_i8_shift(W_q)
    Wk_i8,  sh_k = to_i8_shift(W_k)
    Wv_i8,  sh_v = to_i8_shift(W_v)
    Wo_i8,  sh_o = to_i8_shift(W_o)
    a_r, a_q, a_k, a_v, a_o = (base_addr + i*0x1000 for i in range(5))
    sd_load(ser, a_r, rms_i8.tobytes())
    sd_load(ser, a_q, Wq_i8.reshape(-1).tobytes())
    sd_load(ser, a_k, Wk_i8.reshape(-1).tobytes())
    sd_load(ser, a_v, Wv_i8.reshape(-1).tobytes())
    sd_load(ser, a_o, Wo_i8.reshape(-1).tobytes())
    w_real = dict(rms_w=rms_w, W_q=W_q, W_k=W_k, W_v=W_v, W_o=W_o)
    w_fpga = dict(addr_rms=a_r, addr_q=a_q, addr_k=a_k, addr_v=a_v, addr_o=a_o,
                  sh_rms=sh_r, sh_q=sh_q, sh_k=sh_k, sh_v=sh_v, sh_o=sh_o)
    return w_real, w_fpga

def setup_ffn_weights(ser, rng, base_addr, hidden=64):
    rms_w = np.ones(D, dtype=np.float32)
    W1 = rng.normal(0, 0.1, (hidden, D)).astype(np.float32)
    W3 = rng.normal(0, 0.1, (hidden, D)).astype(np.float32)
    W2 = rng.normal(0, 0.1, (D, hidden)).astype(np.float32)
    rms_i8, sh_r = to_i8_shift(rms_w)
    W1_i8, sh_1  = to_i8_shift(W1)
    W3_i8, sh_3  = to_i8_shift(W3)
    W2_i8, sh_2  = to_i8_shift(W2)
    a_r, a_1, a_3, a_2 = (base_addr + i*0x4000 for i in range(4))   # 16KiB par matrice (chunked = 12 KiB max)
    sd_load(ser, a_r, rms_i8.tobytes())
    sd_load_matrix_chunked(ser, a_1, W1_i8, hidden, D)
    sd_load_matrix_chunked(ser, a_3, W3_i8, hidden, D)
    sd_load_matrix_chunked(ser, a_2, W2_i8, D, hidden)
    w_real = dict(rms_w=rms_w, W1=W1, W3=W3, W2=W2)
    w_fpga = dict(addr_rms=a_r, addr_w1=a_1, addr_w3=a_3, addr_w2=a_2,
                  sh_rms=sh_r, sh_w1=sh_1, sh_w3=sh_3, sh_w2=sh_2)
    return w_real, w_fpga

# ─── API compatibilite anciens tests ──────────────────────────────────────
def attention_block(ser, x_i8, sx, weights, pos=0):
    """Attention block compatible test_layer.py (pos=0, T=1, pas de rope)."""
    xn_b, sh_n = call_fn(ser, x_i8, sx, weights['sh_rms'], weights['addr_rms'])
    xn = np.frombuffer(xn_b, dtype=np.int8)
    Q, shQ = call_fq(ser, H*HS,  sh_n, weights['sh_q'], xn, weights['addr_q'])
    K, shK = call_fq(ser, KH*HS, sh_n, weights['sh_k'], xn, weights['addr_k'])
    V, shV = call_fq(ser, KH*HS, sh_n, weights['sh_v'], xn, weights['addr_v'])
    assert pos == 0
    K_t = K.reshape(1, KH, HS); V_t = V.reshape(1, KH, HS)
    attn, sh_a = call_mm(ser, Q, K_t, V_t, shQ, shK, shV, T=1)
    out_i8, sh_o = call_fq(ser, D, sh_a, weights['sh_o'], attn, weights['addr_o'])
    new_i8, new_sh = to_i8_shift(from_i8_shift(x_i8, sx) + from_i8_shift(out_i8, sh_o))
    return new_i8, new_sh

def ffn_block(ser, x_i8, sx, weights, hidden=64):
    return ffn_block_full(ser, x_i8, sx, weights, hidden=hidden)

def transformer_layer(ser, x_i8, sx, attn_w, ffn_w, pos=0):
    x1_i8, sx1 = attention_block(ser, x_i8, sx, attn_w, pos=pos)
    x2_i8, sx2 = ffn_block(ser, x1_i8, sx1, ffn_w, hidden=64)
    return x2_i8, sx2

def transformer_layer_ref(x, attn_w, ffn_w):
    # Reference simple sans KV cache (pour test_layer.py et test_multi_layer.py)
    xn = rmsnorm_f(x, attn_w['rms_w'])
    Q = (attn_w['W_q'] @ xn).reshape(H, HS)
    K = (attn_w['W_k'] @ xn).reshape(KH, HS)
    V = (attn_w['W_v'] @ xn).reshape(KH, HS)
    Kr = np.repeat(K, N_REP, axis=0); Vr = np.repeat(V, N_REP, axis=0)
    out = np.zeros(D, dtype=np.float32)
    for h in range(H):
        out[h*HS:(h+1)*HS] = Vr[h]   # softmax(score scalaire) = 1
    out = attn_w['W_o'] @ out
    x1 = x + out
    return ffn_block_ref(x1, ffn_w)
