#!/usr/bin/env python3
# =============================================================================
# infer_fpga.py - Real stories260K inference on the Tang Nano 20K FPGA.
# All the heavy compute (matmul, rmsnorm, silu, rope, attention) runs on FPGA.
# The PC orchestrates, loads the weights into SDRAM, and performs a few minor
# operations (elementwise multiply in the FFN, residuals, argmax sampling).
# =============================================================================

import os, struct, re, time, sys
import numpy as np
import serial

from test_sdram_diag import sd_load, call_fn, call_fq
from transformer_ops import (
    attention_block_full, ffn_block_full, lm_head,
    sd_load_matrix_chunked, matvec_chunked_N,
    D, H, KH, HS, FQ_MAX_N, FQ_K,
)
from v4_quant import to_i8_shift, from_i8_shift

PORT = "COM6"; BAUD = 1_000_000
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "..", "..", "dsp_coproc", "host", "models", "stories260K.bin")
TOK   = os.path.join(HERE, "..", "..", "dsp_coproc", "host", "models", "tok512.bin")

# ─── Loader ──────────────────────────────────────────────────────────────
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
    L,DD,Hh,KHh,HSh,HID,V,S = (cfg['n_layers'],cfg['dim'],cfg['n_heads'],
        cfg['n_kv_heads'],cfg['head_size'],cfg['hidden_dim'],cfg['vocab_size'],cfg['seq_len'])
    return {'cfg':cfg, 'tok_emb':t(V,DD), 'rms_att':t(L,DD),
            'wq':t(L,Hh*HSh,DD), 'wk':t(L,KHh*HSh,DD), 'wv':t(L,KHh*HSh,DD), 'wo':t(L,DD,Hh*HSh),
            'rms_ffn':t(L,DD), 'w1':t(L,HID,DD), 'w2':t(L,DD,HID), 'w3':t(L,HID,DD),
            'rms_final':t(DD),
            'freq_cis_real':t(S,HSh//2), 'freq_cis_imag':t(S,HSh//2)}

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

# ─── Chargement des poids en SDRAM ───────────────────────────────────────
def quantize_and_load_weights(ser, m):
    """Quantize tous les poids stories260K et charge en SDRAM.
       Retourne un dict avec les addresses et shifts.

       Layout SDRAM (offsets en hex) :
         0x000000 : tok_emb [512, 64] chunked (8 N-chunks de [64, 64])      = 32 KiB
         0x010000 : layer 0   (block 64 KiB)
         0x020000 : layer 1
         ...
         0x050000 : layer 4
         0x060000 : rms_final
    """
    cfg = m['cfg']
    L = cfg['n_layers']; HID = cfg['hidden_dim']; V = cfg['vocab_size']
    weights = {}

    print("  - tok_emb (chunked)...")
    tok_i8, sh_emb = to_i8_shift(m['tok_emb'])
    addr_emb = 0x000000
    sd_load_matrix_chunked(ser, addr_emb, tok_i8, V, D)
    weights['tok_emb_real'] = m['tok_emb']   # garde en float pour le lookup d'embedding
    weights['lm_head'] = dict(addr_emb=addr_emb, sh_emb=sh_emb,
                              addr_rms=0x060000, sh_rms=0)  # rms_final addr fixe ici

    print("  - rms_final...")
    rms_fin_i8, sh_rf = to_i8_shift(m['rms_final'])
    sd_load(ser, 0x060000, rms_fin_i8.tobytes())
    weights['lm_head']['sh_rms'] = sh_rf

    weights['layers'] = []
    for l in range(L):
        base = 0x010000 + l * 0x10000
        print(f"  - layer {l} @ {base:06x}...")
        # attn (addresses in les 8 premiers KiB)
        rms_att_i8, sh_ra = to_i8_shift(m['rms_att'][l])
        wq_i8, sh_q = to_i8_shift(m['wq'][l])
        wk_i8, sh_k = to_i8_shift(m['wk'][l])
        wv_i8, sh_v = to_i8_shift(m['wv'][l])
        wo_i8, sh_o = to_i8_shift(m['wo'][l])
        a_rms_a = base + 0x0000
        a_wq    = base + 0x0100
        a_wk    = base + 0x1100
        a_wv    = base + 0x1900
        a_wo    = base + 0x2100
        sd_load(ser, a_rms_a, rms_att_i8.tobytes())
        sd_load(ser, a_wq,    wq_i8.reshape(-1).tobytes())
        sd_load(ser, a_wk,    wk_i8.reshape(-1).tobytes())
        sd_load(ser, a_wv,    wv_i8.reshape(-1).tobytes())
        sd_load(ser, a_wo,    wo_i8.reshape(-1).tobytes())
        # ffn (rms_ffn + W1/W3/W2 chunked, beaucoup plus gros)
        rms_ffn_i8, sh_rf2 = to_i8_shift(m['rms_ffn'][l])
        w1_i8, sh_w1 = to_i8_shift(m['w1'][l])
        w3_i8, sh_w3 = to_i8_shift(m['w3'][l])
        w2_i8, sh_w2 = to_i8_shift(m['w2'][l])
        a_rms_f = base + 0x3100
        a_w1    = base + 0x3200    # chunks de [64,64] * 3 = 12 KiB
        a_w3    = base + 0x6200
        a_w2    = base + 0x9200
        sd_load(ser, a_rms_f, rms_ffn_i8.tobytes())
        sd_load_matrix_chunked(ser, a_w1, w1_i8, HID, D)
        sd_load_matrix_chunked(ser, a_w3, w3_i8, HID, D)
        sd_load_matrix_chunked(ser, a_w2, w2_i8, D, HID)
        weights['layers'].append({
            'attn': dict(addr_rms=a_rms_a, addr_q=a_wq, addr_k=a_wk, addr_v=a_wv, addr_o=a_wo,
                         sh_rms=sh_ra, sh_q=sh_q, sh_k=sh_k, sh_v=sh_v, sh_o=sh_o),
            'ffn':  dict(addr_rms=a_rms_f, addr_w1=a_w1, addr_w3=a_w3, addr_w2=a_w2,
                         sh_rms=sh_rf2, sh_w1=sh_w1, sh_w3=sh_w3, sh_w2=sh_w2),
        })
    return weights

# ─── loop d'inference ──────────────────────────────────────────────────
def forward_fpga(ser, m, w, token, kv_caches, pos, freq_cis):
    """Un step forward complet : token -> logits[vocab]."""
    cfg = m['cfg']
    L = cfg['n_layers']; HID = cfg['hidden_dim']; V = cfg['vocab_size']
    # 1. Embed (lookup direct from tok_emb float en RAM PC)
    x_real = w['tok_emb_real'][token].astype(np.float32).copy()
    x_i8, sx = to_i8_shift(x_real)
    # 2. 5 layers : attn + ffn
    for l in range(L):
        x_i8, sx = attention_block_full(ser, x_i8, sx, w['layers'][l]['attn'],
                                         pos, kv_caches[l], freq_cis)
        x_i8, sx = ffn_block_full(ser, x_i8, sx, w['layers'][l]['ffn'], hidden=HID)
    # 3. lm_head : rms_final + matmul to vocab=512
    logits = lm_head(ser, x_i8, sx, w['lm_head'], vocab=V)
    return logits

def main():
    print(f"Chargement modele : {MODEL}")
    m = load_model(MODEL); cfg = m['cfg']
    print(f"  dim={cfg['dim']} hidden={cfg['hidden_dim']} layers={cfg['n_layers']} "
          f"heads={cfg['n_heads']}/{cfg['n_kv_heads']} vocab={cfg['vocab_size']} seq_len={cfg['seq_len']}\n")
    vocab = load_tok(TOK, cfg['vocab_size'])

    ser = serial.Serial(PORT, BAUD, timeout=15.0)
    time.sleep(0.5); ser.reset_input_buffer()

    print("Chargement poids stories260K en SDRAM (peut prendre ~5s)...")
    t0 = time.time()
    w = quantize_and_load_weights(ser, m)
    print(f"  -> charge en {time.time()-t0:.1f}s\n")

    L, KHh, HSh, S = cfg['n_layers'], cfg['n_kv_heads'], cfg['head_size'], cfg['seq_len']
    kv_caches = [{'K':  np.zeros((S, KHh, HSh), dtype=np.int8),
                  'sK': np.zeros(S, dtype=np.int32),
                  'V':  np.zeros((S, KHh, HSh), dtype=np.int8),
                  'sV': np.zeros(S, dtype=np.int32)} for _ in range(L)]
    freq_cis = (m['freq_cis_real'], m['freq_cis_imag'])

    print("Generation (greedy argmax, 17 tokens) :")
    print("  Reference v3e attendue : 'Once upon a time, there was a girl named Lily. She loved'\n")
    tokens = [1]; text = b""; nxt = 1
    N_TOK = 17
    for pos in range(N_TOK):
        t_step = time.time()
        logits = forward_fpga(ser, m, w, nxt, kv_caches, pos, freq_cis)
        prev = nxt
        nxt = int(np.argmax(logits))
        tokens.append(nxt)
        p = decode(prev, nxt, vocab); text += p
        dt = time.time() - t_step
        print(f"  pos={pos:2d}  tok={nxt:4d}  ({dt:.2f}s)  {repr(p.decode('utf-8','replace'))}")

    print(f"\nTexte FPGA : {text.decode('utf-8','replace')!r}")
    print(f"Tokens    : {tokens}")
    ser.close()

if __name__ == "__main__":
    main()
