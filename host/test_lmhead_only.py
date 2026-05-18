#!/usr/bin/env python3
# Debug en 2 etapes :
#   1. v4sim full forward jusqu'a x_norm pre-lm_head -> donne x_norm_i8 et sxn
#   2a. Sans FPGA : matvec_q reference Python -> logits_ref + argmax_ref
#   2b. FPGA chunked : matvec_chunked_N sur tok_emb -> logits_fpga + argmax_fpga
#   Compare. Isolement complet de la chunked matmul (pas de rmsnorm FPGA).

import time, sys
import numpy as np
import serial

from infer_v4sim import load_model, MODEL_PATH as MODEL, TOK_PATH as TOK, load_tok
from infer_v4sim import (rmsnorm_q, matvec_q, silu_q, mul_q, softmax_q,
                         apply_rope_q, to_i8_shift, from_i8_shift)
from transformer_ops import matvec_chunked_N, sd_load_matrix_chunked, D, FQ_MAX_N, FQ_K
from test_sdram_diag import sd_load, call_fq

PORT = "COM6"; BAUD = 1_000_000

def v4sim_forward_capture(m, token, kv_caches, pos):
    cfg = m['cfg']
    Hh, KHh, HSh, DD = cfg['n_heads'], cfg['n_kv_heads'], cfg['head_size'], cfg['dim']
    n_rep = Hh // KHh
    x = m['tok_emb'][token].astype(np.float32).copy()
    for l in range(cfg['n_layers']):
        x_norm_i8, sxn = rmsnorm_q(*to_i8_shift(x), m['rms_att'][l])
        Q_i8, sQ = matvec_q(m['wq'][l], x_norm_i8, sxn)
        K_i8, sK = matvec_q(m['wk'][l], x_norm_i8, sxn)
        V_i8, sV = matvec_q(m['wv'][l], x_norm_i8, sxn)
        Q = from_i8_shift(Q_i8, sQ).reshape(Hh, HSh).astype(np.float32)
        K = from_i8_shift(K_i8, sK).reshape(KHh, HSh).astype(np.float32)
        V = from_i8_shift(V_i8, sV).reshape(KHh, HSh).astype(np.float32)
        fr = m['freq_cis_real'][pos]; fi = m['freq_cis_imag'][pos]
        Q_i8, sQ = apply_rope_q(*to_i8_shift(Q), fr, fi)
        K_i8, sK = apply_rope_q(*to_i8_shift(K), fr, fi)
        V_i8_2d, sV2 = to_i8_shift(V)
        kv_caches[l]['K'][pos]  = K_i8; kv_caches[l]['sK'][pos] = sK
        kv_caches[l]['V'][pos]  = V_i8_2d; kv_caches[l]['sV'][pos] = sV2
        Q_f = from_i8_shift(Q_i8, sQ)
        Ks = np.array([from_i8_shift(kv_caches[l]['K'][p], kv_caches[l]['sK'][p]) for p in range(pos+1)])
        Vs = np.array([from_i8_shift(kv_caches[l]['V'][p], kv_caches[l]['sV'][p]) for p in range(pos+1)])
        Ks_q = np.repeat(Ks, n_rep, axis=1); Vs_q = np.repeat(Vs, n_rep, axis=1)
        scores = np.einsum('hd,thd->ht', Q_f, Ks_q) / np.sqrt(HSh)
        attn_i8, sa = softmax_q(*to_i8_shift(scores), axis=-1)
        attn = from_i8_shift(attn_i8, sa)
        out = np.einsum('ht,thd->hd', attn, Vs_q).reshape(DD)
        out_i8, so = matvec_q(m['wo'][l], *to_i8_shift(out))
        x = x + from_i8_shift(out_i8, so)
        x_norm_i8, sxn = rmsnorm_q(*to_i8_shift(x), m['rms_ffn'][l])
        gate_i8, sg = matvec_q(m['w1'][l], x_norm_i8, sxn)
        up_i8,   su = matvec_q(m['w3'][l], x_norm_i8, sxn)
        silu_g_i8, ssg = silu_q(gate_i8, sg)
        h_i8, sh = mul_q(silu_g_i8, ssg, up_i8, su)
        out_i8, so = matvec_q(m['w2'][l], h_i8, sh)
        x = x + from_i8_shift(out_i8, so)
    # Final rmsnorm en Python (on saute le FPGA rmsnorm pour isoler le matmul)
    x_norm_i8, sxn = rmsnorm_q(*to_i8_shift(x), m['rms_final'])
    return x_norm_i8, sxn

def main():
    m = load_model(MODEL); cfg = m['cfg']
    vocab = load_tok(TOK, cfg['vocab_size'])
    print(f"Model: dim={cfg['dim']} hidden={cfg['hidden_dim']} vocab={cfg['vocab_size']}")

    L, KHh, HSh, S = cfg['n_layers'], cfg['n_kv_heads'], cfg['head_size'], cfg['seq_len']
    kv_caches = [{'K': np.zeros((S, KHh, HSh), dtype=np.int8),
                  'sK': np.zeros(S, dtype=np.int32),
                  'V': np.zeros((S, KHh, HSh), dtype=np.int8),
                  'sV': np.zeros(S, dtype=np.int32)} for _ in range(L)]

    print("\nv4sim forward + rmsnorm final (Python) jusqu'a x_norm pre-lm_head...")
    xn_i8, sxn = v4sim_forward_capture(m, token=1, kv_caches=kv_caches, pos=0)
    print(f"  x_norm: min={xn_i8.min()} max={xn_i8.max()} shift={sxn}")

    # 2a. Reference Python : logits = tok_emb @ x_norm avec quantif power-of-2
    print("\n--- Reference Python (v4sim matvec_q) ---")
    logits_ref_i8, sl_ref = matvec_q(m['tok_emb'], xn_i8, sxn)
    logits_ref = from_i8_shift(logits_ref_i8, sl_ref)
    top5_ref = np.argsort(logits_ref)[-5:][::-1]
    print(f"  argmax = {int(top5_ref[0])}  ({vocab[int(top5_ref[0])]!r})")
    print(f"  top5   = {[(int(i), vocab[int(i)].decode('utf-8','replace')) for i in top5_ref]}")
    print(f"  range: [{logits_ref.min():.2f}, {logits_ref.max():.2f}]")

    # 2b. FPGA : chunked matvec sur tok_emb
    print("\n--- FPGA matvec_chunked_N (tok_emb [512,64] en 8 chunks N=64) ---")
    ser = serial.Serial(PORT, BAUD, timeout=15.0)
    time.sleep(0.5); ser.reset_input_buffer()

    tok_emb_i8, sh_emb = to_i8_shift(m['tok_emb'])
    addr_emb = 0x100000   # base differente de 0 (0 semblait poser souci)
    sd_load_matrix_chunked(ser, addr_emb, tok_emb_i8, cfg['vocab_size'], D)
    print(f"  tok_emb load: shift={sh_emb}")

    # Appel direct du chunked matmul (PAS de rmsnorm FPGA)
    logits_fpga_i8, sl_fpga = matvec_chunked_N(ser, xn_i8, sxn, sh_emb, addr_emb, cfg['vocab_size'])
    logits_fpga = from_i8_shift(logits_fpga_i8, sl_fpga)
    top5_fpga = np.argsort(logits_fpga)[-5:][::-1]
    print(f"  argmax = {int(top5_fpga[0])}  ({vocab[int(top5_fpga[0])]!r})")
    print(f"  top5   = {[(int(i), vocab[int(i)].decode('utf-8','replace')) for i in top5_fpga]}")
    print(f"  range: [{logits_fpga.min():.2f}, {logits_fpga.max():.2f}]")

    # Comparaison detaillee
    print(f"\n--- Comparaison ---")
    if int(top5_ref[0]) == int(top5_fpga[0]):
        print(f"  argmax MATCH : {vocab[int(top5_ref[0])]!r}")
    else:
        print(f"  argmax DIFFERE : ref={vocab[int(top5_ref[0])]!r} fpga={vocab[int(top5_fpga[0])]!r}")
    ref_norm = logits_ref / np.linalg.norm(logits_ref)
    fpga_norm = logits_fpga / np.linalg.norm(logits_fpga)
    print(f"  cosine similarity = {np.dot(ref_norm, fpga_norm):.4f}")
    print(f"  spearman top10 ref vs fpga: top10_ref={np.argsort(logits_ref)[-10:][::-1].tolist()}")
    print(f"                              top10_fpga={np.argsort(logits_fpga)[-10:][::-1].tolist()}")

    # Si argmax differe : explore d'ou vient le bug
    if int(top5_ref[0]) != int(top5_fpga[0]):
        argmax_ref = int(top5_ref[0])
        # Quel score donne le FPGA au token 403 (la bonne reponse) ?
        print(f"\n  Token correct {argmax_ref}: ref_logit={logits_ref[argmax_ref]:.2f}, fpga_logit={logits_fpga[argmax_ref]:.2f}")
        # Rang du token correct dans le ranking FPGA
        sort_fpga = np.argsort(-logits_fpga)
        rank_correct = np.where(sort_fpga == argmax_ref)[0][0]
        print(f"  Token correct est rank {rank_correct} dans le FPGA top")

    ser.close()

if __name__ == "__main__":
    main()
