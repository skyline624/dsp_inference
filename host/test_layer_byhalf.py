#!/usr/bin/env python3
# Debug pousse : compare x apres CHAQUE demi-layer (attn puis ffn) FPGA vs v4sim.
# Permet d'identifier exactement ou la divergence apparait.

import time
import numpy as np
import serial

from infer_v4sim import load_model, MODEL_PATH as MODEL
from infer_v4sim import (rmsnorm_q, matvec_q, silu_q, mul_q, softmax_q,
                         apply_rope_q, to_i8_shift, from_i8_shift)
from transformer_ops import attention_block_full, ffn_block_full, D, H, KH, HS
from infer_fpga import quantize_and_load_weights

PORT = "COM6"; BAUD = 1_000_000

def v4sim_attn(m, x, l, pos, kv):
    cfg = m['cfg']
    Hh, KHh, HSh, DD = cfg['n_heads'], cfg['n_kv_heads'], cfg['head_size'], cfg['dim']
    n_rep = Hh // KHh
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
    kv[l]['K'][pos] = K_i8; kv[l]['sK'][pos] = sK
    kv[l]['V'][pos] = V_i8_2d; kv[l]['sV'][pos] = sV2
    Q_f = from_i8_shift(Q_i8, sQ)
    Ks = np.array([from_i8_shift(kv[l]['K'][p], kv[l]['sK'][p]) for p in range(pos+1)])
    Vs = np.array([from_i8_shift(kv[l]['V'][p], kv[l]['sV'][p]) for p in range(pos+1)])
    Ks_q = np.repeat(Ks, n_rep, axis=1); Vs_q = np.repeat(Vs, n_rep, axis=1)
    scores = np.einsum('hd,thd->ht', Q_f, Ks_q) / np.sqrt(HSh)
    attn_i8, sa = softmax_q(*to_i8_shift(scores), axis=-1)
    attn = from_i8_shift(attn_i8, sa)
    out = np.einsum('ht,thd->hd', attn, Vs_q).reshape(DD)
    out_i8, so = matvec_q(m['wo'][l], *to_i8_shift(out))
    return x + from_i8_shift(out_i8, so)

def v4sim_ffn(m, x, l):
    x_norm_i8, sxn = rmsnorm_q(*to_i8_shift(x), m['rms_ffn'][l])
    gate_i8, sg = matvec_q(m['w1'][l], x_norm_i8, sxn)
    up_i8,   su = matvec_q(m['w3'][l], x_norm_i8, sxn)
    silu_g_i8, ssg = silu_q(gate_i8, sg)
    h_i8, sh = mul_q(silu_g_i8, ssg, up_i8, su)
    out_i8, so = matvec_q(m['w2'][l], h_i8, sh)
    return x + from_i8_shift(out_i8, so)

def cmp(label, fpga_real, ref_real):
    diff = np.abs(fpga_real - ref_real)
    rel = diff.max() / max(np.abs(ref_real).max(), 1e-9) * 100
    cos = np.dot(fpga_real, ref_real) / (np.linalg.norm(fpga_real) * np.linalg.norm(ref_real) + 1e-9)
    sign_match = (np.sign(fpga_real) == np.sign(ref_real)).sum()
    flag = " <<<< BAD" if cos < 0.95 else " ok" if cos > 0.99 else ""
    print(f"  {label:30s}: cos={cos:.4f}  diff_max={diff.max():.3f}  rel={rel:.1f}%  sign={sign_match}/{len(ref_real)}{flag}")

def main():
    m = load_model(MODEL); cfg = m['cfg']
    L, HID, S = cfg['n_layers'], cfg['hidden_dim'], cfg['seq_len']
    print(f"Model: L={L} hidden={HID}")

    ser = serial.Serial(PORT, BAUD, timeout=15.0)
    time.sleep(0.5); ser.reset_input_buffer()
    print("\nChargement poids FPGA...")
    w = quantize_and_load_weights(ser, m)
    print()

    kv_ref  = [{'K': np.zeros((S, KH, HS), dtype=np.int8), 'sK': np.zeros(S, dtype=np.int32),
                'V': np.zeros((S, KH, HS), dtype=np.int8), 'sV': np.zeros(S, dtype=np.int32)} for _ in range(L)]
    kv_fpga = [{'K': np.zeros((S, KH, HS), dtype=np.int8), 'sK': np.zeros(S, dtype=np.int32),
                'V': np.zeros((S, KH, HS), dtype=np.int8), 'sV': np.zeros(S, dtype=np.int32)} for _ in range(L)]
    freq_cis = (m['freq_cis_real'], m['freq_cis_imag'])

    # Init
    x_ref  = m['tok_emb'][1].astype(np.float32).copy()
    x_fpga_real = x_ref.copy()  # meme depart
    pos = 0
    print(f"Embed init : x[:4] = {x_ref[:4].round(3)}\n")

    for l in range(L):
        # v4sim
        x_ref = v4sim_attn(m, x_ref, l, pos, kv_ref)
        x_ref_after_attn = x_ref.copy()
        x_ref = v4sim_ffn(m, x_ref, l)

        # FPGA -- attn
        x_i8, sx = to_i8_shift(x_fpga_real)
        x_i8, sx = attention_block_full(ser, x_i8, sx, w['layers'][l]['attn'],
                                         pos, kv_fpga[l], freq_cis)
        x_fpga_after_attn = from_i8_shift(x_i8, sx)
        # FPGA -- ffn
        x_i8, sx = ffn_block_full(ser, x_i8, sx, w['layers'][l]['ffn'], hidden=HID)
        x_fpga_real = from_i8_shift(x_i8, sx)

        print(f"--- Layer {l} ---")
        cmp(f"L{l} apres attn", x_fpga_after_attn, x_ref_after_attn)
        cmp(f"L{l} apres ffn", x_fpga_real, x_ref)
        print()

    ser.close()

if __name__ == "__main__":
    main()
