#!/usr/bin/env python3
# Debug : compare x after 5 layers FPGA vs after 5 layers v4sim Python.
# Si x match (a la quantif pres), le bug est in rms_final ou lm_head.
# Si x differe drastiquement, le bug est in les layers.

import time, struct
import numpy as np
import serial

from infer_v4sim import load_model, MODEL_PATH as MODEL, TOK_PATH as TOK, load_tok
from infer_v4sim import (rmsnorm_q, matvec_q, silu_q, mul_q, softmax_q,
                         apply_rope_q, to_i8_shift, from_i8_shift)
from transformer_ops import attention_block_full, ffn_block_full, sd_load_matrix_chunked, D, H, KH, HS
from test_sdram_diag import sd_load

PORT = "COM6"; BAUD = 1_000_000

def v4sim_5layers(m, token, kv_caches, pos):
    """v4sim forward - retourne x APRES les 5 layers, AVANT rms_final."""
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
    return x

def fpga_5layers(ser, m, token, kv_caches, pos, w, freq_cis):
    """FPGA forward - retourne x APRES les 5 layers, AVANT rms_final."""
    x_real = m['tok_emb'][token].astype(np.float32).copy()
    x_i8, sx = to_i8_shift(x_real)
    for l in range(m['cfg']['n_layers']):
        x_i8, sx = attention_block_full(ser, x_i8, sx, w['layers'][l]['attn'],
                                         pos, kv_caches[l], freq_cis)
        x_i8, sx = ffn_block_full(ser, x_i8, sx, w['layers'][l]['ffn'], hidden=m['cfg']['hidden_dim'])
    return from_i8_shift(x_i8, sx)

def main():
    m = load_model(MODEL); cfg = m['cfg']
    L = cfg['n_layers']; HID = cfg['hidden_dim']; S = cfg['seq_len']
    print(f"Model: D={cfg['dim']} hidden={HID} layers={L}")

    # --- reference v4sim ---
    kv_ref = [{'K': np.zeros((S, KH, HS), dtype=np.int8),
               'sK': np.zeros(S, dtype=np.int32),
               'V': np.zeros((S, KH, HS), dtype=np.int8),
               'sV': np.zeros(S, dtype=np.int32)} for _ in range(L)]
    print("\nv4sim forward (5 layers) pos=0, token=1...")
    x_ref = v4sim_5layers(m, 1, kv_ref, 0)
    print(f"  x_ref[:8] = {x_ref[:8].round(3)}")
    print(f"  x_ref stats: min={x_ref.min():.2f} max={x_ref.max():.2f} mean={x_ref.mean():.3f}")

    # --- FPGA forward ---
    ser = serial.Serial(PORT, BAUD, timeout=15.0)
    time.sleep(0.5); ser.reset_input_buffer()

    # Charger tous les poids (comme infer_fpga)
    from infer_fpga import quantize_and_load_weights
    print("\nChargement poids FPGA...")
    t0 = time.time()
    w = quantize_and_load_weights(ser, m)
    print(f"  charge en {time.time()-t0:.1f}s")

    kv_fpga = [{'K':  np.zeros((S, KH, HS), dtype=np.int8),
                'sK': np.zeros(S, dtype=np.int32),
                'V':  np.zeros((S, KH, HS), dtype=np.int8),
                'sV': np.zeros(S, dtype=np.int32)} for _ in range(L)]
    freq_cis = (m['freq_cis_real'], m['freq_cis_imag'])

    print("\nFPGA forward (5 layers) pos=0, token=1...")
    t0 = time.time()
    x_fpga = fpga_5layers(ser, m, 1, kv_fpga, 0, w, freq_cis)
    print(f"  ({time.time()-t0:.2f}s)")
    print(f"  x_fpga[:8] = {x_fpga[:8].round(3)}")
    print(f"  x_fpga stats: min={x_fpga.min():.2f} max={x_fpga.max():.2f} mean={x_fpga.mean():.3f}")

    # --- compare ---
    print(f"\n--- Comparaison ---")
    diff = np.abs(x_fpga - x_ref)
    rel = diff / max(np.abs(x_ref).max(), 1e-9) * 100
    print(f"  diff abs   : max={diff.max():.3f} mean={diff.mean():.3f}")
    print(f"  diff rel   : max={rel.max():.1f}% mean={rel.mean():.1f}%")
    print(f"  cosine sim : {np.dot(x_fpga, x_ref) / (np.linalg.norm(x_fpga) * np.linalg.norm(x_ref)):.4f}")
    print(f"  sign match : {(np.sign(x_fpga) == np.sign(x_ref)).sum()}/{len(x_ref)}")

    if np.dot(x_fpga, x_ref) / (np.linalg.norm(x_fpga) * np.linalg.norm(x_ref)) > 0.9:
        print("\n  ==> x correspond bien : bug est dans rms_final ou lm_head FPGA")
    else:
        print("\n  ==> x ne correspond PAS : bug est dans les layers (attn ou ffn)")

    ser.close()

if __name__ == "__main__":
    main()
