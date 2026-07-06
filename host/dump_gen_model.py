#!/usr/bin/env python3
# =============================================================================
# dump_gen_model.py - dump the stories260K model, QUANTISED, for the cocotb tb.
#
# numpy is NOT available in the sim Docker, so the tb cannot run infer_v5gen_ref.
# This host script (numpy OK) quantises every weight with to_i8_shift (exactly what
# the node's FQ expects), captures the per-layer shifts, the freq_cis (rope cos/sin)
# per position in Q15, and the 17 oracle tokens, and dumps them to a pure-Python
# JSON that sim/link/test_gen_F*.py loads with NO numpy.
#
# Layout is the gen_seq RTL layout (base_l = 0x010000 + layer*0x10000, offsets
# rms_att@0, wq@0x100, wk@0x1100, wv@0x1900, wo@0x2100, rms_ffn@0x3100, w1@0x3200,
# w3@0x6200, w2@0x9200 ; tok_emb@0 ; rms_final@0x060000). The tb does the padding/
# chunking arrangement (same as test_gen_D/E), so this dump keeps raw quantised
# matrices + shifts.
# =============================================================================
import json
import os
import numpy as np

import infer_v4sim as S
from infer_v4sim import to_i8_shift
import infer_v5gen_ref as R

HERE = os.path.dirname(os.path.abspath(__file__))
NPOS = 17


def q(W):
    wi8, s = to_i8_shift(np.asarray(W))
    return wi8.astype(int).tolist(), int(s)


def q15(v):
    return int(max(-32768, min(32767, round(float(v) * 32768.0))))


def main():
    m = S.load_model(S.MODEL_PATH)
    cfg = m['cfg']
    L = cfg['n_layers']

    data = {'NL': L, 'D': cfg['dim'], 'HID': cfg['hidden_dim'],
            'H': cfg['n_heads'], 'KH': cfg['n_kv_heads'], 'HS': cfg['head_size'],
            'VOCAB': cfg['vocab_size'], 'layers': []}

    for l in range(L):
        layer = {}
        for key in ('wq', 'wk', 'wv', 'wo', 'w1', 'w3', 'w2', 'rms_att', 'rms_ffn'):
            i8, s = q(m[key][l])
            layer[key] = i8
            layer[key + '_s'] = s
        data['layers'].append(layer)

    data['tok_emb'], data['tok_emb_s'] = q(m['tok_emb'])
    data['rms_final'], data['rms_final_s'] = q(m['rms_final'])

    # rope cos/sin per position (freq_cis_real = cos, freq_cis_imag = sin), Q15
    data['cos'] = [[q15(x) for x in m['freq_cis_real'][p]] for p in range(NPOS)]
    data['sin'] = [[q15(x) for x in m['freq_cis_imag'][p]] for p in range(NPOS)]

    txt, toks = R.generate(NPOS)
    data['tokens'] = list(map(int, toks))     # [1, 403, 407, ...] (18 : input + 17 gen)
    data['text'] = txt

    out = os.path.join(HERE, 'gen_model_dump.json')
    with open(out, 'w') as f:
        json.dump(data, f)
    print(f"dumped {out}")
    print(f"  tokens ({len(toks)}): {data['tokens']}")
    print(f"  text  : {txt!r}")
    print(f"  tok_emb shift={data['tok_emb_s']}  rms_final shift={data['rms_final_s']}")
    print(f"  layer0 shifts: wq={data['layers'][0]['wq_s']} wk={data['layers'][0]['wk_s']} "
          f"w1={data['layers'][0]['w1_s']} w2={data['layers'][0]['w2_s']}")


if __name__ == "__main__":
    main()
