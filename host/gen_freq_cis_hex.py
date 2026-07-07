#!/usr/bin/env python3
"""Generate src/freq_cis_{cos,sin}.hex for gen_seq's on-chip rope ROM (Session 6b).

The values are the EXACT Q15 freq_cis of the generation oracle, taken verbatim
from host/gen_model_dump.json (m['cos']/m['sin'] : NPOS=17 positions x HS/2=4
pairs) - the SAME values test_gen_F used to drive the cos_q15/sin_q15 ports.

Relocating them from a flat 1088-bit input port (whose variable part-select
cos_q15[pos*64 + 16*rcp +: 16] synthesised into two 68:1 muxes + a high-fanout
1088-bit bus = the GW2AR-18C routing congestion) into two internal BSRAM ROMs
read by (pos, pair) removes that mux from the fabric. See gen_seq.v rope states.

ROM layout : idx = pos*(HS/2) + pair, one signed 16-bit (Q15) value per line,
matching gen_seq's  cos_rom[pos_reg*(HS/2) + rcp].
At pos=0 : cos=0x7fff (~1.0 in Q15), sin=0x0000 -> the rope identity that
test_gen_B already validated. Same numbers, just read from ROM instead of a port.
"""
import json
import os

HS = 8
HERE = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(HERE, "gen_model_dump.json")) as f:
    m = json.load(f)


def dump(name, table):
    path = os.path.join(HERE, "..", "src", name)
    with open(path, "w") as f:
        for pos in range(len(table)):
            for pair in range(HS // 2):
                f.write("%04x\n" % (table[pos][pair] & 0xFFFF))
    return path


c = dump("freq_cis_cos.hex", m["cos"])
s = dump("freq_cis_sin.hex", m["sin"])
print("wrote %s and %s : %d pos x %d pairs = %d entries each"
      % (c, s, len(m["cos"]), HS // 2, len(m["cos"]) * (HS // 2)))
