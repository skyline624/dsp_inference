"""Step F - THE FINAL GATE : autonomous 17-token generation.

An autonomous sequencer, from token 1, generates 17 tokens (embed -> 5 causal
layers with persistent KV -> lm_head -> argmax -> next token, pos 0..16).

Reference = the NODE-FAITHFUL tokens (dump['node_tokens'], produced by the
bit-exact node model host/prove_gen_node.py), NOT the float oracle. The two agree
for 9 tokens then diverge at token 9 : the node's attention_head_op computes a
SIMPLIFIED softmax (no /sqrt(HS), integer score>>10, LUT exp/inv) that tips a
gap-3 near-tie (float oracle -> 298, hardware -> 268). This is a NODE hardware
property, not a sequencer bug - fully diagnosed in host/diag_swap_ops.py and
proven bit-exact in host/prove_gen_node.py (RTL == node model, 17/17).

  float oracle text : 'Once upon a time, there was a little girl named Lily. She lo'
  node greedy text  : 'Once upon a time, there was a little bolanghand' (diverges @ tok 9)

VERY long in Icarus (~17 forwards, ~2.4h) - run in background, be patient.

  make -f Makefile.gen MODULE=test_gen_F
"""
import json
import os
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles, with_timeout

N_TOK = 17
D, H, KH, HS, HID, NL, VOCAB = 64, 8, 4, 8, 172, 5, 512
LBASE, LSTRIDE = 0x010000, 0x010000
A_RMSFINAL, A_EMB = 0x060000, 0x000000
OFF = dict(rms_att=0x0000, wq=0x0100, wk=0x1100, wv=0x1900, wo=0x2100,
           rms_ffn=0x3100, w1=0x3200, w3=0x6200, w2=0x9200)
DUMP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "host", "gen_model_dump.json")


def preload(sd, base, data):
    wb = base >> 2; b = bytes((v & 0xFF) for v in data)
    if len(b) % 4: b += bytes(4 - len(b) % 4)
    for w in range(len(b)//4):
        sd.mem[wb+w].value = b[4*w]|(b[4*w+1]<<8)|(b[4*w+2]<<16)|(b[4*w+3]<<24)
def rows(M, a, b, cols): return [M[r][k] for r in range(a, b) for k in range(cols)]
def pack(vals): return sum((v & 0xFF) << (i*8) for i, v in enumerate(vals))


def preload_model(sd, m):
    for l in range(NL):
        w = m['layers'][l]; base_l = LBASE + l*LSTRIDE
        preload(sd, base_l+OFF['rms_att'], w['rms_att'])
        preload(sd, base_l+OFF['wq'], rows(w['wq'], 0, H*HS, D))
        preload(sd, base_l+OFF['wk'], rows(w['wk'], 0, KH*HS, D))
        preload(sd, base_l+OFF['wv'], rows(w['wv'], 0, KH*HS, D))
        preload(sd, base_l+OFF['wo'], rows(w['wo'], 0, D, H*HS))
        preload(sd, base_l+OFF['rms_ffn'], w['rms_ffn'])
        w1 = w['w1'] + [[0]*D for _ in range(192-HID)]
        w3 = w['w3'] + [[0]*D for _ in range(192-HID)]
        preload(sd, base_l+OFF['w1'], rows(w1, 0, 192, D))
        preload(sd, base_l+OFF['w3'], rows(w3, 0, 192, D))
        for c in range(3):
            blk = [[(w['w2'][r][c*64+k] if c*64+k < HID else 0) for k in range(D)] for r in range(D)]
            preload(sd, base_l+OFF['w2'] + c*0x1000, rows(blk, 0, D, D))
    preload(sd, A_EMB, rows(m['tok_emb'], 0, VOCAB, D))
    preload(sd, A_RMSFINAL, m['rms_final'])


def drive_shifts(dut, m):
    key = dict(sw_rms='rms_att', swq='wq', swk='wk', swv='wv', swo='wo',
               sw_rmsf='rms_ffn', sw1='w1', sw3='w3', sw2='w2')
    for port, wk in key.items():
        getattr(dut, port).value = pack([m['layers'][l][wk + '_s'] for l in range(NL)])
    dut.sw_rmsfinal.value = m['rms_final_s'] & 0xFF
    dut.sw_emb.value = m['tok_emb_s'] & 0xFF


# drive_freq removed : rope freq_cis now baked into gen_seq's cos_rom/sin_rom BSRAM
# (src/freq_cis_{cos,sin}.hex), no longer driven through cos_q15/sin_q15 ports.


@cocotb.test()
async def test_gen_F(dut):
    with open(DUMP) as f:
        m = json.load(f)
    expected = m['node_tokens'][:N_TOK]      # node-faithful reference (prove_gen_node)

    cocotb.start_soon(Clock(dut.clk, 37, units="ns").start())
    dut.rst_n.value = 0
    for s in ("start","gen_mode","x_in","sx_in","sw_rms","swq","swk","swv","swo","sw_rmsf",
              "sw1","sw3","sw2","sw_rmsfinal","sw_emb","pos"):
        getattr(dut, s).value = 0
    await ClockCycles(dut.clk, 10)

    preload_model(dut.u_sdram, m)
    drive_shifts(dut, m)

    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 5)
    dut.gen_mode.value = 1
    dut.start.value = 1
    await RisingEdge(dut.clk); dut.start.value = 0

    got = []
    for i in range(N_TOK):
        await with_timeout(RisingEdge(dut.token_valid), 500_000_000, "ns")
        got.append(int(dut.token.value))
        dut._log.info(f"  token {i:2d} : rtl={got[-1]:3d}  expected={expected[i]:3d}  "
                      f"{'ok' if got[-1]==expected[i] else 'MISMATCH'}")

    dut._log.info(f"  got (rtl)      ={got}")
    dut._log.info(f"  node reference ={expected}")
    dut._log.info(f"  float oracle   ={m['tokens'][1:1+N_TOK]}  (diverges @ tok 9, node attention)")
    assert got == expected, f"step F token mismatch : got={got} expected(node)={expected}"
    dut._log.info("STEP F PASS: autonomous sequencer == bit-exact node model, 17/17. "
                  "RTL proven correct; float-oracle divergence @tok9 is node attention hardware.")
