"""Step A gate : gen_seq embedding. Send EE(tok), check vfile[XB] == tok_emb[tok].

The node's EE reads SDRAM[tok*64] (ADDR_TOK_EMB = 0, no base), so tok_emb is
preloaded at 0x000000. This validates the embedding phase of the LUT-lean
generation sequencer (register-file vfile) in isolation.

  make -f Makefile.gen MODULE=test_gen_A
"""
import random
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles

D = 64
VOCAB = 512


def to_i8(b): return b - 256 if b >= 128 else b
def preload(sd, base, data):
    wb = base >> 2; b = bytes(data)
    if len(b) % 4: b += bytes(4 - len(b) % 4)
    for w in range(len(b)//4):
        sd.mem[wb+w].value = b[4*w]|(b[4*w+1]<<8)|(b[4*w+2]<<16)|(b[4*w+3]<<24)


@cocotb.test()
async def test_gen_A(dut):
    cocotb.start_soon(Clock(dut.clk, 37, units="ns").start())
    dut.rst_n.value = 0; dut.start.value = 0; dut.cur_tok.value = 0
    await ClockCycles(dut.clk, 10)

    # tok_emb [512,64] int8, random, at 0x000000
    rng = random.Random(202)
    emb = [[rng.randrange(-128, 128) for _ in range(D)] for _ in range(VOCAB)]
    preload(dut.u_sdram, 0x000000, bytes((emb[r][k] & 0xFF) for r in range(VOCAB) for k in range(D)))

    TOK = 100
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 5)
    dut.cur_tok.value = TOK
    dut.start.value = 1
    await RisingEdge(dut.clk); dut.start.value = 0

    for _ in range(2000000):
        await RisingEdge(dut.clk)
        if int(dut.done.value) == 1:
            break
    else:
        assert False, "gen_seq step A never finished"

    xv = int(dut.xb_out.value)
    got = [to_i8((xv >> (i*8)) & 0xFF) for i in range(D)]
    ref = emb[TOK]

    bad = [i for i in range(D) if got[i] != ref[i]]
    dut._log.info(f"  embed tok={TOK} : got[:6]={got[:6]} ref[:6]={ref[:6]}")
    assert not bad, f"embedding mismatch at {bad[:5]} (got {got[bad[0]]} ref {ref[bad[0]]})"
    dut._log.info("STEP A PASS: gen_seq embedding -> vfile[XB] == tok_emb[tok], bit-exact.")
