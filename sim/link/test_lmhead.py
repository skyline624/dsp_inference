"""G3 gate : the lm_head + argmax sequencer (one node, ss_link).

Seeds tok_emb (shared classifier, [512,64]) and rms_final, drives a known x[64],
and checks the sequencer's argmax token equals the integer reference (rms_final +
8 FQ chunks + integer re-align + argmax), matching prove_lmhead_argmax.

  make -f Makefile.lmhead MODULE=test_lmhead
"""
import math
import random
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles

D = 64
VOCAB = 512
A_RMS = 0x060000
A_EMB = 0x000000


def to_i8(b): return b - 256 if b >= 128 else b
def to_i8_shift(x):
    m = max((abs(v) for v in x), default=0.0)
    if m == 0: return [0]*len(x), 0
    s = math.ceil(math.log2(m/127.0))
    return [max(-128, min(127, round(v/(2.0**s)))) for v in x], s
def vint(bs):
    v = 0
    for i, b in enumerate(bs): v |= (b & 0xFF) << (i*8)
    return v
def preload(sd, base, data):
    wb = base >> 2; b = bytes(data)
    if len(b) % 4: b += bytes(4 - len(b) % 4)
    for w in range(len(b)//4):
        sd.mem[wb+w].value = b[4*w]|(b[4*w+1]<<8)|(b[4*w+2]<<16)|(b[4*w+3]<<24)


# node primitive references (fixed-point, mirror rmsnorm_op / FQ)
def rms_ref(x_i8, sx, w):
    xr = [x_i8[i]*(2.0**sx) for i in range(D)]
    inv = 1.0/math.sqrt(sum(v*v for v in xr)/D + 1e-5)
    y = [xr[i]*w[i]*inv for i in range(D)]
    return to_i8_shift(y)
def fq_ref(Wc_i8, x_i8, sx, sw):
    N = len(Wc_i8)
    y = [sum(Wc_i8[r][k]*x_i8[k] for k in range(D)) for r in range(N)]
    m = max((abs(v) for v in y), default=0)
    add = max(0, (m.bit_length()-1)-6) if m else 0
    half = (1 << (add-1)) if add > 0 else 0
    y8 = [max(-128, min(127, (v+half) >> add if add > 0 else v)) for v in y]
    return y8, sx+sw+add


@cocotb.test()
async def test_lmhead(dut):
    cocotb.start_soon(Clock(dut.clk, 37, units="ns").start())
    dut.rst_n.value = 0; dut.start.value = 0
    for s in ("x_in","sx","sw_rms","sw_emb"):
        getattr(dut, s).value = 0
    await ClockCycles(dut.clk, 10)

    rng = random.Random(123)
    x_f   = [rng.gauss(0, 1.0) for _ in range(D)]
    rms_w = [1.0]*D
    # tok_emb : [512,64] random, quantized with ONE shift (shared classifier)
    emb_f = [[rng.gauss(0, 0.1) for _ in range(D)] for _ in range(VOCAB)]
    x_i8, sx = to_i8_shift(x_f)
    rms_i8, swr = to_i8_shift(rms_w)
    flat = [v for row in emb_f for v in row]; _, sw_emb = to_i8_shift(flat)
    emb_i8 = [[max(-128, min(127, round(emb_f[r][k]/(2.0**sw_emb)))) for k in range(D)] for r in range(VOCAB)]

    sd = dut.u_sdram
    preload(sd, A_RMS, bytes((v & 0xFF) for v in rms_i8))
    preload(sd, A_EMB, bytes((emb_i8[r][k] & 0xFF) for r in range(VOCAB) for k in range(D)))

    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 5)
    dut.x_in.value = vint([v & 0xFF for v in x_i8]); dut.sx.value = sx & 0xFF
    dut.sw_rms.value = swr & 0xFF; dut.sw_emb.value = sw_emb & 0xFF
    dut.start.value = 1
    await RisingEdge(dut.clk); dut.start.value = 0

    for _ in range(2000000):
        await RisingEdge(dut.clk)
        if int(dut.done.value) == 1:
            break
    else:
        assert False, "lmhead seq never finished"
    tok_hw = int(dut.token.value)

    # integer reference : rms_final -> 8 FQ chunks -> re-align -> argmax
    xn_i8, sxn = rms_ref(x_i8, sx, rms_w)
    chunks = []; shifts = []
    for c in range(VOCAB // 64):
        Wc = emb_i8[c*64:(c+1)*64]
        y8, sy = fq_ref(Wc, xn_i8, sxn, sw_emb)
        chunks.append(y8); shifts.append(sy)
    sref = max(shifts)
    logits = []
    for c in range(VOCAB // 64):
        for j in range(64):
            logits.append(chunks[c][j] >> (sref - shifts[c]))
    tok_ref = max(range(VOCAB), key=lambda i: logits[i])

    dut._log.info(f"  lm_head : token_hw={tok_hw}  token_ref={tok_ref}")
    assert tok_hw == tok_ref, f"argmax mismatch : hw {tok_hw} vs ref {tok_ref}"
    dut._log.info("LM_HEAD PASS: sequencer FN(rms_final)+8xFQ+re-align+argmax -> correct token.")
