"""Autonomous HEAD-PARALLEL attention over 2 brick nodes (no PC), pos=0/T=1.

The on-chip attn_tp_seq splits attention across 2 nodes: rmsnorm broadcast,
Wq/Wk/Wv head-parallel, gather (vec_alu ADD with padding), real MM core, Wo
row-parallel, gather, residual. Weights split across the two SDRAMs. Validated
within tolerance vs the float attention reference (pos=0: attn = GQA-expand(V)).
"""

import math
import os
import random
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles

CLK_NS = 37
D = 64
H, KH, HS = 8, 4, 8
NREP = H // KH
NN = int(os.environ.get("NN", "2"))   # match -Pattn_tp_top.NN
A_RMS, A_WQ, A_WK, A_WV, A_WO = 0x100000, 0x101000, 0x102000, 0x103000, 0x104000


def to_i8(b): return b - 256 if b >= 128 else b


def to_i8_shift(x):
    m = max((abs(v) for v in x), default=0.0)
    if m == 0:
        return [0]*len(x), 0
    s = math.ceil(math.log2(m/127.0))
    return [max(-128, min(127, round(v/(2.0**s)))) for v in x], s


def quantize_matrix(M):
    flat = [v for row in M for v in row]
    _, s = to_i8_shift(flat)
    return [[max(-128, min(127, round(v/(2.0**s)))) for v in row] for row in M], s


def deq(i8, s): return [v*(2.0**s) for v in i8]
def rmsnorm_f(x, w, eps=1e-5):
    inv = 1.0/math.sqrt(sum(v*v for v in x)/len(x)+eps); return [x[i]*w[i]*inv for i in range(len(x))]
def matvec_f(Wm, x): return [sum(Wm[r][k]*x[k] for k in range(len(x))) for r in range(len(Wm))]


def vec_to_int(bs):
    v = 0
    for i, b in enumerate(bs):
        v |= (b & 0xFF) << (i*8)
    return v


def preload(sd, base, data):
    wbase = base >> 2
    b = bytes(data)
    if len(b) % 4:
        b += bytes(4 - len(b) % 4)
    for w in range(len(b)//4):
        sd.mem[wbase+w].value = b[4*w] | (b[4*w+1]<<8) | (b[4*w+2]<<16) | (b[4*w+3]<<24)


@cocotb.test()
async def test_attn_tp(dut):
    cocotb.start_soon(Clock(dut.clk, CLK_NS, units="ns").start())
    dut.rst_n.value = 0; dut.start.value = 0
    for s in ("x_in","sx","sw_rms","swq","swk","swv","swo"):
        getattr(dut, s).value = 0
    await ClockCycles(dut.clk, 10)

    rng = random.Random(51)
    x_f   = [rng.gauss(0, 1.0) for _ in range(D)]
    rms_w = [1.0]*D
    Wq_f  = [[rng.gauss(0, 0.1) for _ in range(D)] for _ in range(H*HS)]
    Wk_f  = [[rng.gauss(0, 0.1) for _ in range(D)] for _ in range(KH*HS)]
    Wv_f  = [[rng.gauss(0, 0.1) for _ in range(D)] for _ in range(KH*HS)]
    Wo_f  = [[rng.gauss(0, 0.1) for _ in range(H*HS)] for _ in range(D)]

    x_i8, sx    = to_i8_shift(x_f)
    rms_i8, swr = to_i8_shift(rms_w)
    Wq_i8, swq  = quantize_matrix(Wq_f)
    Wk_i8, swk  = quantize_matrix(Wk_f)
    Wv_i8, swv  = quantize_matrix(Wv_f)
    Wo_i8, swo  = quantize_matrix(Wo_f)

    QPN, KPN, OPN = 64 // NN, 32 // NN, 64 // NN

    def sdram_of(c):
        node = dut.nodes[c]
        for h in (getattr(getattr(node, "u_n", node), "u_sdram", None), getattr(node, "u_sdram", None)):
            if h is not None and hasattr(h, "mem"):
                return h
        return node.u_n.u_sdram
    sd = [sdram_of(c) for c in range(NN)]

    def rows(M, a, b, cols): return bytes((M[r][k] & 0xFF) for r in range(a, b) for k in range(cols))

    # rmsnorm weight on node 0 ; Wq/Wk/Wv/Wo split by chunk (QPN/KPN/OPN rows each)
    preload(sd[0], A_RMS, bytes((v & 0xFF) for v in rms_i8))
    for c in range(NN):
        preload(sd[c], A_WQ, rows(Wq_i8, c*QPN, (c+1)*QPN, D))
        preload(sd[c], A_WK, rows(Wk_i8, c*KPN, (c+1)*KPN, D))
        preload(sd[c], A_WV, rows(Wv_i8, c*KPN, (c+1)*KPN, D))
        preload(sd[c], A_WO, rows(Wo_i8, c*OPN, (c+1)*OPN, H*HS))

    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 5)
    dut.x_in.value = vec_to_int([v & 0xFF for v in x_i8])
    dut.sx.value = sx & 0xFF; dut.sw_rms.value = swr & 0xFF
    dut.swq.value = swq & 0xFF; dut.swk.value = swk & 0xFF
    dut.swv.value = swv & 0xFF; dut.swo.value = swo & 0xFF

    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    for _ in range(900000):
        await RisingEdge(dut.clk)
        if int(dut.done.value) == 1:
            break
    else:
        assert False, "attn_tp_seq never finished"

    rv = int(dut.result.value); rsh = to_i8(int(dut.result_sh.value))
    got = [to_i8((rv >> (i*8)) & 0xFF) * (2.0**rsh) for i in range(D)]

    # float reference (pos=0/T=1: attn = GQA-expand(V))
    xr = deq(x_i8, sx); rmsr = deq(rms_i8, swr)
    Wvr = [[Wv_i8[r][k]*(2.0**swv) for k in range(D)] for r in range(KH*HS)]
    Wor = [[Wo_i8[r][k]*(2.0**swo) for k in range(H*HS)] for r in range(D)]
    xn = rmsnorm_f(xr, rmsr)
    V  = matvec_f(Wvr, xn)
    attn = [V[(h//NREP)*HS + d] for h in range(H) for d in range(HS)]
    o  = matvec_f(Wor, attn)
    ref = [xr[i] + o[i] for i in range(D)]

    ma = max(abs(v) for v in ref) or 1.0
    err = max(abs(got[i]-ref[i]) for i in range(D))
    dut._log.info(f"  autonomous TP attention (NN={NN} node(s)): max_err={err:.4f} ({100*err/ma:.1f}% of range)")
    assert err/ma < 0.35, f"TP attention output too far from reference ({err})"

    dut._log.info(
        f"ATTENTION TP (autonomous, NN={NN}) PASS: on-chip seq ran attention "
        "(Q/K/V head-parallel, real MM, Wo row-parallel + gather), zero PC. "
        + ("mono-card." if NN == 1 else f"{NN}-node head-parallel.")
    )
