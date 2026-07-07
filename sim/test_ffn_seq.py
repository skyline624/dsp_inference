"""COMPLETE autonomous FFN on a real brick node (final integration).

The on-chip ffn_seq runs the whole FFN by itself, zero PC:
  FN(rmsnorm) -> FQ(W1) -> FQ(W3) -> SS(silu) -> vec_alu MUL(silu.h3)
              -> FQ(W2) -> vec_alu ADD(residual x + .) -> y
driving the unmodified node over UART + using vec_alu for the glue, with handoff
and shift threading. Weights backdoor-loaded into SDRAM. Validated within
quantization tolerance vs the float FFN reference (with the same quantized weights).
"""

import math
import random
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles

CLK_NS = 37
D = 64
K = 64
A_RMS, A_W1, A_W3, A_W2 = 0x100000, 0x101000, 0x102000, 0x103000


def to_i8(b): return b - 256 if b >= 128 else b


def to_i8_shift(x):
    m = max((abs(v) for v in x), default=0.0)
    if m == 0:
        return [0] * len(x), 0
    s = math.ceil(math.log2(m / 127.0))
    return [max(-128, min(127, round(v / (2.0 ** s)))) for v in x], s


def quantize_matrix(M):
    flat = [v for row in M for v in row]
    _, s = to_i8_shift(flat)
    return [[max(-128, min(127, round(v / (2.0 ** s)))) for v in row] for row in M], s


def deq(i8, s): return [v * (2.0 ** s) for v in i8]
def rmsnorm_f(x, w, eps=1e-5):
    inv = 1.0 / math.sqrt(sum(v*v for v in x)/len(x) + eps)
    return [x[i]*w[i]*inv for i in range(len(x))]
def silu_f(v): return v / (1.0 + math.exp(-v))
def matvec_f(Wm, x): return [sum(Wm[r][k]*x[k] for k in range(len(x))) for r in range(len(Wm))]


def vec_to_int(bs):
    v = 0
    for i, b in enumerate(bs):
        v |= (b & 0xFF) << (i * 8)
    return v


def preload(sdram, base, data):
    wbase = base >> 2
    b = bytes(data)
    if len(b) % 4:
        b += bytes(4 - len(b) % 4)
    for w in range(len(b)//4):
        sdram.mem[wbase + w].value = b[4*w] | (b[4*w+1]<<8) | (b[4*w+2]<<16) | (b[4*w+3]<<24)


@cocotb.test()
async def test_ffn_seq(dut):
    cocotb.start_soon(Clock(dut.clk, CLK_NS, units="ns").start())
    dut.rst_n.value = 0
    dut.start.value = 0
    for s in ("x_in", "sx", "sw_rms", "sw1", "sw3", "sw2"):
        getattr(dut, s).value = 0
    await ClockCycles(dut.clk, 10)

    rng = random.Random(31)
    x_f   = [rng.gauss(0, 1.0) for _ in range(D)]
    rms_w = [1.0] * D
    W1_f  = [[rng.gauss(0, 0.1) for _ in range(K)] for _ in range(D)]
    W3_f  = [[rng.gauss(0, 0.1) for _ in range(K)] for _ in range(D)]
    W2_f  = [[rng.gauss(0, 0.1) for _ in range(K)] for _ in range(D)]

    x_i8, sx    = to_i8_shift(x_f)
    rms_i8, swr = to_i8_shift(rms_w)
    W1_i8, sw1  = quantize_matrix(W1_f)
    W3_i8, sw3  = quantize_matrix(W3_f)
    W2_i8, sw2  = quantize_matrix(W2_f)

    sd = dut.u_node.u_sdram
    preload(sd, A_RMS, bytes((v & 0xFF) for v in rms_i8))
    preload(sd, A_W1, bytes((W1_i8[r][k] & 0xFF) for r in range(D) for k in range(K)))
    preload(sd, A_W3, bytes((W3_i8[r][k] & 0xFF) for r in range(D) for k in range(K)))
    preload(sd, A_W2, bytes((W2_i8[r][k] & 0xFF) for r in range(D) for k in range(K)))

    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 5)
    dut.x_in.value = vec_to_int([v & 0xFF for v in x_i8])
    dut.sx.value = sx & 0xFF
    dut.sw_rms.value = swr & 0xFF
    dut.sw1.value = sw1 & 0xFF
    dut.sw3.value = sw3 & 0xFF
    dut.sw2.value = sw2 & 0xFF

    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    for _ in range(800000):
        await RisingEdge(dut.clk)
        if int(dut.done.value) == 1:
            break
    else:
        assert False, "ffn_seq never finished"

    rv = int(dut.result.value)
    rsh = to_i8(int(dut.result_sh.value))
    got = [to_i8((rv >> (i*8)) & 0xFF) * (2.0 ** rsh) for i in range(D)]

    # float FFN reference (same quantized weights, dequantized)
    xr   = deq(x_i8, sx)
    rmsr = deq(rms_i8, swr)
    W1r  = [[W1_i8[r][k] * (2.0**sw1) for k in range(K)] for r in range(D)]
    W3r  = [[W3_i8[r][k] * (2.0**sw3) for k in range(K)] for r in range(D)]
    W2r  = [[W2_i8[r][k] * (2.0**sw2) for k in range(K)] for r in range(D)]
    xn   = rmsnorm_f(xr, rmsr)
    h1   = matvec_f(W1r, xn); h3 = matvec_f(W3r, xn)
    hg   = [silu_f(h1[i]) * h3[i] for i in range(D)]
    o    = matvec_f(W2r, hg)
    ref  = [xr[i] + o[i] for i in range(D)]

    ma = max(abs(v) for v in ref) or 1.0
    err = max(abs(got[i] - ref[i]) for i in range(D))
    dut._log.info(f"  full autonomous FFN: max_err={err:.4f} ({100*err/ma:.1f}% of range)")
    assert err / ma < 0.35, f"FFN output too far from reference ({err})"

    dut._log.info(
        "FFN SEQUENCER (FINAL) PASS: the on-chip sequencer ran the COMPLETE FFN "
        "(rmsnorm->W1/W3->silu->mul->W2->residual) on the real node, zero PC."
    )
