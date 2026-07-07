"""Autonomous tensor-parallel matmul node (real mini-GG, increment 1).

The testbench sends x[K] over the link; the node autonomously runs y = W@x on its
slice with the REAL DSP (mac18) + project requantize, and sends y[N] back -- no
per-op command. Weights are backdoor-loaded (never cross the link). Validated
bit-exact vs the project reference (fq_ref).
"""

import random
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles

A_NS = 37
B_NS = 31
K = 64
N = 32


def to_i8(b): return b - 256 if b >= 128 else b


def fq_ref(W, x, n):
    y = [sum(W[r][k] * x[k] for k in range(K)) for r in range(n)]
    ma = max(abs(v) for v in y)
    if ma == 0:
        return [0] * n
    sh = max(0, (ma.bit_length() - 1) - 6)
    rnd = (1 << (sh - 1)) if sh > 0 else 0
    return [max(-128, min(127, ((v + rnd) >> sh) if sh > 0 else v)) for v in y]


def vec_to_int(bs):
    v = 0
    for i, b in enumerate(bs):
        v |= (b & 0xFF) << (i * 8)
    return v


def int_to_vec(v, n):
    return [to_i8((v >> (i * 8)) & 0xFF) for i in range(n)]


@cocotb.test()
async def test_tp_mm(dut):
    cocotb.start_soon(Clock(dut.clk_a, A_NS, units="ns").start())
    cocotb.start_soon(Clock(dut.clk_b, B_NS, units="ns").start())
    dut.rst_a_n.value = 0; dut.rst_b_n.value = 0
    dut.start_a.value = 0; dut.vec_a_tx.value = 0
    await ClockCycles(dut.clk_a, 5)
    await ClockCycles(dut.clk_b, 5)

    # backdoor-load the node's weight slice (row-major), never crosses the link
    rng = random.Random(4)
    W = [[rng.randrange(-50, 50) for _ in range(K)] for _ in range(N)]
    for r in range(N):
        for k in range(K):
            dut.u_node.wmem[r * K + k].value = W[r][k] & 0xFF

    dut.rst_a_n.value = 1; dut.rst_b_n.value = 1
    await ClockCycles(dut.clk_a, 2)

    x = [rng.randrange(-50, 50) for _ in range(K)]
    dut.vec_a_tx.value = vec_to_int([v & 0xFF for v in x])

    dut.start_a.value = 1
    await RisingEdge(dut.clk_a)
    dut.start_a.value = 0

    got = None
    for _ in range(60000):
        await RisingEdge(dut.clk_a)
        if int(dut.done_a.value) == 1:
            got = int_to_vec(int(dut.vec_a_rx.value), N)
            break
    assert got is not None, "no autonomous result received"

    ref = fq_ref(W, x, N)
    diffs = [(i, got[i], ref[i]) for i in range(N) if got[i] != ref[i]]
    assert not diffs, f"matmul mismatch vs fq_ref, first={diffs[0]}"

    dut._log.info(
        f"REAL MINI-GG (inc.1) PASS: node autonomously computed y=W@x[{N}x{K}] "
        f"with real DSP (mac18) + requantize, link-driven, zero PC, bit-exact vs fq_ref"
    )
