"""Fully autonomous tensor-parallel matmul across N nodes (real mini-GG, inc.3.1).

x is broadcast to NN nodes; each node autonomously computes its NR-row matmul
slice (real DSP) and the ring all-gather distributes the slices so EVERY node
ends with the full output. No PC. Validated bit-exact vs fq_ref per slice, for
NN=2 and NN=4 (scalability).

Env: NODES=NN, ROWS_PER=NR (build with -Ptp_ring_cluster.NN/.NR).
"""

import os
import random
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles

CLK_NS = 37
W = 8
K = 64
NN = int(os.environ.get("NODES", "4"))
NR = int(os.environ.get("ROWS_PER", "16"))
ROWS = NN * NR


def to_i8(b): return b - 256 if b >= 128 else b


def fq_ref(Wrows, x):
    y = [sum(Wrows[r][k] * x[k] for k in range(K)) for r in range(len(Wrows))]
    ma = max(abs(v) for v in y)
    if ma == 0:
        return [0] * len(Wrows)
    sh = max(0, (ma.bit_length() - 1) - 6)
    rnd = (1 << (sh - 1)) if sh > 0 else 0
    return [max(-128, min(127, ((v + rnd) >> sh) if sh > 0 else v)) for v in y]


def vec_to_int(bs):
    v = 0
    for i, b in enumerate(bs):
        v |= (b & 0xFF) << (i * 8)
    return v


@cocotb.test()
async def test_tp_ring(dut):
    cocotb.start_soon(Clock(dut.clk, CLK_NS, units="ns").start())
    dut.rst_n.value = 0
    dut.start.value = 0
    dut.x.value = 0
    await ClockCycles(dut.clk, 5)

    rng = random.Random(6)
    Wm = [[rng.randrange(-50, 50) for _ in range(K)] for _ in range(ROWS)]
    x = [rng.randrange(-50, 50) for _ in range(K)]

    # backdoor-load each node's weight slice (rows g*NR .. (g+1)*NR-1)
    for g in range(NN):
        for r in range(NR):
            for k in range(K):
                dut.nodes[g].u_node.wmem[r * K + k].value = Wm[g * NR + r][k] & 0xFF

    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)

    dut.x.value = vec_to_int([v & 0xFF for v in x])
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    for _ in range(200000):
        await RisingEdge(dut.clk)
        if int(dut.done.value) == (1 << NN) - 1:
            break
    else:
        assert False, f"not all nodes done (done={int(dut.done.value):0{NN}b})"
    await ClockCycles(dut.clk, 2)

    # expected full output = concat of each node's per-slice fq_ref
    expected = []
    for g in range(NN):
        expected += fq_ref(Wm[g * NR:(g + 1) * NR], x)

    fv = int(dut.full_ys.value)
    node_bits = W * ROWS
    for g in range(NN):
        nv = (fv >> (g * node_bits)) & ((1 << node_bits) - 1)
        node_vec = [to_i8((nv >> (i * 8)) & 0xFF) for i in range(ROWS)]
        bad = [i for i in range(ROWS) if node_vec[i] != expected[i]]
        assert not bad, f"node {g} wrong at row {bad[0]} ({node_vec[bad[0]]} != {expected[bad[0]]})"

    dut._log.info(
        f"REAL MINI-GG (inc.3.1) PASS: autonomous tensor-parallel matmul on NN={NN} "
        f"nodes (NR={NR} rows each) -- real DSP compute + ring all-gather, zero PC, "
        f"every node holds the full {ROWS}-row output, bit-exact vs fq_ref"
    )
