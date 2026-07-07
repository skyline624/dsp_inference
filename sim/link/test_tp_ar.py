"""Autonomous column-parallel matmul + ring all-reduce (real mini-GG, inc.3.2).

W is split by columns across NN nodes; each node computes a partial output, and
the ring all-reduce sums them so EVERY node holds the full y = W@x. Summing raw
int32 partials == the full dot product -> bit-exact vs fq_ref over the full W,x.
Autonomous (no PC), scalable (NN=2 and NN=4).

Env: NODES=NN, KS=K/NN (build with -Ptp_ar_cluster.NN/.KS).
"""

import os
import random
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles

CLK_NS = 37
W = 8
D = 64
NN = int(os.environ.get("NODES", "2"))
KS = int(os.environ.get("KS", "32"))
K = NN * KS


def to_i8(b): return b - 256 if b >= 128 else b


def fq_ref(Wm, x):
    y = [sum(Wm[d][k] * x[k] for k in range(K)) for d in range(D)]
    ma = max(abs(v) for v in y)
    if ma == 0:
        return [0] * D
    sh = max(0, (ma.bit_length() - 1) - 6)
    rnd = (1 << (sh - 1)) if sh > 0 else 0
    return [max(-128, min(127, ((v + rnd) >> sh) if sh > 0 else v)) for v in y]


def vec_to_int(bs):
    v = 0
    for i, b in enumerate(bs):
        v |= (b & 0xFF) << (i * 8)
    return v


@cocotb.test()
async def test_tp_ar(dut):
    cocotb.start_soon(Clock(dut.clk, CLK_NS, units="ns").start())
    dut.rst_n.value = 0
    dut.start.value = 0
    dut.x_slices.value = 0
    await ClockCycles(dut.clk, 5)

    rng = random.Random(8)
    Wm = [[rng.randrange(-50, 50) for _ in range(K)] for _ in range(D)]
    x = [rng.randrange(-50, 50) for _ in range(K)]

    # backdoor: node g holds W[:, g*KS:(g+1)*KS] as wmem[d*KS+k]
    for g in range(NN):
        for d in range(D):
            for k in range(KS):
                dut.nodes[g].u_node.wmem[d * KS + k].value = Wm[d][g * KS + k] & 0xFF

    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)

    # x_slices : node g gets x[g*KS:(g+1)*KS]
    flat = []
    for g in range(NN):
        flat += [x[g * KS + k] & 0xFF for k in range(KS)]
    dut.x_slices.value = vec_to_int(flat)

    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    for _ in range(300000):
        await RisingEdge(dut.clk)
        if int(dut.done.value) == (1 << NN) - 1:
            break
    else:
        assert False, f"not all nodes done (done={int(dut.done.value):0{NN}b})"
    await ClockCycles(dut.clk, 2)

    expected = fq_ref(Wm, x)
    ov = int(dut.outs.value)
    node_bits = W * D
    for g in range(NN):
        nv = (ov >> (g * node_bits)) & ((1 << node_bits) - 1)
        node_vec = [to_i8((nv >> (i * 8)) & 0xFF) for i in range(D)]
        bad = [i for i in range(D) if node_vec[i] != expected[i]]
        assert not bad, f"node {g} all-reduce wrong at {bad[0]} ({node_vec[bad[0]]} != {expected[bad[0]]})"

    dut._log.info(
        f"REAL MINI-GG (inc.3.2) PASS: autonomous COLUMN-parallel matmul + ring "
        f"ALL-REDUCE on NN={NN} nodes (KS={KS}) -- real DSP, zero PC, every node "
        f"holds the full y=W@x, bit-exact vs fq_ref"
    )
