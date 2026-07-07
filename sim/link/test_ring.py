"""Scalable ring all-gather across N nodes (real mini-GG, increment 2).

N nodes in a ring each start with their own slice; after the autonomous ring
all-gather, EVERY node holds the full vector (all N slices). One in + one out
link per node -> the topology scales to any N (run with N=2 and N=4 to prove it).

Set N via env NODES (matches the -Pring_cluster.N=... build).
"""

import os
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles

CLK_NS = 37
W = 8
S = 32
N = int(os.environ.get("NODES", "2"))
SB = W * S


def vec_to_int(bs):
    v = 0
    for i, b in enumerate(bs):
        v |= (b & 0xFF) << (i * 8)
    return v


def int_to_bytes(v, n):
    return [(v >> (i * 8)) & 0xFF for i in range(n)]


@cocotb.test()
async def test_ring(dut):
    cocotb.start_soon(Clock(dut.clk, CLK_NS, units="ns").start())
    dut.rst_n.value = 0
    dut.start.value = 0
    dut.my_slices.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)

    # node g's slice : S bytes, distinct per node so we can check placement
    slices = [[(g * 40 + j) & 0xFF for j in range(S)] for g in range(N)]
    full_expected = [b for g in range(N) for b in slices[g]]   # concat of all slices

    all_bytes = [b for g in range(N) for b in slices[g]]       # my_slices flat (node0 first)
    dut.my_slices.value = vec_to_int(all_bytes)

    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    # wait until all nodes report done
    for _ in range(50000):
        await RisingEdge(dut.clk)
        if int(dut.done.value) == (1 << N) - 1:
            break
    else:
        assert False, f"not all nodes done (done={int(dut.done.value):0{N}b})"

    await ClockCycles(dut.clk, 2)

    # every node must hold the full vector (all N slices, in order)
    fv = int(dut.full_vecs.value)
    node_bits = SB * N
    for g in range(N):
        node_vec = int_to_bytes((fv >> (g * node_bits)) & ((1 << node_bits) - 1), S * N)
        bad = [i for i in range(S * N) if node_vec[i] != full_expected[i]]
        assert not bad, f"node {g} all-gather wrong at byte {bad[0]}"

    dut._log.info(
        f"REAL MINI-GG (inc.2) PASS: ring all-gather over N={N} nodes "
        f"(1 in + 1 out link/node), every node holds the full {S*N}-byte vector, autonomous"
    )
