"""Per-node autonomous mini-GG over the link (task 4).

Node A sends x[64] to node B and waits for a reply. Node B is a self-contained
mini_gg: it autonomously receives, runs its 2-stage op chain (scale x2+1, then
saturate to int8) and sends the result back -- with NO per-op command issued by
anyone during the exchange. Independent clocks. Validates that the "zero-PC"
GG-style sequencing works per node, link-driven.
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles

A_NS = 37
B_NS = 31
L    = 64


def to_i8(b): return b - 256 if b >= 128 else b
def sat8(v):  return max(-128, min(127, v))


def vec_to_int(bs):
    v = 0
    for i, b in enumerate(bs):
        v |= (b & 0xFF) << (i * 8)
    return v


def int_to_vec(v, n):
    return [to_i8((v >> (i * 8)) & 0xFF) for i in range(n)]


@cocotb.test()
async def test_minigg(dut):
    cocotb.start_soon(Clock(dut.clk_a, A_NS, units="ns").start())
    cocotb.start_soon(Clock(dut.clk_b, B_NS, units="ns").start())
    dut.rst_a_n.value = 0; dut.rst_b_n.value = 0
    dut.start_a.value = 0; dut.vec_a_tx.value = 0
    await ClockCycles(dut.clk_a, 5)
    await ClockCycles(dut.clk_b, 5)
    dut.rst_a_n.value = 1; dut.rst_b_n.value = 1
    await ClockCycles(dut.clk_a, 2)

    # input activation (signed int8) and the expected autonomous transform
    x = [to_i8((i * 5 + 9) & 0xFF) for i in range(L)]
    expected = [sat8(xi * 2 + 1) for xi in x]

    dut.vec_a_tx.value = vec_to_int([v & 0xFF for v in x])

    # fire node A once; node B then runs recv->compute->send entirely on its own
    dut.start_a.value = 1
    await RisingEdge(dut.clk_a)
    dut.start_a.value = 0

    got = None
    for _ in range(40000):
        await RisingEdge(dut.clk_a)
        if int(dut.done_a.value) == 1:
            got = int_to_vec(int(dut.vec_a_rx.value), L)
            break
    assert got is not None, "node A never received B's autonomous reply"

    diffs = [(i, got[i], expected[i]) for i in range(L) if got[i] != expected[i]]
    assert not diffs, f"mini_gg transform mismatch, first={diffs[0]}"

    dut._log.info(
        f"TASK 4 PASS: node B autonomously RECV->chain(scale,saturate)->SEND "
        f"x[{L}] over the link, zero PC commands, result bit-exact across independent clocks"
    )
