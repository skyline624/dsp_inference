"""Cluster inter-node transport over the source-synchronous link (task 1).

Two nodes on INDEPENDENT clocks exchange activation vectors x[64] full-duplex
(A->B and B->A simultaneously) through link_send -> ss_link (async FIFO) ->
link_recv. Validates the parallel link as the cluster's node-to-node transport
carrying real activation payloads across asynchronous clock domains.
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles

A_NS = 37     # node A oscillator (~27.0 MHz)
B_NS = 31     # node B oscillator (~32.3 MHz)
L    = 64     # activation vector length (bytes)


def vec_to_int(bs):
    v = 0
    for i, b in enumerate(bs):
        v |= (b & 0xFF) << (i * 8)
    return v


def int_to_vec(v, n):
    return [(v >> (i * 8)) & 0xFF for i in range(n)]


async def pulse(clk, sig):
    sig.value = 1
    await RisingEdge(clk)
    sig.value = 0


async def wait_recv(clk, done_sig, vec_sig, n, timeout=20000):
    for _ in range(timeout):
        await RisingEdge(clk)
        if int(done_sig.value) == 1:
            return int_to_vec(int(vec_sig.value), n)
    assert False, "receive timed out"


@cocotb.test()
async def test_cluster_link(dut):
    cocotb.start_soon(Clock(dut.clk_a, A_NS, units="ns").start())
    cocotb.start_soon(Clock(dut.clk_b, B_NS, units="ns").start())

    dut.rst_a_n.value = 0
    dut.rst_b_n.value = 0
    dut.start_a.value = 0
    dut.start_b.value = 0
    dut.vec_a_tx.value = 0
    dut.vec_b_tx.value = 0
    await ClockCycles(dut.clk_a, 5)
    await ClockCycles(dut.clk_b, 5)
    dut.rst_a_n.value = 1
    dut.rst_b_n.value = 1
    await ClockCycles(dut.clk_a, 2)

    # node A -> B : an attention-like activation slice ; B -> A : a different one
    xA = [(i * 3 + 1) & 0xFF for i in range(L)]
    xB = [(200 - i * 5) & 0xFF for i in range(L)]
    dut.vec_a_tx.value = vec_to_int(xA)
    dut.vec_b_tx.value = vec_to_int(xB)

    # launch the receivers, then fire both senders (full-duplex)
    rb = cocotb.start_soon(wait_recv(dut.clk_b, dut.done_b, dut.vec_b_rx, L))  # B gets A's vec
    ra = cocotb.start_soon(wait_recv(dut.clk_a, dut.done_a, dut.vec_a_rx, L))  # A gets B's vec
    cocotb.start_soon(pulse(dut.clk_a, dut.start_a))
    cocotb.start_soon(pulse(dut.clk_b, dut.start_b))

    got_b = await rb
    got_a = await ra

    assert got_b == xA, f"A->B corrupted: first diff {next(i for i in range(L) if got_b[i]!=xA[i])}"
    assert got_a == xB, f"B->A corrupted: first diff {next(i for i in range(L) if got_a[i]!=xB[i])}"

    dut._log.info(
        f"TASK 1 PASS: full-duplex activation exchange x[{L}] between 2 nodes on "
        f"independent clocks ({1000/A_NS:.1f}/{1000/B_NS:.1f} MHz) over ss_link, both directions bit-exact"
    )
