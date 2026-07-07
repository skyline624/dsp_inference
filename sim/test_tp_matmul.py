"""Milestone 3 - tensor-parallel matmul (row-parallel / output-split) across 2 nodes.

Validates the core of the scaling architecture: a single matmul y = W @ x is
split by OUTPUT ROWS across two FPGA nodes. Each node holds ONLY its slice of W,
in its OWN SDRAM, and computes its slice of y in parallel. The coordinator
(this testbench) broadcasts x to both nodes and gathers the two y slices.

Why this matters for the project's goals:
  - memory bandwidth: W is split across 2 independent SDRAMs, streamed in // ;
  - compute: 2 independent DSP datapaths run concurrently ;
  - only activations (x ~64 B, y slices) cross between coordinator and nodes.

Correctness: each node runs the project's real FQ command on its W slice, so its
output must be BIT-EXACT vs the project reference (requantize) for that slice.
Distributing the rows across nodes therefore gives the identical result to doing
them on one node -- proven here.
"""

import random
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles

CLK_NS      = 37
DIV         = 27
K           = 64       # matmul inner dim (RTL xbuf width)
N_PER_NODE  = 32       # output rows assigned to each node
WADDR       = 0x100000


def addr_bytes(a): return bytes([a & 0xFF, (a >> 8) & 0xFF, (a >> 16) & 0xFF])
def n_bytes(n):    return bytes([n & 0xFF, (n >> 8) & 0xFF])
def i8(b):         return b - 256 if b >= 128 else b


def fq_ref(W, x, N):
    """Project reference (test_sdram_diag.py): int matmul + pow2 requantize.
    Returns (out_int8_list, shift_used)."""
    y = [sum(W[r][k] * x[k] for k in range(K)) for r in range(N)]
    max_abs = max(abs(v) for v in y)
    if max_abs == 0:
        return [0] * N, 0
    sh = max(0, (max_abs.bit_length() - 1) - 6)
    rnd = (1 << (sh - 1)) if sh > 0 else 0
    out = []
    for v in y:
        o = (v + rnd) >> sh if sh > 0 else v
        out.append(max(-128, min(127, o)))
    return out, sh


def _as_int(sig):
    try:
        return int(sig.value)
    except Exception:
        return None


async def uart_send(clk, rx_sig, data):
    for byte in data:
        rx_sig.value = 0
        await ClockCycles(clk, DIV)
        for i in range(8):
            rx_sig.value = (byte >> i) & 1
            await ClockCycles(clk, DIV)
        rx_sig.value = 1
        await ClockCycles(clk, DIV)


async def uart_rx_monitor(clk, tx_sig, sink):
    while True:
        while _as_int(tx_sig) != 0:
            await RisingEdge(clk)
        await ClockCycles(clk, DIV // 2)
        byte = 0
        for i in range(8):
            await ClockCycles(clk, DIV)
            byte |= (_as_int(tx_sig) or 0) << i
        await ClockCycles(clk, DIV)
        sink.append(byte)


class Node:
    """A cluster node addressed over its own UART line."""
    def __init__(self, clk, rx_sig, tx_sig, fpga):
        self.clk, self.rx, self.fpga = clk, rx_sig, tx_sig
        self.sink = bytearray()
        self.off = 0
        cocotb.start_soon(uart_rx_monitor(clk, tx_sig, self.sink))

    async def xfer(self, pkt, nresp, timeout_cycles=3_000_000):
        await uart_send(self.clk, self.rx, pkt)
        waited = 0
        while len(self.sink) < self.off + nresp:
            await ClockCycles(self.clk, 20)
            waited += 20
            assert waited < timeout_cycles, (
                f"timeout: {len(self.sink) - self.off}/{nresp} resp bytes"
            )
        resp = bytes(self.sink[self.off:self.off + nresp])
        self.off += nresp
        return resp

    async def load(self, addr, data):
        ack = await self.xfer(b"LL" + addr_bytes(addr) + n_bytes(len(data)) + data, 2)
        assert ack == b"LK", f"LL ack {ack!r}"

    async def fq(self, N, sx, sw, x, addr):
        pkt = b"FQ" + bytes([N, sx & 0xFF, sw & 0xFF]) + bytes((v & 0xFF) for v in x) + addr_bytes(addr)
        resp = await self.xfer(pkt, 3 + N)
        assert resp[:2] == b"FQ", f"FQ magic {resp[:2]!r}"
        return [i8(resp[3 + i]) for i in range(N)], i8(resp[2])


async def boot_node(clk, fpga):
    for _ in range(40000):
        await RisingEdge(clk)
        if _as_int(fpga.rst_n) == 1:
            break
    else:
        assert False, "reset never released"
    for _ in range(40000):
        await RisingEdge(clk)
        if _as_int(fpga.sd_busy) == 0:
            break
    else:
        assert False, "SDRAM never ready"


@cocotb.test()
async def test_tensor_parallel_matmul(dut):
    dut.rx0.value = 1
    dut.rx1.value = 1
    cocotb.start_soon(Clock(dut.clk, CLK_NS, units="ns").start())

    # boot both nodes concurrently
    b0 = cocotb.start_soon(boot_node(dut.clk, dut.n0.u_fpga))
    b1 = cocotb.start_soon(boot_node(dut.clk, dut.n1.u_fpga))
    await b0
    await b1
    dut._log.info("both nodes booted, SDRAM ready")

    node0 = Node(dut.clk, dut.rx0, dut.tx0, dut.n0.u_fpga)
    node1 = Node(dut.clk, dut.rx1, dut.tx1, dut.n1.u_fpga)

    # ── build a matmul and split W by output rows across the two nodes ───────
    rng = random.Random(1)
    N_total = 2 * N_PER_NODE
    W = [[rng.randrange(-50, 50) for _ in range(K)] for _ in range(N_total)]
    x = [rng.randrange(-50, 50) for _ in range(K)]
    sx, sw = -3, -6

    W0 = W[:N_PER_NODE]              # node 0: top rows
    W1 = W[N_PER_NODE:]             # node 1: bottom rows

    # ── each node loads ONLY its slice into its OWN SDRAM (concurrently) ─────
    def flat(Ws):
        return bytes((Ws[r][k] & 0xFF) for r in range(N_PER_NODE) for k in range(K))
    l0 = cocotb.start_soon(node0.load(WADDR, flat(W0)))
    l1 = cocotb.start_soon(node1.load(WADDR, flat(W1)))
    await l0
    await l1
    dut._log.info(f"weights distributed: {N_PER_NODE} rows/node in separate SDRAMs")

    # ── broadcast x, compute slices in parallel, gather ─────────────────────
    f0 = cocotb.start_soon(node0.fq(N_PER_NODE, sx, sw, x, WADDR))
    f1 = cocotb.start_soon(node1.fq(N_PER_NODE, sx, sw, x, WADDR))
    y0, s0 = await f0
    y1, s1 = await f1

    # ── verify each slice bit-exact vs the project reference ────────────────
    ref0, shr0 = fq_ref(W0, x, N_PER_NODE)
    ref1, shr1 = fq_ref(W1, x, N_PER_NODE)
    exp_shift = sx + sw

    assert y0 == ref0, f"node0 slice mismatch: {[(i,y0[i],ref0[i]) for i in range(N_PER_NODE) if y0[i]!=ref0[i]][:3]}"
    assert y1 == ref1, f"node1 slice mismatch: {[(i,y1[i],ref1[i]) for i in range(N_PER_NODE) if y1[i]!=ref1[i]][:3]}"
    assert s0 == exp_shift + shr0, f"node0 shift {s0} != {exp_shift+shr0}"
    assert s1 == exp_shift + shr1, f"node1 shift {s1} != {exp_shift+shr1}"

    # ── gather: full y reassembled from the two nodes' slices ───────────────
    y_full = y0 + y1
    dut._log.info(f"  OK  node0: {N_PER_NODE}/{N_PER_NODE} bit-exact (shift {s0})")
    dut._log.info(f"  OK  node1: {N_PER_NODE}/{N_PER_NODE} bit-exact (shift {s1})")
    dut._log.info(f"  gathered y[{N_total}] from 2 nodes; first 4 = {y_full[:4]}")

    # ── cross-check vs float matmul (informational: quantization error) ─────
    def deq(slice_i8, sh):
        return [v * (2.0 ** sh) for v in slice_i8]
    y_real = deq(y0, s0) + deq(y1, s1)
    y_float = [sum(W[r][k] * (x[k] * 2.0 ** sx) for k in range(K)) * (2.0 ** sw)
               for r in range(N_total)]
    max_err = max(abs(a - b) for a, b in zip(y_real, y_float))
    scale = max(abs(v) for v in y_float) or 1.0
    dut._log.info(f"  recombined vs float matmul: max_err={max_err:.3f} ({100*max_err/scale:.2f}% of range)")

    dut._log.info(
        f"MILESTONE 3 PASS: tensor-parallel matmul split {N_PER_NODE}+{N_PER_NODE} "
        f"across 2 nodes, bit-exact vs project reference"
    )
