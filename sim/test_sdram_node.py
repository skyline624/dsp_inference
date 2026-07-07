"""Milestone 2b - validate the two hand-written sim models against the project.

SiLU (test_silu_node.py) exercised neither the SDRAM nor the DSP. This test
covers exactly those - the two models I wrote by hand:

  1. SDRAM round-trip  : LL (write) + CC (read back)        -> sdram_chip.v
  2. FQ matmul         : LL (load W) + FQ (W@x from SDRAM)  -> sdram_chip.v read
                                                              + MULTALU18X18.v (DSP)

The FQ reference is the project's own (host/test_sdram_diag.py, test 3):
integer matmul + power-of-2 requantize. A bit-exact match proves the SDRAM
read/write path and the DSP MAC behave like the real hardware.
"""

import random
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles

CLK_NS = 37
DIV    = 27
K      = 64


# ─── byte framing (mirrors host/test_sdram_diag.py) ─────────────────────────
def addr_bytes(a): return bytes([a & 0xFF, (a >> 8) & 0xFF, (a >> 16) & 0xFF])
def n_bytes(n):    return bytes([n & 0xFF, (n >> 8) & 0xFF])
def i8(b):         return b - 256 if b >= 128 else b


# ─── project reference for FQ (test_sdram_diag.py, test 3) ──────────────────
def fq_ref(W, x, N):
    y = [sum(W[r][k] * x[k] for k in range(K)) for r in range(N)]
    max_abs = max(abs(v) for v in y)
    if max_abs == 0:
        return [0] * N
    sh = max(0, (max_abs.bit_length() - 1) - 6)
    rnd = (1 << (sh - 1)) if sh > 0 else 0
    out = []
    for v in y:
        o = (v + rnd) >> sh if sh > 0 else v
        out.append(max(-128, min(127, o)))
    return out


# ─── UART helpers ───────────────────────────────────────────────────────────
def _as_int(sig):
    try:
        return int(sig.value)
    except Exception:
        return None


async def uart_send(dut, data):
    for byte in data:
        dut.head_rx.value = 0
        await ClockCycles(dut.clk, DIV)
        for i in range(8):
            dut.head_rx.value = (byte >> i) & 1
            await ClockCycles(dut.clk, DIV)
        dut.head_rx.value = 1
        await ClockCycles(dut.clk, DIV)


async def uart_rx_monitor(dut, sink):
    while True:
        while _as_int(dut.tail_tx) != 0:
            await RisingEdge(dut.clk)
        await ClockCycles(dut.clk, DIV // 2)
        byte = 0
        for i in range(8):
            await ClockCycles(dut.clk, DIV)
            byte |= (_as_int(dut.tail_tx) or 0) << i
        await ClockCycles(dut.clk, DIV)
        sink.append(byte)


class Link:
    """Sequential request/response over the simulated UART (like pyserial)."""
    def __init__(self, dut, sink):
        self.dut, self.sink, self.off = dut, sink, 0

    async def xfer(self, pkt, nresp, timeout_cycles=2_000_000):
        await uart_send(self.dut, pkt)
        waited = 0
        while len(self.sink) < self.off + nresp:
            await ClockCycles(self.dut.clk, 20)
            waited += 20
            assert waited < timeout_cycles, (
                f"timeout: {len(self.sink) - self.off}/{nresp} resp bytes"
            )
        resp = bytes(self.sink[self.off:self.off + nresp])
        self.off += nresp
        return resp


async def boot(dut):
    dut.head_rx.value = 1
    cocotb.start_soon(Clock(dut.clk, CLK_NS, units="ns").start())
    fpga = dut.nodes[0].u_node.u_fpga
    for _ in range(40000):
        await RisingEdge(dut.clk)
        if _as_int(fpga.rst_n) == 1:
            break
    else:
        assert False, "reset never released"

    # Wait for the SDRAM controller to finish its 200 us power-on config
    # (busy -> 0). It only starts that countdown AFTER rst_n, so there is a
    # window where the FSM is live but SDRAM is not ready. On real hardware the
    # host always finds SDRAM ready (ms elapse before the first command); in sim
    # we send commands within us of reset, so we must wait for it explicitly.
    for _ in range(40000):
        await RisingEdge(dut.clk)
        if _as_int(fpga.sd_busy) == 0:
            break
    else:
        assert False, "SDRAM controller never became ready (sd_busy stuck high)"
    await ClockCycles(dut.clk, 50)
    dut._log.info("node booted, SDRAM ready (sd_busy=0)")


@cocotb.test()
async def test_sdram_and_dsp(dut):
    await boot(dut)
    sink = bytearray()
    cocotb.start_soon(uart_rx_monitor(dut, sink))
    link = Link(dut, sink)
    rng = random.Random(0)

    # ── 1. SDRAM round-trip: LL write + CC read back ────────────────────────
    addr = 0x100000
    data = bytes(rng.randrange(256) for _ in range(64))
    ack = await link.xfer(b"LL" + addr_bytes(addr) + n_bytes(len(data)) + data, 2)
    assert ack == b"LK", f"LL ack {ack!r}"
    dump = await link.xfer(b"CC" + addr_bytes(addr) + n_bytes(64), 2 + 64)
    assert dump[:2] == b"CK", f"CC magic {dump[:2]!r}"
    got = dump[2:]
    assert got == data, (
        f"SDRAM round-trip mismatch: {sum(a != b for a, b in zip(got, data))}/64 bytes differ"
    )
    dut._log.info("  OK  SDRAM round-trip (LL+CC) -> 64/64 bytes match")

    # ── 2. FQ matmul: load W to SDRAM, compute W@x, check vs project ref ─────
    N = 8
    W = [[rng.randrange(-50, 50) for _ in range(K)] for _ in range(N)]
    x = [rng.randrange(-50, 50) for _ in range(K)]
    waddr = 0x100400
    W_flat = bytes((W[r][k] & 0xFF) for r in range(N) for k in range(K))
    ack = await link.xfer(b"LL" + addr_bytes(waddr) + n_bytes(len(W_flat)) + W_flat, 2)
    assert ack == b"LK", f"LL(W) ack {ack!r}"

    x_bytes = bytes((v & 0xFF) for v in x)
    sx, sw = -3, -6
    pkt = b"FQ" + bytes([N, sx & 0xFF, sw & 0xFF]) + x_bytes + addr_bytes(waddr)
    resp = await link.xfer(pkt, 3 + N)
    assert resp[:2] == b"FQ", f"FQ magic {resp[:2]!r}"
    out = [i8(resp[3 + i]) for i in range(N)]
    ref = fq_ref(W, x, N)
    mism = [(i, out[i], ref[i]) for i in range(N) if out[i] != ref[i]]
    assert not mism, f"FQ matmul mismatch vs project ref: {mism}"
    dut._log.info(f"  OK  FQ matmul (SDRAM read + DSP) -> {N}/{N} int8 bit-exact")

    dut._log.info("MILESTONE 2b PASS: SDRAM model and DSP model match the project")
