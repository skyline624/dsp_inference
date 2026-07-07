"""SD card read path (SPI controller + behavioral SD model), bit-exact.

Stores a known pattern on the (model) SD card, runs the SD controller to stream
NBLK blocks, and checks the captured bytes match exactly. First brick of the
memory hierarchy: weights live on the SD, stream out correctly.
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles

CLK_NS = 20
NBLK = 2
N = NBLK * 512


def pat(i): return (i * 73 + 19) & 0xFF


@cocotb.test()
async def test_sd_stream(dut):
    cocotb.start_soon(Clock(dut.clk, CLK_NS, units="ns").start())
    dut.rstn.value = 0
    dut.start.value = 0

    # preload the SD card model with a known pattern
    for i in range(N):
        dut.u_sd.mem[i].value = pat(i)

    await ClockCycles(dut.clk, 10)
    dut.rstn.value = 1
    await ClockCycles(dut.clk, 5)

    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    for _ in range(200000):
        await RisingEdge(dut.clk)
        if int(dut.done.value) == 1:
            break
    else:
        assert False, "SD read never finished"

    await ClockCycles(dut.clk, 2)
    bad = []
    for i in range(N):
        got = int(dut.cap[i].value)
        if got != pat(i):
            bad.append((i, got, pat(i)))
            if len(bad) > 5:
                break
    assert not bad, f"SD stream mismatch: {bad[:6]}"

    dut._log.info(
        f"SD READ PATH PASS: {NBLK} blocks ({N} bytes) streamed from the SD card "
        "model via the SPI controller, bit-exact. First brick of the SD->SDRAM hierarchy."
    )
