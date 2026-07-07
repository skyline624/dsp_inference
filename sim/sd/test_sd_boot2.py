"""sd_boot : autonomous model loader SD -> real SDRAM, multi-block, with init.

Validates the integration-ready bootloader: it inits the card once, streams
NBLK blocks, and writes them into the REAL SDRAM (controller + chip). Checks the
SDRAM content byte-exact. This is the loader that gets wired into top.v.
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles

CLK_NS = 20
NBLK = 2
N = NBLK * 512


def pat(i): return (i * 37 + 5) & 0xFF


@cocotb.test()
async def test_sd_boot2(dut):
    cocotb.start_soon(Clock(dut.clk, CLK_NS, units="ns").start())
    dut.rstn.value = 0
    dut.start.value = 0
    for i in range(N):
        dut.u_card.mem[i].value = pat(i)

    await ClockCycles(dut.clk, 10)
    dut.rstn.value = 1
    await ClockCycles(dut.clk, 5)
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    for _ in range(500000):
        await RisingEdge(dut.clk)
        if int(dut.done.value) == 1:
            break
    else:
        assert False, "sd_boot never finished"

    await ClockCycles(dut.clk, 4)
    bad = []
    for a in range(N):
        word = int(dut.u_chip.mem[a >> 2].value)
        byte = (word >> (8 * (a & 3))) & 0xFF
        if byte != pat(a):
            bad.append((a, byte, pat(a)))
            if len(bad) > 5:
                break
    assert not bad, f"SDRAM content wrong after boot-load: {bad[:6]}"

    dut._log.info(
        f"SD-6 PASS: sd_boot inited the card once + streamed {NBLK} blocks ({N} B) "
        "into the real SDRAM, byte-exact. Integration-ready loader for top.v."
    )
