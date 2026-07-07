"""Validate the standalone SD bring-up harness (src/sd_bringup.v) in sim.

Drives the real harness against the SD card model: checks the harness runs its
power-on reset, inits the card, reads block 0, and captures it correctly (the
part that then gets streamed over UART + shown on the LEDs). Uses a FAT-like
block (0x55 0xAA at offset 510) so it mirrors a real card's boot sector.
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles

CLK_NS = 37   # 27 MHz


def blk(i):
    if i == 510: return 0x55
    if i == 511: return 0xAA
    return (i * 91 + 13) & 0xFF


@cocotb.test()
async def test_sd_bringup(dut):
    cocotb.start_soon(Clock(dut.clk, CLK_NS, units="ns").start())
    for i in range(512):
        dut.u_card.mem[i].value = blk(i)

    # wait through power-on reset + init + block read (slow SPI, HALF=34)
    ok = False
    for _ in range(320):
        await ClockCycles(dut.clk, 2500)
        if int(dut.u_dut.status.value) == 0x01:
            ok = True
            break
    assert ok, "harness never reached status=OK (init/read failed in sim)"

    await ClockCycles(dut.clk, 50)
    bad = [(i, int(dut.u_dut.cap[i].value), blk(i)) for i in range(512)
           if int(dut.u_dut.cap[i].value) != blk(i)]
    assert not bad, f"captured block wrong: {bad[:6]}"

    assert int(dut.u_dut.ready_l.value) == 1, "ready flag not set (init did not complete)"

    dut._log.info(
        "SD BRING-UP HARNESS PASS: power-on reset -> SD init -> read block 0 -> "
        "captured 512 bytes bit-exact (incl. 0x55AA sig). Ready to route + flash."
    )
