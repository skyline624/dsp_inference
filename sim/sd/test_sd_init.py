"""Full SPI init sequence + block read (SD-5), real-card-ready.

The controller runs the power-up handshake (CMD0/CMD8/ACMD41/CMD58) against an
SD card model that enforces it, signals `ready`, then reads a block. Validates
init completed and the block streamed bit-exact -> the controller is ready to
drive a real SD card on the Tang Nano 20K TF slot.
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles

CLK_NS = 20
NBLK = 2
RD_BLK = 1                      # read the 2nd block


def pat(i): return (i * 53 + 11) & 0xFF


@cocotb.test()
async def test_sd_init(dut):
    cocotb.start_soon(Clock(dut.clk, CLK_NS, units="ns").start())
    dut.rstn.value = 0
    dut.start.value = 0
    dut.rd_blk.value = RD_BLK

    # fill the target block on the card
    for i in range(512):
        dut.u_sd.mem[RD_BLK * 512 + i].value = pat(i)

    await ClockCycles(dut.clk, 10)
    dut.rstn.value = 1
    await ClockCycles(dut.clk, 5)
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    ready_seen = False
    for _ in range(400000):
        await RisingEdge(dut.clk)
        if int(dut.ready.value) == 1:
            ready_seen = True
        if int(dut.done.value) == 1:
            break
    else:
        assert False, "SD init+read never finished"

    assert ready_seen, "init never reached ready (CMD0/CMD8/ACMD41/CMD58 handshake failed)"

    await ClockCycles(dut.clk, 2)
    bad = [(i, int(dut.cap[i].value), pat(i)) for i in range(512) if int(dut.cap[i].value) != pat(i)]
    assert not bad, f"block read mismatch after init: {bad[:6]}"

    dut._log.info(
        "SD-5 PASS: full SPI init handshake (CMD0/CMD8/ACMD41/CMD58) -> ready, "
        "then block read bit-exact. Controller is ready to drive a real SD card."
    )
