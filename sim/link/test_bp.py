"""almost_full robustness under inter-chip round-trip backpressure delay (task 2).

A free-running writer is throttled by a backpressure signal delayed by DLY cycles
(modeling the sender<-receiver pin round-trip). A slow reader keeps the FIFO
pressured. Plain `full` loses bytes during the blind window; `almost_full`
(AFMARGIN > DLY) is lossless.
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles

W_NS = 37     # writer/forwarded clock
R_NS = 31     # reader clock (independent)


async def slow_reader(dut, every):
    """Consume one word every `every` reader cycles -> keeps the FIFO near full."""
    cnt = 0
    while True:
        dut.rd_strobe.value = 1 if (cnt % every == 0) else 0
        await RisingEdge(dut.rclk)
        cnt += 1


@cocotb.test()
async def test_bp(dut):
    cocotb.start_soon(Clock(dut.wclk, W_NS, units="ns").start())
    cocotb.start_soon(Clock(dut.rclk, R_NS, units="ns").start())
    dut.wrst_n.value = 0
    dut.rrst_n.value = 0
    dut.rd_strobe.value = 0
    await ClockCycles(dut.wclk, 5)
    await ClockCycles(dut.rclk, 5)
    dut.wrst_n.value = 1
    dut.rrst_n.value = 1

    cocotb.start_soon(slow_reader(dut, every=4))   # reader much slower than writer
    await ClockCycles(dut.wclk, 3000)

    plain_lost = int(dut.plain_lost.value)
    afull_lost = int(dut.afull_lost.value)
    plain_wrote = int(dut.plain_wrote.value)
    afull_wrote = int(dut.afull_wrote.value)

    dut._log.info(f"plain full     : wrote={plain_wrote}  lost={plain_lost}")
    dut._log.info(f"almost_full    : wrote={afull_wrote}  lost={afull_lost}")

    assert plain_lost > 0, "expected plain `full` to lose bytes under delayed backpressure"
    assert afull_lost == 0, f"almost_full must be lossless, lost={afull_lost}"

    dut._log.info(
        f"TASK 2 PASS: almost_full prevents the {plain_lost} byte losses that plain "
        f"`full` suffers under a {3}-cycle round-trip backpressure delay"
    )
