"""Full top.v integration: node boots its model from SD into SDRAM (SD-7).

The routable node (top.v with SD_BOOT) inits the SD card at power-on and loads
the model image SD -> SDRAM through its OWN (muxed) SDRAM controller, no PC. We
check the SDRAM ended up with the SD image byte-exact, and that boot_done
asserted (so the command FSM regains the SDRAM for inference).
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles

CLK_NS = 37
N = 2 * 512


def pat(i): return (i * 61 + 7) & 0xFF


@cocotb.test()
async def test_node_sd(dut):
    cocotb.start_soon(Clock(dut.clk, CLK_NS, units="ns").start())
    dut.uart_rx.value = 1
    for i in range(N):
        dut.u_card.mem[i].value = pat(i)

    # wait for the boot loader to finish (internal reset + SD init + load)
    ok = False
    for _ in range(400):
        await ClockCycles(dut.clk, 2000)
        try:
            if int(dut.u_fpga.boot_done.value) == 1:
                ok = True
                break
        except Exception:
            pass
    assert ok, "boot_done never asserted (SD->SDRAM boot failed)"

    await ClockCycles(dut.clk, 20)
    bad = []
    for a in range(N):
        word = int(dut.u_sdram.mem[a >> 2].value)
        byte = (word >> (8 * (a & 3))) & 0xFF
        if byte != pat(a):
            bad.append((a, byte, pat(a)))
            if len(bad) > 5:
                break
    assert not bad, f"SDRAM not loaded correctly from SD: {bad[:6]}"

    dut._log.info(
        "SD-7 PASS: top.v node booted its model from SD -> SDRAM (through its own "
        "muxed controller), byte-exact, boot_done asserted -> FSM ready for inference. "
        "Autonomous-from-SD integration validated."
    )
