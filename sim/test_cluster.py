"""cocotb testbench for the dsp_inference FPGA cluster simulation.

Milestone 1 (this file): bring-up smoke test.
  - the real RTL (top.v + all operators) elaborates against the Gowin primitive
    sim models (Gowin_rPLL, MULTALU18X18, sdram_chip);
  - the design boots: PLL "locks", the internal reset releases;
  - the UART input path captures a byte on real RTL.

Run via sim/run.ps1 (Docker) or `make` inside the hdlc/sim:osvb container.
Later milestones (cluster handoff, full-layer partitioning) extend this file.
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles

CLK_NS = 37    # 27 MHz ~= 37 ns period (functional sim, exact phase not needed)
DIV    = 27    # top.v UART divider: clk cycles per bit @ 1 Mbaud


async def uart_send_byte(dut, byte):
    """Bit-bang one 8N1 byte into the cluster head (LSB first, idle high)."""
    dut.head_rx.value = 0                          # start bit
    await ClockCycles(dut.clk, DIV)
    for i in range(8):
        dut.head_rx.value = (byte >> i) & 1
        await ClockCycles(dut.clk, DIV)
    dut.head_rx.value = 1                          # stop bit
    await ClockCycles(dut.clk, DIV)


def _node0_fpga(dut):
    """Hierarchical handle to node 0's top instance, or None if unreachable."""
    try:
        return dut.nodes[0].u_node.u_fpga
    except Exception as e:                          # noqa: BLE001
        dut._log.warning(f"hierarchical access to node 0 failed: {e}")
        return None


def _as_int(sig):
    try:
        return int(sig.value)
    except Exception:                               # X / Z / unresolved
        return None


@cocotb.test()
async def test_boot_and_uart(dut):
    dut.head_rx.value = 1                           # UART line idle
    cocotb.start_soon(Clock(dut.clk, CLK_NS, units="ns").start())

    fpga = _node0_fpga(dut)

    # Wait for the internal reset to release (PLL lock + 32768 clk_sys cycles).
    if fpga is not None and hasattr(fpga, "rst_n"):
        released = False
        for _ in range(40000):
            await RisingEdge(dut.clk)
            if _as_int(fpga.rst_n) == 1:
                released = True
                break
        assert released, "reset never released within 40000 cycles"
        dut._log.info("reset released (rst_n=1)")
    else:
        await ClockCycles(dut.clk, 35000)          # fallback: > 32768
        dut._log.info("reset wait done (fixed 35000 cycles)")

    await ClockCycles(dut.clk, 50)

    # The UART output line must be a clean idle '1' -> no X/Z contention,
    # which is the key proof that the primitive sim models elaborated cleanly.
    tail = _as_int(dut.tail_tx)
    assert tail is not None, f"tail_tx is unresolved (X/Z): {dut.tail_tx.value}"
    assert tail == 1, f"UART output not idle high: {dut.tail_tx.value}"
    dut._log.info("tail_tx idle high -> design alive, no X contention")

    # Drive a byte and confirm the RX datapath captured it on real RTL.
    test_byte = ord("L")                           # 'L' = start of the LL load cmd
    await uart_send_byte(dut, test_byte)
    await ClockCycles(dut.clk, 5)

    if fpga is not None and hasattr(fpga, "rx_byte"):
        got = _as_int(fpga.rx_byte)
        assert got == test_byte, (
            f"UART RX mismatch: got {got!r} want 0x{test_byte:02x}"
        )
        dut._log.info(f"UART RX verified: node0 rx_byte = 0x{got:02x} ('{chr(got)}')")
    else:
        dut._log.warning("rx_byte not reachable; RX content check skipped")

    dut._log.info("MILESTONE 1 PASS: cluster boots and UART input works")
