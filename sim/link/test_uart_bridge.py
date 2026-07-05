"""Phase-1 transport gate: UART-style adapters over the parallel ss_link.

Proves tx8_link/rx8_link move bytes in order across a CDC, honoring busy (sender
throttle) and the fifo full/empty handshake -- so the command/response FSMs can
keep their UART interface while the transport underneath becomes parallel.

  make TOPLEVEL=uart_bridge_tb MODULE=test_uart_bridge
"""

import random
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge


async def send_byte(dut, val):
    # wait until the sender adapter is free (mirror of "ignore send while busy")
    while int(dut.tx_busy.value) == 1:
        await RisingEdge(dut.clk_tx)
    dut.tx_byte.value = val & 0xFF
    dut.tx_send.value = 1
    await RisingEdge(dut.clk_tx)
    dut.tx_send.value = 0
    await RisingEdge(dut.clk_tx)


@cocotb.test()
async def test_uart_bridge(dut):
    # two independent clocks -> real CDC through the async fifo
    cocotb.start_soon(Clock(dut.clk_tx, 37, units="ns").start())
    cocotb.start_soon(Clock(dut.clk_rx, 53, units="ns").start())
    dut.rst.value = 1
    dut.tx_send.value = 0
    dut.tx_byte.value = 0
    for _ in range(10):
        await RisingEdge(dut.clk_tx)
    dut.rst.value = 0
    for _ in range(5):
        await RisingEdge(dut.clk_tx)

    rng = random.Random(7)
    payload = [rng.randrange(256) for _ in range(200)]
    received = []

    # collect on the receiver clock: every rx_valid pulse = one byte out
    async def collector():
        while True:
            await RisingEdge(dut.clk_rx)
            if int(dut.rx_valid.value) == 1:
                received.append(int(dut.rx_byte.value) & 0xFF)

    cocotb.start_soon(collector())

    for v in payload:
        await send_byte(dut, v)

    # let the tail drain through the CDC
    for _ in range(400):
        await RisingEdge(dut.clk_rx)

    assert len(received) == len(payload), \
        f"count mismatch: sent {len(payload)}, got {len(received)}"
    bad = [i for i in range(len(payload)) if received[i] != payload[i]]
    assert not bad, f"byte mismatch at {bad[:5]} (sent {payload[bad[0]]}, got {received[bad[0]]})"

    dut._log.info(
        f"PHASE-1 TRANSPORT GATE PASS: {len(payload)} bytes crossed the parallel "
        "ss_link in order via the UART-style adapters, across two clocks."
    )
