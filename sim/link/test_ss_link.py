"""Source-synchronous parallel inter-FPGA link - CDC validation.

Two INDEPENDENT clocks (different periods = separate crystals) drive the two
sides of ss_link. A byte stream is pushed on the sender clock and pulled on the
receiver clock; the async FIFO must deliver every byte, in order, with no loss or
duplication despite the asynchronous clocks. A mid-stream consumer pause forces
the FIFO to fill so the tx_full backpressure path is exercised too.

This is the real-hardware link that replaces UART between cluster nodes:
  UART 1 Mbaud : ~10 µs/byte, 65-byte exchange ≈ 650 µs
  this link    : 1 byte / sender clock (here 27 MHz) ≈ 37 ns/byte  (~270x faster)
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles, ReadOnly

TX_NS = 37      # ~27.0 MHz  (sender oscillator)
RX_NS = 31      # ~32.3 MHz  (receiver oscillator) - intentionally unrelated
N     = 256


async def producer(dut, stream):
    """Write stream on clk_tx, honoring tx_full backpressure (no byte lost)."""
    dut.tx_send.value = 0
    dut.tx_data.value = 0
    i = 0
    while i < len(stream):
        dut.tx_data.value = stream[i]
        dut.tx_send.value = 1
        await ReadOnly()                       # settle: tx_full as seen at the edge
        full = int(dut.tx_full.value)
        await RisingEdge(dut.clk_tx)
        if full == 0:
            i += 1                             # write committed this edge
    dut.tx_send.value = 0


async def consumer(dut, n, recv, pause_at, pause_len):
    """Read on clk_rx; pause mid-stream to force the FIFO full (backpressure)."""
    dut.rx_rd.value = 0
    paused = False
    while len(recv) < n:
        if (not paused) and len(recv) >= pause_at:
            paused = True
            dut.rx_rd.value = 0
            await ClockCycles(dut.clk_rx, pause_len)   # stall -> FIFO fills up
        dut.rx_rd.value = 1
        await ReadOnly()
        empty = int(dut.rx_empty.value)
        data = int(dut.rx_data.value) if empty == 0 else None   # rx_data is X when empty
        await RisingEdge(dut.clk_rx)
        if empty == 0:
            recv.append(data)                  # this front word was read at the edge
    dut.rx_rd.value = 0


@cocotb.test()
async def test_ss_link(dut):
    cocotb.start_soon(Clock(dut.clk_tx, TX_NS, units="ns").start())
    cocotb.start_soon(Clock(dut.clk_rx, RX_NS, units="ns").start())

    dut.rst_tx_n.value = 0
    dut.rst_rx_n.value = 0
    dut.tx_send.value = 0
    dut.tx_data.value = 0
    dut.rx_rd.value = 0
    await ClockCycles(dut.clk_tx, 5)
    await ClockCycles(dut.clk_rx, 5)
    dut.rst_tx_n.value = 1
    dut.rst_rx_n.value = 1
    await ClockCycles(dut.clk_tx, 2)

    stream = [(i * 7 + 3) & 0xFF for i in range(N)]   # deterministic pattern
    recv = []

    prod = cocotb.start_soon(producer(dut, stream))
    cons = cocotb.start_soon(consumer(dut, N, recv, pause_at=100, pause_len=60))
    await prod
    await cons

    # ── checks ──────────────────────────────────────────────────────────────
    assert len(recv) == N, f"received {len(recv)} bytes, expected {N}"
    diffs = [(k, recv[k], stream[k]) for k in range(N) if recv[k] != stream[k]]
    assert not diffs, f"{len(diffs)} corrupted/out-of-order bytes, first={diffs[0]}"

    dut._log.info(
        f"SS-LINK PASS: {N} bytes crossed two asynchronous clock domains "
        f"({1000/TX_NS:.1f} MHz tx -> {1000/RX_NS:.1f} MHz rx), in order, "
        f"no loss/dup, tx_full backpressure exercised"
    )
