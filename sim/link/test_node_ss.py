"""Phase-1 node-side gate: top.v (LINK_SS) driven over the parallel link.

Exercises the command RX path and the response TX path of the real node FSM with
its serial UART PHY swapped for the fifo adapters. Uses the simplest round-trip
that touches both directions: WW (write a byte to SDRAM) then BB (read it back).
Bit-exact by construction -> isolates the transport swap from any compute.

  make TOPLEVEL=node_ss_top MODULE=test_node_ss
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge


async def push(dut, byte):
    while int(dut.cmd_full.value) == 1:
        await RisingEdge(dut.clk)
    dut.cmd_data.value = byte & 0xFF
    dut.cmd_wr.value = 1
    await RisingEdge(dut.clk)
    dut.cmd_wr.value = 0
    await RisingEdge(dut.clk)


async def pull(dut, timeout=200000):
    for _ in range(timeout):
        await RisingEdge(dut.clk)
        if int(dut.resp_empty.value) == 0:
            b = int(dut.resp_data.value) & 0xFF
            dut.resp_rd.value = 1
            await RisingEdge(dut.clk)
            dut.resp_rd.value = 0
            await RisingEdge(dut.clk)
            return b
    assert False, "no response byte (timeout)"


async def ww(dut, addr, data):
    # 'W' 'W' addr[0] addr[1] addr[2] data  -> 'W' 'K'
    for b in (ord('W'), ord('W'), addr & 0xFF, (addr >> 8) & 0xFF, (addr >> 16) & 0x7F, data & 0xFF):
        await push(dut, b)
    ack = [await pull(dut), await pull(dut)]
    assert ack == [ord('W'), ord('K')], f"WW ack {ack!r}"


async def bb(dut, addr):
    # 'B' 'B' addr[0] addr[1] addr[2]  -> 'B' 'K' data
    for b in (ord('B'), ord('B'), addr & 0xFF, (addr >> 8) & 0xFF, (addr >> 16) & 0x7F):
        await push(dut, b)
    hdr = [await pull(dut), await pull(dut)]
    assert hdr == [ord('B'), ord('K')], f"BB hdr {hdr!r}"
    return await pull(dut)


@cocotb.test()
async def test_node_ss(dut):
    cocotb.start_soon(Clock(dut.clk, 37, units="ns").start())
    dut.rst.value = 1
    dut.cmd_wr.value = 0
    dut.cmd_data.value = 0
    dut.resp_rd.value = 0
    # pulse reset so the testbench-side fifo Gray pointers leave X
    for _ in range(10):
        await RisingEdge(dut.clk)
    dut.rst.value = 0
    # let the node come out of its internal reset (PLL lock + init counter)
    for _ in range(40000):
        await RisingEdge(dut.clk)

    # write a handful of distinct bytes at distinct addresses, then read them back
    testset = [(0x000010, 0x5A), (0x000011, 0xA5), (0x000200, 0x3C), (0x001000, 0xFF)]
    for addr, val in testset:
        await ww(dut, addr, val)
    for addr, val in testset:
        got = await bb(dut, addr)
        assert got == val, f"addr {addr:#x}: wrote {val:#x}, read {got:#x}"

    dut._log.info(
        "PHASE-1 NODE GATE PASS: top.v speaks the WW/BB protocol over the parallel "
        "link exactly as over UART -- serial PHY -> fifo adapters, node FSM unchanged."
    )
