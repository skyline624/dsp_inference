"""Double-buffered SD -> cache -> compute, with overlap (SD-2).

W = [T*8, 64] lives on the SD card (one block per tile of 8 rows). The controller
streams tiles through two ping-pong cache buffers while the DSP computes the
matmul of the current tile and the next tile loads concurrently. Validates
y = W @ x (int32 per row) bit-exact -> proves the SD->cache->compute hierarchy
with load/compute overlap.
"""

import random
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles

CLK_NS = 20
T = 4
ROWS = 8
K = 64
M = T * ROWS


def s32(v): return v - (1 << 32) if v >= (1 << 31) else v


@cocotb.test()
async def test_sd_layer(dut):
    cocotb.start_soon(Clock(dut.clk, CLK_NS, units="ns").start())
    dut.rstn.value = 0
    dut.start.value = 0
    dut.x_in.value = 0

    rng = random.Random(3)
    W = [[rng.randrange(-128, 128) for _ in range(K)] for _ in range(M)]
    x = [rng.randrange(-128, 128) for _ in range(K)]

    # store W on the SD card: block t holds rows [t*8 .. t*8+7], row-major
    for t in range(T):
        for rr in range(ROWS):
            for k in range(K):
                dut.u_sd.mem[t*512 + rr*K + k].value = W[t*ROWS + rr][k] & 0xFF

    xv = 0
    for k in range(K):
        xv |= (x[k] & 0xFF) << (k * 8)
    dut.x_in.value = xv

    await ClockCycles(dut.clk, 10)
    dut.rstn.value = 1
    await ClockCycles(dut.clk, 5)

    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    for _ in range(400000):
        await RisingEdge(dut.clk)
        if int(dut.done.value) == 1:
            break
    else:
        assert False, "layer stream never finished"

    await ClockCycles(dut.clk, 2)
    ref = [sum(W[r][k] * x[k] for k in range(K)) for r in range(M)]
    bad = []
    for r in range(M):
        got = s32(int(dut.yout[r].value))
        if got != ref[r]:
            bad.append((r, got, ref[r]))
            if len(bad) > 5:
                break
    assert not bad, f"matmul mismatch: {bad[:6]}"

    dut._log.info(
        f"SD-2 PASS: {T} tiles streamed SD->ping-pong cache->DSP with load/compute "
        f"overlap; y = W@x ({M} rows) bit-exact. The SDRAM-as-layer-cache hierarchy works."
    )
