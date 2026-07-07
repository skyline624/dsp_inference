"""Model BIGGER than SDRAM, run by streaming layers from SD (SD-4).

L layers (each W_l = [8,64]) live on the SD card. The SDRAM holds only ONE layer
at a time (512 bytes, reused). For each layer the weights stream SD->SDRAM, then
y_l = W_l @ x is computed reading W_l back from SDRAM. Validated bit-exact.
Model footprint on SD = L*512 bytes >> the 512-byte SDRAM cache region.
"""

import random
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles

CLK_NS = 20
L = 4
M = 8
K = 64


def s32(v): return v - (1 << 32) if v >= (1 << 31) else v


@cocotb.test()
async def test_sd_model(dut):
    cocotb.start_soon(Clock(dut.clk, CLK_NS, units="ns").start())
    dut.rstn.value = 0
    dut.start.value = 0
    dut.x_in.value = 0

    rng = random.Random(9)
    W = [[[rng.randrange(-128, 128) for _ in range(K)] for _ in range(M)] for _ in range(L)]
    x = [rng.randrange(-128, 128) for _ in range(K)]

    # store all L layers on the SD card (block l = layer l, row-major)
    for l in range(L):
        for r in range(M):
            for k in range(K):
                dut.u_sd.mem[l*512 + r*K + k].value = W[l][r][k] & 0xFF

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

    for _ in range(800000):
        await RisingEdge(dut.clk)
        if int(dut.done.value) == 1:
            break
    else:
        assert False, "model stream never finished"

    await ClockCycles(dut.clk, 2)
    bad = []
    for l in range(L):
        for r in range(M):
            ref = sum(W[l][r][k] * x[k] for k in range(K))
            got = s32(int(dut.yout[l*M + r].value))
            if got != ref:
                bad.append((l, r, got, ref))
                if len(bad) > 5:
                    break
    assert not bad, f"layer-streamed matmul mismatch: {bad[:6]}"

    dut._log.info(
        f"SD-4 PASS: {L}-layer model ({L*512} B on SD) run through a single 512 B "
        f"SDRAM cache region, layer by layer; all {L*M} outputs bit-exact. A model "
        "bigger than the fast memory runs by streaming weights per layer."
    )
