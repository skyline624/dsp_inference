"""On-chip sequencer drives the brick datapath (mini-GG increment 1).

The seq host issues an SS (silu) command to the unmodified node over its own UART
command interface and captures the result -- no PC. Validated bit-exact vs the
project silu_ref. Proves the on-chip sequencer <-> brick mechanism end to end.
"""

import math
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles

CLK_NS = 37
D = 64


def to_i8(b): return b - 256 if b >= 128 else b


def _load_silu_lut():
    for path in ("silu_lut.hex", "../src/silu_lut.hex"):
        try:
            with open(path) as f:
                vals = []
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("//"):
                        v = int(line, 16)
                        vals.append(v - 0x10000 if v >= 0x8000 else v)
                return vals[:256]
        except OSError:
            continue
    raise FileNotFoundError("silu_lut.hex")


SILU_LUT = _load_silu_lut()


def silu_ref(x_i8, sx):
    out = []
    shx_p4 = sx + 4
    osh = 11 + sx
    for xs in x_i8:
        x16 = (xs << shx_p4) if shx_p4 >= 0 else (xs >> (-shx_p4))
        idx = max(0, min(255, x16 + 128))
        sv = SILU_LUT[idx]
        o = ((sv + (1 << (osh - 1))) >> osh) if osh > 0 else (sv << (-osh)) if osh < 0 else sv
        out.append(max(-128, min(127, o)))
    return out


def vec_to_int(bs):
    v = 0
    for i, b in enumerate(bs):
        v |= (b & 0xFF) << (i * 8)
    return v


@cocotb.test()
async def test_seq(dut):
    cocotb.start_soon(Clock(dut.clk, CLK_NS, units="ns").start())
    dut.rst_n.value = 0
    dut.start.value = 0
    dut.x_in.value = 0
    dut.sx.value = 0
    await ClockCycles(dut.clk, 10)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 5)

    x = list(range(-32, 32))     # ramp through silu's interesting region
    sx = -4
    dut.x_in.value = vec_to_int([v & 0xFF for v in x])
    dut.sx.value = sx & 0xFF

    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    for _ in range(200000):       # boot wait + on-chip UART round trip
        await RisingEdge(dut.clk)
        if int(dut.done.value) == 1:
            break
    else:
        assert False, "sequencer never finished"

    rv = int(dut.result.value)
    got = [to_i8((rv >> (i * 8)) & 0xFF) for i in range(D)]
    ref = silu_ref(x, sx)
    bad = [i for i in range(D) if got[i] != ref[i]]
    assert not bad, f"on-chip seq result wrong at {bad[0]} ({got[bad[0]]} != {ref[bad[0]]})"

    dut._log.info(
        "MINI-GG SEQUENCER (inc.1) PASS: on-chip FSM issued an SS command to the "
        "unmodified brick node over its UART interface, captured the result, "
        "bit-exact vs silu_ref -- zero PC."
    )
