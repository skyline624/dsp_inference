"""On-chip sequencer chains 2 ops with handoff (mini-GG increment 2).

The seq issues SS, then feeds that output back as the input to a second SS, all
on-chip (the output of op1 becomes the input of op2 via the internal `curx`
buffer -- the FFN handoff). Validated bit-exact vs silu applied twice. Build with
-Pseq_node.N_OPS=2.
"""

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
async def test_seq2(dut):
    cocotb.start_soon(Clock(dut.clk, CLK_NS, units="ns").start())
    dut.rst_n.value = 0
    dut.start.value = 0
    dut.x_in.value = 0
    dut.sx.value = 0
    await ClockCycles(dut.clk, 10)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 5)

    x = list(range(-32, 32))
    sx = -4
    dut.x_in.value = vec_to_int([v & 0xFF for v in x])
    dut.sx.value = sx & 0xFF

    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    for _ in range(300000):
        await RisingEdge(dut.clk)
        if int(dut.done.value) == 1:
            break
    else:
        assert False, "sequencer never finished"

    # reference: silu applied twice (silu_op preserves the shift -> 2nd pass uses sx)
    out1 = silu_ref(x, sx)
    expected = silu_ref(out1, sx)

    rv = int(dut.result.value)
    got = [to_i8((rv >> (i * 8)) & 0xFF) for i in range(D)]
    bad = [i for i in range(D) if got[i] != expected[i]]
    assert not bad, f"2-op chain wrong at {bad[0]} ({got[bad[0]]} != {expected[bad[0]]})"

    dut._log.info(
        "MINI-GG SEQUENCER (inc.2) PASS: on-chip FSM chained SS -> SS with handoff "
        "(op1 output became op2 input internally), bit-exact vs double-silu, zero PC."
    )
