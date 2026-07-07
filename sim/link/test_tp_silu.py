"""Autonomous chain: real-DSP matmul -> real silu_op (FFN building block).

A node autonomously computes h1 = W@x (mac18 + fq_ref requant) then runs the REAL
silu_op on it, with correct shift bookkeeping. Validated bit-exact vs the project
references (fq_ref then silu_ref). Proves a real elementwise operator integrates
into the autonomous FSM -- the missing piece for assembling the full FFN.
"""

import random
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles

CLK_NS = 37
W = 8
K = 64
NR = 32


def to_i8(b): return b - 256 if b >= 128 else b


def _load_silu_lut():
    for path in ("silu_lut.hex", "../../src/silu_lut.hex"):
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


def fq_ref(Wrows, x):
    y = [sum(Wrows[r][k] * x[k] for k in range(K)) for r in range(len(Wrows))]
    ma = max(abs(v) for v in y)
    if ma == 0:
        return [0] * len(Wrows), 0
    sh = max(0, (ma.bit_length() - 1) - 6)
    rnd = (1 << (sh - 1)) if sh > 0 else 0
    out = [max(-128, min(127, ((v + rnd) >> sh) if sh > 0 else v)) for v in y]
    return out, sh


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
async def test_tp_silu(dut):
    cocotb.start_soon(Clock(dut.clk, CLK_NS, units="ns").start())
    dut.rst_n.value = 0
    dut.start.value = 0
    dut.x.value = 0
    dut.sx.value = 0
    dut.sw.value = 0
    await ClockCycles(dut.clk, 5)

    rng = random.Random(9)
    Wm = [[rng.randrange(-50, 50) for _ in range(K)] for _ in range(NR)]
    x = [rng.randrange(-50, 50) for _ in range(K)]
    sx, sw = -3, -6

    for r in range(NR):
        for k in range(K):
            dut.wmem[r * K + k].value = Wm[r][k] & 0xFF

    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)
    dut.x.value = vec_to_int([v & 0xFF for v in x])
    dut.sx.value = sx & 0xFF
    dut.sw.value = sw & 0xFF
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    for _ in range(60000):
        await RisingEdge(dut.clk)
        if int(dut.done.value) == 1:
            break
    else:
        assert False, "no result"

    h1, sh = fq_ref(Wm, x)
    sh1 = sx + sw + sh
    expected = silu_ref(h1, sh1)

    ov = int(dut.out.value)
    got = [to_i8((ov >> (i * 8)) & 0xFF) for i in range(NR)]
    bad = [i for i in range(NR) if got[i] != expected[i]]
    assert not bad, f"silu chain mismatch at {bad[0]} ({got[bad[0]]} != {expected[bad[0]]})"
    assert to_i8(int(dut.out_shift.value)) == sh1, "shift_out mismatch"

    dut._log.info(
        f"FFN BUILDING-BLOCK PASS: autonomous matmul(mac18) -> real silu_op chain, "
        f"correct shift ({sh1}), bit-exact vs fq_ref+silu_ref. Real operator integrated."
    )
