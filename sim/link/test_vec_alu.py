"""vec_alu : the sequencer's multiply/residual ALU (FFN glue).

Validates both ops bit-exact vs a Python reference using the same integer
requantize (round-half-up) as the brick datapath:
  MUL : out = a*b   (multiply done on the real DSP / mac18)
  ADD : out = a+b   (shift-aligned residual)
"""

import random
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles

CLK_NS = 37
D = 64


def to_i8(b): return b - 256 if b >= 128 else b


def requant(vals):
    ma = max((abs(v) for v in vals), default=0)
    sh = max(0, (ma.bit_length() - 1) - 6) if ma else 0
    rnd = (1 << (sh - 1)) if sh > 0 else 0
    out = [max(-128, min(127, ((v + rnd) >> sh) if sh > 0 else v)) for v in vals]
    return out, sh


def vmul(a, sa, b, sb):
    prod = [a[i] * b[i] for i in range(D)]
    out, sh = requant(prod)
    return out, sa + sb + sh


def vadd(a, sa, b, sb):
    smin = min(sa, sb)
    s = [(a[i] << (sa - smin)) + (b[i] << (sb - smin)) for i in range(D)]
    out, sh = requant(s)
    return out, smin + sh


def vec_to_int(bs):
    v = 0
    for i, x in enumerate(bs):
        v |= (x & 0xFF) << (i * 8)
    return v


async def run_op(dut, op, a, sa, b, sb):
    dut.op.value = op
    dut.a.value = vec_to_int([v & 0xFF for v in a]); dut.sa.value = sa & 0xFF
    dut.b.value = vec_to_int([v & 0xFF for v in b]); dut.sb.value = sb & 0xFF
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0
    for _ in range(5000):
        await RisingEdge(dut.clk)
        if int(dut.done.value) == 1:
            break
    else:
        assert False, "vec_alu timeout"
    ov = int(dut.out.value)
    got = [to_i8((ov >> (i * 8)) & 0xFF) for i in range(D)]
    return got, to_i8(int(dut.out_sh.value))


@cocotb.test()
async def test_vec_alu(dut):
    cocotb.start_soon(Clock(dut.clk, CLK_NS, units="ns").start())
    dut.rst_n.value = 0
    dut.start.value = 0
    dut.op.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)

    rng = random.Random(12)
    a = [rng.randrange(-128, 128) for _ in range(D)]
    b = [rng.randrange(-128, 128) for _ in range(D)]
    sa, sb = -5, -3

    # MUL (via DSP)
    got, gsh = await run_op(dut, 0, a, sa, b, sb)
    ref, rsh = vmul(a, sa, b, sb)
    bad = [i for i in range(D) if got[i] != ref[i]]
    assert not bad and gsh == rsh, f"MUL mismatch at {bad[:3]} / sh {gsh}!={rsh}"
    dut._log.info(f"  OK  MUL (DSP): 64/64 bit-exact, shift {gsh}")

    # ADD (residual)
    got, gsh = await run_op(dut, 1, a, sa, b, sb)
    ref, rsh = vadd(a, sa, b, sb)
    bad = [i for i in range(D) if got[i] != ref[i]]
    assert not bad and gsh == rsh, f"ADD mismatch at {bad[:3]} / sh {gsh}!={rsh}"
    dut._log.info(f"  OK  ADD (residual): 64/64 bit-exact, shift {gsh}")

    dut._log.info(
        "FFN GLUE ALU PASS: multiply (on the real DSP) + residual add, both "
        "bit-exact -- the last new brick for the autonomous FFN sequencer."
    )
