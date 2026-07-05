"""vec_alu2 : SERIALIZED (LUT-lean) variant of the sequencer's multiply/residual ALU.

Same contract as vec_alu (test_vec_alu.py) — must be bit-exact vs the same Python
reference (round-half-up requantize):
  MUL : out = a*b   (multiply on the real DSP / mac18)
  ADD : out = a+b   (shift-aligned residual)

vec_alu2 runs one element/cycle instead of 64 in parallel, so it just takes more
cycles; the numeric result must match vec_alu exactly. This is the Phase-0 gate:
prove the LUT-lean refactor (5 juillet) computes correctly, not only that it routes.
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
    # serialized ALU: allow more cycles than the parallel one (64 elems x passes)
    for _ in range(20000):
        await RisingEdge(dut.clk)
        if int(dut.done.value) == 1:
            break
    else:
        assert False, "vec_alu2 timeout"
    ov = int(dut.out.value)
    got = [to_i8((ov >> (i * 8)) & 0xFF) for i in range(D)]
    return got, to_i8(int(dut.out_sh.value))


@cocotb.test()
async def test_vec_alu2(dut):
    cocotb.start_soon(Clock(dut.clk, CLK_NS, units="ns").start())
    dut.rst_n.value = 0
    dut.start.value = 0
    dut.op.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)

    # Several random vectors + shift pairs, to exercise requantize corners.
    for seed, (sa, sb) in enumerate([(-5, -3), (-2, -7), (0, 0), (-8, -1)]):
        rng = random.Random(100 + seed)
        a = [rng.randrange(-128, 128) for _ in range(D)]
        b = [rng.randrange(-128, 128) for _ in range(D)]

        got, gsh = await run_op(dut, 0, a, sa, b, sb)
        ref, rsh = vmul(a, sa, b, sb)
        bad = [i for i in range(D) if got[i] != ref[i]]
        assert not bad and gsh == rsh, f"MUL[{seed}] mismatch at {bad[:3]} / sh {gsh}!={rsh}"

        got, gsh = await run_op(dut, 1, a, sa, b, sb)
        ref, rsh = vadd(a, sa, b, sb)
        bad = [i for i in range(D) if got[i] != ref[i]]
        assert not bad and gsh == rsh, f"ADD[{seed}] mismatch at {bad[:3]} / sh {gsh}!={rsh}"

    dut._log.info(
        "vec_alu2 PASS: serialized MUL (DSP) + ADD (residual) bit-exact vs vec_alu "
        "reference across 4 shift regimes -- LUT-lean refactor is numerically correct."
    )
