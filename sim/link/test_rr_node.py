"""Sub-gate 2 (G3) : the node rope_op (RR) primitive, ISOLATED, over the parallel link.

RoPE has never been exercised in the cocotb lineage (attention ran at pos=0 =
identity rope), so per G3 it must be validated alone before the causal-attention
sequencer orchestrates it. Drives node_ss_top with real RR packets and checks the
node against the exact fixed-point reference host/test_rope.py::rope_ref :
  * cos/sin quantized to Q15,
  * integer products, round (+1<<14) >> 15, clip int8,
  * shift_out == shift_x (rope preserves the shift).

  Protocol  : 'R''R' sx x[8] cos[4]i16LE sin[4]i16LE                 (27 bytes)
  Response  : 'R''K' so dbg_real[4]LE dbg_imag[4]LE out[8]           (19 bytes)

  make -f Makefile.nodess TOPLEVEL=node_ss_top MODULE=test_rr_node
"""

import math
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

HS, HALF = 8, 4


def i8(b):
    return b - 256 if b >= 128 else b


def to_q15(x):
    return int(max(-32768, min(32767, round(x * 32768.0))))


def rope_ref(x_i8, cos_f, sin_f):
    """Exact node rope_op behaviour (mirror of host/test_rope.py::rope_ref)."""
    out = [0] * HS
    for i in range(HALF):
        xr = int(x_i8[2 * i]); xi = int(x_i8[2 * i + 1])
        cq = to_q15(cos_f[i]); sq = to_q15(sin_f[i])
        nr = (xr * cq - xi * sq + 16384) >> 15
        ni = (xr * sq + xi * cq + 16384) >> 15
        out[2 * i]     = max(-128, min(127, nr))
        out[2 * i + 1] = max(-128, min(127, ni))
    return out


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


async def call_rr(dut, x_i8, sx, cos_f, sin_f):
    cos_b = b"".join(int.to_bytes(to_q15(c) & 0xFFFF, 2, "little") for c in cos_f)
    sin_b = b"".join(int.to_bytes(to_q15(s) & 0xFFFF, 2, "little") for s in sin_f)
    pkt = b"RR" + bytes([sx & 0xFF]) + bytes((v & 0xFF) for v in x_i8) + cos_b + sin_b
    assert len(pkt) == 27, len(pkt)
    for b in pkt:
        await push(dut, b)
    resp = [await pull(dut) for _ in range(19)]
    assert resp[0] == ord("R") and resp[1] == ord("K"), f"RR magic {resp[:2]!r}"
    so = i8(resp[2])
    out = [i8(resp[11 + j]) for j in range(HS)]
    return out, so


@cocotb.test()
async def test_rr_node(dut):
    cocotb.start_soon(Clock(dut.clk, 37, units="ns").start())
    dut.rst.value = 1
    dut.cmd_wr.value = 0
    dut.cmd_data.value = 0
    dut.resp_rd.value = 0
    for _ in range(10):
        await RisingEdge(dut.clk)
    dut.rst.value = 0
    for _ in range(40000):        # PLL lock + node init, like test_node_ss
        await RisingEdge(dut.clk)

    cases = [
        ("identity theta=0", [10, 20, -30, 40, 50, -60, 70, -80], -3, [1.0] * HALF, [0.0] * HALF),
        ("pi/2 (cos0 sin1)", [10, 20, -30, 40, 50, -60, 70, -80], -3, [0.0] * HALF, [1.0] * HALF),
        ("pi/4",             [100, 0, 0, 100, -100, 0, 0, -100],   -3, [math.sqrt(0.5)] * HALF, [math.sqrt(0.5)] * HALF),
        ("LLM-like freqs",   [20, 40, 60, -20, -40, -60, 80, -80], -3,
         [math.cos(t) for t in (0.0, 0.3, 0.6, 0.9)], [math.sin(t) for t in (0.0, 0.3, 0.6, 0.9)]),
        ("shift +2",         [12, -34, 56, -78, 90, -100, 111, -120], 2,
         [math.cos(t) for t in (0.1, 0.4, 0.7, 1.0)], [math.sin(t) for t in (0.1, 0.4, 0.7, 1.0)]),
    ]

    fails = 0
    for name, x_i8, sx, cos_f, sin_f in cases:
        out, so = await call_rr(dut, x_i8, sx, cos_f, sin_f)
        ref = rope_ref(x_i8, cos_f, sin_f)
        ok_out = (out == ref)
        ok_sh = (so == sx)
        if not (ok_out and ok_sh):
            fails += 1
            dut._log.error(f"[{name}] out={out} ref={ref} ok_out={ok_out} | so={so} sx={sx} ok_sh={ok_sh}")
        else:
            dut._log.info(f"[{name}] OK  out={out} so={so:+d}")

    assert fails == 0, f"{fails}/{len(cases)} RR cases failed"
    dut._log.info(
        f"SUB-GATE 2 (RR isolated) PASS: node rope_op matches the Q15 fixed-point "
        f"reference on {len(cases)} cases, shift preserved. Rope primitive validated "
        "before orchestration."
    )
