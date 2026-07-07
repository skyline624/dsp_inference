"""Milestone 2 - foundation validation.

Replays the project's real `SS` (SiLU) UART command against the *simulated*
node and checks the result is BIT-EXACT against the project's own reference.

Strategy (highest fidelity, no real board needed):
  - drive the exact byte framing of host/transformer_ops.py:call_ss
    (b'SS' + shift + x[64]) into the cluster head over a real UART link;
  - decode the 70-byte response off tail_tx;
  - reference = the project's silu_ref logic from host/test_silu.py, but using
    the *same* silu_lut.hex the RTL loads -> truly bit-exact, not approximate.

If the simulated int8 output equals the reference exactly, the simulation
reproduces the project's RTL numerics, command protocol, and UART timing.
"""

import math
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles

CLK_NS = 37
DIV    = 27     # top.v UART divider (clk cycles per bit)
D      = 64


# ─── project reference (bit-exact: uses the same LUT the RTL reads) ──────────
def _load_silu_lut():
    for path in ("silu_lut.hex", "../src/silu_lut.hex"):
        try:
            f = open(path)
        except OSError:
            continue
        with f:
            vals = []
            for line in f:
                line = line.strip()
                if not line or line.startswith("//"):
                    continue
                v = int(line, 16)
                if v >= 0x8000:
                    v -= 0x10000          # signed 16-bit
                vals.append(v)
            assert len(vals) >= 256, f"silu_lut.hex: {len(vals)} entries"
            return vals[:256]
    raise FileNotFoundError("silu_lut.hex not found")


SILU_LUT = _load_silu_lut()


def silu_ref(x_i8, sx):
    """Reproduces silu_op.v exactly, using the RTL's own LUT values."""
    out = []
    shx_p4 = sx + 4
    out_shift = 11 + sx
    for xs in x_i8:
        xs = int(xs)
        x16 = (xs << shx_p4) if shx_p4 >= 0 else (xs >> (-shx_p4))  # arith. shift
        idx = max(0, min(255, x16 + 128))
        silu_val = SILU_LUT[idx]
        if out_shift > 0:
            o = (silu_val + (1 << (out_shift - 1))) >> out_shift
        elif out_shift < 0:
            o = silu_val << (-out_shift)
        else:
            o = silu_val
        out.append(max(-128, min(127, o)))
    return out


# ─── UART helpers (real serial framing over the simulated link) ─────────────
def _as_int(sig):
    try:
        return int(sig.value)
    except Exception:
        return None


async def uart_send_byte(dut, byte):
    dut.head_rx.value = 0                       # start bit
    await ClockCycles(dut.clk, DIV)
    for i in range(8):
        dut.head_rx.value = (byte >> i) & 1     # LSB first
        await ClockCycles(dut.clk, DIV)
    dut.head_rx.value = 1                       # stop bit
    await ClockCycles(dut.clk, DIV)


async def uart_send(dut, data):
    for b in data:
        await uart_send_byte(dut, b)


async def uart_rx_monitor(dut, sink):
    """Continuously decode bytes off tail_tx into the bytearray `sink`."""
    while True:
        while _as_int(dut.tail_tx) != 0:        # wait for a start bit
            await RisingEdge(dut.clk)
        await ClockCycles(dut.clk, DIV // 2)    # to middle of start bit
        byte = 0
        for i in range(8):
            await ClockCycles(dut.clk, DIV)     # to middle of data bit i
            byte |= (_as_int(dut.tail_tx) or 0) << i
        await ClockCycles(dut.clk, DIV)         # stop bit
        sink.append(byte)


async def read_resp(dut, sink, off, n, timeout_cycles=400000):
    waited = 0
    while len(sink) < off + n:
        await ClockCycles(dut.clk, 20)
        waited += 20
        assert waited < timeout_cycles, (
            f"timeout: got {len(sink) - off}/{n} response bytes"
        )
    return bytes(sink[off:off + n])


def i8(b):
    return b - 256 if b >= 128 else b


async def boot(dut):
    dut.head_rx.value = 1
    cocotb.start_soon(Clock(dut.clk, CLK_NS, units="ns").start())
    fpga = dut.nodes[0].u_node.u_fpga
    for _ in range(40000):
        await RisingEdge(dut.clk)
        if _as_int(fpga.rst_n) == 1:
            break
    else:
        assert False, "reset never released"
    await ClockCycles(dut.clk, 50)
    dut._log.info("node booted (rst_n=1)")


@cocotb.test()
async def test_silu_bit_exact(dut):
    await boot(dut)

    sink = bytearray()
    cocotb.start_soon(uart_rx_monitor(dut, sink))

    # Same deterministic cases as the project's host/test_silu.py (the ones that
    # need no float quantization): zero, +saturation, -saturation, and a ramp
    # through the interesting region.
    cases = [
        ("x=0  sx=-3",            [0] * D,                       -3),
        ("x=+64 (=8.0) sx=-3",    [64] * D,                      -3),
        ("x=-64 (=-8.0) sx=-3",   [-64] * D,                     -3),
        ("ramp x=[-32..31] sx=-4", list(range(-32, 32)),         -4),
    ]

    off = 0
    n_pass = 0
    for name, x_i8, sx in cases:
        x_bytes = bytes((v & 0xFF) for v in x_i8)
        pkt = b"SS" + bytes([sx & 0xFF]) + x_bytes        # exactly call_ss framing
        await uart_send(dut, pkt)

        resp = await read_resp(dut, sink, off, 70)
        off += 70

        assert resp[:2] == b"SK", f"{name}: bad magic {resp[:2]!r}"
        so = i8(resp[2])
        out_i8 = [i8(resp[6 + i]) for i in range(D)]
        ref = silu_ref(x_i8, sx)

        assert so == sx, f"{name}: shift_out {so} != shift_x {sx}"
        mism = [(i, out_i8[i], ref[i]) for i in range(D) if out_i8[i] != ref[i]]
        assert not mism, (
            f"{name}: {len(mism)} bit mismatches vs project ref, first={mism[0]}"
        )
        dut._log.info(f"  OK  {name:24s} -> 64/64 int8 bit-exact, shift_out={so}")
        n_pass += 1

    dut._log.info(
        f"MILESTONE 2 PASS: {n_pass}/{len(cases)} cases bit-exact vs project reference"
    )
