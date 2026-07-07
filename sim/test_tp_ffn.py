"""Milestone 3b - full FFN block, tensor-parallel across 2 nodes.

Implements the canonical Megatron-style tensor-parallel FFN on the real RTL:

  xn = rmsnorm(x)                              (node 0, broadcast)
  h1 = W1 @ xn ; h3 = W3 @ xn                  ROW-parallel  (each node: hidden/k rows)
  hg = silu(h1) * h3                           local on each node's hidden slice
  out = W2 @ hg                                COLUMN-parallel -> partials, ALL-REDUCE
  y  = x + out                                 residual

Only activations cross between coordinator and nodes; the weights stay split
across the two SDRAMs (memory bandwidth x2) and the two DSP datapaths run in
parallel (compute x2). hidden=128 splits cleanly into 64 rows/cols per node.

Validation:
  - every matmul slice is BIT-EXACT vs the project reference (fq_ref) ;
  - silu slices BIT-EXACT vs the RTL LUT reference ;
  - the final FFN output matches the float reference (infer_v4sim ffn) within
    quantization tolerance.

Weights are preloaded into each node's SDRAM via a backdoor write to the chip
model's memory (the LL write path is already validated in M2b), to keep sim time
reasonable.
"""

import math
import random
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles

CLK_NS = 37
DIV    = 27
D      = 64
K      = 64
HIDDEN = 128
NPN    = HIDDEN // 2          # 64 hidden rows/cols per node

ADDR_RMS = 0x100000
ADDR_W1  = 0x101000
ADDR_W3  = 0x102000
ADDR_W2  = 0x103000


# ─── pure-python numeric helpers (mirror v4_quant / infer_v4sim) ────────────
def to_i8_shift(x):
    m = max((abs(v) for v in x), default=0.0)
    if m == 0:
        return [0] * len(x), 0
    s = math.ceil(math.log2(m / 127.0))
    q = [max(-128, min(127, round(v / (2.0 ** s)))) for v in x]
    return q, s


def deq(x_i8, s):
    return [v * (2.0 ** s) for v in x_i8]


def quantize_matrix(M):
    flat = [v for row in M for v in row]
    _, s = to_i8_shift(flat)
    Mi = [[max(-128, min(127, round(v / (2.0 ** s)))) for v in row] for row in M]
    return Mi, s


def fq_ref(W, x, N):
    y = [sum(W[r][k] * x[k] for k in range(K)) for r in range(N)]
    max_abs = max(abs(v) for v in y)
    if max_abs == 0:
        return [0] * N, 0
    sh = max(0, (max_abs.bit_length() - 1) - 6)
    rnd = (1 << (sh - 1)) if sh > 0 else 0
    out = [max(-128, min(127, ((v + rnd) >> sh) if sh > 0 else v)) for v in y]
    return out, sh


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
    out_shift = 11 + sx
    for xs in x_i8:
        x16 = (xs << shx_p4) if shx_p4 >= 0 else (xs >> (-shx_p4))
        idx = max(0, min(255, x16 + 128))
        sv = SILU_LUT[idx]
        if out_shift > 0:
            o = (sv + (1 << (out_shift - 1))) >> out_shift
        elif out_shift < 0:
            o = sv << (-out_shift)
        else:
            o = sv
        out.append(max(-128, min(127, o)))
    return out


def rmsnorm_f(x, w, eps=1e-5):
    ms = sum(v * v for v in x) / len(x)
    inv = 1.0 / math.sqrt(ms + eps)
    return [x[i] * w[i] * inv for i in range(len(x))]


def silu_f(v):
    return v / (1.0 + math.exp(-v))


def matvec_f(W, x):
    return [sum(W[r][k] * x[k] for k in range(len(x))) for r in range(len(W))]


# ─── byte framing ───────────────────────────────────────────────────────────
def addr_bytes(a): return bytes([a & 0xFF, (a >> 8) & 0xFF, (a >> 16) & 0xFF])
def i8(b):         return b - 256 if b >= 128 else b
def bb(vals):      return bytes((v & 0xFF) for v in vals)


def _as_int(sig):
    try:
        return int(sig.value)
    except Exception:
        return None


async def uart_send(clk, rx_sig, data):
    for byte in data:
        rx_sig.value = 0
        await ClockCycles(clk, DIV)
        for i in range(8):
            rx_sig.value = (byte >> i) & 1
            await ClockCycles(clk, DIV)
        rx_sig.value = 1
        await ClockCycles(clk, DIV)


async def uart_rx_monitor(clk, tx_sig, sink):
    while True:
        while _as_int(tx_sig) != 0:
            await RisingEdge(clk)
        await ClockCycles(clk, DIV // 2)
        byte = 0
        for i in range(8):
            await ClockCycles(clk, DIV)
            byte |= (_as_int(tx_sig) or 0) << i
        await ClockCycles(clk, DIV)
        sink.append(byte)


class Node:
    def __init__(self, clk, rx_sig, tx_sig, fpga, sdram):
        self.clk, self.rx, self.fpga, self.sdram = clk, rx_sig, fpga, sdram
        self.sink = bytearray()
        self.off = 0
        cocotb.start_soon(uart_rx_monitor(clk, tx_sig, self.sink))

    async def xfer(self, pkt, nresp, timeout_cycles=2_000_000):
        await uart_send(self.clk, self.rx, pkt)
        waited = 0
        while len(self.sink) < self.off + nresp:
            await ClockCycles(self.clk, 20)
            waited += 20
            assert waited < timeout_cycles, f"timeout {len(self.sink)-self.off}/{nresp}"
        resp = bytes(self.sink[self.off:self.off + nresp])
        self.off += nresp
        return resp

    def preload(self, base_addr, data):
        """Backdoor-write bytes into the SDRAM model (4 bytes/word, LE)."""
        assert base_addr % 4 == 0
        wbase = base_addr >> 2
        b = bytes(data)
        if len(b) % 4:
            b += bytes(4 - len(b) % 4)
        for w in range(len(b) // 4):
            word = b[4*w] | (b[4*w+1] << 8) | (b[4*w+2] << 16) | (b[4*w+3] << 24)
            self.sdram.mem[wbase + w].value = word

    async def fn(self, x_i8, sx, sw, addr):
        pkt = b"FN" + bytes([sx & 0xFF, sw & 0xFF]) + bb(x_i8) + addr_bytes(addr)
        r = await self.xfer(pkt, 75)
        assert r[:2] == b"FK", f"FN magic {r[:2]!r}"
        return [i8(r[11 + i]) for i in range(D)], i8(r[2])

    async def fq(self, N, sx, sw, x_i8, addr):
        pkt = b"FQ" + bytes([N, sx & 0xFF, sw & 0xFF]) + bb(x_i8) + addr_bytes(addr)
        r = await self.xfer(pkt, 3 + N)
        assert r[:2] == b"FQ", f"FQ magic {r[:2]!r}"
        return [i8(r[3 + i]) for i in range(N)], i8(r[2])

    async def ss(self, x_i8, sx):
        full = list(x_i8) + [0] * (D - len(x_i8))
        pkt = b"SS" + bytes([sx & 0xFF]) + bb(full)
        r = await self.xfer(pkt, 70)
        assert r[:2] == b"SK", f"SS magic {r[:2]!r}"
        return [i8(r[6 + i]) for i in range(len(x_i8))], i8(r[2])


async def boot_node(clk, fpga):
    for _ in range(40000):
        await RisingEdge(clk)
        if _as_int(fpga.rst_n) == 1:
            break
    else:
        assert False, "reset never released"
    for _ in range(40000):
        await RisingEdge(clk)
        if _as_int(fpga.sd_busy) == 0:
            break
    else:
        assert False, "SDRAM never ready"


@cocotb.test()
async def test_tp_ffn(dut):
    dut.rx0.value = 1
    dut.rx1.value = 1
    cocotb.start_soon(Clock(dut.clk, CLK_NS, units="ns").start())
    b0 = cocotb.start_soon(boot_node(dut.clk, dut.n0.u_fpga))
    b1 = cocotb.start_soon(boot_node(dut.clk, dut.n1.u_fpga))
    await b0
    await b1
    node0 = Node(dut.clk, dut.rx0, dut.tx0, dut.n0.u_fpga, dut.n0.u_sdram)
    node1 = Node(dut.clk, dut.rx1, dut.tx1, dut.n1.u_fpga, dut.n1.u_sdram)
    dut._log.info("both nodes booted, SDRAM ready")

    # ── build a random FFN (like setup_ffn_weights) ─────────────────────────
    rng = random.Random(3)
    def g(n):   return [rng.gauss(0, 0.1) for _ in range(n)]
    def gm(r, c): return [[rng.gauss(0, 0.1) for _ in range(c)] for _ in range(r)]
    x_f   = g(D)
    rms_w = [1.0] * D
    W1_f  = gm(HIDDEN, D)
    W3_f  = gm(HIDDEN, D)
    W2_f  = gm(D, HIDDEN)

    x_i8, sx = to_i8_shift(x_f)
    rms_i8, swr = to_i8_shift(rms_w)
    W1_i8, sw1 = quantize_matrix(W1_f)
    W3_i8, sw3 = quantize_matrix(W3_f)
    W2_i8, sw2 = quantize_matrix(W2_f)

    def rows(M, a, b):       return [M[r] for r in range(a, b)]
    def cols(M, a, b):       return [[M[r][k] for k in range(a, b)] for r in range(len(M))]
    def flat(M):             return [v for row in M for v in row]

    # ── distribute weights into each node's own SDRAM (backdoor) ────────────
    node0.preload(ADDR_RMS, bb(rms_i8))
    node0.preload(ADDR_W1, bb(flat(rows(W1_i8, 0, NPN))))
    node0.preload(ADDR_W3, bb(flat(rows(W3_i8, 0, NPN))))
    node0.preload(ADDR_W2, bb(flat(cols(W2_i8, 0, NPN))))      # W2[:,0:64]
    node1.preload(ADDR_W1, bb(flat(rows(W1_i8, NPN, HIDDEN))))
    node1.preload(ADDR_W3, bb(flat(rows(W3_i8, NPN, HIDDEN))))
    node1.preload(ADDR_W2, bb(flat(cols(W2_i8, NPN, HIDDEN))))  # W2[:,64:128]
    await ClockCycles(dut.clk, 10)
    dut._log.info(f"weights distributed: {NPN} hidden/node across 2 SDRAMs")

    # ── rmsnorm on node 0, broadcast xn ─────────────────────────────────────
    xn_i8, sxn = await node0.fn(x_i8, sx, swr, ADDR_RMS)

    # ── W1 / W3 : row-parallel. One UART per node -> sequential WITHIN a node,
    #    parallel ACROSS nodes.
    async def w1w3(node):
        h1, s1 = await node.fq(NPN, sxn, sw1, xn_i8, ADDR_W1)
        h3, s3 = await node.fq(NPN, sxn, sw3, xn_i8, ADDR_W3)
        return h1, s1, h3, s3
    r0 = cocotb.start_soon(w1w3(node0))
    r1 = cocotb.start_soon(w1w3(node1))
    h1_0, s10, h3_0, s30 = await r0
    h1_1, s11, h3_1, s31 = await r1

    for (label, got, W, sh) in [
        ("W1 node0", h1_0, rows(W1_i8, 0, NPN), s10),
        ("W1 node1", h1_1, rows(W1_i8, NPN, HIDDEN), s11),
        ("W3 node0", h3_0, rows(W3_i8, 0, NPN), s30),
        ("W3 node1", h3_1, rows(W3_i8, NPN, HIDDEN), s31)]:
        ref, shu = fq_ref(W, xn_i8, NPN)
        assert got == ref, f"{label} matmul mismatch"
    dut._log.info("  OK  W1/W3 row-parallel matmuls bit-exact (4 slices)")

    # ── silu (concurrent) bit-exact, then local multiply silu(h1)*h3 ────────
    ts = [cocotb.start_soon(node0.ss(h1_0, s10)), cocotb.start_soon(node1.ss(h1_1, s11))]
    (sg0, ss0), (sg1, ss1) = [await x for x in ts]
    assert sg0 == silu_ref(h1_0, s10) and sg1 == silu_ref(h1_1, s11), "silu mismatch"
    dut._log.info("  OK  silu slices bit-exact")

    hg0_i8, shg0 = to_i8_shift([a * b for a, b in zip(deq(sg0, ss0), deq(h3_0, s30))])
    hg1_i8, shg1 = to_i8_shift([a * b for a, b in zip(deq(sg1, ss1), deq(h3_1, s31))])

    # ── W2 : column-parallel partials (concurrent) + all-reduce ─────────────
    tw = [cocotb.start_soon(node0.fq(D, shg0, sw2, hg0_i8, ADDR_W2)),
          cocotb.start_soon(node1.fq(D, shg1, sw2, hg1_i8, ADDR_W2))]
    (p0, sp0), (p1, sp1) = [await x for x in tw]
    assert p0 == fq_ref(cols(W2_i8, 0, NPN), hg0_i8, D)[0], "W2 node0 mismatch"
    assert p1 == fq_ref(cols(W2_i8, NPN, HIDDEN), hg1_i8, D)[0], "W2 node1 mismatch"
    dut._log.info("  OK  W2 column-parallel partials bit-exact")

    out_real = [deq(p0, sp0)[d] + deq(p1, sp1)[d] for d in range(D)]    # all-reduce
    out_i8, so = to_i8_shift(out_real)
    y = [x_f[d] + (out_i8[d] * (2.0 ** so)) for d in range(D)]          # residual

    # ── float reference (infer_v4sim ffn) within tolerance ──────────────────
    xn_f = rmsnorm_f(x_f, rms_w)
    gate = matvec_f(W1_f, xn_f)
    up   = matvec_f(W3_f, xn_f)
    hgt  = [silu_f(gate[i]) * up[i] for i in range(HIDDEN)]
    out_f = matvec_f(W2_f, hgt)
    y_ref = [x_f[d] + out_f[d] for d in range(D)]

    max_abs = max(abs(v) for v in y_ref) or 1.0
    abs_err = max(abs(y[d] - y_ref[d]) for d in range(D))
    dut._log.info(f"  cluster FFN vs float ref: max_abs_err={abs_err:.4f} "
                  f"({100*abs_err/max_abs:.2f}% of range)")
    assert abs_err / max_abs < 0.25, f"FFN output too far from reference: {abs_err}"

    dut._log.info(
        "MILESTONE 3b PASS: full FFN tensor-parallel across 2 nodes "
        "(W1/W3 row-parallel, W2 column-parallel + all-reduce), validated"
    )
