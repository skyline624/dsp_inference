"""Milestone 3c - attention block, head-parallel across 2 nodes.

Attention heads are independent, so the block splits HEAD-parallel:
  - Wq/Wk/Wv projections: each node computes ONLY its heads (its own weight
    slice in its own SDRAM) -> parallel compute + distributed weights ;
  - attention core (scores/softmax/weighted-sum): the project's MM command is
    monolithic over all H heads and is NEGLIGIBLE compute (~2k ops vs ~30k for
    the matmuls), so the gathered Q/K/V run through the real MM engine centrally ;
  - Wo output projection: row-parallel (each node a slice of the output rows).

Scope: pos=0, T=1 -> rope is identity and there is no KV-cache history, so the
attention core reduces to GQA V-passthrough. This validates the distributed
projections + Wo + the real MM datapath. Multi-position softmax is exercised at
M4 (generation, real KV cache).

Validation: every projection / Wo slice BIT-EXACT vs the project reference;
the MM output bit-exact vs the GQA expansion of V; the full attention block
within quantization tolerance of the float reference (infer_v4sim attention).
Weights preloaded via SDRAM backdoor.
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
H, KH, HS = 8, 4, 8          # heads, kv-heads, head_size  (GQA n_rep = 2)
N_REP  = H // KH
QH_PN  = (H // 2) * HS        # query rows per node = 4 heads * 8 = 32
KvH_PN = (KH // 2) * HS       # kv rows per node    = 2 heads * 8 = 16
WO_PN  = D // 2               # Wo output rows per node = 32

ADDR_RMS = 0x100000
ADDR_WQ  = 0x101000
ADDR_WK  = 0x102000
ADDR_WV  = 0x103000
ADDR_WO  = 0x104000


# ─── numeric helpers (mirror v4_quant / infer_v4sim) ────────────────────────
def to_i8_shift(x):
    m = max((abs(v) for v in x), default=0.0)
    if m == 0:
        return [0] * len(x), 0
    s = math.ceil(math.log2(m / 127.0))
    return [max(-128, min(127, round(v / (2.0 ** s)))) for v in x], s


def deq(x_i8, s):
    return [v * (2.0 ** s) for v in x_i8]


def quantize_matrix(M):
    flat = [v for row in M for v in row]
    _, s = to_i8_shift(flat)
    return [[max(-128, min(127, round(v / (2.0 ** s)))) for v in row] for row in M], s


def fq_ref(W, x, N):
    y = [sum(W[r][k] * x[k] for k in range(K)) for r in range(N)]
    ma = max(abs(v) for v in y)
    if ma == 0:
        return [0] * N, 0
    sh = max(0, (ma.bit_length() - 1) - 6)
    rnd = (1 << (sh - 1)) if sh > 0 else 0
    return [max(-128, min(127, ((v + rnd) >> sh) if sh > 0 else v)) for v in y], sh


def gather(parts):
    real = []
    for v, s in parts:
        real += deq(v, s)
    return to_i8_shift(real)


def rmsnorm_f(x, w, eps=1e-5):
    inv = 1.0 / math.sqrt(sum(v * v for v in x) / len(x) + eps)
    return [x[i] * w[i] * inv for i in range(len(x))]


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
        assert base_addr % 4 == 0
        wbase = base_addr >> 2
        b = bytes(data)
        if len(b) % 4:
            b += bytes(4 - len(b) % 4)
        for w in range(len(b) // 4):
            self.sdram.mem[wbase + w].value = (
                b[4*w] | (b[4*w+1] << 8) | (b[4*w+2] << 16) | (b[4*w+3] << 24))

    async def fn(self, x_i8, sx, sw, addr):
        r = await self.xfer(b"FN" + bytes([sx & 0xFF, sw & 0xFF]) + bb(x_i8) + addr_bytes(addr), 75)
        assert r[:2] == b"FK", f"FN magic {r[:2]!r}"
        return [i8(r[11 + i]) for i in range(D)], i8(r[2])

    async def fq(self, N, sx, sw, x_i8, addr):
        r = await self.xfer(b"FQ" + bytes([N, sx & 0xFF, sw & 0xFF]) + bb(x_i8) + addr_bytes(addr), 3 + N)
        assert r[:2] == b"FQ", f"FQ magic {r[:2]!r}"
        return [i8(r[3 + i]) for i in range(N)], i8(r[2])

    async def mm(self, Q, K, V, sq, sk, sv, T):
        pkt = b"MM" + bytes([sq & 0xFF, sk & 0xFF, sv & 0xFF, T]) + bb(Q) + bb(K) + bb(V)
        r = await self.xfer(pkt, 67)
        assert r[:2] == b"MK", f"MM magic {r[:2]!r}"
        return [i8(r[3 + i]) for i in range(D)], sv


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
async def test_tp_attention(dut):
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

    # ── random attention weights ────────────────────────────────────────────
    rng = random.Random(5)
    def g(n):     return [rng.gauss(0, 0.1) for _ in range(n)]
    def gm(r, c): return [[rng.gauss(0, 0.1) for _ in range(c)] for _ in range(r)]
    x_f    = g(D)
    rms_w  = [1.0] * D
    Wq_f   = gm(H * HS, D)      # [64,64]
    Wk_f   = gm(KH * HS, D)     # [32,64]
    Wv_f   = gm(KH * HS, D)     # [32,64]
    Wo_f   = gm(D, H * HS)      # [64,64]

    x_i8, sx   = to_i8_shift(x_f)
    rms_i8, swr = to_i8_shift(rms_w)
    Wq_i8, swq = quantize_matrix(Wq_f)
    Wk_i8, swk = quantize_matrix(Wk_f)
    Wv_i8, swv = quantize_matrix(Wv_f)
    Wo_i8, swo = quantize_matrix(Wo_f)

    def rows(M, a, b): return [M[r] for r in range(a, b)]
    def flat(M):       return [v for row in M for v in row]

    # ── distribute weights HEAD-parallel into each node's own SDRAM ─────────
    node0.preload(ADDR_RMS, bb(rms_i8))
    node0.preload(ADDR_WQ, bb(flat(rows(Wq_i8, 0, QH_PN))))        # query heads 0-3
    node0.preload(ADDR_WK, bb(flat(rows(Wk_i8, 0, KvH_PN))))       # kv heads 0-1
    node0.preload(ADDR_WV, bb(flat(rows(Wv_i8, 0, KvH_PN))))
    node0.preload(ADDR_WO, bb(flat(rows(Wo_i8, 0, WO_PN))))        # output rows 0-31
    node1.preload(ADDR_WQ, bb(flat(rows(Wq_i8, QH_PN, H * HS))))   # query heads 4-7
    node1.preload(ADDR_WK, bb(flat(rows(Wk_i8, KvH_PN, KH * HS)))) # kv heads 2-3
    node1.preload(ADDR_WV, bb(flat(rows(Wv_i8, KvH_PN, KH * HS))))
    node1.preload(ADDR_WO, bb(flat(rows(Wo_i8, WO_PN, D))))        # output rows 32-63
    await ClockCycles(dut.clk, 10)
    dut._log.info("attention weights distributed head-parallel across 2 SDRAMs")

    # ── rmsnorm (node 0, broadcast) ─────────────────────────────────────────
    xn_i8, sxn = await node0.fn(x_i8, sx, swr, ADDR_RMS)

    # ── Q/K/V projections: each node computes ONLY its heads (parallel) ─────
    async def proj(node):
        q, sq = await node.fq(QH_PN,  sxn, swq, xn_i8, ADDR_WQ)
        k, sk = await node.fq(KvH_PN, sxn, swk, xn_i8, ADDR_WK)
        v, sv = await node.fq(KvH_PN, sxn, swv, xn_i8, ADDR_WV)
        return q, sq, k, sk, v, sv
    r0 = cocotb.start_soon(proj(node0))
    r1 = cocotb.start_soon(proj(node1))
    q0, sq0, k0, sk0, v0, sv0 = await r0
    q1, sq1, k1, sk1, v1, sv1 = await r1

    assert q0 == fq_ref(rows(Wq_i8, 0, QH_PN), xn_i8, QH_PN)[0]
    assert q1 == fq_ref(rows(Wq_i8, QH_PN, H * HS), xn_i8, QH_PN)[0]
    assert k0 == fq_ref(rows(Wk_i8, 0, KvH_PN), xn_i8, KvH_PN)[0]
    assert k1 == fq_ref(rows(Wk_i8, KvH_PN, KH * HS), xn_i8, KvH_PN)[0]
    assert v0 == fq_ref(rows(Wv_i8, 0, KvH_PN), xn_i8, KvH_PN)[0]
    assert v1 == fq_ref(rows(Wv_i8, KvH_PN, KH * HS), xn_i8, KvH_PN)[0]
    dut._log.info("  OK  Q/K/V head-parallel projections bit-exact (6 slices)")

    # ── gather Q/K/V (re-align shifts), run the real MM core (pos=0, T=1) ────
    Q_i8, sQ = gather([(q0, sq0), (q1, sq1)])     # [64]
    K_i8, sK = gather([(k0, sk0), (k1, sk1)])     # [32]
    V_i8, sV = gather([(v0, sv0), (v1, sv1)])     # [32]
    attn, sa = await node0.mm(Q_i8, K_i8, V_i8, sQ, sK, sV, T=1)

    # at T=1 the engine returns GQA-expanded V: head h -> kv head h//N_REP
    exp = [V_i8[(h // N_REP) * HS + d] for h in range(H) for d in range(HS)]
    assert attn == exp, f"MM T=1 output != GQA V-expansion (first diff at "\
        f"{next(i for i in range(D) if attn[i]!=exp[i])})"
    dut._log.info("  OK  MM attention core (T=1) bit-exact = GQA V-passthrough")

    # ── Wo : row-parallel over the gathered attention (broadcast attn) ──────
    w0 = cocotb.start_soon(node0.fq(WO_PN, sa, swo, attn, ADDR_WO))
    w1 = cocotb.start_soon(node1.fq(WO_PN, sa, swo, attn, ADDR_WO))
    o0, so0 = await w0
    o1, so1 = await w1
    assert o0 == fq_ref(rows(Wo_i8, 0, WO_PN), attn, WO_PN)[0]
    assert o1 == fq_ref(rows(Wo_i8, WO_PN, D), attn, WO_PN)[0]
    dut._log.info("  OK  Wo row-parallel output projection bit-exact")

    # ── residual ────────────────────────────────────────────────────────────
    out_real = deq(o0, so0) + deq(o1, so1)        # rows 0-31 ++ rows 32-63
    y = [x_f[d] + out_real[d] for d in range(D)]

    # ── float reference (infer_v4sim attention, pos=0 T=1) ──────────────────
    xn_f = rmsnorm_f(x_f, rms_w)
    Qf = matvec_f(Wq_f, xn_f)
    Vf = matvec_f(Wv_f, xn_f)
    # rope at pos 0 = identity ; T=1 softmax -> out_head = V[kv head]
    out_f_pre = [Vf[(h // N_REP) * HS + d] for h in range(H) for d in range(HS)]
    out_f = matvec_f(Wo_f, out_f_pre)
    y_ref = [x_f[d] + out_f[d] for d in range(D)]

    max_abs = max(abs(v) for v in y_ref) or 1.0
    abs_err = max(abs(y[d] - y_ref[d]) for d in range(D))
    dut._log.info(f"  cluster attention vs float ref: max_abs_err={abs_err:.4f} "
                  f"({100*abs_err/max_abs:.2f}% of range)")
    assert abs_err / max_abs < 0.25, f"attention output too far from ref: {abs_err}"

    dut._log.info(
        "MILESTONE 3c PASS: attention block head-parallel across 2 nodes "
        "(Q/K/V head-parallel, real MM core, Wo row-parallel), validated"
    )
