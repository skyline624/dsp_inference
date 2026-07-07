"""Milestone 4a - full transformer LAYER, tensor-parallel, multi-position.

Chains the two validated blocks (attention head-parallel + FFN tensor-parallel)
into one complete layer, and runs it for several positions with a real KV cache
so the attention core exercises MULTI-POSITION softmax (T>1) -- the last RTL path
not yet covered (M3c was T=1).

Per position p:
  attention: rmsnorm -> Q/K/V (head-parallel) -> rope -> KV cache ->
             MM over t=0..p (real multi-position softmax) -> Wo (row-parallel) -> +residual
  ffn:       rmsnorm -> W1/W3 (row-parallel) -> silu*  -> W2 (col-parallel + all-reduce) -> +residual

Heavy, weight-bearing ops run on the real RTL, distributed across 2 nodes
(weights split across the two SDRAMs). The coordinator does only small activation
glue (rope, multiply, all-reduce sums, residual, KV cache) -- like the real host.
Random weights (the real stories260K.bin is out-of-repo); validated against a
pure-python reference mirroring infer_v4sim.forward (within quantization tolerance).

Note: rope is computed coordinator-side here (could move to the RR command for
full RTL fidelity); the MM core is the real RTL.
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
H, KH, HS = 8, 4, 8
N_REP  = H // KH
HIDDEN = 128
QH_PN  = (H // 2) * HS      # 32
KvH_PN = (KH // 2) * HS     # 16
WO_PN  = D // 2             # 32
HID_PN = HIDDEN // 2        # 64
N_POS  = 3

A_RMS_ATT = 0x100000
A_WQ      = 0x102000
A_WK      = 0x104000
A_WV      = 0x106000
A_WO      = 0x108000
A_RMS_FFN = 0x10A000
A_W1      = 0x10C000
A_W3      = 0x10E000
A_W2      = 0x110000


# ─── numeric helpers ────────────────────────────────────────────────────────
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


def gather(parts):
    real = []
    for v, s in parts:
        real += deq(v, s)
    return to_i8_shift(real)


def rmsnorm_f(x, w, eps=1e-5):
    inv = 1.0 / math.sqrt(sum(v * v for v in x) / len(x) + eps)
    return [x[i] * w[i] * inv for i in range(len(x))]


def silu_f(v):
    return v / (1.0 + math.exp(-v))


def matvec_f(W, x):
    return [sum(W[r][k] * x[k] for k in range(len(x))) for r in range(len(W))]


def freq_cis(pos):
    fr, fi = [], []
    for j in range(HS // 2):
        ang = pos * (10000.0 ** (-(2.0 * j) / HS))
        fr.append(math.cos(ang))
        fi.append(math.sin(ang))
    return fr, fi


def rope_vec(v, n_heads, fr, fi):
    out = list(v)
    for h in range(n_heads):
        for j in range(HS // 2):
            r = v[h * HS + 2 * j]
            im = v[h * HS + 2 * j + 1]
            out[h * HS + 2 * j]     = r * fr[j] - im * fi[j]
            out[h * HS + 2 * j + 1] = r * fi[j] + im * fr[j]
    return out


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

    async def ss(self, x_i8, sx):
        full = list(x_i8) + [0] * (D - len(x_i8))
        r = await self.xfer(b"SS" + bytes([sx & 0xFF]) + bb(full), 70)
        assert r[:2] == b"SK", f"SS magic {r[:2]!r}"
        return [i8(r[6 + i]) for i in range(len(x_i8))], i8(r[2])

    async def mm(self, Q, Kc, Vc, sq, sk, sv, T):
        pkt = b"MM" + bytes([sq & 0xFF, sk & 0xFF, sv & 0xFF, T]) + bb(Q) + bb(Kc) + bb(Vc)
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


def rows(M, a, b):  return [M[r] for r in range(a, b)]
def flat(M):        return [v for row in M for v in row]


@cocotb.test()
async def test_tp_layer(dut):
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

    # ── random layer weights ────────────────────────────────────────────────
    rng = random.Random(7)
    def gm(r, c): return [[rng.gauss(0, 0.1) for _ in range(c)] for _ in range(r)]
    rms_att = [1.0] * D
    rms_ffn = [1.0] * D
    Wq_f, Wk_f, Wv_f = gm(H*HS, D), gm(KH*HS, D), gm(KH*HS, D)
    Wo_f = gm(D, H*HS)
    W1_f, W3_f = gm(HIDDEN, D), gm(HIDDEN, D)
    W2_f = gm(D, HIDDEN)

    rmsa_i8, s_rmsa = to_i8_shift(rms_att)
    rmsf_i8, s_rmsf = to_i8_shift(rms_ffn)
    Wq_i8, swq = quantize_matrix(Wq_f)
    Wk_i8, swk = quantize_matrix(Wk_f)
    Wv_i8, swv = quantize_matrix(Wv_f)
    Wo_i8, swo = quantize_matrix(Wo_f)
    W1_i8, sw1 = quantize_matrix(W1_f)
    W3_i8, sw3 = quantize_matrix(W3_f)
    W2_i8, sw2 = quantize_matrix(W2_f)

    # ── distribute all weights into the two SDRAMs (backdoor) ───────────────
    node0.preload(A_RMS_ATT, bb(rmsa_i8)); node0.preload(A_RMS_FFN, bb(rmsf_i8))
    node0.preload(A_WQ, bb(flat(rows(Wq_i8, 0, QH_PN))))
    node0.preload(A_WK, bb(flat(rows(Wk_i8, 0, KvH_PN))))
    node0.preload(A_WV, bb(flat(rows(Wv_i8, 0, KvH_PN))))
    node0.preload(A_WO, bb(flat(rows(Wo_i8, 0, WO_PN))))
    node0.preload(A_W1, bb(flat(rows(W1_i8, 0, HID_PN))))
    node0.preload(A_W3, bb(flat(rows(W3_i8, 0, HID_PN))))
    node0.preload(A_W2, bb(flat([[W2_i8[n][k] for k in range(0, HID_PN)] for n in range(D)])))
    node1.preload(A_RMS_FFN, bb(rmsf_i8))    # node1 needs rms_ffn too (local rmsnorm option)
    node1.preload(A_WQ, bb(flat(rows(Wq_i8, QH_PN, H*HS))))
    node1.preload(A_WK, bb(flat(rows(Wk_i8, KvH_PN, KH*HS))))
    node1.preload(A_WV, bb(flat(rows(Wv_i8, KvH_PN, KH*HS))))
    node1.preload(A_WO, bb(flat(rows(Wo_i8, WO_PN, D))))
    node1.preload(A_W1, bb(flat(rows(W1_i8, HID_PN, HIDDEN))))
    node1.preload(A_W3, bb(flat(rows(W3_i8, HID_PN, HIDDEN))))
    node1.preload(A_W2, bb(flat([[W2_i8[n][k] for k in range(HID_PN, HIDDEN)] for n in range(D)])))
    await ClockCycles(dut.clk, 10)
    dut._log.info("full layer weights distributed across 2 SDRAMs")

    # cluster + reference KV caches (float)
    cache_K, cache_V = [], []        # cluster: per-position roped-K / V (float, len 32)
    ref_K, ref_V = [], []            # reference
    max_err_seen = 0.0

    async def attn_block(x_f, pos):
        x_i8, sx = to_i8_shift(x_f)
        xn_i8, sxn = await node0.fn(x_i8, sx, s_rmsa, A_RMS_ATT)

        async def proj(node):
            q, sq = await node.fq(QH_PN,  sxn, swq, xn_i8, A_WQ)
            k, sk = await node.fq(KvH_PN, sxn, swk, xn_i8, A_WK)
            v, sv = await node.fq(KvH_PN, sxn, swv, xn_i8, A_WV)
            return q, sq, k, sk, v, sv
        r0 = cocotb.start_soon(proj(node0))
        r1 = cocotb.start_soon(proj(node1))
        q0, sq0, k0, sk0, v0, sv0 = await r0
        q1, sq1, k1, sk1, v1, sv1 = await r1

        Q_i8, sQ = gather([(q0, sq0), (q1, sq1)])      # 64
        K_i8, sK = gather([(k0, sk0), (k1, sk1)])      # 32
        V_i8, sV = gather([(v0, sv0), (v1, sv1)])      # 32

        fr, fi = freq_cis(pos)
        Qr = rope_vec(deq(Q_i8, sQ), H, fr, fi)
        Kr = rope_vec(deq(K_i8, sK), KH, fr, fi)
        Vf = deq(V_i8, sV)
        cache_K.append(Kr)
        cache_V.append(Vf)

        T = pos + 1
        Qr_i8, sQr = to_i8_shift(Qr)
        Ksend_i8, sKs = to_i8_shift([v for p in range(T) for v in cache_K[p]])
        Vsend_i8, sVs = to_i8_shift([v for p in range(T) for v in cache_V[p]])
        attn, sa = await node0.mm(Qr_i8, Ksend_i8, Vsend_i8, sQr, sKs, sVs, T)

        w0 = cocotb.start_soon(node0.fq(WO_PN, sa, swo, attn, A_WO))
        w1 = cocotb.start_soon(node1.fq(WO_PN, sa, swo, attn, A_WO))
        o0, so0 = await w0
        o1, so1 = await w1
        out = deq(o0, so0) + deq(o1, so1)
        return [x_f[d] + out[d] for d in range(D)]

    async def ffn_block(x_f):
        x_i8, sx = to_i8_shift(x_f)
        xn_i8, sxn = await node0.fn(x_i8, sx, s_rmsf, A_RMS_FFN)

        async def w1w3(node):
            h1, s1 = await node.fq(HID_PN, sxn, sw1, xn_i8, A_W1)
            h3, s3 = await node.fq(HID_PN, sxn, sw3, xn_i8, A_W3)
            return h1, s1, h3, s3
        r0 = cocotb.start_soon(w1w3(node0))
        r1 = cocotb.start_soon(w1w3(node1))
        h1_0, s10, h3_0, s30 = await r0
        h1_1, s11, h3_1, s31 = await r1

        ts = [cocotb.start_soon(node0.ss(h1_0, s10)), cocotb.start_soon(node1.ss(h1_1, s11))]
        (sg0, ss0), (sg1, ss1) = [await x for x in ts]
        hg0_i8, shg0 = to_i8_shift([a*b for a, b in zip(deq(sg0, ss0), deq(h3_0, s30))])
        hg1_i8, shg1 = to_i8_shift([a*b for a, b in zip(deq(sg1, ss1), deq(h3_1, s31))])

        tw = [cocotb.start_soon(node0.fq(D, shg0, sw2, hg0_i8, A_W2)),
              cocotb.start_soon(node1.fq(D, shg1, sw2, hg1_i8, A_W2))]
        (p0, sp0), (p1, sp1) = [await x for x in tw]
        out = [deq(p0, sp0)[d] + deq(p1, sp1)[d] for d in range(D)]
        return [x_f[d] + out[d] for d in range(D)]

    # ── reference (pure python, mirrors infer_v4sim one layer) ──────────────
    def ref_layer(x, pos):
        xn = rmsnorm_f(x, rms_att)
        Q = matvec_f(Wq_f, xn); Kf = matvec_f(Wk_f, xn); Vf = matvec_f(Wv_f, xn)
        fr, fi = freq_cis(pos)
        Qr = rope_vec(Q, H, fr, fi); Kr = rope_vec(Kf, KH, fr, fi)
        ref_K.append(Kr); ref_V.append(Vf)
        T = pos + 1
        out = [0.0] * D
        for h in range(H):
            kvh = h // N_REP
            qh = Qr[h*HS:(h+1)*HS]
            sc = [sum(qh[d] * ref_K[t][kvh*HS + d] for d in range(HS)) / math.sqrt(HS)
                  for t in range(T)]
            mmax = max(sc)
            e = [math.exp(s - mmax) for s in sc]
            Z = sum(e)
            a = [ei / Z for ei in e]
            for d in range(HS):
                out[h*HS + d] = sum(a[t] * ref_V[t][kvh*HS + d] for t in range(T))
        ao = matvec_f(Wo_f, out)
        x1 = [x[d] + ao[d] for d in range(D)]
        xn2 = rmsnorm_f(x1, rms_ffn)
        g = matvec_f(W1_f, xn2); u = matvec_f(W3_f, xn2)
        hg = [silu_f(g[i]) * u[i] for i in range(HIDDEN)]
        o = matvec_f(W2_f, hg)
        return [x1[d] + o[d] for d in range(D)]

    # ── run several positions through the full layer ────────────────────────
    rin = random.Random(11)
    for pos in range(N_POS):
        x_f = [rin.gauss(0, 1.0) for _ in range(D)]      # fresh input activation
        x1 = await attn_block(x_f, pos)
        y_cluster = await ffn_block(x1)
        y_ref = ref_layer(x_f, pos)
        ma = max(abs(v) for v in y_ref) or 1.0
        err = max(abs(y_cluster[d] - y_ref[d]) for d in range(D))
        max_err_seen = max(max_err_seen, err / ma)
        dut._log.info(f"  pos={pos} (T={pos+1}): full layer err={100*err/ma:.2f}% of range")
        assert err / ma < 0.30, f"pos {pos}: layer output too far from ref ({err})"

    dut._log.info(
        f"MILESTONE 4a PASS: full transformer layer tensor-parallel over "
        f"{N_POS} positions (multi-position KV cache, real MM softmax), "
        f"max err {100*max_err_seen:.2f}% vs float reference"
    )
