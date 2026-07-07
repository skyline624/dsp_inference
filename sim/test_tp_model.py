"""Milestone 4b - full REAL model forward pass, tensor-parallel.

Runs the real stories260K (5 layers, dim=64, hidden=172, vocab=512) for the
first token through the 2-node cluster and checks the argmax matches the project
reference (infer_v4sim): BOS(1) -> token 403 ('Once').

Everything heavy runs on the real RTL, distributed across the two SDRAMs/DSPs:
  - all matmuls via a generic chunked tensor-parallel helper (chunks of <=64
    round-robined across the two nodes) ;
  - attention core via the real MM command (gathered Q/K/V) ;
  - rmsnorm (FN) and silu (SS) on the real RTL.
Coordinator does only small activation glue (embedding lookup, rope, multiply,
all-reduce sums, residual, KV cache, argmax). Weights are quantized in pure
python (container has no numpy) and preloaded per layer via the SDRAM backdoor.

Scope: one forward pass (pos=0). Generating the full sentence is the same loop,
just longer sim time (~15 min/token cycle-accurate).
"""

import math
import struct
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles

CLK_NS = 37
DIV    = 27
D, K   = 64, 64
H, KH, HS = 8, 4, 8
N_REP  = H // KH
HID    = 172
VOCAB  = 512
NLAY   = 5
EXPECT_TOK0 = 403          # infer_v4sim: BOS -> 'Once'
BASE   = 0x100000


# ─── numeric helpers ────────────────────────────────────────────────────────
def to_i8_shift(x):
    m = max((abs(v) for v in x), default=0.0)
    if m == 0:
        return [0] * len(x), 0
    s = math.ceil(math.log2(m / 127.0))
    return [max(-128, min(127, round(v / (2.0 ** s)))) for v in x], s


def deq(x_i8, s):
    return [v * (2.0 ** s) for v in x_i8]


def qmat(M):
    flat = [v for row in M for v in row]
    _, s = to_i8_shift(flat)
    return [[max(-128, min(127, round(v / (2.0 ** s)))) for v in row] for row in M], s


def rmsnorm_f(x, w, eps=1e-5):
    inv = 1.0 / math.sqrt(sum(v * v for v in x) / len(x) + eps)
    return [x[i] * w[i] * inv for i in range(len(x))]


def rope_vec(v, n_heads, fr, fi):
    out = list(v)
    for h in range(n_heads):
        for j in range(HS // 2):
            r = v[h*HS + 2*j]; im = v[h*HS + 2*j + 1]
            out[h*HS + 2*j]     = r*fr[j] - im*fi[j]
            out[h*HS + 2*j + 1] = r*fi[j] + im*fr[j]
    return out


# ─── model loader (pure python) ─────────────────────────────────────────────
def load_model(path):
    with open(path, "rb") as f:
        blob = f.read()
    dim, hidden, L, nh, nkv, vocab, seqlen = struct.unpack("<7i", blob[:28])
    vocab = abs(vocab)
    fl = struct.unpack(f"<{(len(blob)-28)//4}f", blob[28:])
    p = [0]
    def take(*shape):
        n = 1
        for s in shape:
            n *= s
        chunk = fl[p[0]:p[0]+n]; p[0] += n
        if len(shape) == 1:
            return list(chunk)
        if len(shape) == 2:
            r, c = shape
            return [list(chunk[i*c:(i+1)*c]) for i in range(r)]
        a, r, c = shape
        return [[list(chunk[(x*r + i)*c:(x*r + i)*c + c]) for i in range(r)] for x in range(a)]
    m = {}
    m['tok_emb'] = take(vocab, dim)
    m['rms_att'] = take(L, dim)
    m['wq'] = take(L, H*HS, dim); m['wk'] = take(L, KH*HS, dim); m['wv'] = take(L, KH*HS, dim)
    m['wo'] = take(L, dim, H*HS)
    m['rms_ffn'] = take(L, dim)
    m['w1'] = take(L, hidden, dim); m['w2'] = take(L, dim, hidden); m['w3'] = take(L, hidden, dim)
    m['rms_final'] = take(dim)
    m['fcr'] = take(seqlen, HS//2); m['fci'] = take(seqlen, HS//2)
    return m


# ─── UART / node ────────────────────────────────────────────────────────────
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
        self.sink = bytearray(); self.off = 0
        self.next = BASE
        cocotb.start_soon(uart_rx_monitor(clk, tx_sig, self.sink))

    def alloc(self, nbytes):
        a = self.next
        self.next += (nbytes + 3) & ~3
        return a

    def reset_alloc(self):
        self.next = BASE

    def preload(self, base_addr, data):
        wbase = base_addr >> 2
        b = bytes(data)
        if len(b) % 4:
            b += bytes(4 - len(b) % 4)
        for w in range(len(b) // 4):
            self.sdram.mem[wbase + w].value = (
                b[4*w] | (b[4*w+1] << 8) | (b[4*w+2] << 16) | (b[4*w+3] << 24))

    async def xfer(self, pkt, nresp, timeout_cycles=2_000_000):
        await uart_send(self.clk, self.rx, pkt)
        waited = 0
        while len(self.sink) < self.off + nresp:
            await ClockCycles(self.clk, 20)
            waited += 20
            assert waited < timeout_cycles, f"timeout {len(self.sink)-self.off}/{nresp}"
        resp = bytes(self.sink[self.off:self.off + nresp]); self.off += nresp
        return resp

    async def fn(self, x_i8, sx, sw, addr):
        r = await self.xfer(b"FN" + bytes([sx & 0xFF, sw & 0xFF]) + bb(x_i8) + addr_bytes(addr), 75)
        assert r[:2] == b"FK"
        return [i8(r[11 + i]) for i in range(D)], i8(r[2])

    async def fq(self, N, sx, sw, x_i8, addr):
        r = await self.xfer(b"FQ" + bytes([N, sx & 0xFF, sw & 0xFF]) + bb(x_i8) + addr_bytes(addr), 3 + N)
        assert r[:2] == b"FQ"
        return [i8(r[3 + i]) for i in range(N)], i8(r[2])

    async def ss(self, x_i8, sx):
        full = list(x_i8) + [0] * (D - len(x_i8))
        r = await self.xfer(b"SS" + bytes([sx & 0xFF]) + bb(full), 70)
        assert r[:2] == b"SK"
        return [i8(r[6 + i]) for i in range(len(x_i8))], i8(r[2])

    async def mm(self, Q, Kc, Vc, sq, sk, sv, T):
        r = await self.xfer(b"MM" + bytes([sq & 0xFF, sk & 0xFF, sv & 0xFF, T]) + bb(Q) + bb(Kc) + bb(Vc), 67)
        assert r[:2] == b"MK"
        return [i8(r[3 + i]) for i in range(D)], sv


async def boot_node(clk, fpga):
    for _ in range(40000):
        await RisingEdge(clk)
        if _as_int(fpga.rst_n) == 1:
            break
    for _ in range(40000):
        await RisingEdge(clk)
        if _as_int(fpga.sd_busy) == 0:
            break


@cocotb.test()
async def test_tp_model(dut):
    dut.rx0.value = 1; dut.rx1.value = 1
    cocotb.start_soon(Clock(dut.clk, CLK_NS, units="ns").start())
    await cocotb.start_soon(boot_node(dut.clk, dut.n0.u_fpga))
    await cocotb.start_soon(boot_node(dut.clk, dut.n1.u_fpga))
    n0 = Node(dut.clk, dut.rx0, dut.tx0, dut.n0.u_fpga, dut.n0.u_sdram)
    n1 = Node(dut.clk, dut.rx1, dut.tx1, dut.n1.u_fpga, dut.n1.u_sdram)
    nodes = [n0, n1]
    dut._log.info("both nodes booted, SDRAM ready")

    m = load_model("models/stories260K.bin") if False else load_model(
        "../host/models/stories260K.bin")
    dut._log.info("model loaded (5 layers, hidden=172, vocab=512)")

    # ── generic chunked tensor-parallel matmul helpers ──────────────────────
    def register_row(W_i8, sw):
        """Row-chunk W[N][64] into <=64-row blocks, round-robin to nodes."""
        N = len(W_i8); chunks = []; pos = 0; ci = 0
        while pos < N:
            n = min(64, N - pos)
            node = nodes[ci % 2]
            addr = node.alloc(n * K)
            node.preload(addr, bb([v for r in range(pos, pos + n) for v in W_i8[r]]))
            chunks.append((ci, node, addr, n)); pos += n; ci += 1
        return {"chunks": chunks, "sw": sw, "N": N}

    def register_col(W_i8, sw):
        """Col-chunk W[D][Kt] into <=64-wide blocks (padded), round-robin."""
        Kt = len(W_i8[0]); chunks = []; k = 0; ci = 0
        while k < Kt:
            kc = min(64, Kt - k)
            node = nodes[ci % 2]
            addr = node.alloc(D * 64)
            block = []
            for n in range(D):
                block += [W_i8[n][k + j] if j < kc else 0 for j in range(64)]
            node.preload(addr, bb(block))
            chunks.append((ci, node, addr, k, kc)); k += kc; ci += 1
        return {"chunks": chunks, "sw": sw, "Kt": Kt}

    async def tp_row(spec, x_i8, sx):
        sw = spec["sw"]
        async def run(node, items):
            out = {}
            for (ci, addr, n) in items:
                out[ci] = await node.fq(n, sx, sw, x_i8, addr)
            return out
        by = {0: [], 1: []}
        for (ci, node, addr, n) in spec["chunks"]:
            by[nodes.index(node)].append((ci, addr, n))
        t0 = cocotb.start_soon(run(n0, by[0]))
        t1 = cocotb.start_soon(run(n1, by[1]))
        r0 = await t0; r1 = await t1
        res = {**r0, **r1}
        real = []
        for ci in sorted(res):
            y, s = res[ci]
            real += deq(y, s)
        return to_i8_shift(real)

    async def tp_col(spec, x_full_i8, sx):
        sw = spec["sw"]
        async def run(node, items):
            out = []
            for (ci, addr, k0, kc) in items:
                xc = list(x_full_i8[k0:k0 + kc]) + [0] * (64 - kc)
                out.append(await node.fq(D, sx, sw, xc, addr))
            return out
        by = {0: [], 1: []}
        for (ci, node, addr, k0, kc) in spec["chunks"]:
            by[nodes.index(node)].append((ci, addr, k0, kc))
        t0 = cocotb.start_soon(run(n0, by[0]))
        t1 = cocotb.start_soon(run(n1, by[1]))
        parts = (await t0) + (await t1)
        acc = [0.0] * D
        for (y, s) in parts:
            for d in range(D):
                acc[d] += y[d] * (2.0 ** s)
        return to_i8_shift(acc)

    async def silu_full(h_i8, sh):
        real = []
        pos = 0
        while pos < len(h_i8):
            n = min(64, len(h_i8) - pos)
            y, s = await n0.ss(h_i8[pos:pos + n], sh)
            real += deq(y, s); pos += n
        return to_i8_shift(real)

    # ── forward, pos = 0, token = BOS(1) ────────────────────────────────────
    pos = 0
    tok = 1
    x = list(m['tok_emb'][tok])     # embedding lookup (float)
    cache_K, cache_V = [[] for _ in range(NLAY)], [[] for _ in range(NLAY)]

    for l in range(NLAY):
        n0.reset_alloc(); n1.reset_alloc()
        # preload attention weights for this layer
        rmsa_i8, s_rmsa = to_i8_shift(m['rms_att'][l])
        a_rmsa = n0.alloc(D); n0.preload(a_rmsa, bb(rmsa_i8))
        wq = register_row(*qmat(m['wq'][l]))
        wk = register_row(*qmat(m['wk'][l]))
        wv = register_row(*qmat(m['wv'][l]))
        wo = register_row(*qmat(m['wo'][l]))

        x_i8, sx = to_i8_shift(x)
        xn_i8, sxn = await n0.fn(x_i8, sx, s_rmsa, a_rmsa)
        Q_i8, sQ = await tp_row(wq, xn_i8, sxn)
        Kk_i8, sK = await tp_row(wk, xn_i8, sxn)
        V_i8, sV = await tp_row(wv, xn_i8, sxn)

        fr, fi = m['fcr'][pos], m['fci'][pos]
        Qr = rope_vec(deq(Q_i8, sQ), H, fr, fi)
        Kr = rope_vec(deq(Kk_i8, sK), KH, fr, fi)
        cache_K[l].append(Kr); cache_V[l].append(deq(V_i8, sV))
        T = pos + 1
        Qr_i8, sQr = to_i8_shift(Qr)
        Ks_i8, sKs = to_i8_shift([v for p in range(T) for v in cache_K[l][p]])
        Vs_i8, sVs = to_i8_shift([v for p in range(T) for v in cache_V[l][p]])
        attn, sa = await n0.mm(Qr_i8, Ks_i8, Vs_i8, sQr, sKs, sVs, T)
        ao_i8, sao = await tp_row(wo, attn, sa)
        x = [x[d] + deq(ao_i8, sao)[d] for d in range(D)]

        # FFN
        n0.reset_alloc(); n1.reset_alloc()
        rmsf_i8, s_rmsf = to_i8_shift(m['rms_ffn'][l])
        a_rmsf = n0.alloc(D); n0.preload(a_rmsf, bb(rmsf_i8))
        w1 = register_row(*qmat(m['w1'][l]))
        w3 = register_row(*qmat(m['w3'][l]))
        w2 = register_col(*qmat(m['w2'][l]))

        x_i8, sx = to_i8_shift(x)
        xn_i8, sxn = await n0.fn(x_i8, sx, s_rmsf, a_rmsf)
        h1_i8, sh1 = await tp_row(w1, xn_i8, sxn)
        h3_i8, sh3 = await tp_row(w3, xn_i8, sxn)
        sg_i8, ssg = await silu_full(h1_i8, sh1)
        hg_i8, shg = to_i8_shift([a*b for a, b in zip(deq(sg_i8, ssg), deq(h3_i8, sh3))])
        o_i8, so = await tp_col(w2, hg_i8, shg)
        x = [x[d] + deq(o_i8, so)[d] for d in range(D)]
        dut._log.info(f"  layer {l} done")

    # final rmsnorm + lm_head (shared tok_emb)
    n0.reset_alloc(); n1.reset_alloc()
    rmsfin_i8, s_rmsfin = to_i8_shift(m['rms_final'])
    a_fin = n0.alloc(D); n0.preload(a_fin, bb(rmsfin_i8))
    emb = register_row(*qmat(m['tok_emb']))     # [512][64]
    x_i8, sx = to_i8_shift(x)
    xn_i8, sxn = await n0.fn(x_i8, sx, s_rmsfin, a_fin)
    logits_i8, slog = await tp_row(emb, xn_i8, sxn)
    logits = deq(logits_i8, slog)
    tok0 = max(range(VOCAB), key=lambda i: logits[i])
    top5 = sorted(range(VOCAB), key=lambda i: logits[i], reverse=True)[:5]
    dut._log.info(f"  argmax token = {tok0} (expected {EXPECT_TOK0}); top5 = {top5}")

    assert tok0 == EXPECT_TOK0, f"first token {tok0} != reference {EXPECT_TOK0}"
    dut._log.info(
        f"MILESTONE 4b PASS: real stories260K (5 layers) forward on the cluster "
        f"-> first token {tok0} = 'Once', matches infer_v4sim"
    )
