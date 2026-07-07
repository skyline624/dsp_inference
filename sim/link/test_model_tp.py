"""Autonomous MULTI-LAYER tensor-parallel forward pass over 2 nodes (no PC).

model_tp_top loops the full TP transformer layer (attention + FFN) over NL=2
layers, each layer's weights at SDRAM offset l*0x20000, split across the 2 nodes.
Per-layer weight shifts come in as packed buses. Validated within tolerance vs
the float NL-layer reference. One forward pass through 2 layers, entirely on-chip.
"""

import math
import os
import random
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles

CLK_NS = 37
D = 64
H, KH, HS = 8, 4, 8
NREP = H // KH
NN = int(os.environ.get("NN", "2"))   # match -Pmodel_tp_top.NN
HID = NN * D
NL = 2
STRIDE = 0x20000


def to_i8(b): return b - 256 if b >= 128 else b
def to_i8_shift(x):
    m = max((abs(v) for v in x), default=0.0)
    if m == 0: return [0]*len(x), 0
    s = math.ceil(math.log2(m/127.0))
    return [max(-128, min(127, round(v/(2.0**s)))) for v in x], s
def quantize_matrix(M):
    flat = [v for row in M for v in row]; _, s = to_i8_shift(flat)
    return [[max(-128, min(127, round(v/(2.0**s)))) for v in row] for row in M], s
def deq(i8, s): return [v*(2.0**s) for v in i8]
def deqm(Mi8, s): return [[v*(2.0**s) for v in row] for row in Mi8]
def rmsnorm_f(x, w, eps=1e-5):
    inv = 1.0/math.sqrt(sum(v*v for v in x)/len(x)+eps); return [x[i]*w[i]*inv for i in range(len(x))]
def silu_f(v): return v/(1.0+math.exp(-v))
def matvec_f(Wm, x): return [sum(Wm[r][k]*x[k] for k in range(len(x))) for r in range(len(Wm))]
def vec_to_int(bs):
    v = 0
    for i, b in enumerate(bs): v |= (b & 0xFF) << (i*8)
    return v
def preload(sd, base, data):
    wbase = base >> 2; b = bytes(data)
    if len(b) % 4: b += bytes(4 - len(b) % 4)
    for w in range(len(b)//4):
        sd.mem[wbase+w].value = b[4*w]|(b[4*w+1]<<8)|(b[4*w+2]<<16)|(b[4*w+3]<<24)
def rowsb(M, a, b, cols): return bytes((M[r][k] & 0xFF) for r in range(a, b) for k in range(cols))
def pack(lst): return sum((v & 0xFF) << (i*8) for i, v in enumerate(lst))


@cocotb.test()
async def test_model_tp(dut):
    cocotb.start_soon(Clock(dut.clk, CLK_NS, units="ns").start())
    dut.rst_n.value = 0; dut.start.value = 0
    for s in ("x_in","sx","sw_rms_a","swq","swk","swv","swo","sw_rms_f","sw1","sw3","sw2"):
        getattr(dut, s).value = 0
    await ClockCycles(dut.clk, 10)

    def sdram_of(c):
        node = dut.nodes[c]
        for h in (getattr(getattr(node, "u_n", node), "u_sdram", None), getattr(node, "u_sdram", None)):
            if h is not None and hasattr(h, "mem"):
                return h
        return node.u_n.u_sdram
    sd = [sdram_of(c) for c in range(NN)]
    QPN, KPN, OPN, Dh = 64 // NN, 32 // NN, 64 // NN, D
    sh = {k: [] for k in ('ra','q','k','v','o','rf','w1','w3','w2')}
    layers = []

    for l in range(NL):
        rng = random.Random(100 + l)
        rms_a = [1.0]*D; rms_f = [1.0]*D
        Wq = [[rng.gauss(0,0.1) for _ in range(D)] for _ in range(H*HS)]
        Wk = [[rng.gauss(0,0.1) for _ in range(D)] for _ in range(KH*HS)]
        Wv = [[rng.gauss(0,0.1) for _ in range(D)] for _ in range(KH*HS)]
        Wo = [[rng.gauss(0,0.1) for _ in range(H*HS)] for _ in range(D)]
        W1 = [[rng.gauss(0,0.1) for _ in range(D)] for _ in range(HID)]
        W3 = [[rng.gauss(0,0.1) for _ in range(D)] for _ in range(HID)]
        W2 = [[rng.gauss(0,0.1) for _ in range(HID)] for _ in range(D)]

        ra_i8, sra = to_i8_shift(rms_a); rf_i8, srf = to_i8_shift(rms_f)
        Wq_i8, sq = quantize_matrix(Wq); Wk_i8, sk = quantize_matrix(Wk)
        Wv_i8, sv = quantize_matrix(Wv); Wo_i8, so = quantize_matrix(Wo)
        W1_i8, s1 = quantize_matrix(W1); W3_i8, s3 = quantize_matrix(W3); W2_i8, s2 = quantize_matrix(W2)

        b = l * STRIDE
        # each node c holds its slice of every weight (attention 0x10xxxx, ffn 0x11xxxx)
        preload(sd[0], 0x100000+b, bytes((v & 0xFF) for v in ra_i8))
        preload(sd[0], 0x110000+b, bytes((v & 0xFF) for v in rf_i8))
        for c in range(NN):
            preload(sd[c], 0x101000+b, rowsb(Wq_i8, c*QPN, (c+1)*QPN, D))
            preload(sd[c], 0x102000+b, rowsb(Wk_i8, c*KPN, (c+1)*KPN, D))
            preload(sd[c], 0x103000+b, rowsb(Wv_i8, c*KPN, (c+1)*KPN, D))
            preload(sd[c], 0x104000+b, rowsb(Wo_i8, c*OPN, (c+1)*OPN, H*HS))
            preload(sd[c], 0x111000+b, rowsb(W1_i8, c*Dh, (c+1)*Dh, D))
            preload(sd[c], 0x112000+b, rowsb(W3_i8, c*Dh, (c+1)*Dh, D))
            preload(sd[c], 0x113000+b, bytes((W2_i8[r][k] & 0xFF) for r in range(D) for k in range(c*Dh, (c+1)*Dh)))

        for kk, vv in (('ra',sra),('q',sq),('k',sk),('v',sv),('o',so),('rf',srf),('w1',s1),('w3',s3),('w2',s2)):
            sh[kk].append(vv)
        layers.append(dict(
            rms_a=deq(ra_i8,sra), rms_f=deq(rf_i8,srf),
            Wv=deqm(Wv_i8,sv), Wo=deqm(Wo_i8,so),
            W1=deqm(W1_i8,s1), W3=deqm(W3_i8,s3), W2=deqm(W2_i8,s2)))

    rng = random.Random(7)
    x_f = [rng.gauss(0, 1.0) for _ in range(D)]
    x_i8, sx = to_i8_shift(x_f)

    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 5)
    dut.x_in.value = vec_to_int([v & 0xFF for v in x_i8]); dut.sx.value = sx & 0xFF
    dut.sw_rms_a.value = pack(sh['ra']); dut.swq.value = pack(sh['q']); dut.swk.value = pack(sh['k'])
    dut.swv.value = pack(sh['v']); dut.swo.value = pack(sh['o'])
    dut.sw_rms_f.value = pack(sh['rf']); dut.sw1.value = pack(sh['w1'])
    dut.sw3.value = pack(sh['w3']); dut.sw2.value = pack(sh['w2'])

    dut.start.value = 1
    await RisingEdge(dut.clk); dut.start.value = 0

    for _ in range(3000000):
        await RisingEdge(dut.clk)
        if int(dut.done.value) == 1:
            break
    else:
        assert False, "model never finished"

    rv = int(dut.result.value); rsh = to_i8(int(dut.result_sh.value))
    got = [to_i8((rv >> (i*8)) & 0xFF) * (2.0**rsh) for i in range(D)]

    # float reference : NL layers
    x = deq(x_i8, sx)
    for ly in layers:
        xn_a = rmsnorm_f(x, ly['rms_a'])
        V = matvec_f(ly['Wv'], xn_a)
        attn = [V[(h//NREP)*HS + d] for h in range(H) for d in range(HS)]
        x1 = [x[i] + matvec_f(ly['Wo'], attn)[i] for i in range(D)]
        xn_f = rmsnorm_f(x1, ly['rms_f'])
        h1 = matvec_f(ly['W1'], xn_f); h3 = matvec_f(ly['W3'], xn_f)
        hg = [silu_f(h1[i])*h3[i] for i in range(HID)]
        x = [x1[i] + matvec_f(ly['W2'], hg)[i] for i in range(D)]

    ma = max(abs(v) for v in x) or 1.0
    err = max(abs(got[i]-x[i]) for i in range(D))
    dut._log.info(f"  autonomous {NL}-LAYER TP forward: max_err={err:.4f} ({100*err/ma:.1f}% of range)")
    assert err/ma < 0.60, f"{NL}-layer output too far from reference ({err})"

    dut._log.info(
        f"MULTI-LAYER TP (autonomous, NN={NN}) PASS: {NL} transformer layers "
        "(attn+FFN each) chained in a loop, zero PC. "
        + ("mono-card." if NN == 1 else f"{NN}-node tensor-parallel.")
    )
