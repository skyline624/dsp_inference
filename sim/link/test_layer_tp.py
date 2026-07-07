"""COMPLETE autonomous tensor-parallel transformer LAYER over 2 nodes (no PC).

The layer controller runs attn_tp_seq then ffn_tp_seq (handing x1 forward),
each splitting its block across the 2 nodes. attention weights at 0x10xxxx, FFN
weights at 0x11xxxx. Validated within tolerance vs the float layer reference
(attention pos=0/T=1 + FFN).
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
NN = int(os.environ.get("NN", "2"))   # match -Player_tp_top.NN
HID = NN * D
# attention addrs
A_RMS_A, A_WQ, A_WK, A_WV, A_WO = 0x100000, 0x101000, 0x102000, 0x103000, 0x104000
# ffn addrs
A_RMS_F, A_W1, A_W3, A_W2 = 0x110000, 0x111000, 0x112000, 0x113000


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


@cocotb.test()
async def test_layer_tp(dut):
    cocotb.start_soon(Clock(dut.clk, CLK_NS, units="ns").start())
    dut.rst_n.value = 0; dut.start.value = 0
    for s in ("x_in","sx","sw_rms_a","swq","swk","swv","swo","sw_rms_f","sw1","sw3","sw2"):
        getattr(dut, s).value = 0
    await ClockCycles(dut.clk, 10)

    rng = random.Random(61)
    x_f    = [rng.gauss(0, 1.0) for _ in range(D)]
    rms_a  = [1.0]*D; rms_f = [1.0]*D
    Wq_f = [[rng.gauss(0,0.1) for _ in range(D)] for _ in range(H*HS)]
    Wk_f = [[rng.gauss(0,0.1) for _ in range(D)] for _ in range(KH*HS)]
    Wv_f = [[rng.gauss(0,0.1) for _ in range(D)] for _ in range(KH*HS)]
    Wo_f = [[rng.gauss(0,0.1) for _ in range(H*HS)] for _ in range(D)]
    W1_f = [[rng.gauss(0,0.1) for _ in range(D)] for _ in range(HID)]
    W3_f = [[rng.gauss(0,0.1) for _ in range(D)] for _ in range(HID)]
    W2_f = [[rng.gauss(0,0.1) for _ in range(HID)] for _ in range(D)]

    x_i8, sx     = to_i8_shift(x_f)
    rma_i8, swra = to_i8_shift(rms_a); rmf_i8, swrf = to_i8_shift(rms_f)
    Wq_i8, swq = quantize_matrix(Wq_f); Wk_i8, swk = quantize_matrix(Wk_f)
    Wv_i8, swv = quantize_matrix(Wv_f); Wo_i8, swo = quantize_matrix(Wo_f)
    W1_i8, sw1 = quantize_matrix(W1_f); W3_i8, sw3 = quantize_matrix(W3_f)
    W2_i8, sw2 = quantize_matrix(W2_f)

    def sdram_of(c):
        node = dut.nodes[c]
        for h in (getattr(getattr(node, "u_n", node), "u_sdram", None), getattr(node, "u_sdram", None)):
            if h is not None and hasattr(h, "mem"):
                return h
        return node.u_n.u_sdram
    sd = [sdram_of(c) for c in range(NN)]

    QPN, KPN, OPN = 64 // NN, 32 // NN, 64 // NN   # attention rows per node
    Dh = D                                          # ffn hidden chunk = D rows/node
    preload(sd[0], A_RMS_A, bytes((v & 0xFF) for v in rma_i8))
    preload(sd[0], A_RMS_F, bytes((v & 0xFF) for v in rmf_i8))
    for c in range(NN):
        # attention slices
        preload(sd[c], A_WQ, rowsb(Wq_i8, c*QPN, (c+1)*QPN, D))
        preload(sd[c], A_WK, rowsb(Wk_i8, c*KPN, (c+1)*KPN, D))
        preload(sd[c], A_WV, rowsb(Wv_i8, c*KPN, (c+1)*KPN, D))
        preload(sd[c], A_WO, rowsb(Wo_i8, c*OPN, (c+1)*OPN, H*HS))
        # ffn slices : W1/W3 rows[c*D:(c+1)*D], W2 cols[c*D:(c+1)*D]
        preload(sd[c], A_W1, rowsb(W1_i8, c*Dh, (c+1)*Dh, D))
        preload(sd[c], A_W3, rowsb(W3_i8, c*Dh, (c+1)*Dh, D))
        preload(sd[c], A_W2, bytes((W2_i8[r][k] & 0xFF) for r in range(D) for k in range(c*Dh, (c+1)*Dh)))

    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 5)
    dut.x_in.value = vec_to_int([v & 0xFF for v in x_i8])
    dut.sx.value = sx & 0xFF
    dut.sw_rms_a.value = swra & 0xFF; dut.swq.value = swq & 0xFF; dut.swk.value = swk & 0xFF
    dut.swv.value = swv & 0xFF; dut.swo.value = swo & 0xFF
    dut.sw_rms_f.value = swrf & 0xFF; dut.sw1.value = sw1 & 0xFF
    dut.sw3.value = sw3 & 0xFF; dut.sw2.value = sw2 & 0xFF

    dut.start.value = 1
    await RisingEdge(dut.clk); dut.start.value = 0

    for _ in range(1500000):
        await RisingEdge(dut.clk)
        if int(dut.done.value) == 1:
            break
    else:
        assert False, "layer never finished"

    rv = int(dut.result.value); rsh = to_i8(int(dut.result_sh.value))
    got = [to_i8((rv >> (i*8)) & 0xFF) * (2.0**rsh) for i in range(D)]

    # float reference : full layer (attention pos=0 + FFN)
    xr = deq(x_i8, sx)
    Wvr = [[Wv_i8[r][k]*(2.0**swv) for k in range(D)] for r in range(KH*HS)]
    Wor = [[Wo_i8[r][k]*(2.0**swo) for k in range(H*HS)] for r in range(D)]
    xn_a = rmsnorm_f(xr, deq(rma_i8, swra))
    V = matvec_f(Wvr, xn_a)
    attn = [V[(h//NREP)*HS + d] for h in range(H) for d in range(HS)]
    x1 = [xr[i] + matvec_f(Wor, attn)[i] for i in range(D)]
    W1r = [[W1_i8[r][k]*(2.0**sw1) for k in range(D)] for r in range(HID)]
    W3r = [[W3_i8[r][k]*(2.0**sw3) for k in range(D)] for r in range(HID)]
    W2r = [[W2_i8[r][k]*(2.0**sw2) for k in range(HID)] for r in range(D)]
    xn_f = rmsnorm_f(x1, deq(rmf_i8, swrf))
    h1 = matvec_f(W1r, xn_f); h3 = matvec_f(W3r, xn_f)
    hg = [silu_f(h1[i])*h3[i] for i in range(HID)]
    x2 = [x1[i] + matvec_f(W2r, hg)[i] for i in range(D)]

    ma = max(abs(v) for v in x2) or 1.0
    err = max(abs(got[i]-x2[i]) for i in range(D))
    dut._log.info(f"  autonomous TP LAYER (attn+FFN, NN={NN}): max_err={err:.4f} ({100*err/ma:.1f}% of range)")
    assert err/ma < 0.40, f"layer output too far from reference ({err})"

    dut._log.info(
        f"TRANSFORMER LAYER TP (autonomous, NN={NN}) PASS: attention + FFN chained, "
        "zero PC -- a full layer runs autonomously. "
        + ("mono-card." if NN == 1 else f"{NN}-node tensor-parallel.")
    )
