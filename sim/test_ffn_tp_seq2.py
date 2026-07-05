"""Phase-0 gate : autonomous TENSOR-PARALLEL FFN over 2 brick nodes, LUT-lean seq.

Identical scenario to test_ffn_tp.py (which validates ffn_tp_seq v1) but the DUT is
ffn_tp_seq2_top -> ffn_tp_seq2 (register-file / vec_alu2 variant, the 5-juillet
refactor). Same float FFN reference, same 0.35 tolerance. Proves the LUT-lean
sequencer computes correctly, not only that it routes.

  make TOPLEVEL=ffn_tp_seq2_top MODULE=test_ffn_tp_seq2
"""

import math
import random
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles

CLK_NS = 37
D = 64
K = 64
HID = 128
A_RMS, A_W1, A_W3, A_W2 = 0x100000, 0x101000, 0x102000, 0x103000


def to_i8(b): return b - 256 if b >= 128 else b


def to_i8_shift(x):
    m = max((abs(v) for v in x), default=0.0)
    if m == 0:
        return [0]*len(x), 0
    s = math.ceil(math.log2(m/127.0))
    return [max(-128, min(127, round(v/(2.0**s)))) for v in x], s


def quantize_matrix(M):
    flat = [v for row in M for v in row]
    _, s = to_i8_shift(flat)
    return [[max(-128, min(127, round(v/(2.0**s)))) for v in row] for row in M], s


def deq(i8, s): return [v*(2.0**s) for v in i8]
def rmsnorm_f(x, w, eps=1e-5):
    inv = 1.0/math.sqrt(sum(v*v for v in x)/len(x)+eps); return [x[i]*w[i]*inv for i in range(len(x))]
def silu_f(v): return v/(1.0+math.exp(-v))
def matvec_f(Wm, x): return [sum(Wm[r][k]*x[k] for k in range(len(x))) for r in range(len(Wm))]


def vec_to_int(bs):
    v = 0
    for i, b in enumerate(bs):
        v |= (b & 0xFF) << (i*8)
    return v


def preload(sd, base, data):
    wbase = base >> 2
    b = bytes(data)
    if len(b) % 4:
        b += bytes(4 - len(b) % 4)
    for w in range(len(b)//4):
        sd.mem[wbase+w].value = b[4*w] | (b[4*w+1]<<8) | (b[4*w+2]<<16) | (b[4*w+3]<<24)


@cocotb.test()
async def test_ffn_tp_seq2(dut):
    cocotb.start_soon(Clock(dut.clk, CLK_NS, units="ns").start())
    dut.rst_n.value = 0; dut.start.value = 0
    for s in ("x_in", "sx", "sw_rms", "sw1", "sw3", "sw2"):
        getattr(dut, s).value = 0
    await ClockCycles(dut.clk, 10)

    rng = random.Random(41)
    x_f   = [rng.gauss(0, 1.0) for _ in range(D)]
    rms_w = [1.0]*D
    W1_f  = [[rng.gauss(0, 0.1) for _ in range(K)] for _ in range(HID)]
    W3_f  = [[rng.gauss(0, 0.1) for _ in range(K)] for _ in range(HID)]
    W2_f  = [[rng.gauss(0, 0.1) for _ in range(HID)] for _ in range(D)]

    x_i8, sx    = to_i8_shift(x_f)
    rms_i8, swr = to_i8_shift(rms_w)
    W1_i8, sw1  = quantize_matrix(W1_f)
    W3_i8, sw3  = quantize_matrix(W3_f)
    W2_i8, sw2  = quantize_matrix(W2_f)

    H = HID // 2
    n0, n1 = dut.u_n0.u_sdram, dut.u_n1.u_sdram
    # node0 : rms, W1 rows[0:64], W3 rows[0:64], W2 cols[0:64]
    preload(n0, A_RMS, bytes((v & 0xFF) for v in rms_i8))
    preload(n0, A_W1, bytes((W1_i8[r][k] & 0xFF) for r in range(0, H) for k in range(K)))
    preload(n0, A_W3, bytes((W3_i8[r][k] & 0xFF) for r in range(0, H) for k in range(K)))
    preload(n0, A_W2, bytes((W2_i8[r][k] & 0xFF) for r in range(D) for k in range(0, H)))
    # node1 : W1 rows[64:128], W3 rows[64:128], W2 cols[64:128]
    preload(n1, A_W1, bytes((W1_i8[r][k] & 0xFF) for r in range(H, HID) for k in range(K)))
    preload(n1, A_W3, bytes((W3_i8[r][k] & 0xFF) for r in range(H, HID) for k in range(K)))
    preload(n1, A_W2, bytes((W2_i8[r][k] & 0xFF) for r in range(D) for k in range(H, HID)))

    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 5)
    dut.x_in.value = vec_to_int([v & 0xFF for v in x_i8])
    dut.sx.value = sx & 0xFF; dut.sw_rms.value = swr & 0xFF
    dut.sw1.value = sw1 & 0xFF; dut.sw3.value = sw3 & 0xFF; dut.sw2.value = sw2 & 0xFF

    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    # ffn_tp_seq2 adds BOOT=40000 startup + serialized vec_alu2 passes -> more cycles.
    for _ in range(2000000):
        await RisingEdge(dut.clk)
        if int(dut.done.value) == 1:
            break
    else:
        assert False, "ffn_tp_seq2 never finished"

    rv = int(dut.result.value); rsh = to_i8(int(dut.result_sh.value))
    got = [to_i8((rv >> (i*8)) & 0xFF) * (2.0**rsh) for i in range(D)]

    xr = deq(x_i8, sx); rmsr = deq(rms_i8, swr)
    W1r = [[W1_i8[r][k]*(2.0**sw1) for k in range(K)] for r in range(HID)]
    W3r = [[W3_i8[r][k]*(2.0**sw3) for k in range(K)] for r in range(HID)]
    W2r = [[W2_i8[r][k]*(2.0**sw2) for k in range(HID)] for r in range(D)]
    xn = rmsnorm_f(xr, rmsr)
    h1 = matvec_f(W1r, xn); h3 = matvec_f(W3r, xn)
    hg = [silu_f(h1[i])*h3[i] for i in range(HID)]
    o  = matvec_f(W2r, hg)
    ref = [xr[d] + o[d] for d in range(D)]

    ma = max(abs(v) for v in ref) or 1.0
    err = max(abs(got[i]-ref[i]) for i in range(D))
    dut._log.info(f"  autonomous TP-FFN v2 (2 nodes): max_err={err:.4f} ({100*err/ma:.1f}% of range)")
    assert err/ma < 0.35, f"TP-FFN v2 output too far from reference ({err})"

    dut._log.info(
        "PHASE-0 GATE PASS: ffn_tp_seq2 (LUT-lean) splits the FFN across 2 nodes "
        "(W1/W3 row-parallel, W2 col-parallel + all-reduce), matches float FFN, zero PC."
    )
