"""Cluster + SD boot: each node loads ITS slice from ITS OWN SD, then TP-FFN.

2 nodes; each SD card holds only that node's weight slice (compact layout
rms@0, W1@0x1000, W3@0x2000, W2@0x3000). At power-on each node boots its slice
SD->SDRAM (in parallel), then the FFN-tensor-parallel coordinator runs the FFN.
Validated within tolerance vs the full float FFN. The model is DISTRIBUTED across
the two SD cards; zero PC, zero backdoor SDRAM load.
"""

import math
import random
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles

CLK_NS = 37
D = 64
HID = 128


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


@cocotb.test()
async def test_ffn_tp_sd(dut):
    cocotb.start_soon(Clock(dut.clk, CLK_NS, units="ns").start())
    dut.rst_n.value = 0; dut.start.value = 0
    for s in ("x_in","sx","sw_rms","sw1","sw3","sw2"):
        getattr(dut, s).value = 0

    rng = random.Random(41)
    x_f   = [rng.gauss(0, 1.0) for _ in range(D)]
    rms_w = [1.0]*D
    W1_f  = [[rng.gauss(0, 0.1) for _ in range(D)] for _ in range(HID)]
    W3_f  = [[rng.gauss(0, 0.1) for _ in range(D)] for _ in range(HID)]
    W2_f  = [[rng.gauss(0, 0.1) for _ in range(HID)] for _ in range(D)]

    x_i8, sx    = to_i8_shift(x_f)
    rms_i8, swr = to_i8_shift(rms_w)
    W1_i8, sw1  = quantize_matrix(W1_f)
    W3_i8, sw3  = quantize_matrix(W3_f)
    W2_i8, sw2  = quantize_matrix(W2_f)

    # per-node SD image (compact: rms@0, W1@0x1000, W3@0x2000, W2@0x3000)
    def build_img(g):
        img = bytearray(32 * 512)
        for i in range(D): img[i] = rms_i8[i] & 0xFF
        for r in range(64):
            for k in range(64):
                img[0x1000 + r*64 + k] = W1_i8[g*64 + r][k] & 0xFF
                img[0x2000 + r*64 + k] = W3_i8[g*64 + r][k] & 0xFF
                img[0x3000 + r*64 + k] = W2_i8[r][g*64 + k] & 0xFF
        return img

    for card, g in ((dut.u_n0.u_card, 0), (dut.u_n1.u_card, 1)):
        img = build_img(g)
        for i in range(len(img)):
            card.mem[i].value = img[i]

    await ClockCycles(dut.clk, 10)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 5)
    dut.x_in.value = vec_to_int([v & 0xFF for v in x_i8])
    dut.sx.value = sx & 0xFF; dut.sw_rms.value = swr & 0xFF
    dut.sw1.value = sw1 & 0xFF; dut.sw3.value = sw3 & 0xFF; dut.sw2.value = sw2 & 0xFF

    dut.start.value = 1
    await RisingEdge(dut.clk); dut.start.value = 0

    done_flag = {"v": False}
    async def watch_done():
        await RisingEdge(dut.done); done_flag["v"] = True
    cocotb.start_soon(watch_done())

    b0 = b1 = False
    dumped = False
    fin = False
    last = None
    for i in range(200):
        await ClockCycles(dut.clk, 10000)
        if not b0 and int(dut.u_n0.u_fpga.boot_done.value) == 1:
            b0 = True; dut._log.info(f"  node0 boot_done @ ~{(i+1)*10000} cyc")
        if not b1 and int(dut.u_n1.u_fpga.boot_done.value) == 1:
            b1 = True; dut._log.info(f"  node1 boot_done @ ~{(i+1)*10000} cyc")
        if b0 and b1 and not dumped:
            dumped = True
            for nd, u in (("n0", dut.u_n0), ("n1", dut.u_n1)):
                mem = u.u_sdram.mem
                xwords = [idx for idx in range(4200) if 'x' in mem[idx].value.binstr or 'X' in mem[idx].value.binstr]
                ndef = 4200 - len(xwords)
                holes = []
                if xwords:
                    s = xwords[0]; p = s
                    for x in xwords[1:]:
                        if x == p+1: p = x
                        else: holes.append((s,p)); s = x; p = x
                    holes.append((s,p))
                dut._log.info(f"  {nd} SDRAM: {ndef}/4200 words defined, {len(xwords)} X. X-ranges={holes[:6]}")
            for nd, u in (("n0", dut.u_n0), ("n1", dut.u_n1)):
                try:
                    nwe = int(u.u_fpga.u_boot.dbg_nwe.value)
                    nwr = int(u.u_fpga.u_boot.dbg_nwr.value)
                    dut._log.info(f"  {nd} boot: SD_bytes={nwe}  SDRAM_writes={nwr}  (expect 16384 / 16384)")
                except Exception as e:
                    dut._log.info(f"  {nd} boot counters unavailable: {e}")
        try:
            cur = (int(dut.u_seq.st.value), int(dut.u_seq.step.value), int(dut.u_seq.tgt.value))
        except Exception:
            cur = None
        if cur is not None and cur != last:
            dut._log.info(f"  seq st={cur[0]} step={cur[1]} tgt={cur[2]} @ ~{(i+1)*10000} cyc")
            last = cur
        if done_flag["v"]:
            fin = True; dut._log.info(f"  done pulse caught @ ~{(i+1)*10000} cyc"); break
    if not fin:
        dut._log.info(f"  TIMEOUT: node0={b0} node1={b1} ffn_done=0 last_seq={last}")
        assert False, "cluster+SD never finished"

    rbin = dut.result.value.binstr; shbin = dut.result_sh.value.binstr
    dut._log.info(f"  result.binstr = {rbin}")
    dut._log.info(f"  result_sh.binstr = {shbin}")
    nx = rbin.count('x') + rbin.count('X')
    nxs = shbin.count('x') + shbin.count('X')
    dut._log.info(f"  X-counts: result={nx}/{len(rbin)}  result_sh={nxs}/{len(shbin)}")
    if nx or nxs:
        assert False, f"result has X bits (result={nx}, result_sh={nxs}) — datapath X-propagation"
    rv = int(dut.result.value); rsh = to_i8(int(dut.result_sh.value))
    got = [to_i8((rv >> (i*8)) & 0xFF) * (2.0**rsh) for i in range(D)]

    xr = deq(x_i8, sx); rmsr = deq(rms_i8, swr)
    W1r = [[W1_i8[r][k]*(2.0**sw1) for k in range(D)] for r in range(HID)]
    W3r = [[W3_i8[r][k]*(2.0**sw3) for k in range(D)] for r in range(HID)]
    W2r = [[W2_i8[r][k]*(2.0**sw2) for k in range(HID)] for r in range(D)]
    xn = rmsnorm_f(xr, rmsr)
    h1 = matvec_f(W1r, xn); h3 = matvec_f(W3r, xn)
    hg = [silu_f(h1[i])*h3[i] for i in range(HID)]
    o  = matvec_f(W2r, hg)
    ref = [xr[d] + o[d] for d in range(D)]

    ma = max(abs(v) for v in ref) or 1.0
    err = max(abs(got[i]-ref[i]) for i in range(D))
    dut._log.info(f"  cluster+SD TP-FFN: max_err={err:.4f} ({100*err/ma:.1f}% of range)")
    assert err/ma < 0.35, f"output too far from reference ({err})"

    dut._log.info(
        "CLUSTER+SD PASS: each node booted ITS slice from ITS OWN SD card, then ran "
        "the FFN tensor-parallel across the 2 nodes. Model distributed on 2 SDs, zero PC."
    )
