"""Step B gate : gen_seq causal attention block (vfile LUT-lean), at pos=0.

Drives x[64] + weights, runs the attention phases (FN, Wq/Wk/Wv, rope, KV write,
MM, Wo, residual) and checks result == x + attn_out against a float reference
(pos=0/T=1 : rope identity, attn = GQA V-passthrough). Same scenario as the
Phase-5b gate test_attn_causal, but gen_seq ALSO adds the residual.

  make -f Makefile.gen MODULE=test_gen_B
"""
import math
import random
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles

D = 64
H, KH, HS = 8, 4, 8
NREP = H // KH
A_RMS, A_WQ, A_WK, A_WV, A_WO = 0x100000, 0x101000, 0x102000, 0x103000, 0x104000


def to_i8(b): return b - 256 if b >= 128 else b
def to_i8_shift(x):
    m = max((abs(v) for v in x), default=0.0)
    if m == 0: return [0]*len(x), 0
    s = math.ceil(math.log2(m/127.0))
    return [max(-128, min(127, round(v/(2.0**s)))) for v in x], s
def qm(M):
    flat = [v for row in M for v in row]; _, s = to_i8_shift(flat)
    return [[max(-128, min(127, round(v/(2.0**s)))) for v in row] for row in M], s
def deq(i8, s): return [v*(2.0**s) for v in i8]
def rms_f(x, w, eps=1e-5):
    inv = 1.0/math.sqrt(sum(v*v for v in x)/len(x)+eps); return [x[i]*w[i]*inv for i in range(len(x))]
def mv(Wm, x): return [sum(Wm[r][k]*x[k] for k in range(len(x))) for r in range(len(Wm))]
def vint(bs):
    v = 0
    for i, b in enumerate(bs): v |= (b & 0xFF) << (i*8)
    return v
def preload(sd, base, data):
    wb = base >> 2; b = bytes(data)
    if len(b) % 4: b += bytes(4 - len(b) % 4)
    for w in range(len(b)//4):
        sd.mem[wb+w].value = b[4*w]|(b[4*w+1]<<8)|(b[4*w+2]<<16)|(b[4*w+3]<<24)
def rowsb(M, a, b, cols): return bytes((M[r][k] & 0xFF) for r in range(a, b) for k in range(cols))
def to_q15(v): return int(max(-32768, min(32767, round(v*32768.0))))


@cocotb.test()
async def test_gen_B(dut):
    cocotb.start_soon(Clock(dut.clk, 37, units="ns").start())
    dut.rst_n.value = 0; dut.start.value = 0
    for s in ("x_in","sx_in","sw_rms","swq","swk","swv","swo","pos","cos_q15","sin_q15"):
        getattr(dut, s).value = 0
    await ClockCycles(dut.clk, 10)

    rng = random.Random(77)
    x_f   = [rng.gauss(0, 1.0) for _ in range(D)]
    rms_w = [1.0]*D
    Wq = [[rng.gauss(0,0.1) for _ in range(D)] for _ in range(H*HS)]
    Wk = [[rng.gauss(0,0.1) for _ in range(D)] for _ in range(KH*HS)]
    Wv = [[rng.gauss(0,0.1) for _ in range(D)] for _ in range(KH*HS)]
    Wo = [[rng.gauss(0,0.1) for _ in range(H*HS)] for _ in range(D)]
    x_i8, sx = to_i8_shift(x_f)
    rms_i8, swr = to_i8_shift(rms_w)
    Wq_i8, swq = qm(Wq); Wk_i8, swk = qm(Wk); Wv_i8, swv = qm(Wv); Wo_i8, swo = qm(Wo)

    sd = dut.u_sdram
    preload(sd, A_RMS, bytes((v & 0xFF) for v in rms_i8))
    preload(sd, A_WQ, rowsb(Wq_i8, 0, H*HS, D))
    preload(sd, A_WK, rowsb(Wk_i8, 0, KH*HS, D))
    preload(sd, A_WV, rowsb(Wv_i8, 0, KH*HS, D))
    preload(sd, A_WO, rowsb(Wo_i8, 0, D, H*HS))

    cos = [1.0]*(HS//2); sin = [0.0]*(HS//2)   # pos=0 : rope identity

    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 5)
    dut.x_in.value = vint([v & 0xFF for v in x_i8]); dut.sx_in.value = sx & 0xFF
    dut.sw_rms.value = swr & 0xFF; dut.swq.value = swq & 0xFF; dut.swk.value = swk & 0xFF
    dut.swv.value = swv & 0xFF; dut.swo.value = swo & 0xFF
    dut.pos.value = 0
    dut.cos_q15.value = vint(b"".join(int.to_bytes(to_q15(c) & 0xFFFF, 2, "little") for c in cos))
    dut.sin_q15.value = vint(b"".join(int.to_bytes(to_q15(s) & 0xFFFF, 2, "little") for s in sin))
    dut.start.value = 1
    await RisingEdge(dut.clk); dut.start.value = 0

    for _ in range(3000000):
        await RisingEdge(dut.clk)
        if int(dut.done.value) == 1:
            break
    else:
        assert False, "gen_seq step B never finished"

    rv = int(dut.result.value); rsh = to_i8(int(dut.result_sh.value))
    got = [to_i8((rv >> (i*8)) & 0xFF) * (2.0**rsh) for i in range(D)]

    # reference : pos=0 attention + residual
    xr = deq(x_i8, sx); rmsr = deq(rms_i8, swr)
    Wvr = [[Wv_i8[r][k]*(2.0**swv) for k in range(D)] for r in range(KH*HS)]
    Wor = [[Wo_i8[r][k]*(2.0**swo) for k in range(H*HS)] for r in range(D)]
    xn = rms_f(xr, rmsr)
    V = mv(Wvr, xn)
    attn = [V[(h//NREP)*HS + d] for h in range(H) for d in range(HS)]
    out = mv(Wor, attn)
    ref = [xr[i] + out[i] for i in range(D)]     # residual included

    ma = max(abs(v) for v in ref) or 1.0
    err = max(abs(got[i]-ref[i]) for i in range(D))
    dut._log.info(f"  gen step B (attn+residual, pos=0) : max_err={err:.4f} ({100*err/ma:.1f}%)")
    dut._log.info(f"  got[:4]={[round(g,3) for g in got[:4]]} ref[:4]={[round(r,3) for r in ref[:4]]}")
    assert err/ma < 0.35, f"step B wrong ({err})"
    dut._log.info("STEP B PASS: LUT-lean causal attention block (FN/FQ/rope/KV/MM/Wo/residual).")
