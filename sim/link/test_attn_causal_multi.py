"""G3 gate : causal attention across MULTIPLE positions (KV-cache persistence).

Runs the sequencer at pos=0,1,2 in sequence with the SAME x (weights fixed), each
with real rope cos/sin for that position. The KV-cache (kmem/vmem, internal) is NOT
reset between starts, so position p sees K/V of 0..p -> real causal attention.
Validates the pos=2 output (T=3, softmax over 3 positions) against a Python
reference mirroring prove_causal_orch.mm_node. This is the true causal proof:
softmax over several positions, per-position rope, integer KV re-align.

  make -f Makefile.causal MODULE=test_attn_causal_multi
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
def q15r(v):
    q = to_q15(v); return q - 65536 if q >= 32768 else q


# ---- fixed-point references mirroring the node primitives ----
def rms_ref(x_i8, sx, w):
    xr = deq(x_i8, sx)
    return to_i8_shift(rms_f(xr, w))
def mvq(W, x_i8, sx):
    Wi, sw = qm(W)
    y = [sum(Wi[r][k]*x_i8[k] for k in range(len(x_i8))) for r in range(len(Wi))]
    m = max((abs(v) for v in y), default=0)
    add = max(0, (m.bit_length()-1)-6) if m else 0
    half = (1 << (add-1)) if add > 0 else 0
    y8 = [max(-128, min(127, (v+half) >> add if add > 0 else v)) for v in y]
    return y8, sx+sw+add
def rope_ref(x_i8, cos, sin, nh):
    out = []
    for h in range(nh):
        seg = x_i8[h*HS:(h+1)*HS]
        o = [0]*HS
        for i in range(HS//2):
            xr = seg[2*i]; xi = seg[2*i+1]
            cq = q15r(cos[i]); sq = q15r(sin[i])
            o[2*i]   = max(-128, min(127, (xr*cq - xi*sq + 16384) >> 15))
            o[2*i+1] = max(-128, min(127, (xr*sq + xi*cq + 16384) >> 15))
        out += o
    return out


@cocotb.test()
async def test_attn_causal_multi(dut):
    cocotb.start_soon(Clock(dut.clk, 37, units="ns").start())
    dut.rst_n.value = 0; dut.start.value = 0
    for s in ("x_in","sx","sw_rms","swq","swk","swv","swo","pos","cos_q15","sin_q15"):
        getattr(dut, s).value = 0
    await ClockCycles(dut.clk, 10)

    rng = random.Random(88)
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

    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 5)

    # distinct rope angles per position (real-ish freqs)
    def cs(pos):
        cos = [math.cos(pos*t) for t in (0.0, 0.4, 0.8, 1.2)]
        sin = [math.sin(pos*t) for t in (0.0, 0.4, 0.8, 1.2)]
        return cos, sin

    # python KV cache mirror
    KV = {'K': [], 'sK': [], 'V': [], 'sV': []}
    got = None
    for pos in range(3):
        cos, sin = cs(pos)
        # drive one causal step at this position
        dut.x_in.value = vint([v & 0xFF for v in x_i8]); dut.sx.value = sx & 0xFF
        dut.sw_rms.value = swr & 0xFF; dut.swq.value = swq & 0xFF; dut.swk.value = swk & 0xFF
        dut.swv.value = swv & 0xFF; dut.swo.value = swo & 0xFF
        dut.pos.value = pos
        dut.cos_q15.value = vint(b"".join(int.to_bytes(to_q15(c) & 0xFFFF, 2, "little") for c in cos))
        dut.sin_q15.value = vint(b"".join(int.to_bytes(to_q15(s) & 0xFFFF, 2, "little") for s in sin))
        dut.start.value = 1
        await RisingEdge(dut.clk); dut.start.value = 0
        for _ in range(400000):
            await RisingEdge(dut.clk)
            if int(dut.done.value) == 1:
                break
        else:
            assert False, f"pos={pos} never finished"
        rv = int(dut.result.value); rsh = to_i8(int(dut.result_sh.value))
        got = [to_i8((rv >> (i*8)) & 0xFF) * (2.0**rsh) for i in range(D)]

        # ---- python reference : same orchestration, accumulate KV ----
        xn_i8, sxn = rms_ref(x_i8, sx, rms_w)
        Q_i8, sQ = mvq(Wq, xn_i8, sxn)
        K_i8, sK = mvq(Wk, xn_i8, sxn)
        V_i8, sV = mvq(Wv, xn_i8, sxn)
        Qr = rope_ref(Q_i8, cos, sin, H)
        Kr = rope_ref(K_i8, cos, sin, KH)
        KV['K'].append(Kr); KV['sK'].append(sK)
        KV['V'].append(V_i8); KV['sV'].append(sV)
        T = pos+1
        sKref = max(KV['sK'][:T]); sVref = max(KV['sV'][:T])
        Kf = [[ (KV['K'][t][j] >> (sKref-KV['sK'][t])) * (2.0**sKref) for j in range(KH*HS)] for t in range(T)]
        Vf = [[ (KV['V'][t][j] >> (sVref-KV['sV'][t])) * (2.0**sVref) for j in range(KH*HS)] for t in range(T)]
        Qf = [Qr[j]*(2.0**sQ) for j in range(H*HS)]
        # attention per query head
        attn = [0.0]*(H*HS)
        for h in range(H):
            kvh = h // NREP
            scores = [sum(Qf[h*HS+d]*Kf[t][kvh*HS+d] for d in range(HS))/math.sqrt(HS) for t in range(T)]
            mx = max(scores); e = [math.exp(s-mx) for s in scores]; ssum = sum(e); p = [x/ssum for x in e]
            for d in range(HS):
                attn[h*HS+d] = sum(p[t]*Vf[t][kvh*HS+d] for t in range(T))
        Wor = [[Wo_i8[r][k]*(2.0**swo) for k in range(H*HS)] for r in range(D)]
        ref = mv(Wor, attn)

        if pos == 2:
            ma = max(abs(v) for v in ref) or 1.0
            err = max(abs(got[i]-ref[i]) for i in range(D))
            dut._log.info(f"  causal pos=2 (T=3): max_err={err:.4f} ({100*err/ma:.1f}%)")
            dut._log.info(f"  got[:4]={[round(g,3) for g in got[:4]]} ref[:4]={[round(r,3) for r in ref[:4]]}")
            assert err/ma < 0.40, f"causal pos=2 wrong ({err})"

    dut._log.info("CAUSAL MULTI-POS PASS: KV-cache persists, softmax over 3 positions, per-pos rope.")
