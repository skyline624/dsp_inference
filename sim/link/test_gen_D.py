"""Step D gate : gen_seq 5-layer loop (attention + FFN per layer), at pos=0.

Wraps the validated single layer (step C) in a loop over NL=5 layers. Two real
difficulties handled by the RTL and exercised here :
  1. PER-LAYER KV-cache : kmem/vmem/ksh/vsh indexed by (layer,pos) so layer 1
     does not clobber layer 0's KV.
  2. INTERNAL base_l : the sequencer computes base_l = 0x010000 + layer*0x10000
     itself (no base port), reading each layer's weights at the real stories260K
     SDRAM offsets (rms_att@0, wq@0x100, wk@0x1100, wv@0x1900, wo@0x2100,
     rms_ffn@0x3100, w1@0x3200, w3@0x6200, w2@0x9200).

pos is FIXED at 0 (single token) : this is the LAYER loop, not the token loop (F).

Weight shifts : the RTL feeds the node a single weight shift per op type (static
sw_* ports), so every layer must share the SAME shift per weight type. We quantise
each weight type with a SHARED shift computed over all 5 layers (per-layer shifts
would need a wider interface, deferred). Reference : float forward through the 5
layers, stop BEFORE lm_head, return x. Tolerance ~0.60 (G2 measured 0.093).

  make -f Makefile.gen MODULE=test_gen_D
"""
import math
import random
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles

D = 64
H, KH, HS = 8, 4, 8
HID = 172
NREP = H // KH
NL = 5
LBASE, LSTRIDE = 0x010000, 0x010000
OFF = dict(rms=0x0000, wq=0x0100, wk=0x1100, wv=0x1900, wo=0x2100,
           rmsff=0x3100, w1=0x3200, w3=0x6200, w2=0x9200)


def to_i8(b): return b - 256 if b >= 128 else b
def shift_of(vals):
    m = max((abs(v) for v in vals), default=0.0)
    return 0 if m == 0 else math.ceil(math.log2(m/127.0))
def q_at(x, s): return [max(-128, min(127, round(v/(2.0**s)))) for v in x]
def qm_at(M, s): return [q_at(row, s) for row in M]
def deq(i8, s): return [v*(2.0**s) for v in i8]
def deqm(M_i8, s): return [[v*(2.0**s) for v in row] for row in M_i8]
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
def flat_any(x):
    out = []
    for e in x:
        if isinstance(e, list): out.extend(flat_any(e))
        else: out.append(e)
    return out
def shared_shift(mats): return shift_of(flat_any(mats))   # over all layers (matrices or 1D)


def make_layer(rng):
    return dict(
        rms=[1.0]*D, rmsff=[1.0]*D,
        Wq=[[rng.gauss(0,0.1) for _ in range(D)] for _ in range(H*HS)],
        Wk=[[rng.gauss(0,0.1) for _ in range(D)] for _ in range(KH*HS)],
        Wv=[[rng.gauss(0,0.1) for _ in range(D)] for _ in range(KH*HS)],
        Wo=[[rng.gauss(0,0.1) for _ in range(H*HS)] for _ in range(D)],
        W1=[[rng.gauss(0,0.1) for _ in range(D)] for _ in range(HID)],
        W3=[[rng.gauss(0,0.1) for _ in range(D)] for _ in range(HID)],
        W2=[[rng.gauss(0,0.1) for _ in range(HID)] for _ in range(D)])


def layer_ref(x, r):
    xn = rms_f(x, r['rms'])
    V = mv(r['Wv'], xn)
    attn = [V[(h//NREP)*HS + d] for h in range(H) for d in range(HS)]
    x = [x[i] + mv(r['Wo'], attn)[i] for i in range(D)]
    xn2 = rms_f(x, r['rmsff'])
    g = mv(r['W1'], xn2); u = mv(r['W3'], xn2)
    sg = [gi/(1.0+math.exp(-gi)) for gi in g]
    hh = [sg[i]*u[i] for i in range(HID)]
    o = mv(r['W2'], hh)
    return [x[i] + o[i] for i in range(D)]


@cocotb.test()
async def test_gen_D(dut):
    cocotb.start_soon(Clock(dut.clk, 37, units="ns").start())
    dut.rst_n.value = 0; dut.start.value = 0
    for s in ("gen_mode","x_in","sx_in","sw_rms","swq","swk","swv","swo",
              "sw_rmsf","sw1","sw3","sw2","sw_rmsfinal","sw_emb","pos","cos_q15","sin_q15"):
        getattr(dut, s).value = 0
    await ClockCycles(dut.clk, 10)

    rng = random.Random(77)
    x_f = [rng.gauss(0, 1.0) for _ in range(D)]
    sx = shift_of(x_f); x_i8 = q_at(x_f, sx)
    L = [make_layer(rng) for _ in range(NL)]

    # per-layer, per-type quantization : each layer gets its OWN shift, packed into
    # the [NL*8-1:0] buses -> exercises the per-layer shift selection in the RTL.
    sh = {k: [] for k in ('rms','rmsff','wq','wk','wv','wo','w1','w3','w2')}
    sd = dut.u_sdram
    refs = []
    for l in range(NL):
        w = L[l]; base_l = LBASE + l*LSTRIDE
        srm = shift_of(w['rms']);          rms_i8  = q_at(w['rms'],  srm)
        srf = shift_of(w['rmsff']);        rmsf_i8 = q_at(w['rmsff'], srf)
        sq  = shift_of(flat_any(w['Wq'])); Wq_i8 = qm_at(w['Wq'], sq)
        sk  = shift_of(flat_any(w['Wk'])); Wk_i8 = qm_at(w['Wk'], sk)
        sv  = shift_of(flat_any(w['Wv'])); Wv_i8 = qm_at(w['Wv'], sv)
        so  = shift_of(flat_any(w['Wo'])); Wo_i8 = qm_at(w['Wo'], so)
        s1  = shift_of(flat_any(w['W1'])); W1_i8 = qm_at(w['W1'], s1)
        s3  = shift_of(flat_any(w['W3'])); W3_i8 = qm_at(w['W3'], s3)
        s2  = shift_of(flat_any(w['W2'])); W2_i8 = qm_at(w['W2'], s2)
        preload(sd, base_l+OFF['rms'], bytes((v & 0xFF) for v in rms_i8))
        preload(sd, base_l+OFF['wq'], rowsb(Wq_i8, 0, H*HS, D))
        preload(sd, base_l+OFF['wk'], rowsb(Wk_i8, 0, KH*HS, D))
        preload(sd, base_l+OFF['wv'], rowsb(Wv_i8, 0, KH*HS, D))
        preload(sd, base_l+OFF['wo'], rowsb(Wo_i8, 0, D, H*HS))
        preload(sd, base_l+OFF['rmsff'], bytes((v & 0xFF) for v in rmsf_i8))
        preload(sd, base_l+OFF['w1'], rowsb(W1_i8 + [[0]*D for _ in range(192-HID)], 0, 192, D))
        preload(sd, base_l+OFF['w3'], rowsb(W3_i8 + [[0]*D for _ in range(192-HID)], 0, 192, D))
        for c in range(3):
            blk = [[(W2_i8[r][c*64+k] if c*64+k < HID else 0) for k in range(D)] for r in range(D)]
            preload(sd, base_l+OFF['w2'] + c*0x1000, rowsb(blk, 0, D, D))
        for k, s in (('rms',srm),('rmsff',srf),('wq',sq),('wk',sk),('wv',sv),
                     ('wo',so),('w1',s1),('w3',s3),('w2',s2)):
            sh[k].append(s)
        refs.append(dict(rms=deq(rms_i8, srm), rmsff=deq(rmsf_i8, srf),
                         Wv=deqm(Wv_i8, sv), Wo=deqm(Wo_i8, so),
                         W1=deqm(W1_i8, s1), W3=deqm(W3_i8, s3), W2=deqm(W2_i8, s2)))

    # gen_seq now always runs the lm_head after the 5 layers ; this gate checks the
    # 5-layer x (captured in `result` BEFORE the head) and ignores `token`. Preload
    # zeros for rms_final / tok_emb so the head runs cleanly (no X).
    preload(sd, 0x060000, bytes(D))
    preload(sd, 0x000000, bytes(512*D))

    def pack(vals): return sum((v & 0xFF) << (i*8) for i, v in enumerate(vals))
    cos = [1.0]*(HS//2); sin = [0.0]*(HS//2)   # pos=0 : rope identity

    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 5)
    dut.x_in.value = vint([v & 0xFF for v in x_i8]); dut.sx_in.value = sx & 0xFF
    dut.sw_rms.value = pack(sh['rms']); dut.swq.value = pack(sh['wq']); dut.swk.value = pack(sh['wk'])
    dut.swv.value = pack(sh['wv']); dut.swo.value = pack(sh['wo'])
    dut.sw_rmsf.value = pack(sh['rmsff'])
    dut.sw1.value = pack(sh['w1']); dut.sw3.value = pack(sh['w3']); dut.sw2.value = pack(sh['w2'])
    dut.pos.value = 0
    dut.cos_q15.value = vint(b"".join(int.to_bytes(int(max(-32768,min(32767,round(c*32768.0))))&0xFFFF,2,"little") for c in cos))
    dut.sin_q15.value = vint(b"".join(int.to_bytes(int(max(-32768,min(32767,round(s*32768.0))))&0xFFFF,2,"little") for s in sin))
    dut.start.value = 1
    await RisingEdge(dut.clk); dut.start.value = 0

    for _ in range(25000000):
        await RisingEdge(dut.clk)
        if int(dut.done.value) == 1:
            break
    else:
        assert False, "gen_seq step D never finished"

    rv = int(dut.result.value); rsh = to_i8(int(dut.result_sh.value))
    got = [to_i8((rv >> (i*8)) & 0xFF) * (2.0**rsh) for i in range(D)]

    # reference : float forward through the 5 layers (no lm_head)
    x = deq(x_i8, sx)
    for l in range(NL):
        x = layer_ref(x, refs[l])
    ref = x

    ma = max(abs(v) for v in ref) or 1.0
    err = max(abs(got[i]-ref[i]) for i in range(D))
    dut._log.info(f"  gen step D (5 layers, pos=0) : max_err={err:.4f} ({100*err/ma:.1f}%)")
    dut._log.info(f"  got[:4]={[round(g,3) for g in got[:4]]} ref[:4]={[round(r,3) for r in ref[:4]]}")
    assert err/ma < 0.60, f"step D wrong ({err})"
    dut._log.info("STEP D PASS: 5-layer transformer stack (per-layer KV, internal base_l).")
