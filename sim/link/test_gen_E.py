"""Step E gate : gen_seq 5 layers + lm_head + argmax -> next token, at pos=0.

After the 5-layer stack (step D), the sequencer runs the generation head :
  FN(x, rms_final @ 0x060000) -> xn ; 8 x FQ(xn, tok_emb chunk c @ c*0x1000, N=64)
  -> 512 logits + 8 shifts ; integer re-align to sref=max(shifts) ; argmax -> token.

The RTL captures `result` = x-after-5-layers BEFORE the head (so test_gen_D still
passes), then produces `token`. This test reads BOTH : it recomputes the integer
lm_head argmax ON THE RTL's OWN result x (same x, same global-sw_emb dataflow the
RTL uses) and requires an EXACT token match (argmax is integer, no tolerance).
That isolates the lm_head+argmax RTL ; the 5-layer x itself is checked by test_gen_D.

  make -f Makefile.gen MODULE=test_gen_E
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
VOCAB, NCHUNK = 512, 8
LBASE, LSTRIDE = 0x010000, 0x010000
A_RMSFINAL, A_EMB = 0x060000, 0x000000
OFF = dict(rms=0x0000, wq=0x0100, wk=0x1100, wv=0x1900, wo=0x2100,
           rmsff=0x3100, w1=0x3200, w3=0x6200, w2=0x9200)


def to_i8(b): return b - 256 if b >= 128 else b
def shift_of(vals):
    m = max((abs(v) for v in vals), default=0.0)
    return 0 if m == 0 else math.ceil(math.log2(m/127.0))
def q_at(x, s): return [max(-128, min(127, round(v/(2.0**s)))) for v in x]
def qm_at(M, s): return [q_at(row, s) for row in M]
def deq(i8, s): return [v*(2.0**s) for v in i8]
def deqm(M, s): return [[v*(2.0**s) for v in row] for row in M]
def flat_any(x):
    out = []
    for e in x:
        if isinstance(e, list): out.extend(flat_any(e))
        else: out.append(e)
    return out
def rms_f(x, w, eps=1e-5):
    inv = 1.0/math.sqrt(sum(v*v for v in x)/len(x)+eps); return [x[i]*w[i]*inv for i in range(len(x))]
def to_i8s(x): s = shift_of(x); return q_at(x, s), s
def requant_i32(y, sin):
    m = max((abs(int(v)) for v in y), default=0)
    if m == 0: return [0]*len(y), sin
    add = max(0, math.ceil(math.log2(m/127.0)))
    if add == 0: return [max(-128, min(127, int(v))) for v in y], sin
    half = 1 << (add-1)
    return [max(-128, min(127, (int(v)+half) >> add)) for v in y], sin+add
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
def mv(Wm, x): return [sum(Wm[r][k]*x[k] for k in range(len(x))) for r in range(len(Wm))]


def make_layer(rng):
    return dict(rms=[1.0]*D, rmsff=[1.0]*D,
        Wq=[[rng.gauss(0,0.1) for _ in range(D)] for _ in range(H*HS)],
        Wk=[[rng.gauss(0,0.1) for _ in range(D)] for _ in range(KH*HS)],
        Wv=[[rng.gauss(0,0.1) for _ in range(D)] for _ in range(KH*HS)],
        Wo=[[rng.gauss(0,0.1) for _ in range(H*HS)] for _ in range(D)],
        W1=[[rng.gauss(0,0.1) for _ in range(D)] for _ in range(HID)],
        W3=[[rng.gauss(0,0.1) for _ in range(D)] for _ in range(HID)],
        W2=[[rng.gauss(0,0.1) for _ in range(HID)] for _ in range(D)])


def load_lut(path):
    lut = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("//"): lut.append(int(line, 16))
    return lut


def node_rmsnorm(x_i8, w_i8, shift_w, lut):
    """Bit-exact replica of src/rmsnorm_op.v (LUT 1/sqrt, parity, apply_shift).
    Output shift = shift_w (NOT dynamic). rmsnorm is scale-invariant so shift_x
    is unused. Needed because a float rmsnorm differs by ~1 LSB -> flips tiny
    argmax gaps."""
    D = len(x_i8)
    acc = sum(xi*xi for xi in x_i8) & 0xFFFFFF       # 24-bit
    p = 0
    for ii in range(24):
        if (acc >> ii) & 1: p = ii
    shift_amt = 8 - p                                  # signed, -15..8
    acc_norm = ((acc << shift_amt) & 0xFFFFFFFF) if shift_amt >= 0 else (acc >> (-shift_amt))
    raw_inv = lut[acc_norm & 0xFF]                     # Q1.15
    if shift_amt & 1:                                  # odd -> * sqrt(2)
        raw_inv = ((raw_inv * 0xB505) >> 15) & 0xFFFF
    apply_shift = (shift_amt >> 1) - 16                # arithmetic >> (floor)
    out = []
    for i in range(D):
        prod = x_i8[i] * w_i8[i] * raw_inv
        if apply_shift >= 0:
            shifted = prod << apply_shift
        else:
            na = -apply_shift
            rounding = 0 if na == 1 else (1 << (na-1))
            shifted = (prod + rounding) >> na
        out.append(max(-128, min(127, shifted)))
    return out, shift_w


def lmhead_argmax(x_i8, rmsfinal_i8, sw_rmsfinal, tok_emb_i8, s_emb, lut):
    """Faithful integer lm_head : node rmsnorm(rms_final) + 8 chunked FQ (global emb
    shift) + integer re-align + argmax. Bit-exact with the RTL; run on the RTL's x."""
    xn_i8, sxn = node_rmsnorm(x_i8, rmsfinal_i8, sw_rmsfinal, lut)
    logits = []; csh = []
    for c in range(NCHUNK):
        chunk = tok_emb_i8[c*64:(c+1)*64]                       # [64][64] int8
        y = [sum(chunk[r][k]*xn_i8[k] for k in range(D)) for r in range(D)]
        yi8, sy = requant_i32(y, sxn + s_emb)
        logits.extend(yi8); csh.append(sy)
    sref = max(csh)
    aligned = [logits[i] >> (sref - csh[i//64]) for i in range(VOCAB)]   # >> is arithmetic in py
    best = max(range(VOCAB), key=lambda i: aligned[i])
    return best, aligned


@cocotb.test()
async def test_gen_E(dut):
    cocotb.start_soon(Clock(dut.clk, 37, units="ns").start())
    dut.rst_n.value = 0; dut.start.value = 0
    for s in ("x_in","sx_in","sw_rms","swq","swk","swv","swo","sw_rmsf","sw1","sw3","sw2",
              "sw_rmsfinal","sw_emb","pos","cos_q15","sin_q15"):
        getattr(dut, s).value = 0
    await ClockCycles(dut.clk, 10)

    rng = random.Random(77)
    x_f = [rng.gauss(0, 1.0) for _ in range(D)]
    sx = shift_of(x_f); x_i8 = q_at(x_f, sx)
    Lw = [make_layer(rng) for _ in range(NL)]
    # lm_head weights : rms_final and tok_emb (shared classifier) [VOCAB,64]
    rmsfinal = [1.0]*D
    tok_emb = [[rng.gauss(0, 0.5) for _ in range(D)] for _ in range(VOCAB)]

    sh = {k: [] for k in ('rms','rmsff','wq','wk','wv','wo','w1','w3','w2')}
    sd = dut.u_sdram
    for l in range(NL):
        w = Lw[l]; base_l = LBASE + l*LSTRIDE
        srm = shift_of(w['rms']);          rms_i8  = q_at(w['rms'],  srm)
        srf = shift_of(w['rmsff']);        rmsf_i8 = q_at(w['rmsff'], srf)
        sq  = shift_of(flat_any(w['Wq'])); sk = shift_of(flat_any(w['Wk']))
        sv  = shift_of(flat_any(w['Wv'])); so = shift_of(flat_any(w['Wo']))
        s1  = shift_of(flat_any(w['W1'])); s3 = shift_of(flat_any(w['W3'])); s2 = shift_of(flat_any(w['W2']))
        preload(sd, base_l+OFF['rms'], bytes((v & 0xFF) for v in rms_i8))
        preload(sd, base_l+OFF['wq'], rowsb(qm_at(w['Wq'], sq), 0, H*HS, D))
        preload(sd, base_l+OFF['wk'], rowsb(qm_at(w['Wk'], sk), 0, KH*HS, D))
        preload(sd, base_l+OFF['wv'], rowsb(qm_at(w['Wv'], sv), 0, KH*HS, D))
        preload(sd, base_l+OFF['wo'], rowsb(qm_at(w['Wo'], so), 0, D, H*HS))
        preload(sd, base_l+OFF['rmsff'], bytes((v & 0xFF) for v in rmsf_i8))
        preload(sd, base_l+OFF['w1'], rowsb(qm_at(w['W1'], s1) + [[0]*D for _ in range(192-HID)], 0, 192, D))
        preload(sd, base_l+OFF['w3'], rowsb(qm_at(w['W3'], s3) + [[0]*D for _ in range(192-HID)], 0, 192, D))
        W2_i8 = qm_at(w['W2'], s2)
        for c in range(3):
            blk = [[(W2_i8[r][c*64+k] if c*64+k < HID else 0) for k in range(D)] for r in range(D)]
            preload(sd, base_l+OFF['w2'] + c*0x1000, rowsb(blk, 0, D, D))
        for k, s in (('rms',srm),('rmsff',srf),('wq',sq),('wk',sk),('wv',sv),
                     ('wo',so),('w1',s1),('w3',s3),('w2',s2)):
            sh[k].append(s)

    # lm_head weights
    srmf = shift_of(rmsfinal); rmsfinal_i8 = q_at(rmsfinal, srmf)
    semb = shift_of(flat_any(tok_emb)); tok_emb_i8 = qm_at(tok_emb, semb)
    preload(sd, A_RMSFINAL, bytes((v & 0xFF) for v in rmsfinal_i8))
    preload(sd, A_EMB, rowsb(tok_emb_i8, 0, VOCAB, D))     # [512,64] contiguous, chunk c @ c*0x1000

    def pack(vals): return sum((v & 0xFF) << (i*8) for i, v in enumerate(vals))
    cos = [1.0]*(HS//2); sin = [0.0]*(HS//2)

    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 5)
    dut.x_in.value = vint([v & 0xFF for v in x_i8]); dut.sx_in.value = sx & 0xFF
    dut.sw_rms.value = pack(sh['rms']); dut.swq.value = pack(sh['wq']); dut.swk.value = pack(sh['wk'])
    dut.swv.value = pack(sh['wv']); dut.swo.value = pack(sh['wo']); dut.sw_rmsf.value = pack(sh['rmsff'])
    dut.sw1.value = pack(sh['w1']); dut.sw3.value = pack(sh['w3']); dut.sw2.value = pack(sh['w2'])
    dut.sw_rmsfinal.value = srmf & 0xFF; dut.sw_emb.value = semb & 0xFF
    dut.pos.value = 0
    dut.cos_q15.value = vint(b"".join(int.to_bytes(int(max(-32768,min(32767,round(c*32768.0))))&0xFFFF,2,"little") for c in cos))
    dut.sin_q15.value = vint(b"".join(int.to_bytes(int(max(-32768,min(32767,round(s*32768.0))))&0xFFFF,2,"little") for s in sin))
    dut.start.value = 1
    await RisingEdge(dut.clk); dut.start.value = 0

    for _ in range(30000000):
        await RisingEdge(dut.clk)
        if int(dut.done.value) == 1:
            break
    else:
        assert False, "gen_seq step E never finished"

    tok_rtl = int(dut.token.value)
    rv = int(dut.result.value)
    x5_i8 = [to_i8((rv >> (i*8)) & 0xFF) for i in range(D)]   # RTL's x after 5 layers (int8)

    # reference : bit-exact integer lm_head + argmax on the RTL's OWN x
    lut = load_lut("rsqrt_lut.hex")
    tok_ref, aligned = lmhead_argmax(x5_i8, rmsfinal_i8, srmf, tok_emb_i8, semb, lut)
    ranked = sorted(range(VOCAB), key=lambda i: aligned[i], reverse=True)
    rank_rtl = ranked.index(tok_rtl)
    top = [aligned[i] for i in ranked[:3]]
    dut._log.info(f"  gen step E : token_rtl={tok_rtl} (ref rank {rank_rtl}, logit {aligned[tok_rtl]}) token_ref={tok_ref}")
    dut._log.info(f"  ref top3 idx={ranked[:3]} logits={top} gap={top[0]-top[1]}")
    assert tok_rtl == tok_ref, f"step E token mismatch : rtl={tok_rtl} ref={tok_ref}"
    dut._log.info("STEP E PASS: 5 layers + lm_head + integer argmax -> next token.")
