# Implementation plan — `GG` command: 100% FPGA autonomous inference

**End goal**: the Tang Nano 20K generates N stories260K tokens from a seed token, with zero PC compute during generation. The PC loads weights once (~5 s), then sends `GG start_token N`; the FPGA returns N tokens.

## Current status (2026-05-18)

- ✅ **GG v0**: embed + rmsnorm L0 → x_norm[64]
- ✅ **GG v1**: + Wq → Q[64], cos > 0.99
- ✅ **GG v2**: + Wk, Wv → K[32], V[32]
- ✅ **GG v3**: + multi-head attention (T=1, pos=0)
- ✅ **GG v4**: + Wo + residual
- ✅ **GG v5a-f**: + rmsnorm FFN + W1/W3 chunked + silu chunked + elementwise multiply
- 🔴 **GG v5g**: W2 chunked + final residual — output 25× too small, needs debugging
- 🚧 **v6-v8**: remaining

**Validated pattern**: RTL FSM that chains embed → rmsnorm → copy obuf→xbuf → setup FQ → FQ → TX. The `gg_active` flag routes the return of the existing FQ flow to the GG branch. Avoids duplicating RTL.

## Target architecture

```
PC command:    GG start_tok N
               ↓
FPGA FSM:      pos=0, tok=start_tok
               ┌──── loop N times ────────────────────┐
               │ 1. embed lookup → x[64]              │
               │ 2. ┌── loop 5 layers ─────────────┐  │
               │    │ ATT: rmsnorm → Q/K/V → rope  │  │
               │    │      → write KV cache SDRAM  │  │
               │    │      → read KV[0..pos] SDRAM │  │
               │    │      → MM → Wo → residual    │  │
               │    │ FFN: rmsnorm → W1/W3 chunked │  │
               │    │      → silu → multiply       │  │
               │    │      → W2 chunked → residual │  │
               │    └───────────────────────────────┘  │
               │ 3. final rmsnorm → lm_head → argmax  │
               │ 4. TX tok, tok=new_tok, pos++        │
               └──────────────────────────────────────┘
               TX_DONE
```

## Hardcoded addresses (already in top.v or to add)

| Const | Address | Content |
|---|---|---|
| ADDR_TOK_EMB | 0x000000 | tok_emb [512, 64] (32 KiB) |
| ADDR_RMS_ATT_LX | 0x010000 + L*0x10000 | rms_att[L] (64 B) |
| ADDR_WQ_LX | base + 0x0100 | wq[L] [64, 64] (4 KiB) |
| ADDR_WK_LX | base + 0x1100 | wk[L] [32, 64] (2 KiB) |
| ADDR_WV_LX | base + 0x1900 | wv[L] [32, 64] (2 KiB) |
| ADDR_WO_LX | base + 0x2100 | wo[L] [64, 64] (4 KiB) |
| ADDR_RMS_FFN_LX | base + 0x3100 | rms_ffn[L] (64 B) |
| ADDR_W1_LX | base + 0x3200 | w1[L] [172, 64] chunked (12 KiB) |
| ADDR_W3_LX | base + 0x6200 | w3[L] [172, 64] chunked (12 KiB) |
| ADDR_W2_LX | base + 0x9200 | w2[L] [64, 172] chunked (12 KiB) |
| ADDR_RMS_FINAL | 0x060000 | rms_final (64 B) |
| ADDR_KV_K | 0x300000 | KV cache K: 5 layers × 32 pos × 32 B = 5 KiB |
| ADDR_KV_V | 0x301400 | KV cache V: idem |

`base = 0x010000 + L*0x10000` for layer L

## Remaining incremental steps

### GG v5g — debug W2 chunked + residual (in progress)

**Symptom**: output 25× too small (fpga[1]=0.062 vs ref -1.403, cos < 0).

**Hypotheses**:
1. W2 chunked load h_gated → xbuf: 1-cycle BSRAM read may be insufficient
2. 3-way sum via sequential BSRAM reads through 6 sub-states FRR_*P0/P1/P2: cycle timing issue
3. Requantize after sum: `rms_shift_x = sh_min` formula may be wrong

**Debug strategy**: add intermediate TX states to dump partials, sum_i32, etc.

### GG v6 — 5-layer loop (~45 min)

**Goal**: run v2-v5 for layer 0, then layer 1, etc. up to 4.

**RTL**:
- Counter `layer_idx [2:0]` (0..4)
- Per-layer address computation:
  ```verilog
  wire [22:0] base_layer = 23'h010000 + ({4'd0, layer_idx, 16'd0});
  wire [22:0] addr_rms_att_L = base_layer + 23'h0000;
  wire [22:0] addr_wq_L      = base_layer + 23'h0100;
  // ... etc
  ```
- Replace all ADDR_*_L0 constants with addr_*_L in the v2-v5 FSM
- At end of v5 (x_after_ffn in x_save_packed):
  - if layer_idx < 4: layer_idx++, x_save_packed becomes the next layer's input, restart at rmsnorm_att
  - else: jump to v7 (final norm + lm_head)
- Note: embed lookup runs ONLY at layer 0. Layers 1..4 start with rmsnorm_att using x_save_packed as input.

### GG v7 — final norm + lm_head + argmax → token (~45 min)

**Goal**: after 5 layers, compute the next token.

**RTL**:
- rms_final: sd_addr=ADDR_RMS_FINAL, run rmsnorm
- lm_head: vocab=512 matmul with weights = tok_emb (chunked into 8 sub-matmuls of N=64):
  - 8 sub-matmuls of N=64 (input dim 64, output 64)
  - Each sub-matmul produces logits[chunk*64..(chunk+1)*64] with its own shift
- **Argmax with mismatched shifts**:
  - Regs: `argmax_val [signed 8:0]`, `argmax_idx [9:0]`, `argmax_shift [signed 7:0]`
  - Initialize argmax_val = -128, argmax_shift = -128 (very small)
  - For each chunk c, for each i=0..63:
    - logit_int = obuf[i], logit_shift = fq_shift_total
    - Compare (logit_int, logit_shift) vs (argmax_val, argmax_shift):
      - if logit_shift > argmax_shift: rescale argmax_val = argmax_val >> (logit_shift - argmax_shift), argmax_shift = logit_shift
      - else if logit_shift < argmax_shift: rescale logit_int = logit_int >> (argmax_shift - logit_shift)
      - then compare int values
    - if new > current: update argmax_val, argmax_idx = c*64+i, argmax_shift
  - After all chunks: argmax_idx = token_id
- TX: 'GK' tok_lo tok_hi = 4 bytes

**Tradeoff**: cross-shift comparison is fiddly. A simpler approach forces all chunks to use the same shift (pass shift_force to FQ) — loses precision but simplifies.

### GG v8 — KV cache SDRAM + N-token loop (~1 h)

**Goal**: full autonomy. PC sends `GG start_token N`, FPGA returns N tokens.

**RTL**:
- Reg `pos [5:0]` (0..31) position counter
- Reg `n_tokens [5:0]` (N to generate)
- Reg `current_token [9:0]` (next token to embed)
- Extended RX: 'GG' tok_lo tok_hi N + shifts (15+ bytes)
- Each generation iteration:
  - Reset gg_active sub-flags
  - Embed current_token → x_save_packed
  - **For each layer**: ATT (with KV cache) then FFN
- **KV cache**: at each attention, after rope:
  - WRITE K[pos], V[pos] to `ADDR_KV_K + L*32*32 + pos*32` (32 bytes for the 4 heads × 8 HS)
  - DMA write 32 bytes to SDRAM
- **KV cache read for MM**:
  - Before MM, fetch K[0..pos] from SDRAM to xbuf: (pos+1)*32 bytes = DMA loop
  - Fetch V[0..pos] to wbuf: same
  - Launch MM with attn_T = pos+1
- **Rope for pos>0**:
  - Load freq_cis_real/imag for position pos (from SDRAM table or recompute via LUT)
  - For each Q head H and each K head KH, launch rope_op
  - **Simpler alternative**: precompute freq_cis for pos=0..31 and store in SDRAM (32 × 4 × 4 = 512 bytes for Q15 cos+sin)
- Argmax → next_token, TX next_token, current_token = next_token, pos++
- if pos == n_tokens: send DONE marker, return to IDLE
- else: restart the loop

**TX format**: `GS` (Generation Start) then N × 2 bytes (each token), then `GD` (Generation Done). PC receives a stream.

**Main difficulty**: KV cache orchestration + multi-head rope + state persistence between tokens.

## Test strategy per step

For each GG vX:
1. Code RTL (~30-60 min)
2. Build (`gw_sh build.tcl`) — 6 min
3. Reflash (`programmer_cli`) — 10 sec
4. Python test `test_gg_vX.py` — compare with the infer_v4sim Python reference
5. If OK: add to `run_regression.py`
6. Update memory

## Known pitfalls to avoid

1. **`cur_shift_out` mux**: any new op that produces an output shift must be in the mux around line 671 of top.v
2. **`rx_consume` list**: any new RX state must be listed around line 641
3. **SDRAM addresses modulo BSRAM_SZ=1024**: xbuf/wbuf are 1024 bytes now, not 128
4. **MM T_MAX=32**: OK for seq_len 32 (see v4.5t)
5. **Quantization shift on boundaries**: can give ±1 bit diff between FPGA and Python ref (the 1/sqrt LUT approximates). Tolerance > 5% is acceptable.
6. **`gg_active` clear**: remember to reset it to 0 in S_TX_O_W when returning to S_IDLE
7. **op_sel restore**: when switching to op_fm for FQ from GG, remember to restore op_sel=10 before the GG TX states

## Time budget (estimate)

| Step | Time |
|---|---|
| GG v5g debug | 1 h |
| GG v6 | 45 min |
| GG v7 | 45 min |
| GG v8 | 1 h |
| **Total** | **~3.5 h of effective work** |

Add ~24 min of builds (4 × 6 min) on top.

## Success criterion

```bash
# PC (just to load weights once):
python load_weights.py   # ~5 s, loads everything into SDRAM
# Then:
python generate.py 17    # sends GG 1 17, waits, receives 17 tokens, prints text

# Expected output:
"Once upon a time, there was a little gir mommy. The bo"
# (matches infer_fpga.py but WITHOUT any Python compute during generation)
```

Estimated throughput: ~5-10 tok/s (UART-bound for TX only, no per-token RTT).
Vs current infer_fpga: ~0.36 tok/s (30 RTTs per token).
**Speedup: ×10-30**.
