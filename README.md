# dsp_inference

Inference of the **stories260K** transformer (Karpathy, 5 layers, 260K params) on a **Tang Nano 20K** (Gowin GW2AR-18).

All heavy compute runs on the FPGA via DSP cores. The PC loads weights into SDRAM, then orchestrates inference over UART (1 Mbaud).

## Status

✅ **Generates text**: `Once upon a time, there was a little gir mommy. The bo` (100% FPGA compute, including multi-head attention in RTL)

See `host/infer_fpga.py` for the full pipeline, `PLAN_GG_AUTONOMIE.md` for the roadmap toward zero-PC autonomy.

## Architecture

**RTL** (`src/`):
- `top.v` — main FSM, UART, command dispatch, SDRAM fetch, incremental `GG` FSM (v0..v5g)
- `rmsnorm_op.v`, `silu_op.v`, `rope_op.v`, `softmax_op.v` — LUT-based operators
- `attention_head_op.v` — single-head multi-head attention (T_MAX=32)
- `mac18.v` — multiply-accumulate via the `MULTALU18X18` DSP primitive
- `sdram.v` — SDR-SDRAM controller (NESTang)
- `uart_*.v`, `gowin_rpll.v` — UART + PLL

**Host** (`host/`):
- `infer_fpga.py` — full stories260K inference (PC-orchestrated, ~3 s/token)
- `infer_v4sim.py` — pure Python reference (validation)
- `transformer_ops.py` — FPGA primitives (call_fn, call_fq, call_mm, etc.)
- `test_*.py` — per-operator unit tests
- `test_gg_v*.py` — tests for the incremental GG FSM
- `run_regression.py` — non-regression suite (~100 s, 10 tests)

## UART commands

| Cmd | What | Format |
|---|---|---|
| `LL/CC/BB/WW` | SDRAM load/check | bulk r/w |
| `NN/SS/RR/XX/AA` | Standalone ops (rmsnorm/silu/rope/softmax/attention) | weights via UART |
| `FN/FQ/FM` | Op with weights from SDRAM | int8+shift matmul |
| `MM` | Multi-head GQA attention | T up to 32 |
| `CN/CS` | Chained rmsnorm+matmul[+silu] | one command |
| `EE` | Embedding lookup (token → x[64]) | hardcoded addr |
| `GG` | Generation FSM (v0..v5g WIP) | fully in RTL |

## Numeric format

**int8 + power-of-2 shift**: each activation = `(int8[D], shift)`, with `real = int8 * 2^shift`. Conversions are pure shifts (hardware-friendly). See `host/v4_quant.py`.

## Build / Run

```bash
# Build bitstream (Gowin EDA)
cd C:/Gowin/.../dsp_inference
gw_sh.exe build.tcl

# Reflash Tang Nano
programmer_cli.exe --device GW2AR-18C --run 2 --fsFile impl/pnr/dsp_inference.fs

# Run inference
cd host
python infer_fpga.py
```

## Known RTL bugs

1. `MM`: 3 bugs fixed in v4.5t (hardcoded shift, T_MAX=8→32, T width 4→6 bits)
2. `GG v5g`: output 25× too small (W2 chunked + 3-way sum + residual — needs debugging)

See commit history for the full timeline.
