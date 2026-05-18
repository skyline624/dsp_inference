# dsp_inference

Inférence transformer **stories260K** (Karpathy, 5 layers, 260K params) sur **Tang Nano 20K** (Gowin GW2AR-18).

Tous les calculs lourds tournent sur FPGA via DSP cores. Le PC charge les poids en SDRAM puis orchestre l'inférence via UART (1 Mbaud).

## État

✅ **Production de texte** : `Once upon a time, there was a little gir mommy. The bo` (FPGA 100% compute, MM RTL inclus)

Voir `host/infer_fpga.py` pour le pipeline complet, `PLAN_GG_AUTONOMIE.md` pour le plan vers zéro-PC.

## Architecture

**RTL** (`src/`) :
- `top.v` — FSM principale, UART, dispatching commandes, fetch SDRAM, FSM `GG` incrémentale (v0..v5g)
- `rmsnorm_op.v`, `silu_op.v`, `rope_op.v`, `softmax_op.v` — opérateurs LUT-based
- `attention_head_op.v` — multi-head attention single-head (T_MAX=32)
- `mac18.v` — multiply-accumulate via primitive DSP `MULTALU18X18`
- `sdram.v` — contrôleur SDR-SDRAM (NESTang)
- `uart_*.v`, `gowin_rpll.v` — UART + PLL

**Host** (`host/`) :
- `infer_fpga.py` — inférence stories260K full pipeline (PC-orchestrated, ~3 s/token)
- `infer_v4sim.py` — référence Python pure (validation)
- `transformer_ops.py` — primitives FPGA (call_fn, call_fq, call_mm, etc.)
- `test_*.py` — suite de tests unitaires par opérateur
- `test_gg_v*.py` — tests de la FSM GG incrémentale
- `run_regression.py` — suite de non-régression (~100s, 10 tests)

## Commandes UART

| Cmd | Quoi | Format |
|---|---|---|
| `LL/CC/BB/WW` | Load/check SDRAM | bulk r/w |
| `NN/SS/RR/XX/AA` | Op standalone (rmsnorm/silu/rope/softmax/attention) | weights via UART |
| `FN/FQ/FM` | Op avec weights depuis SDRAM | matmul int8+shift |
| `MM` | Multi-head attention GQA | T jusqu'à 32 |
| `CN/CS` | Chained rmsnorm+matmul[+silu] | une commande |
| `EE` | Embedding lookup (token → x[64]) | hardcoded addr |
| `GG` | Generation FSM (v0..v5g WIP) | tout en RTL |

## Format numérique

**int8 + power-of-2 shift** : chaque activation = `(int8[D], shift)`, `real = int8 * 2^shift`. Conversions = shifts purs (hardware-friendly). Voir `host/v4_quant.py`.

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

## Bugs RTL connus

1. `MM` : 3 bugs fixés v4.5t (shift hardcoded, T_MAX=8→32, T width 4→6 bits)
2. `GG v5g` : output 25× trop petit (W2 chunked + sum 3-way + résidu — à débugger)

Voir `MEMORY.md` (auto-memory Claude) ou commits pour l'historique complet.
