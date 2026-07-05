#!/bin/bash
# Run the ss_link (parallel-link) sims with the LUT .hex files in place.
# The transformer ops $readmemh their LUTs by relative path, so they must sit in
# this directory or rmsnorm/silu silently produce X. Usage:
#   ./run_ss.sh ffn <NN>    # FFN sequencer over ss_link, NN nodes (default 2)
#   ./run_ss.sh attn <NN>   # attention sequencer over ss_link, NN nodes
#   ./run_ss.sh node        # single node WW/BB over ss_link
set -e
cd "$(dirname "$0")"
cp -f ../../src/*.hex .            # rsqrt_lut, silu_lut, exp_lut, inv_lut
case "${1:-ffn}" in
  ffn)
    cp -f ../test_ffn_tp_seq2.py .
    NN="${2:-2}"
    make -f Makefile.ffnss clean >/dev/null 2>&1 || true
    NN="$NN" make -f Makefile.ffnss NN="$NN"
    ;;
  attn)
    cp -f ../test_attn_tp.py .
    NN="${2:-2}"
    make -f Makefile.attnss clean >/dev/null 2>&1 || true
    NN="$NN" make -f Makefile.attnss NN="$NN"
    ;;
  node)
    make -f Makefile.nodess clean >/dev/null 2>&1 || true
    make -f Makefile.nodess
    ;;
  *) echo "usage: $0 {ffn <NN>|attn <NN>|node}"; exit 1;;
esac
