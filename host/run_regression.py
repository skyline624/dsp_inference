#!/usr/bin/env python3
# Suite de non-regression : lance tous les tests qui devraient always marcher
# after le refactor RTL du 2026-05-18 (T_MAX=32, BSRAM=1024, MM shift fix).
#
# Pour each test : capture stdout, parse les marqueurs OK/FAIL.
# Resume final : tableau de PASS/FAIL with details.

import subprocess
import sys
import time
import re
from pathlib import Path

HERE = Path(__file__).parent

TESTS = [
    # (script, description, pattern de validation, timeout sec)
    ("test_sdram_diag.py",   "SDRAM basics (LL/CC/FN/FQ)",          r"20/20 OK",                              60),
    ("test_attn_block.py",   "Attention block T=1, poids random",   r"diff max final\s*=\s*0\.\d+",           60),
    ("test_ffn_block.py",    "FFN block hidden=64",                  r"diff max final\s*=\s*0\.\d+",           60),
    ("test_ffn_172.py",      "FFN hidden=172 (chunked)",             r"==>\s*OK",                              60),
    ("test_layer.py",        "Couche complete (attn+ffn) T=1",       r"==>\s*OK",                              60),
    ("test_multi_layer.py",  "5 couches enchainees, T=1",            r"==>\s*OK",                              120),
    ("test_attn_pos.py",     "Attention rope+KV cache multi-pos",    r"==>\s*OK",                              120),
    ("test_lmhead_only.py",  "lm_head isole vs Python ref",          r"argmax MATCH",                          60),
    ("test_mm_scale.py",     "MM shift propagation (post-fix RTL)",  r"mean_ratio=\s*0?\.\d+|mean_ratio=1\.0", 60),
    ("infer_fpga.py",        "Inference stories260K 17 tokens",      r"Texte FPGA",                            300),
]

def run_test(script, pattern, timeout):
    """Lance un script Python, retourne (ok, stdout, duree, raison_si_fail)."""
    t0 = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, "-u", script],
            cwd=HERE, capture_output=True, text=True, timeout=timeout, encoding='utf-8', errors='replace'
        )
        dur = time.time() - t0
        out = proc.stdout + proc.stderr
        if proc.returncode != 0:
            return False, out, dur, f"exit code {proc.returncode}"
        if re.search(pattern, out):
            return True, out, dur, ""
        return False, out, dur, f"pattern '{pattern}' non trouve"
    except subprocess.TimeoutExpired:
        return False, "", time.time() - t0, f"timeout >{timeout}s"
    except Exception as e:
        return False, "", time.time() - t0, f"exception: {e}"

def main():
    print(f"{'='*70}")
    print(f" SUITE DE NON-REGRESSION dsp_inference (post refactor MM RTL v4.5t)")
    print(f"{'='*70}\n")

    results = []
    for script, desc, pattern, tout in TESTS:
        path = HERE / script
        if not path.exists():
            results.append((script, desc, False, 0, "fichier absent"))
            print(f"  [SKIP] {script:30s} : fichier absent")
            continue
        print(f"  [RUN ] {script:30s} : {desc}...")
        ok, out, dur, why = run_test(script, pattern, tout)
        flag = "PASS" if ok else "FAIL"
        results.append((script, desc, ok, dur, why))
        # Affiche last line utile en cas de fail
        if not ok:
            last_lines = [l for l in out.strip().splitlines() if l.strip()][-3:]
            print(f"          {flag}  ({dur:.1f}s)  {why}")
            for l in last_lines:
                print(f"            > {l[:100]}")
        else:
            print(f"          {flag}  ({dur:.1f}s)")
        print()

    # Resume
    print(f"\n{'='*70}")
    print(f" RESUME")
    print(f"{'='*70}")
    print(f"{'Test':30s} {'Statut':6s} {'Duree':>8s}  {'Raison':30s}")
    print(f"{'-'*70}")
    n_pass = n_fail = 0
    total_dur = 0.0
    for script, desc, ok, dur, why in results:
        flag = "PASS" if ok else "FAIL"
        print(f"{script:30s} {flag:6s} {dur:>6.1f}s  {why[:30]}")
        if ok: n_pass += 1
        else:  n_fail += 1
        total_dur += dur
    print(f"{'-'*70}")
    print(f"Total: {n_pass} PASS, {n_fail} FAIL  -- duree totale: {total_dur:.0f}s")
    return 0 if n_fail == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
