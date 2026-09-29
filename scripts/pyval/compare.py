"""Compare Julia vs Python spectra from the cross-check runs.

Usage: python compare.py <pydir> <jldir>
"""
import json
import os
import sys

import numpy as np


def metrics(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        rel = np.inf if na != nb else 0.0
        cos = 0.0
    else:
        rel = np.linalg.norm(a - b) / nb
        cos = float(a @ b / (na * nb))
    return rel, cos


def main(pydir, jldir, only=None):
    names = sorted(f[:-5] for f in os.listdir(pydir) if f.endswith(".json"))
    if only:
        names = [n for n in names if n in only]
    print(f"{'method':26s} {'relL2':>10s} {'cos':>8s}  status")
    print("-" * 66)
    fails = []
    for name in names:
        py = json.load(open(os.path.join(pydir, name + ".json")))
        jp = os.path.join(jldir, name + ".json")
        if "error" in py:
            print(f"{name:26s} {'-':>10s} {'-':>8s}  PY-ERR")
            fails.append((name, "py error"))
            continue
        if not os.path.exists(jp):
            print(f"{name:26s} {'-':>10s} {'-':>8s}  JL-MISSING")
            fails.append((name, "jl missing"))
            continue
        jl = json.load(open(jp))
        if "error" in jl:
            print(f"{name:26s} {'-':>10s} {'-':>8s}  JL-ERR: {jl['error'].splitlines()[0][:40]}")
            fails.append((name, "jl error"))
            continue
        rel, cos = metrics(jl["spectrum"], py["spectrum"])
        status = "OK" if (cos > 0.995 and rel < 0.1) else ("WEAK" if cos > 0.9 else "MISMATCH")
        if status == "MISMATCH":
            fails.append((name, f"cos={cos:.3f} rel={rel:.3f}"))
        print(f"{name:26s} {rel:10.4g} {cos:8.5f}  {status}")
    print("-" * 66)
    print(f"{len(fails)} problem(s)")
    for f in fails:
        print("  *", f)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
