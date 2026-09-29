"""Python reference for MAEO cross-check (batch b2).

Usage: python run_python_b2.py <problem.json> <outdir> [seed]
Writes <outdir>/maeo.json and <outdir>/maeo_ensemble.json with
{"spectrum": [...], "chi2": float}.
"""
import json
import os
import sys

import numpy as np
from bssunfold.core.unfold_maeo import solve_maeo, solve_maeo_ensemble


def main(problem_path, outdir, seed=11):
    os.makedirs(outdir, exist_ok=True)
    prob = json.load(open(problem_path))
    A = np.array(prob["A"])
    b = np.array(prob["b"])
    E_MeV = np.array(prob["E_MeV"])

    cfgs = [
        ("maeo", lambda: solve_maeo(A, b, E_MeV, seed=seed)),
        ("maeo_ensemble", lambda: solve_maeo_ensemble(A, b, E_MeV, seed=seed)),
    ]
    for name, fn in cfgs:
        res = fn()
        x = np.asarray(res["spectrum"], dtype=float)
        chi2 = float(np.sum((A @ x - b) ** 2) / np.dot(b, b))
        json.dump({"spectrum": x.tolist(), "chi2": chi2},
                  open(os.path.join(outdir, name + ".json"), "w"))
        print(f"py ok {name} chi2={chi2:.6g}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 11)
