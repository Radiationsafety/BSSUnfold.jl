"""Python reference for QUBO only (bssunfold 0.28.0 solve_qubo_unfold, defaults).

Usage: python run_python_b6.py <problem.json> <outdir> [seed]
Writes <outdir>/qubo.json (and <outdir>/qubo_s<seed>.json when a seed is given)
with {"spectrum": [...], "chi2": ...}.
"""
import json
import os
import sys

import numpy as np
from bssunfold.core.unfold_qubo import solve_qubo_unfold


def dump(outdir, spectrum, chi2, seed):
    os.makedirs(outdir, exist_ok=True)
    payload = {"spectrum": np.asarray(spectrum, dtype=float).tolist(),
               "chi2": chi2}
    json.dump(payload, open(os.path.join(outdir, "qubo.json"), "w"))
    if seed is not None:
        json.dump(payload, open(os.path.join(outdir, f"qubo_s{seed}.json"), "w"))


def main(problem_path, outdir):
    prob = json.load(open(problem_path))
    A = np.array(prob["A"])
    b = np.array(prob["b"])
    x0 = np.array(prob["x0"])
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else None

    spectrum, iterations, converged = solve_qubo_unfold(A, b, x0,
                                                        random_state=seed)
    chi2 = float(np.linalg.norm(A @ spectrum - b) ** 2
                 / np.linalg.norm(b) ** 2)
    print(f"qubo seed={seed} iters={iterations} converged={converged} "
          f"chi2={chi2:.6g}")
    dump(outdir, spectrum, chi2, seed)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
