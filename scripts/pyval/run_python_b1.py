"""Run the seeded solve_eki reference (bssunfold 0.28.0) only.

Usage: python run_python_b1.py <problem.json> <outdir>
Writes <outdir>/eki.json with {"spectrum": [...], "iterations": ..., "converged": ...}.
"""
import json
import os
import sys

import numpy as np
from bssunfold.core.unfold_eki import solve_eki


def main(problem_path, outdir):
    os.makedirs(outdir, exist_ok=True)
    prob = json.load(open(problem_path))
    A = np.array(prob["A"])
    b = np.array(prob["b"])
    x0 = np.array(prob["x0"])

    x, iters, converged = solve_eki(A, b, x0, random_state=7)
    json.dump(
        {
            "spectrum": np.asarray(x, dtype=float).tolist(),
            "iterations": int(iters),
            "converged": bool(converged),
        },
        open(os.path.join(outdir, "eki.json"), "w"),
    )
    print("python b1: eki ok")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
