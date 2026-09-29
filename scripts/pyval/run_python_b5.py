"""Python reference for NN-KSVD only (bssunfold 0.28.0).

Usage: python run_python_b5.py <problem.json> <outdir>
Writes <outdir>/nnksvd.json with {"spectrum": [...]}.
"""
import json
import os
import sys

import numpy as np
from bssunfold.core.unfold_nnksvd import solve_nnksvd_unfold


def main(problem_path, outdir):
    os.makedirs(outdir, exist_ok=True)
    prob = json.load(open(problem_path))
    A = np.array(prob["A"])
    b = np.array(prob["b"])
    x0 = np.array(prob["x0"])

    spectrum, n_iter, converged = solve_nnksvd_unfold(A, b, x0)
    print(f"nnksvd: n_iter={n_iter} converged={converged} "
          f"res={np.linalg.norm(A @ spectrum - b):.6g}")
    json.dump({"spectrum": np.asarray(spectrum, dtype=float).tolist()},
              open(os.path.join(outdir, "nnksvd.json"), "w"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
