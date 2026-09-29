"""Python reference for FRUIT parametric only.

Usage: python run_python_b3.py <problem.json> <outdir>
Writes <outdir>/parametric.json with {"spectrum": [...]}.
"""
import json
import os
import sys

import numpy as np
from bssunfold.core._fruit import solve_parametric as fruit_solve_parametric


def main(problem_path, outdir):
    os.makedirs(outdir, exist_ok=True)
    prob = json.load(open(problem_path))
    A = np.array(prob["A"])
    b = np.array(prob["b"])
    E_MeV = np.array(prob["E_MeV"])
    n = A.shape[1]
    le = np.log10(E_MeV)
    log_steps = np.empty(n)
    log_steps[0] = le[1] - le[0]
    log_steps[-1] = le[-1] - le[-2]
    log_steps[1:-1] = (le[2:] - le[:-2]) / 2

    spectrum, success, message, nfev = fruit_solve_parametric(A, b, E_MeV, log_steps)
    print(f"parametric: success={success} message={message!r} nfev={nfev} "
          f"res={np.linalg.norm(A @ spectrum - b):.6g}")
    json.dump({"spectrum": np.asarray(spectrum, dtype=float).tolist()},
              open(os.path.join(outdir, "parametric.json"), "w"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
