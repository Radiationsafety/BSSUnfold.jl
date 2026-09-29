"""Python reference for BON95 parametric2 only.

Usage: python run_python_b4.py <problem.json> <outdir>
Writes <outdir>/parametric2.json with {"spectrum": [...]}.
"""
import json
import os
import sys

import numpy as np
import bssunfold.core as c


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
    ln_steps = log_steps * np.log(10.0)

    spectrum, success, message, nfev = c.solve_parametric2(A, b, E_MeV, ln_steps)
    print(f"parametric2: success={success} message={message!r} nfev={nfev} "
          f"res={np.linalg.norm(A @ spectrum - b):.6g}")
    json.dump({"spectrum": np.asarray(spectrum, dtype=float).tolist()},
              open(os.path.join(outdir, "parametric2.json"), "w"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
