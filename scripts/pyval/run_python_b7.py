"""Python reference for batch 7 (first-order / convex / classical BSS ports).

Usage: python run_python_b7.py <problem.json> <outdir>
Writes <outdir>/<name>.json with {"spectrum": [...]} for each configuration.
"""
import json
import os
import sys

import numpy as np

from bssunfold.core.unfold_pgd import solve_pgd
from bssunfold.core.unfold_extragradient import solve_extragradient
from bssunfold.core.unfold_coordinate_descent import solve_coordinate_descent
from bssunfold.core.unfold_subgradient import solve_subgradient
from bssunfold.core.unfold_frank_wolfe import solve_frank_wolfe
from bssunfold.core.unfold_admm import solve_admm
from bssunfold.core.unfold_lbfgsb import solve_lbfgsb
from bssunfold.core.unfold_rfsp_jul import solve_rfsp_jul
from bssunfold.core.unfold_louhi import solve_louhi


def main(problem_path, outdir):
    prob = json.load(open(problem_path))
    A = np.array(prob["A"], float)
    b = np.array(prob["b"], float)
    x0 = np.array(prob["x0"], float)
    x_true = np.array(prob["x_true"], float)
    fluence = float(x_true.sum())

    solvers = {
        "pgd": (solve_pgd, {}),
        "pgd_l1": (solve_pgd, {"regularization": 1e-3}),
        "pgd_simplex": (solve_pgd, {"constraint": "simplex",
                                    "total_fluence": fluence}),
        "pgd_backtrack": (solve_pgd, {"backtracking": True}),
        "extragradient": (solve_extragradient, {}),
        "coordinate_descent": (solve_coordinate_descent, {}),
        "coordinate_descent_l1": (solve_coordinate_descent,
                                  {"l1_penalty": 1e-3}),
        "subgradient": (solve_subgradient, {}),
        "subgradient_l1": (solve_subgradient, {"l1_penalty": 1e-3,
                                               "step_policy": "polyak"}),
        "frank_wolfe": (solve_frank_wolfe, {"total_fluence": fluence}),
        "frank_wolfe_away": (solve_frank_wolfe, {"total_fluence": fluence,
                                                 "away_steps": False}),
        "admm_l1": (solve_admm, {"l1_penalty": 1e-3}),
        "admm_tv": (solve_admm, {"tv_penalty": 1e-3}),
        "lbfgsb": (solve_lbfgsb, {}),
        "lbfgsb_smooth": (solve_lbfgsb, {"regularization": 1e-3,
                                         "smoothness": 1e-3}),
        "rfsp": (solve_rfsp_jul, {}),
        "louhi": (solve_louhi, {}),
        "louhi_order2": (solve_louhi, {"smooth_order": 2, "auto_smooth": True}),
    }

    os.makedirs(outdir, exist_ok=True)
    for name, (fn, kw) in solvers.items():
        try:
            out = fn(A, b, x0, **kw)
            x = np.asarray(out[0], float)
            chi2 = float(np.linalg.norm(A @ x - b) ** 2
                         / np.linalg.norm(b) ** 2)
            payload = {"spectrum": x.tolist(), "chi2": chi2,
                       "iterations": int(out[1]), "converged": bool(out[2])}
            print(f"{name:22s} iters={out[1]:5d} converged={out[2]!s:5s} "
                  f"chi2={chi2:.6g}")
        except Exception as exc:  # noqa: BLE001 - record, keep the batch going
            payload = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"{name:22s} ERROR {type(exc).__name__}: {exc}")
        json.dump(payload, open(os.path.join(outdir, name + ".json"), "w"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
