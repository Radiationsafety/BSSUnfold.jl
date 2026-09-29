"""Run bssunfold (Python) reference solvers on the shared problem.

Usage: python run_python.py <problem.json> <outdir>
Writes <outdir>/<method>.json with {"spectrum": [...]} for each method.
"""
import json
import os
import sys
import traceback

import numpy as np
import bssunfold.core as c
from bssunfold.core.unfold_maeo import solve_maeo, solve_maeo_ensemble
from bssunfold.core.unfold_qubo import solve_qubo_unfold
from bssunfold.core._fruit import solve_parametric as fruit_solve_parametric

E = None


def _spectrum(res):
    if isinstance(res, tuple):
        res = res[0]
    if isinstance(res, dict):
        for key in ("spectrum", "x", "result"):
            if key in res:
                res = res[key]
                break
        else:
            res = next(iter(res.values()))
    arr = np.asarray(res, dtype=float)
    if arr.ndim > 1:
        arr = arr[0]
    return arr


def main(problem_path, outdir):
    os.makedirs(outdir, exist_ok=True)
    prob = json.load(open(problem_path))
    A = np.array(prob["A"])
    b = np.array(prob["b"])
    x0 = np.array(prob["x0"])
    E_MeV = np.array(prob["E_MeV"])
    m, n = A.shape
    le = np.log10(E_MeV)
    log_steps = np.empty(n)
    log_steps[0] = le[1] - le[0]
    log_steps[-1] = le[-1] - le[-2]
    log_steps[1:-1] = (le[2:] - le[:-2]) / 2
    ln_steps = log_steps * np.log(10.0)

    cfgs = [
        ("mlem", lambda: c.solve_mlem(A, b, x0)),
        ("gravel", lambda: c.solve_gravel(A, b, x0)),
        ("landweber", lambda: c.solve_landweber(A, b, x0)),
        ("maxed", lambda: c.solve_maxed(A, b, x0)),
        ("bunki", lambda: c.solve_bunki(A, b, x0)),
        ("bunkiut", lambda: c.solve_bunkiut(A, b, x0)),
        ("rebunki", lambda: c.solve_rebunki(A, b, x0)),
        ("nsduaz", lambda: c.solve_nsduaz(A, b, x0)),
        ("sandii", lambda: c.solve_sandii(A, b, x0)),
        ("osem", lambda: c.solve_osem(A, b, x0)),
        ("bsrem", lambda: c.solve_bsrem(A, b, x0)),
        ("cgls", lambda: c.solve_cgls(A, b, x0)),
        ("kaczmarz", lambda: c.solve_kaczmarz(A, b, x0)),
        ("randomized_kaczmarz", lambda: c.solve_randomized_kaczmarz(A, b, x0, random_state=7)),
        ("iterative_refinement", lambda: c.solve_iterative_refinement(A, b, x0)),
        ("lanczos", lambda: c.solve_lanczos(A, b, x0)),
        ("doroshenko", lambda: c.solve_doroshenko(A, b, x0)),
        ("staysl", lambda: c.solve_staysl(A, b, x0)),
        ("sart", lambda: c.solve_sart(A, b, x0)),
        ("mapem", lambda: c.solve_mapem(A, b, x0)),
        ("mlem_stop", lambda: c.solve_mlem_stop(A, b, x0)),
        ("amaxed", lambda: c.solve_amaxed(A, b, x0)),
        ("amaxed_regularization", lambda: c.solve_amaxed_regularization(A, b, x0)),
        ("imaxed", lambda: c.solve_imaxed(A, b, x0)),
        ("bayes", lambda: c.solve_bayes(A, b, x0)),
        ("bayes_spline", lambda: c.solve_bayes_spline(A, b, x0)),
        ("directed_divergence", lambda: c.solve_directed_divergence(A, b, x0)),
        ("ferdor", lambda: c.solve_ferdor(A, b, x0)),
        ("gks", lambda: c.solve_gks(A, b, x0)),
        ("tikhonov_tv", lambda: c.solve_tikhonov_tv(A, b, x0)),
        ("tikhonov_legendre", lambda: c.solve_tikhonov_legendre(A, b, x0)),
        ("tsvd", lambda: c.solve_tsvd(A, b, x0)),
        ("scipy_direct", lambda: c.solve_scipy_direct(A, b, x0)),
        ("statreg", lambda: c.solve_statreg(A, b, x0, E_MeV=E_MeV)),
        ("reconst", lambda: c.solve_reconst(A, b, x0, E_MeV=E_MeV)),
        ("express", lambda: c.solve_express(A, b, E_MeV, x0)),
        ("eki", lambda: c.solve_eki(A, b, x0)),
        ("ensemble", lambda: c.solve_ensemble(A, b, x0)),
        ("crystal_ball", lambda: c.solve_crystal_ball(A, b, x0)),
        ("cvxpy", lambda: c.solve_cvxpy(A, b, 0.01, norm=2, solver="OSQP", x0=x0)),
        ("qpsolvers", lambda: c.solve_qpsolvers(A, b, 0.01, norm=2, solver="osqp", x0=x0)),
        ("cs", lambda: c.solve_cs(A, b, x0)),
        ("nspline", lambda: c.solve_nspline(A, b, x0, E_MeV=E_MeV)),
        ("nspline_full", lambda: c.solve_nspline_full(A, b, x0, E_MeV=E_MeV)),
        ("hybrid_parametric", lambda: c.solve_hybrid_parametric(A, b, E_MeV, ln_steps)),
        ("parametric", lambda: fruit_solve_parametric(A, b, E_MeV, log_steps)),
        ("parametric2", lambda: c.solve_parametric2(A, b, E_MeV, ln_steps)),
        ("genetic", lambda: c.solve_genetic(A, b, x0)),
        ("maeo", lambda: solve_maeo(A, b, x0)),
        ("maeo_ensemble", lambda: solve_maeo_ensemble(A, b, x0)),
        ("qubo", lambda: solve_qubo_unfold(A, b, x0)),
        ("nnksvd", lambda: c.solve_nnksvd_unfold(A, b, x0)),
        ("omp", lambda: c.solve_omp(A, b, 5)),
        ("nn_omp", lambda: c.solve_nn_omp(A, b, 5)),
        ("nnls_topk", lambda: c.solve_nnls_topk(A, b, 5)),
        ("tikhonov_nnls", lambda: c.solve_tikhonov_nnls(A, b)),
        ("sl0", lambda: c.solve_sl0(A, b)),
    ]

    ok, failed = [], []
    for name, fn in cfgs:
        try:
            x = _spectrum(fn())
            json.dump({"spectrum": x.tolist()}, open(os.path.join(outdir, f"{name}.json"), "w"))
            ok.append(name)
        except Exception:
            failed.append(name)
            traceback.print_exc(limit=3)
            json.dump({"error": traceback.format_exc(limit=5)},
                      open(os.path.join(outdir, f"{name}.json"), "w"))
    print(f"python: {len(ok)} ok, {len(failed)} failed: {failed}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
