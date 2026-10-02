# bssunfold-julia (Python bridge)

Drop-in Julia acceleration for [`bssunfold`](https://github.com/Radiationsafety/bssunfold).

If the `bssunfold` package and the Python module `bssunfold_julia` are both
installed, the module patches `bssunfold.solve_*` (and `Detector.unfold_*`) so
that:

1. **Where a Julia implementation is available** — computation is routed to
   `BSSUnfold.jl` (2–25× speedup on typical problems).
2. **Where no Julia implementation exists** or the Julia call fails — a
   fallback to the original Python code of `bssunfold` is performed.

## Installation

```bash
# Step 1. Install the Julia runtime bindings
pip install juliacall

# Step 2. Install the bssunfold-julia bridge
pip install git+https://github.com/Radiationsafety/BSSUnfold.jl#subdirectory=python_bridge

# Step 3. Julia dependencies are resolved via juliapkg.json
export JULIA_PKG_PRECOMPILE_AUTO=1
```

## Usage

```python
import bssunfold
import bssunfold_julia  # activates the Julia patch

# From here on everything works as usual — bssunfold, but fast:
detector = bssunfold.Detector(...)
result = detector.unfold_mlem(readings)   # ← already Julia
result = detector.unfold_gravel(readings) # ← Julia
result = detector.unfold_mcmc(readings)   # ← Python fallback if Turing is unavailable
```

## What is ported to Julia

BSSUnfold.jl (v0.4.0) provides **64+ solvers** (73 including auxiliary
variants); the bridge routes the following
methods through Julia: (all with the `✅ Julia` status):

| Algorithm      | Status    | Speedup |
|----------------|-----------|---------|
| MLEM (+stop)   | ✅ Julia  | 3–25×   |
| GRAVEL         | ✅ Julia  | 2–12×   |
| Landweber      | ✅ Julia  | 2–11×   |
| MAXED (amaxed/imaxed) | ✅ Julia | 2–8× |
| Tikhonov (+tv/nnls/legendre) | ✅ Julia | 5–10× |
| TSVD           | ✅ Julia  | 3–8×    |
| Sandii         | ✅ Julia  | 2–10×   |
| Bunki (+ut/re) | ✅ Julia  | 2–10×   |
| Kaczmarz (+randomized) | ✅ Julia | 2–15×   |
| CGLS           | ✅ Julia  | 2–8×    |
| FISTA          | ✅ Julia  | 2–6×    |
| BSREM          | ✅ Julia  | 3–8×    |
| OSEM           | ✅ Julia  | 4–12×   |
| Staysl         | ✅ Julia  | 2–8×    |
| Doroshenko/Ferdor | ✅ Julia | 2–8×  |
| SART/MAPEM     | ✅ Julia  | 2–8×    |
| Bayes/Bayes-spline | ✅ Julia | 2–6× |
| EKI/Express/Ensemble | ✅ Julia | 2–6× |
| OMP/KSVD/NN-OMP/NNKSVD | ✅ Julia | 2–10× |
| Sl0/CS/Binned/NNLS-topk | ✅ Julia | 2–6× |
| GKS/MAEO/MAEO-ensemble | ✅ Julia | 2–6× |
| NSDUAZ/NSpline | ✅ Julia  | 2–8×    |
| Hybrid GMRES/Parametric | ✅ Julia | 2–6× |
| Parametric/Parametric2 | ✅ Julia | 2–6× |
| Cvxpy/QPsolvers | ✅ Julia (hard deps) | 2–5× |
| MCMC (NUTS, Turing.jl) | ✅ Julia (lazy) | 2–5× |
| Genetic (native PSO/GA/DE/GWO/NSGA-II) | ✅ Julia | 2–6× |
| QUBO (binary + annealing) | ✅ Julia | 2–6× |
| Coordinate descent | ✅ Julia | ~11× |
| L-BFGS-B (native, no Optim.jl) | ✅ Julia | ~2.4× |
| PGD | ✅ Julia | ~1× |
| Extragradient / Subgradient / Frank-Wolfe / LOUHI / RFSP | ✅ Julia | parity (not yet faster than scipy) |
| ADMM | ✅ Julia | 1.3–1.7× (x-update on a Gram active set, see below) |

The batch-4 speedups above are single measurements on a 14×640 response matrix
(`scripts/pyval`, warm JIT). The first-order solvers were ported for coverage
and numerical parity first — results match Python to relL2 ≈ 1e-15, but the
allocation-bound inner loops still need an optimisation pass.

`ADMM` was the exception: handing the augmented design
`[A; sqrt(rho) I; sqrt(rho) D]` to the pure-Julia Lawson-Hanson solver made it
40–80× *slower* than scipy at n>200 (~9 s and ~15 s per iteration on 14×640),
because that solver refactorises its least-squares subproblem from scratch at
every pivot. The x-update now states the same problem on the Gram system with an
incremental Cholesky factor (`_admm_nnls_gram` in `src/algorithms/admm.jl`):
it returns the identical active set (agreement 1e-15), and measured 13.9 s
against scipy's 24.0 s for `l1_penalty=1e-3` (214 iterations both sides) and
148 s against 193 s for `tv_penalty=1e-3`.

## Python fallback

Methods with heavy Python-only dependencies (CPLEX/docplex, SCIP,
z3-solver, zfit + TensorFlow, pyoptexplain, ODL, lmfit-based variants) run via
the Python fallback — see the `PYTHON_FALLBACK` list in
`bssunfold_julia/__init__.py`.
