# BSSUnfold.jl

Julia port of the **bssunfold** package for neutron spectrum unfolding with
Bonner Sphere Spectrometers (BSS).

[![CI](https://github.com/Radiationsafety/BSSUnfold.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/Radiationsafety/BSSUnfold.jl/actions/workflows/CI.yml)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/2418bcb172b34722bef6329fda54d02e)](https://app.codacy.com/gh/Radiationsafety/BSSUnfold.jl/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)
[![Documentation](https://img.shields.io/badge/docs-blue.svg)](https://radiationsafety.github.io/BSSUnfold.jl/)
[![License: GPL-3.0](https://img.shields.io/badge/License-GPL--3.0-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

## Installation

The package is not yet registered in General. Until then:

```julia
using Pkg
Pkg.add(url="https://github.com/Radiationsafety/BSSUnfold.jl")
```

After registration this becomes `Pkg.add("BSSUnfold")`.

## Quick start

```julia
using BSSUnfold

# Default spectrometer: real GSF response functions, ICRP-116 coefficients
detector = Detector()

# Simulated measurement: count rates = response functions folded with a
# plausible AmBe-like flux
flux = exp.(-detector.config.E_MeV ./ 2.0)
rates = [sum(detector.config.sensitivities[name] .* flux)
         for name in detector.config.detector_names]

# Measured count rates keyed by sphere name
readings = Dict{String,Float64}(n => r for
    (n, r) in zip(detector.config.detector_names, rates))

# High-level Detector API
out = unfold_gravel(detector, readings; max_iterations=500)
println("Converged in $(out["iterations"]) iterations")

# Low-level solver on an explicit response matrix
A, b, _ = build_system(readings, detector.config.detector_names,
                       detector.config.sensitivities)
x0 = fill(0.5, size(A, 2))
result = solve_gravel(A, b, x0; max_iterations=500)

# Monte-Carlo uncertainty
mc = monte_carlo_uncertainty(solve_mlem, A, b, x0, 0.01, 100; random_state=42)
println("Mean σ: $(mean(mc.std))")
```

## Available algorithms

BSSUnfold.jl provides **90+ unfolding solvers** (`solve_*`) plus **55+ high-level `Detector` wrappers**
(`unfold_*`). The complete list:

| Category                       | Solvers                                                                                     |
|--------------------------------|----------------------------------------------------------------------------------------------|
| EM-family                      | `solve_mlem`, `solve_mlem_stop`, `solve_osem`, `solve_bsrem`, `solve_mapem`, `solve_sart`     |
| Weighted (BSS classics)        | `solve_gravel`, `solve_maxed`, `solve_amaxed`, `solve_amaxed_regularization`, `solve_imaxed`   |
| Direct / regularized           | `solve_tikhonov`, `solve_tikhonov_nnls`, `solve_tikhonov_tv`, `solve_tikhonov_legendre`, `solve_tsvd`, `solve_direct`, `solve_scipy_direct`, `solve_statreg`, `solve_reconst` |
| Iterative least squares        | `solve_landweber`, `solve_kaczmarz`, `solve_randomized_kaczmarz`, `solve_cgls`, `solve_fista`, `solve_lanczos`, `solve_iterative_refinement`, `solve_hybrid_gmres` |
| Classical BSS                  | `solve_sandii`, `solve_bunki`, `solve_bunkiut`, `solve_rebunki`, `solve_staysl`, `solve_doroshenko`, `solve_ferdor`, `solve_doroshenko` |
| Bayesian                       | `solve_bayes`, `solve_bayes_spline`, `solve_mcmc` (NUTS via Turing.jl, lazy load)              |
| Sparse / dictionary learning   | `solve_omp`, `solve_ksvd`, `solve_nn_omp`, `solve_nnksvd`, `solve_nnls_topk`, `solve_sl0`, `solve_cs` |
| Parametric / hybrid            | `solve_parametric`, `solve_parametric2`, `solve_hybrid_parametric`, `solve_binned`, `solve_crystal_ball` |
| N-splines                      | `solve_nspline`, `solve_nspline_full` (Islamgulov & Lartsev, 2008)                             |
| Catalogue + SPUNIT             | `solve_nsduaz` — automatic initial-spectrum selection from a built-in catalogue                |
| Global optimization            | `solve_genetic` (native PSO/GA/DE/GWO/NSGA-II), `solve_qubo` (binary encoding + simulated annealing), `solve_eki` |
| Constraint programming (CP)    | `solve_seapearl` (SeaPearl.jl feasibility-set enumeration: kσ-compatible spectra + per-bin interval estimates + optional RL value heuristic; lazy load) |
| Convex optimization            | `solve_cvxpy` (Convex.jl + SCS), `solve_qpsolvers` (OSQP.jl), `solve_parametric_cvxpy`, `solve_parametric_qpsolvers` |
| First-order optimization       | `solve_pgd`, `solve_coordinate_descent`, `solve_extragradient` (Korpelevich), `solve_subgradient`, `solve_frank_wolfe` (with away steps), `solve_admm`, `solve_lbfgsb` (native bounded L-BFGS, no Optim.jl) |
| Additional classical BSS       | `solve_rfsp` (Fischer), `solve_louhi` (Routti & Sandberg 1980, LOUHI78) |
| Optional JuMP backend          | `solve_docplex`, `solve_scip`, `solve_commercial` (+ `solve_gurobi`/`mosek`/`cplex`/`copt`/`xpress`), `solve_interval`/`_tol`/`_posterior`, `solve_nnqp`, `solve_qpmad` (enabled when JuMP is loaded; graceful degradation without it) |
| Native-portable additions      | `solve_lavrentiev` (gram/padded/iterated/direct), `solve_mirror_descent` (Bregman: entropy/log/l2/pnorm), `solve_osem_anlm` (OSEM + Adaptive NLM + Immerkaer noise), `solve_tikhonov_sobolev_dp` (generalized discrepancy, Brent / Newton–Kantorovich), `solve_bayesian_parametric` (5-parameter Maxwellian + 1/E + evaporation, Metropolis–Hastings) — no external backend required |
| Other ported methods           | `solve_directed_divergence`, `solve_bunkiut`, `solve_ensemble`, `solve_express`, `solve_gks`, `solve_maeo`, `solve_maeo_ensemble`, `solve_bon95_*` and others |
| Dev-branch (v0.5.0)            | `solve_rfsp_jul` (damped least squares), `solve_amg` (preconditioned Krylov: PCG/BiCGSTAB/GMRES + Jacobi/SOR/SSOR), `solve_uno` (filter-SQP & interior-point NLP presets), `solve_ssr` (sisireg sign-parsimony), `solve_mlem_bs` (B-spline sieve MLEM), `solve_pspline_reml` (REML smoothing selection) |

See [Algorithms](docs/src/algorithms.md) for the full annotated list.

Convex/SCS/OSQP/JSON are **hard dependencies** in `Project.toml` (v0.5.0) and
are installed automatically with the package. Two solvers degrade
gracefully when their optional dependency is missing:

- `solve_mcmc` — loads Turing.jl lazily;
- `solve_seapearl` — loads SeaPearl.jl lazily (CP feasibility-set enumeration;
  SeaPearl 0.4.x declares `julia = "1.8 - 1.9"` upstream. On Julia 1.10 (the
  BSSUnfold requirement) use the compat fork — a declaration-only change,
  no source differences — and the CP solver + the RL heuristic run in ONE
  session:

  ```julia
  julia examples/seapearl_training/setup_seapearl_env.jl
  julia --project=examples/seapearl_training examples/09-seapearl.jl
  ```

  or, in an existing environment:

  ```julia
  Pkg.add(url = "https://github.com/Radiationsafety/SeaPearl.jl",
          rev = "compat/julia-1.10")
  # Registry metadata for GPUCompiler 0.17.3 is stricter than the tag itself;
  # the git pin restores the proven Flux 0.12 + CUDA 3 stack on 1.10:
  Pkg.add(url = "https://github.com/JuliaGPU/GPUCompiler.jl", rev = "v0.17.3")
  # Julia 1.8-1.9 still works with the plain registry version:
  #   Pkg.add(name = "SeaPearl", version = "0.4.5")
  ```

All other solvers work out of the box.

A complete worked example — CP unfolding of the IAEA `ISO_ref_AmBe` reference
spectrum with feasible-set intervals, dose comparison and the optional
CP+RL learned heuristic (training pipeline in `examples/seapearl_training/`)
— is available in [`examples/09-seapearl.jl`](examples/09-seapearl.jl).

## Performance

On a 14×640 response matrix (a typical BSS problem size):

| Algorithm  | Python (ms) | Julia (ms) | Speedup |
|------------|-------------|------------|---------|
| MLEM       | 52          | 6          | **3.6×**|
| GRAVEL     | 12          | 5          | **2.6×**|
| Landweber  | 24          | 10         | **2.4×**|

A systematic benchmark of all methods against IAEA reference spectra is
available in `examples/05-methods_comparison.jl` (metric-based ranking via
`benchmark_unfold_methods`).

## Documentation

- [Tutorial](docs/src/tutorial.md)
- [Algorithms](docs/src/algorithms.md)
- [API Reference](docs/src/api.md)
- [Comparison with Python bssunfold](docs/src/comparison.md)

## Examples

The `examples/` directory contains Pluto.jl notebooks. See
[examples/README.md](examples/README.md).

| Notebook                          | Description                                            |
|-----------------------------------|---------------------------------------------------------|
| `01-basic-example.jl`             | Basic unfolding with GRAVEL                              |
| `02-uncertainty.jl`               | Monte-Carlo uncertainty estimation                       |
| `03-mlem_example.jl`              | MLEM: effect of iterations and x₀                        |
| `04-regularization.jl`            | Tikhonov/TSVD and λ selection                            |
| `05-methods_comparison.jl`        | Comparison of all solvers (metric ranking)               |
| `06-robustness_analysis.jl`       | Robustness analysis (noise, x₀, random seeds)            |
| `07-real-spectra.jl`              | Real IAEA spectra: RF, reference spectra and dose rates  |
| `08-python-julia-parity.ipynb`    | Julia ↔ Python bssunfold parity (Jupyter)                |
| `09-seapearl.jl`                  | SeaPearl CP unfolding with feasible-set intervals        |

### Running Pluto notebooks

```bash
julia -e 'using Pkg; Pkg.add.(["Pluto", "Plots"])'
julia -e 'using Pluto; Pluto.run()'
# Open http://localhost:1234 and pick a notebook from examples/
```

Notebook plotting requires `Plots.jl`; it is intentionally not a package
dependency.

## Tests

Tests live in `test/`, organized to mirror the original `bssunfold/tests/`:

```
test/
├── runtests.jl                   ← main entry point
├── test_detector.jl              ← Detector tests (port of test_detector.py)
├── test_classic_unfolders.jl     ← classic solvers (test_classic_unfolders.py)
├── test_comparison_metrics.jl    ← 52 comparison metrics, matches scipy to 1e-6
├── test_ported_methods.jl        ← 32 ported solvers
├── test_batch3_algorithms.jl     ← NSDUAZ/NSpline/MCMC/Genetic/QUBO
├── test_dev_methods.jl           ← RFSP-JUL/AMG/Uno/SSR/MLEM-BS/P-spline-REML
├── test_montecarlo.jl            ← Monte-Carlo tests
├── test_regularization.jl        ← regularization
├── test_iaea_validation.jl       ← IAEA Compendium validation
└── data/
    ├── IAEA_Compendium_dataset.csv
    └── MonteCarlo_Calculated_spectra_from_IAEA_Comp_for_comparison.csv
```

### Running tests

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

All 100+ tests pass. This includes validation against the IAEA Compendium
(29 reference spectra), smoke tests for all solvers, checks of the 52
comparison metrics against scipy (agreement to 1e-6), and Monte-Carlo
robustness checks.

## Python compatibility

Via `PythonCall.jl` the package can be called from Python:

```python
from juliacall import Main as jl
jl.seval("using BSSUnfold")
result = jl.BSSUnfold.solve_mlem(A, b, x0)
```

See `python_bridge/` for a ready-made drop-in wrapper that transparently
routes `bssunfold` calls to their Julia equivalents.

## Related resources

- [Original Python package bssunfold](https://github.com/Radiationsafety/bssunfold)
- [IAEA Compendium of neutron spectra](https://www-nds.iaea.org/bssunfold/)

## License

GPL-3.0-only — inherited from the original `bssunfold`.

## AI-assisted development

This package was developed with the assistance of large language models
(Claude, via agentic CLI coding tools). Every generated file was reviewed
by the maintainer; the code is not accepted blindly.

Where the LLM was used and how correctness is enforced:

- **Solver ports** (`src/algorithms/`, ~90 methods): translated from the
  reference Python implementation `bssunfold` and from the published
  algorithms they implement. Correctness is checked by parity tests
  against the Python reference on shared problems (e.g.
  `test_ported_methods.jl`, `test_comparison_metrics.jl` — 52 comparison
  metrics agree with SciPy to 1e-6) and by physics validation against the
  IAEA Compendium of neutron spectra (29 reference spectra,
  `test_iaea_*.jl`), all run in CI.
- **Tests, documentation, and README prose**: LLM-drafted, maintainer-edited.

Known limitation: methods whose underlying papers could not be verified
line-by-line against a reference implementation carry fewer guarantees;
their docstrings cite the source they were ported from, and users should
treat them as research code.
