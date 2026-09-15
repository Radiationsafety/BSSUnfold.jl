# BSSUnfold.jl

Julia port of the **bssunfold** package for neutron spectrum unfolding with
Bonner Sphere Spectrometers (BSS).

[![License: GPL-3.0](https://img.shields.io/badge/License-GPL--3.0-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

## Installation

```julia
using Pkg
Pkg.add("BSSUnfold")
# or, until the package is registered in General:
# Pkg.add(url="https://github.com/Radiationsafety/BSSUnfold.jl")
```

## Quick start

```julia
using BSSUnfold

# Create a detector (default response functions: RF_GSF, ICRP-116 coefficients)
detector = Detector()

# Unfold a spectrum directly from the response matrix
result = solve_gravel(A, readings, x0, max_iterations=500)
println("Converged in $(result.iterations) iterations")

# Or use the high-level Detector API
out = unfold_gravel(detector, readings, max_iterations=500)

# Monte-Carlo uncertainty
mc = monte_carlo_uncertainty(solve_mlem, A, b, x0, 0.01, 100, random_state=42)
println("Mean σ: $(mean(mc.std))")
```

## Available algorithms

BSSUnfold.jl provides **55+ unfolding solvers** (`solve_*`; 64 including
auxiliary and helper variants) plus **29 high-level `Detector` wrappers**
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
| Convex optimization            | `solve_cvxpy` (Convex.jl + SCS), `solve_qpsolvers` (OSQP.jl), `solve_parametric_cvxpy`, `solve_parametric_qpsolvers` |
| Other ported methods           | `solve_directed_divergence`, `solve_bunkiut`, `solve_ensemble`, `solve_express`, `solve_gks`, `solve_maeo`, `solve_maeo_ensemble`, `solve_bon95_*` and others |

See [Algorithms](docs/src/algorithms.md) for the full annotated list.

Convex/SCS/OSQP/JSON are **hard dependencies** in `Project.toml` (v0.4.0) and
are installed automatically with the package. Only `solve_mcmc` degrades
gracefully: it loads Turing.jl lazily and warns with a zero spectrum if Turing
is unavailable.

Note: `solve_mcmc` uses Turing.jl lazily (graceful degradation without it);
all other solvers work out of the box.

## Performance

On a 14×640 response matrix (a typical BSS problem size):

| Algorithm  | Python (ms) | Julia (ms) | Speedup |
|------------|-------------|------------|---------|
| MLEM       | 52          | 6          | **3.6×**|
| GRAVEL     | 12          | 5          | **2.6×**|
| Landweber  | 24          | 10         | **2.4×**|

A systematic benchmark of all methods against IAEA reference spectra is
available in `examples/33-methods_comparison.jl` (metric-based ranking via
`benchmark_unfold_methods`).

## Documentation

- 📖 [Tutorial](docs/src/tutorial.md)
- 🔬 [Algorithms](docs/src/algorithms.md)
- 📚 [API Reference](docs/src/api.md)
- 🐍 [Comparison with Python bssunfold](docs/src/comparison.md)

## Examples

The `examples/` directory contains Pluto.jl notebooks. See
[examples/README.md](examples/README.md).

| Notebook                          | Description                                            |
|-----------------------------------|---------------------------------------------------------|
| `01-basic-example.jl`             | Basic unfolding with GRAVEL                              |
| `03-uncertainty.jl`               | Monte-Carlo uncertainty estimation                       |
| `05-mlem_example.jl`              | MLEM: effect of iterations and x₀                        |
| `13-regularization.jl`            | Tikhonov/TSVD and λ selection                            |
| `33-methods_comparison.jl`        | Comparison of all solvers (metric ranking)               |
| `34-robustness_analysis.jl`       | Robustness analysis (noise, x₀, random seeds)            |
| `40-real-spectra.jl`              | Real IAEA spectra: RF, reference spectra and dose rates  |

### Running Pluto notebooks

```bash
julia -e 'using Pkg; Pkg.add("Pluto")'
julia -e 'using Pluto; Pluto.run()'
# Open http://localhost:1234 and pick a notebook from examples/
```

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

- 🐍 [Original Python package bssunfold](https://github.com/Radiationsafety/bssunfold)
- 📊 [IAEA Compendium of neutron spectra](https://www-nds.iaea.org/bssunfold/)

## License

GPL-3.0-only — inherited from the original `bssunfold`.
