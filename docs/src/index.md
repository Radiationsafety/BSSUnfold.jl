# BSSUnfold.jl

**Julia port of the `bssunfold` package for neutron spectrum unfolding with
Bonner Sphere Spectrometers (BSS).**

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

# Create a detector
detector = Detector(detector_names, E_MeV, sensitivities, cc_icrp116)

# Unfold a spectrum
result = unfold_gravel(detector, readings, max_iterations=500)
println("Converged in $(result["iterations"]) iterations")

# Monte-Carlo uncertainty
mc = monte_carlo_uncertainty(solve_mlem, A, b, x0, 0.01, 100, random_state=42)
println("Mean σ: $(mean(mc.std))")
```

## Available algorithms

BSSUnfold.jl provides **55+ unfolding solvers** (64 including auxiliary
variants) and **29 high-level `Comparator`/`Detector` wrappers**:

| Category               | Solvers                                                                                    |
|------------------------|---------------------------------------------------------------------------------------------|
| EM-family              | `solve_mlem`, `solve_osem`, `solve_bsrem`, `solve_mapem`, `solve_sart`                       |
| Weighted               | `solve_gravel`, `solve_maxed`, `solve_amaxed`, `solve_imaxed`                                |
| Iterative              | `solve_landweber`, `solve_kaczmarz`, `solve_cgls`, `solve_fista`                             |
| Regularized            | `solve_tikhonov`, `solve_tsvd`, `solve_tikhonov_tv`, `solve_tikhonov_legendre`, `solve_bsrem`|
| Classical              | `solve_sandii`, `solve_bunki`, `solve_staysl`, `solve_doroshenko`, `solve_ferdor`            |
| Optimization           | `solve_cvxpy` (Convex.jl/SCS), `solve_qpsolvers` (OSQP.jl)                                  |
| Catalogue + SPUNIT     | `solve_nsduaz` — automatic initial spectrum from a catalogue                                 |
| N-splines              | `solve_nspline`, `solve_nspline_full` — Islamgulov & Lartsev (2008)                          |
| Metaheuristics         | `solve_genetic` — native PSO/GA/DE/GWO/NSGA-II                                               |
| QUBO                   | `solve_qubo` — binary encoding + simulated annealing                                         |
| Bayesian               | `solve_mcmc`, `solve_bayes`, `solve_bayes_spline` — NUTS via Turing.jl (lazy load)            |
| Sparse/dictionary      | `solve_omp`, `solve_ksvd`, `solve_nnksvd`, `solve_sl0`, `solve_cs`, `solve_nnls_topk`         |
| Parametric/hybrid      | `solve_parametric`, `solve_parametric2`, `solve_hybrid_parametric`, `solve_hybrid_gmres`      |
| Other                  | `solve_eki`, `solve_ensemble`, `solve_express`, `solve_crystal_ball`, `solve_gks`, `solve_maeo`, `solve_directed_divergence`, `solve_statreg`, `solve_reconst`, ... |

## Real-world data

The package ships the real constants ported from `bssunfold/constants.py`:

| Constant                                                        | Meaning                                      |
|------------------------------------------------------------------|----------------------------------------------|
| `RF_GSF`, `RF_PTB`, `RF_LANL`, `RF_JINR`, `RF_FERMILAB`, `RF_EURADOS`, `RF_IHEP` | Real BSS response functions |
| `ICRP116_COEFF_EFFECTIVE_DOSE`, `ICRP74_COEFF_EFFECTIVE_DOSE`    | Conversion coefficients (effective dose)     |
| `ICRP74_COEFF_OPERATIONAL_QUANTITIES`                            | Operational quantities (H*(10), etc.)        |
| `NRB99_2009_COEFF_EFFECTIVE_DOSE`                                | NRB-99/2009 effective-dose coefficients      |

Infrastructure: `calculate_dose_rates` / `get_coefficients` /
`interpolate_coefficients` (PCHIP, `interpolate_spectrum` /
`discretize_spectra` / `resample_to_log_grid`), Lawson–Hanson NNLS,
`Detector()` with `RF_GSF` default, `get_effective_readings_for_spectra`,
`set_dose_coefficients!`, `upper_bounds` / `max_energy_mask`.

Spectrum comparison: `compare_spectra` with **52 metrics** (exact port of
`utils/comparison.py`; matches scipy to 1e-6), `compare_multiple`, and
`benchmark_unfold_methods` with `BenchmarkResult` ranking.

## Performance

On a 14×640 response matrix (a typical BSS problem size):

| Algorithm  | Python (ms) | Julia (ms) | Speedup |
|------------|-------------|------------|---------|
| MLEM       | 52          | 6          | **3.6×**|
| GRAVEL     | 12          | 5          | **2.6×**|
| Landweber  | 24          | 10         | **2.4×**|

## Related resources

- [Original Python package bssunfold](https://github.com/Radiationsafety/bssunfold)
- [IAEA Compendium of neutron spectra](https://www-nds.iaea.org/bssunfold/)
- [Pluto example notebooks](https://github.com/Radiationsafety/BSSUnfold.jl/tree/main/examples)
