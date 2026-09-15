# Comparison with bssunfold (Python)

BSSUnfold.jl is a Julia port of the Python package
[bssunfold](https://github.com/Radiationsafety/bssunfold). This document
describes the differences and the API correspondence.

## Ported algorithms

As of v0.4.0, **55+ solvers** are implemented natively in Julia (64 including
auxiliary variants), plus **29 `unfold_*` `Detector` wrappers**. All ported
methods have the status "full port" below.

| Python name                 | Julia name              | Status             |
|-----------------------------|-------------------------|--------------------|
| `solve_mlem`                | `solve_mlem`            | ✅ Full port        |
| `solve_gravel`              | `solve_gravel`          | ✅ Full port        |
| `solve_landweber`           | `solve_landweber`       | ✅ Full port        |
| `solve_maxed`               | `solve_maxed`           | ✅ Full port        |
| `solve_tikhonov` (+tv/nnls/legendre) | `solve_tikhonov_*` | ✅ Full port     |
| `solve_tsvd`                | `solve_tsvd`            | ✅ Full port        |
| `solve_sandii`              | `solve_sandii`          | ✅ Full port        |
| `solve_bunki` (+ut/re)      | `solve_bunki*`          | ✅ Full port        |
| `solve_kaczmarz` (+randomized) | `solve_kaczmarz`, `solve_randomized_kaczmarz` | ✅ Full port |
| `solve_cgls`                | `solve_cgls`            | ✅ Full port        |
| `solve_fista`               | `solve_fista`           | ✅ Full port        |
| `solve_bsrem`               | `solve_bsrem`           | ✅ Full port        |
| `solve_osem`                | `solve_osem`            | ✅ Full port        |
| `solve_staysl`              | `solve_staysl`          | ✅ Full port        |
| `solve_doroshenko`          | `solve_doroshenko`      | ✅ Full port        |
| `solve_amaxed` (+reg), `solve_imaxed` | `solve_amaxed*`, `solve_imaxed` | ✅ Full port |
| `solve_sart`, `solve_mapem`, `solve_mlem_stop` | same names | ✅ Full port |
| `solve_ferdor`, `solve_directed_divergence`, `solve_direct` | same names | ✅ Full port |
| `solve_scipy_direct`, `solve_statreg`, `solve_reconst` | same names | ✅ Full port |
| `solve_bayes`, `solve_bayes_spline`, `solve_eki` | same names | ✅ Full port |
| `solve_express`, `solve_crystal_ball`, `solve_ensemble` | same names | ✅ Full port |
| `solve_omp`, `solve_ksvd`, `solve_nn_omp`, `solve_nnksvd` | same names | ✅ Full port |
| `solve_nnls_topk`, `solve_sl0`, `solve_cs`, `solve_binned` | same names | ✅ Full port |
| `solve_gks`, `solve_maeo`, `solve_maeo_ensemble` | same names | ✅ Full port |
| `solve_nsduaz`, `solve_nspline` (+full), `solve_nnksvd` | same names | ✅ Full port |
| `solve_hybrid_gmres`, `solve_hybrid_parametric` | same names | ✅ Full port |
| `solve_parametric`, `solve_parametric2` | same names | ✅ Full port |
| `solve_mcmc` (PyMC in Python) | `solve_mcmc` (Turing.jl, lazy) | ✅ Full port |
| `solve_genetic` (mealpy in Python) | `solve_genetic` (native PSO/GA/DE/GWO/NSGA-II) | ✅ Full port |
| `solve_qubo` (pyqubo/dwave in Python) | `solve_qubo` (binary encoding + simulated annealing) | ✅ Full port |
| `solve_cvxpy`, `solve_qpsolvers` | Convex.jl+SCS, OSQP.jl | ✅ Full port (hard deps) |

## Differences from the original

The Python-side dependencies are replaced by native Julia implementations:

| Python dependency | Julia replacement                            |
|-------------------|-----------------------------------------------|
| scipy             | native Julia (wesserstein/KS via merged ECDF, power divergence formulas) |
| pymoo / mealpy    | native PSO/GA/DE/GWO/NSGA-II in `solve_genetic` |
| pyqubo / dwave-neal | binary encoding + simulated annealing in `solve_qubo` |
| PyMC              | Turing.jl in `solve_mcmc` (lazy load, graceful degradation) |
| custom N-splines  | `solve_nspline` / `solve_nspline_full`        |

Methods that require heavy, license-restricted or ecosystem-specific Python
stacks (CPLEX/docplex, SCIP, z3-solver, zfit+TensorFlow, pyoptexplain, ODL,
lmfit-based variants) remain available through the Python fallback in
`python_bridge/`.

## API comparison

### Python

```python
from bssunfold import Detector
det = Detector.from_response_functions(df)
result = det.unfold_gravel(readings, max_iterations=500, calculate_errors=True)
spectrum = result["spectrum"]
```

### Julia

```julia
using BSSUnfold
det = Detector(detector_names, E_MeV, sensitivities, cc_icrp116)
result = unfold_gravel(det, readings, max_iterations=500, calculate_errors=true)
spectrum = result["spectrum"]
```

### Key differences

1. **Types**: Julia uses `Vector{Float64}` instead of `np.ndarray`,
   `Dict{String,Float64}` instead of Python dicts.
2. **Keyword arguments**: `max_iterations=500` (same syntax as Python kwargs).
3. **Return value**: native solvers return an `UnfoldResult` struct;
   `unfold_*` wrappers return `Dict{String,Any}` for compatibility with the
   Python API.
4. **Indexing**: 1-based in Julia vs 0-based in Python.

## Real data parity

The constants module is a faithful port of `bssunfold/constants.py`:
response functions `RF_GSF`, `RF_PTB`, `RF_LANL`, `RF_JINR`, `RF_FERMILAB`,
`RF_EURADOS`, `RF_IHEP`; conversion coefficient sets ICRP-116, ICRP-74
(effective + operational quantities) and NRB-99/2009; and dose/interpolation
utilities (`calculate_dose_rates`, `get_coefficients`, PCHIP
`interpolate_spectrum`, `discretize_spectra`, `resample_to_log_grid`).

`utils/comparison.py` is ported as `src/comparison.jl`: `compare_spectra`
provides 52 metrics (matches scipy to 1e-6), plus `compare_multiple` and
`benchmark_unfold_methods` (benchmark ranking; see `BenchmarkResult`).

## Using Python and Julia together

Via `PythonCall.jl` both APIs can be used simultaneously:

```julia
using PythonCall
bssunfold_py = pyimport("bssunfold")
result_py = bssunfold_py.Detector.from_response_functions(df).unfold_gravel(readings)

using BSSUnfold
result_jl = unfold_gravel(detector, readings)
```

See `python_bridge/` for the ready-made wrapper.
