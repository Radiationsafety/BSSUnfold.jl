# API Reference

```@meta
CurrentModule = BSSUnfold
```

## Types

```@docs
UnfoldResult
Detector
DetectorConfig
```

## Unfolding algorithms

Basic solvers (v0.1):

```@docs
solve_mlem
solve_gravel
solve_landweber
solve_maxed
solve_tikhonov
solve_tsvd
solve_sandii
solve_bunki
solve_kaczmarz
solve_cgls
solve_fista
solve_bsrem
solve_osem
solve_staysl
solve_doroshenko
```

Extensions (v0.2):

```@docs
solve_lanczos
solve_iterative_refinement
solve_randomized_kaczmarz
solve_cvxpy
solve_qpsolvers
```

Ported algorithms (v0.4 — 55+ solvers in total; representative subset shown,
see the Algorithms page for the full list):

```@docs
solve_mlem_stop
solve_mapem
solve_sart
solve_bunkiut
solve_rebunki
solve_amaxed
solve_amaxed_regularization
solve_imaxed
solve_directed_divergence
solve_ferdor
solve_scipy_direct
solve_direct
solve_tikhonov_tv
solve_tikhonov_nnls
solve_tikhonov_legendre
solve_statreg
solve_reconst
solve_bayes
solve_bayes_spline
solve_eki
solve_express
solve_crystal_ball
solve_ensemble
solve_omp
solve_ksvd
solve_nn_omp
solve_nnksvd
solve_nnls_topk
solve_sl0
solve_cs
solve_binned
solve_gks
solve_maeo
solve_maeo_ensemble
solve_nsduaz
solve_nspline
solve_nspline_full
solve_hybrid_gmres
solve_hybrid_parametric
solve_parametric
solve_parametric2
solve_genetic
solve_qubo
solve_mcmc
```

## Detector methods (high-level)

29 wrappers, one per supported algorithm:

```@docs
unfold_mlem
unfold_gravel
unfold_landweber
unfold_maxed
unfold_tikhonov
unfold_tsvd
unfold_sandii
unfold_bunki
unfold_kaczmarz
unfold_cgls
unfold_fista
unfold_bsrem
unfold_osem
unfold_staysl
unfold_doroshenko
unfold_lanczos
unfold_iterative_refinement
unfold_randomized_kaczmarz
unfold_cvxpy
unfold_qpsolvers
unfold_hybrid_gmres
unfold_hybrid_parametric
unfold_parametric
unfold_parametric2
unfold_nsduaz
unfold_nspline
unfold_mcmc
unfold_genetic
unfold_qubo
```

## Framework

```@docs
run_unfolding
make_solve_wrapper
build_system
normalize_initial
validate_system
```

## Monte-Carlo uncertainty

```@docs
monte_carlo_uncertainty
add_noise
```

## Regularization

```@docs
select_regularization_parameter
lcurve_selection
gcv_selection
```

## Real data and dose calculation

Constants ported from `bssunfold/constants.py`:

| Constant                                                        | Meaning                                     |
|------------------------------------------------------------------|---------------------------------------------|
| `RF_GSF`, `RF_PTB`, `RF_LANL`, `RF_JINR`, `RF_FERMILAB`, `RF_EURADOS`, `RF_IHEP` | Real BSS response functions |
| `ICRP116_COEFF_EFFECTIVE_DOSE`, `ICRP74_COEFF_EFFECTIVE_DOSE`    | Effective dose conversion coefficients      |
| `ICRP74_COEFF_OPERATIONAL_QUANTITIES`                            | Operational quantities                      |
| `NRB99_2009_COEFF_EFFECTIVE_DOSE`                                | NRB-99/2009 dosimetry                       |

```@docs
get_icrp116_coefficients
get_coefficients
interpolate_coefficients
calculate_dose_rates
get_effective_readings_for_spectra
set_dose_coefficients!
max_energy_mask
upper_bounds
```

## Interpolation (PCHIP)

```@docs
interpolate_spectrum
discretize_spectra
resample_to_log_grid
```

## Spectrum comparison (port of `utils/comparison.py`)

`compare_spectra` computes **52 metrics**; the implementation matches scipy to
1e-6 (validated in `test/test_comparison_metrics.jl`).

```@docs
compare_spectra
compare_multiple
benchmark_unfold_methods
total_flux
kl_divergence
cross_entropy
wasserstein_dist
kolmogorov_smirnov_stat
pearson_r
spearman_r
chi_squared
anderson_darling
wilcoxon_test
mannwhitneyu_test
cosine_similarity
r2_score
mean_squared_error
mean_absolute_error
mape
dose_difference_percent
dose_averaged_energy
ambient_dose_equivalent_rate
spectral_shape_similarity
comprehensive_score
available_metrics
```

## Detector utilities

```@docs
save_result!
n_energy_bins
energy_grid
detector_names
load_spectra_csv
```
