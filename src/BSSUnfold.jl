"""
    BSSUnfold

Julia-порт пакета bssunfold для развёртки нейтронных спектров
со спектрометров Боннера (BSS).

Основные экспорты:
- [`solve_mlem`](@ref), [`solve_gravel`](@ref), [`solve_landweber`](@ref)
- [`solve_maxed`](@ref), [`solve_tikhonov`](@ref), [`solve_tsvd`](@ref)
- [`solve_sandii`](@ref), [`solve_bunki`](@ref), [`solve_kaczmarz`](@ref)
- [`solve_cgls`](@ref), [`solve_fista`](@ref), [`solve_bsrem`](@ref), [`solve_osem`](@ref)
- [`solve_staysl`](@ref), [`solve_doroshenko`](@ref)
- [`solve_lanczos`](@ref), [`solve_iterative_refinement`](@ref), [`solve_randomized_kaczmarz`](@ref)
- [`solve_cvxpy`](@ref) (через Convex.jl extension), [`solve_qpsolvers`](@ref) (через OSQP.jl extension)
- [`solve_nsduaz`](@ref), [`solve_nspline`](@ref) (N-сплайны Исламгулова-Ларцева)
- [`solve_mcmc`](@ref) (байесовский NUTS через Turing.jl, опционально),
  [`solve_genetic`](@ref) (нативные PSO/GA/DE/GWO/NSGA-II), [`solve_qubo`](@ref) (QUBO + отжиг)
- [`Detector`](@ref), [`run_unfolding`](@ref)
- [`monte_carlo_uncertainty`](@ref)
- [`select_regularization_parameter`](@ref)
"""
module BSSUnfold

using LinearAlgebra
using Statistics
using Random
using SparseArrays
using Printf
using JSON

# ─── Public API ──────────────────────────────────────────────────────────────
export
    # Типы
    UnfoldResult, Detector, DetectorConfig,
    # Алгоритмы (15 базовых)
    solve_mlem, solve_gravel, solve_landweber,
    solve_maxed, solve_tikhonov, solve_tsvd,
    solve_sandii, solve_bunki, solve_kaczmarz,
    solve_cgls, solve_fista, solve_bsrem, solve_osem,
    solve_staysl, solve_doroshenko,
    # Новые алгоритмы (5 расширений)
    solve_lanczos, solve_iterative_refinement, solve_randomized_kaczmarz,
    solve_cvxpy, solve_qpsolvers,
    solve_gks, solve_maeo, solve_maeo_ensemble, solve_nnksvd,
    solve_nspline, solve_hybrid_gmres, solve_hybrid_parametric,
    solve_parametric, solve_parametric2,
    solve_tikhonov_nnls, solve_nn_omp, solve_nnls_topk,
    solve_ferdor, solve_scipy_direct, solve_direct, solve_tikhonov_tv,
    solve_tikhonov_legendre, solve_statreg, solve_reconst,
    # Портированные алгоритмы (этап D)
    solve_bayes, solve_bayes_spline, solve_eki, solve_express,
    solve_crystal_ball, solve_ensemble,
    solve_omp, solve_ksvd, solve_sl0, solve_cs,
    solve_binned, load_bin_lookup, save_bin_lookup, build_bin_lookup,
    solve_sart, solve_mapem, solve_mlem_stop, calculate_j_factor,
    solve_bunkiut, solve_rebunki, solve_directed_divergence,
    solve_amaxed, solve_amaxed_regularization, solve_imaxed,
    # Новые алгоритмы (партия 3: NSDUAZ, NSpline, MCMC, Genetic, QUBO)
    solve_nsduaz, solve_nspline_full,
    solve_mcmc, solve_genetic, solve_qubo,
    select_catalogue_initial, builtin_catalogue,
    nsduaz_builtin_catalogue, nsduaz_reference_index, nsduaz_select_catalogue_initial,
    directed_divergence,
    auto_knots, build_continuity_matrix, fit_nspline, nspline_eval,
    NSPLINE_KNOT_PRESETS,
    coarsen_columns, split_coarse, apply_smoother,
    spectrum_to_binary, binary_to_spectrum,
    # Framework
    run_unfolding, make_solve_wrapper,
    # Detector methods (генерируются макросом в detector.jl)
    unfold_mlem, unfold_gravel, unfold_landweber,
    unfold_maxed, unfold_tikhonov, unfold_tsvd,
    unfold_sandii, unfold_bunki, unfold_kaczmarz,
    unfold_cgls, unfold_fista, unfold_bsrem, unfold_osem,
    unfold_staysl, unfold_doroshenko,
    unfold_lanczos, unfold_iterative_refinement, unfold_randomized_kaczmarz,
    unfold_cvxpy, unfold_qpsolvers,
    unfold_hybrid_gmres, unfold_hybrid_parametric,
    unfold_parametric, unfold_parametric2,
    unfold_nsduaz, unfold_nspline, unfold_mcmc, unfold_genetic, unfold_qubo,
    # Detector API (реальные RF)
    get_effective_readings_for_spectra, set_dose_coefficients!,
    max_energy_mask, upper_bounds,
    n_energy_bins, energy_grid, detector_names,
    save_result!,
    # Monte-Carlo
    monte_carlo_uncertainty, add_noise,
    # Regularization
    select_regularization_parameter, lcurve_selection, gcv_selection,
    # Utilities
    validate_system, build_system, normalize_initial,
    # Constants (реальные данные из bssunfold/constants.py)
    ICRP116_COEFF_EFFECTIVE_DOSE, ICRP74_COEFF_EFFECTIVE_DOSE,
    ICRP74_COEFF_OPERATIONAL_QUANTITIES, NRB99_2009_COEFF_EFFECTIVE_DOSE,
    RF_GSF, RF_PTB, RF_LANL, RF_JINR, RF_FERMILAB, RF_EURADOS, RF_IHEP,
    DOSE_COEFFICIENTS_NAMES,
    # Dose calculation
    get_icrp116_coefficients, get_coefficients,
    interpolate_coefficients, calculate_dose_rates,
    # Interpolation
    interpolate_spectrum, discretize_spectra, resample_to_log_grid,
    # Data loading
    load_spectra_csv,
    # Re-exports из LinearAlgebra
    norm

# ─── Include submodules ──────────────────────────────────────────────────────
include("types.jl")
include("utils.jl")
include("constants.jl")
include("interpolation.jl")
include("dose_calculation.jl")
include("montecarlo.jl")
include("regularization.jl")
include("base_unfolder.jl")
include("detector.jl")

include("algorithms/_nnls.jl")

# Алгоритмы (каждый в своём файле для удобства поддержки)
include("algorithms/mlem.jl")
include("algorithms/gravel.jl")
include("algorithms/landweber.jl")
include("algorithms/maxed.jl")
include("algorithms/tikhonov.jl")
include("algorithms/tsvd.jl")
include("algorithms/sandii.jl")
include("algorithms/bunki.jl")
include("algorithms/kaczmarz.jl")
include("algorithms/cgls.jl")
include("algorithms/fista.jl")
include("algorithms/bsrem.jl")
include("algorithms/osem.jl")
include("algorithms/staysl.jl")
include("algorithms/doroshenko.jl")
include("algorithms/lanczos.jl")
include("algorithms/iterative_refinement.jl")
include("algorithms/randomized_kaczmarz.jl")
include("algorithms/cvxpy.jl")
include("algorithms/qpsolvers.jl")

include("algorithms/mcmc.jl")
include("algorithms/genetic.jl")
include("algorithms/qubo.jl")

include("algorithms/gks.jl")
include("algorithms/maeo.jl")
include("algorithms/nsduaz.jl")
include("algorithms/nspline.jl")
include("algorithms/nnksvd.jl")
include("algorithms/hybrid_gmres.jl")
include("algorithms/parametric.jl")
include("algorithms/hybrid_parametric.jl")
include("algorithms/parametric2.jl")
include("algorithms/ferdor.jl")
include("algorithms/scipy_direct.jl")
include("algorithms/tikhonov_tv.jl")
include("algorithms/tikhonov_legendre.jl")
include("algorithms/statreg.jl")
include("algorithms/reconst.jl")
include("algorithms/bayes.jl")
include("algorithms/bayes_spline.jl")
include("algorithms/eki.jl")
include("algorithms/express.jl")
include("algorithms/crystal_ball.jl")
include("algorithms/ensemble.jl")
include("algorithms/cs.jl")
include("algorithms/binned.jl")
include("algorithms/sart.jl")
include("algorithms/mapem.jl")
include("algorithms/mlem_stop.jl")
include("algorithms/bunkiut.jl")
include("algorithms/rebunki.jl")
include("algorithms/directed_divergence.jl")
include("algorithms/amaxed.jl")
include("algorithms/amaxed_regularization.jl")
include("algorithms/imaxed.jl")

# ─── Version ────────────────────────────────────────────────────────────────
const VERSION = v"0.4.0"

end # module
