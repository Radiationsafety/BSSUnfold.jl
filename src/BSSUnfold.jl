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
    # Новые алгоритмы (партия 3: NSDUAZ, NSpline, MCMC, Genetic, QUBO)
    solve_nsduaz, solve_nspline, solve_nspline_full,
    solve_mcmc, solve_genetic, solve_qubo,
    select_catalogue_initial, builtin_catalogue,
    auto_knots, build_continuity_matrix, fit_nspline, nspline_eval,
    directed_divergence, NSPLINE_KNOT_PRESETS,
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
    unfold_nsduaz, unfold_nspline, unfold_mcmc, unfold_genetic, unfold_qubo,
    n_energy_bins, energy_grid, detector_names,
    save_result!,
    # Monte-Carlo
    monte_carlo_uncertainty, add_noise,
    # Regularization
    select_regularization_parameter, lcurve_selection, gcv_selection,
    # Utilities
    validate_system, build_system, normalize_initial,
    # Re-exports из LinearAlgebra
    norm

# ─── Include submodules ──────────────────────────────────────────────────────
include("types.jl")
include("utils.jl")
include("montecarlo.jl")
include("regularization.jl")
include("base_unfolder.jl")
include("detector.jl")

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
include("algorithms/nsduaz.jl")
include("algorithms/nspline.jl")
include("algorithms/mcmc.jl")
include("algorithms/genetic.jl")
include("algorithms/qubo.jl")

# ─── Version ────────────────────────────────────────────────────────────────
const VERSION = v"0.3.0"

end # module
