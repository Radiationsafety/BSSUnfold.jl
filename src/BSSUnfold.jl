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
    # Алгоритмы
    solve_mlem, solve_gravel, solve_landweber,
    solve_maxed, solve_tikhonov, solve_tsvd,
    solve_sandii, solve_bunki, solve_kaczmarz,
    solve_cgls, solve_fista, solve_bsrem, solve_osem,
    solve_staysl, solve_doroshenko,
    # Framework
    run_unfolding, make_solve_wrapper,
    # Detector methods (генерируются макросом в detector.jl)
    unfold_mlem, unfold_gravel, unfold_landweber,
    unfold_maxed, unfold_tikhonov, unfold_tsvd,
    unfold_sandii, unfold_bunki, unfold_kaczmarz,
    unfold_cgls, unfold_fista, unfold_bsrem, unfold_osem,
    unfold_staysl, unfold_doroshenko,
    n_energy_bins, energy_grid, detector_names,
    # Monte-Carlo
    monte_carlo_uncertainty,
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

# ─── Version ────────────────────────────────────────────────────────────────
const VERSION = v"0.1.0"

end # module
