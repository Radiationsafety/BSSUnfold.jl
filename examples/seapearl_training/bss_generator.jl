# bss_generator.jl — SeaPearl problem generator for Bonner-sphere (BSS)
# spectrum-unfolding instances.
#
# This file is part of the CP+RL training pipeline for `solve_seapearl`
# (see examples/45-seapearl.jl and train_seapearl_bss.jl). It follows the
# methodology of corail-research/learning-generic-csp ("instance
# distribution → generic CSP encoding → GNN value-selection agent"), but is
# implemented natively on top of SeaPearl.jl's `AbstractModelGenerator`
# interface so that the trained `LearnedHeuristic` transfers 1:1 to the
# CP models built by `BSSUnfold.solve_seapearl`.
#
# Two generators are provided:
#   * `BSSUnfoldingGenerator` — samples randomized physically-flavored BSS
#     instances (mixture spectra + smooth response matrices + noise). This is
#     the training distribution.
#   * `FixedInstanceGenerator` — wraps ONE concrete instance (e.g. the
#     materialized IAEA reference problem in
#     examples/data/seapearl_iaea_instance.json) for evaluation/benchmarking.
#
# IMPORTANT — keep in sync: the CP encoding below mirrors
# src/algorithms/seapearl_csp.jl::_seapearl_build_model (flux quantization,
# kσ-compatibility weighted sums over IntVarViewMul views, adjacent-bin
# smoothing over IntVarViewOffset views). Any change there must be reflected
# here and vice versa, otherwise a learned heuristic will not transfer.

using SeaPearl
using Random
using Statistics
using LinearAlgebra

# ─── Shared CP encoding (mirror of _seapearl_build_model) ────────────────────

"""
    add_bss_cp_constraints!(model, A, b, sigma, mv, n_levels, k_sigma,
                            smooth_bins; integer_scale=1e8)

Build the BSS unfolding CSP inside `model`:

  * `n` branchable `IntVar`s `q_j ∈ 0..n_levels-1` (flux quantization grid,
    `Δ = mv/(n_levels-1)`, names `"phi_1" ... "phi_n"`);
  * per response row `i`: `b_i - kσ_i ≤ Σ_j A_ij·Δ·q_j ≤ b_i + kσ_i`
    expressed as an integer weighted sum `Σ_j c_ij·q_j ∈ [lo_i, hi_i]`
    (`c_ij = round(A_ij·Δ·s)`) over `IntVarViewMul` views (negative
    coefficients via `IntVarViewOpposite`), with the window widened by the
    worst-case coefficient-rounding tolerance `0.5·n·(n_levels-1)`;
  * smoothing `|q_{j+1} - q_j| ≤ smooth_bins` via two `LessOrEqual`
    constraints on `IntVarViewOffset` views.

Returns `(qs, delta)` — the vector of quantized-flux variables and Δ.
"""
function add_bss_cp_constraints!(model::SeaPearl.CPModel,
                                 A::Matrix{Float64}, b::Vector{Float64},
                                 sigma::Vector{Float64}, mv::Float64,
                                 n_levels::Int, k_sigma::Float64,
                                 smooth_bins::Int;
                                 integer_scale::Float64=1e8)
    m, n = size(A)
    delta = mv / (n_levels - 1)
    s = Float64(integer_scale)

    C = round.(Int, A .* (delta * s))
    c_max = isempty(C) ? 0 : maximum(abs.(C))
    max_pred = Float64(n) * (n_levels - 1) * c_max
    max_pred < 2e18 || throw(ArgumentError(
        "Integer coefficients overflow risk (max prediction $max_pred); " *
        "reduce `integer_scale` or `n_levels`."))

    tol = 0.5 * n * (n_levels - 1)

    trailer = model.trailer
    qs = SeaPearl.IntVar[]
    for j in 1:n
        q = SeaPearl.IntVar(0, n_levels - 1, "phi_$j", trailer)
        SeaPearl.addVariable!(model, q)
        push!(qs, q)
    end

    infeasible_flag = Ref(false)
    for i in 1:m
        lo = ceil(Int, (b[i] - k_sigma * sigma[i]) * s - tol)
        hi = floor(Int, (b[i] + k_sigma * sigma[i]) * s + tol)
        lo > hi && continue
        terms = SeaPearl.AbstractIntVar[]
        for j in 1:n
            c = C[i, j]
            c == 0 && continue
            if c > 0
                push!(terms, SeaPearl.IntVarViewMul(qs[j], c, "w_$(i)_$(j)"))
            else
                opp = SeaPearl.IntVarViewOpposite(qs[j], "n_$(i)_$(j)")
                push!(terms, SeaPearl.IntVarViewMul(opp, -c, "w_$(i)_$(j)_opp"))
            end
        end
        if isempty(terms)
            (lo ≤ 0 ≤ hi) || (infeasible_flag[] = true)
            continue
        end
        SeaPearl.addConstraint!(model, SeaPearl.SumGreaterThan(terms, lo, trailer))
        SeaPearl.addConstraint!(model, SeaPearl.SumLessThan(terms, hi, trailer))
    end
    if infeasible_flag[]
        SeaPearl.addConstraint!(model,
            SeaPearl.SumGreaterThan([qs[1]], n_levels, trailer))
    end

    if smooth_bins > 0
        for j in 1:n-1
            up = SeaPearl.IntVarViewOffset(qs[j], smooth_bins, "off_$(j)_up")
            SeaPearl.addConstraint!(model, SeaPearl.LessOrEqual(qs[j+1], up, trailer))
            dn = SeaPearl.IntVarViewOffset(qs[j+1], smooth_bins, "off_$(j)_dn")
            SeaPearl.addConstraint!(model, SeaPearl.LessOrEqual(qs[j], dn, trailer))
        end
    end
    return qs, delta
end

"""
    flux_ceiling(A, b, max_value) -> Float64

Flux-ceiling estimator — identical logic to
`BSSUnfold._seapearl_flux_ceiling` (2 × max of the clipped minimum-norm LS
solution, with a flux-matched fallback), so that a model built here is
indistinguishable from one built by `solve_seapearl`.
"""
function flux_ceiling(A::Matrix{Float64}, b::Vector{Float64},
                      max_value::Union{Nothing,Float64}=nothing)
    max_value !== nothing && return max_value
    n = size(A, 2)
    mv = 0.0
    try
        x_est = max.(A \ b, 0.0)
        mv = 2.0 * maximum(x_est)
    catch
        mv = 0.0
    end
    if !(mv > 0) || !isfinite(mv)
        denom = sum(A)
        mv = denom > 0 ? 2.0 * sum(b) / denom : 0.0
    end
    return mv > 0 && isfinite(mv) ? mv : 1.0
end

# ─── Random physically-flavored instance sampling ────────────────────────────

"""
    sample_bss_instance(rng, n_spheres, n_bins; noise_level=0.02, n_levels=8)

Sample one randomized BSS instance whose quantized spectrum is **provably
feasible** for the CP encoding used by `solve_seapearl`:

  1. sample a 3-component spectrum (Maxwellian fission `E·exp(-E/T)`, `1/E`
     intermediate tail, evaporation `exp(-E/T)`) on a logarithmic energy
     grid `1e-9 … 630.96 MeV`, plus `n_spheres` smooth log-normal response
     bumps (a caricature of Bonner-sphere response functions);
  2. compute the flux ceiling `mv` with the *same estimator* as the solver
     (`2 × max` of the clipped minimum-norm LS solution) on a provisional
     clean fold, and quantize the spectrum onto that grid
     (`Δ = mv/(n_levels-1)`);
  3. project the quantized spectrum onto the adjacent-bin smoothness band
     (forward/backward clamping, `|q_{j+1}-q_j| ≤ smooth_bins`) so the truth
     satisfies every constraint;
  4. fold exactly: `b = A·φ_q·(1 + ε)`, `|ε| ≤ noise_level` → the true
     quantized spectrum lies inside the kσ window for any `k ≥ 1`.

Returns `(A, b, sigma, mv, phi_quant)`.
"""
function sample_bss_instance(rng::AbstractRNG, n_spheres::Int, n_bins::Int;
                             noise_level::Float64=0.02, n_levels::Int=8,
                             smooth_bins::Int=max(1, n_levels ÷ 3))
    E = collect(10.0 .^ range(log10(1e-9), log10(630.9573444801944),
                              length=n_bins))

    # 3-component mixture spectrum (fission + 1/E + evaporation)
    w = rand(rng, 3); w ./= sum(w)
    T_f  = exp10(rand(rng) * (log10(1.6) - log10(0.8)) + log10(0.8))   # fission temperature
    T_e  = exp10(rand(rng) * (log10(2.5) - log10(0.5)) + log10(0.5))   # evaporation temperature
    fission = E .* exp.(-E ./ T_f)
    invE    = 1.0 ./ E
    evap    = exp.(-E ./ T_e)
    phi = w[1] .* fission .+ w[2] .* invE .+ w[3] .* evap
    phi ./= maximum(phi)
    phi .*= 0.4 + 0.6 * rand(rng)

    # Smooth log-normal response bumps, peaks spread over the grid
    A = zeros(Float64, n_spheres, n_bins)
    for i in 1:n_spheres
        center_log = log10(1e-9) + (log10(630.9573444801944) - log10(1e-9)) *
                     (i - 0.5 + 0.4 * randn(rng)) / n_spheres
        width = 0.7 + 0.7 * rand(rng)
        amp = 0.6 + 1.0 * rand(rng)
        for j in 1:n_bins
            dl = log10(E[j]) - center_log
            A[i, j] = amp * exp(-dl^2 / (2 * width^2))
        end
    end

    # Pass 1: provisional clean fold → ceiling with the solver's estimator
    mv = flux_ceiling(A, A * phi, nothing)
    delta = mv / (n_levels - 1)

    # Pass 2: quantize onto that grid and enforce the smoothness band so the
    # truth satisfies every constraint of the CP model
    phi_q = clamp.(round.(phi ./ delta) .* delta, 0.0, mv)
    for _ in 1:2   # forward + backward clamping sweeps
        for j in 1:n_bins-1
            phi_q[j+1] = clamp(phi_q[j+1], phi_q[j] - smooth_bins * delta,
                               phi_q[j] + smooth_bins * delta)
        end
        for j in n_bins-1:-1:1
            phi_q[j] = clamp(phi_q[j], phi_q[j+1] - smooth_bins * delta,
                             phi_q[j+1] + smooth_bins * delta)
        end
    end

    # Pass 3: exact fold of the quantized truth + relative noise
    b_clean = A * phi_q
    b = b_clean .* (1.0 .+ noise_level .* randn(rng, length(b_clean)))
    sigma = max.(noise_level .* abs.(b), 1e-9 * maximum(abs.(b)))

    return A, b, sigma, mv, phi_q
end

# ─── SeaPearl generators ─────────────────────────────────────────────────────

"""
    BSSUnfoldingGenerator <: SeaPearl.AbstractModelGenerator

Random BSS unfolding instances for RL training. Each episode builds a fresh
instance from `sample_bss_instance` and encodes it with the same CSP as
`solve_seapearl`. Search is bounded by `max_solutions` / `time_limit_s`
(written into `cpmodel.limit`) so training episodes always terminate.
"""
struct BSSUnfoldingGenerator <: SeaPearl.AbstractModelGenerator
    n_spheres::Int
    n_bins::Int
    n_levels::Int
    k_sigma::Float64
    noise_level::Float64
    smooth_bins::Int
    integer_scale::Float64
    max_solutions::Int
    time_limit_s::Int
end

BSSUnfoldingGenerator(n_spheres::Int, n_bins::Int, n_levels::Int=8;
                      k_sigma::Float64=2.0, noise_level::Float64=0.02,
                      smooth_bins::Int=max(1, n_levels ÷ 3),
                      integer_scale::Float64=1e8,
                      max_solutions::Int=32, time_limit_s::Int=20) =
    BSSUnfoldingGenerator(n_spheres, n_bins, n_levels, k_sigma, noise_level,
                          smooth_bins, integer_scale, max_solutions,
                          time_limit_s)

function SeaPearl.fill_with_generator!(cpmodel::SeaPearl.CPModel,
                                       gen::BSSUnfoldingGenerator;
                                       rng::AbstractRNG=MersenneTwister())
    A, b, sigma, mv, _ = sample_bss_instance(rng, gen.n_spheres, gen.n_bins;
                                             noise_level=gen.noise_level,
                                             n_levels=gen.n_levels,
                                             smooth_bins=gen.smooth_bins)
    add_bss_cp_constraints!(cpmodel, A, b, sigma, mv, gen.n_levels,
                            gen.k_sigma, gen.smooth_bins;
                            integer_scale=gen.integer_scale)
    cpmodel.limit = SeaPearl.Limit(nothing, gen.max_solutions,
                                   gen.time_limit_s, nothing)
    return cpmodel
end

"""
    FixedInstanceGenerator <: SeaPearl.AbstractModelGenerator

Wraps one concrete unfolding instance (already measured/quantized) as a
SeaPearl generator — used to benchmark a trained heuristic against the
`BasicHeuristic` on a fixed problem (e.g. the materialized IAEA reference
instance).
"""
struct FixedInstanceGenerator <: SeaPearl.AbstractModelGenerator
    A::Matrix{Float64}
    b::Vector{Float64}
    sigma::Vector{Float64}
    mv::Float64
    n_levels::Int
    k_sigma::Float64
    smooth_bins::Int
    integer_scale::Float64
    max_solutions::Int
    time_limit_s::Int
end

function SeaPearl.fill_with_generator!(cpmodel::SeaPearl.CPModel,
                                       gen::FixedInstanceGenerator;
                                       rng::AbstractRNG=MersenneTwister())
    qs, delta = add_bss_cp_constraints!(cpmodel, gen.A, gen.b, gen.sigma,
                                        gen.mv, gen.n_levels, gen.k_sigma,
                                        gen.smooth_bins;
                                        integer_scale=gen.integer_scale)
    cpmodel.limit = SeaPearl.Limit(nothing, gen.max_solutions,
                                   gen.time_limit_s, nothing)
    return cpmodel
end
