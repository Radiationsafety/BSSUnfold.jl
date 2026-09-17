"""
Constraint-Programming (CP) neutron spectrum unfolding based on **SeaPearl.jl**
(https://github.com/corail-research/SeaPearl.jl — a hybrid CP solver whose
branching heuristics can be replaced by a Reinforcement-Learning agent).

Unlike the iterative (MLEM/GRAVEL/...) or variational (Tikhonov/...) methods,
the CP formulation treats unfolding as a *finite-domain feasibility problem*:

  * the flux of every energy bin is quantized onto `n_levels` levels
    `φ_j ∈ {0, Δ, 2Δ, ..., (n_levels-1)Δ}` and represented by an integer
    variable `q_j ∈ {0, ..., n_levels-1}`;
  * every Bonner sphere reading `b_i` produces two *kσ-compatibility*
    constraints `b_i - k·σ_i ≤ Σ_j A_ij·φ_j ≤ b_i + k·σ_i`, expressed as
    integer sums after scaling the response matrix onto the integer grid;
  * a smoothing constraint `|φ_{j+1} - φ_j| ≤ δ` (physically: adjacent-bin
    flux jumps are bounded) is expressed with binary `LessOrEqual` constraints
    over shifted variable views.

The solver then **enumerates the set of all admissible spectra** (up to
`max_solutions`) instead of returning a single point estimate. This yields
what no other method in the package provides: per-bin *interval estimates*
(the min/max flux of every bin over the whole feasible set) together with the
minimum-χ² member of the feasible set. The intervals characterize the
null-space uncertainty directly — the answer to "which part of the spectrum
is actually determined by the data?".

Search strategies: DFS (`strategy=:dfs`) and Iterative Limited Discrepancy
Search (`strategy=:ilds`). The value heuristic is a `BasicHeuristic` by
default (lowest level first — canonical for enumeration), which makes the
method work out of the box with **no trained agent**. A pre-trained
SeaPearl `LearnedHeuristic` (e.g. trained with the
[learning-generic-csp](https://github.com/corail-research/learning-generic-csp)
tooling) can be injected via the `learned_heuristic` keyword, turning the
method into a full CP+RL pipeline.

Dependency: SeaPearl.jl, **lazily loaded** on the first call (same pattern as
`solve_mcmc` / Turing.jl). SeaPearl 0.4.x supports Julia 1.8–1.9; install it
in a Julia 1.9 environment (or a compat-patched fork on newer Julia):

    Pkg.add(name = "SeaPearl", version = "0.4.5")   # on Julia 1.9

Without SeaPearl the function issues a warning and returns a zero spectrum.
"""

const _SEAPEARL_LOADED = Ref(false)

"""
    _try_load_seapearl() -> Bool

Lazy load of SeaPearl.jl into `Main` on the first call of `solve_seapearl`
(the package cannot `using` a non-direct dependency from its own namespace,
but it can load it into the user environment — same pattern as Turing.jl in
`solve_mcmc`). If SeaPearl is already loaded by the user (`using SeaPearl`),
it is simply reused. Returns `true` if SeaPearl is available in `Main`.
"""
function _try_load_seapearl()
    _SEAPEARL_LOADED[] && return true
    try
        Base.eval(Main, :(using SeaPearl))
    catch err
        @warn "SeaPearl.jl could not be loaded; solve_seapearl is unavailable. " *
              "Install via: Pkg.add(name=\"SeaPearl\", version=\"0.4.5\") on Julia 1.9 " *
              "(SeaPearl 0.4.x supports Julia 1.8–1.9)." exception=err
        return false
    end
    _SEAPEARL_LOADED[] = true
    return true
end

_seapearl() = getfield(Main, :SeaPearl)

"""
    seapearl_available() -> Bool

Return `true` if SeaPearl.jl can be loaded in the current session
(i.e. `solve_seapearl` is usable). Loading is attempted at most once and
cached, mirroring `_try_load_seapearl`.
"""
function seapearl_available()
    _SEAPEARL_LOADED[] && return true
    # Probe without warning spam: a failed probe stays uncached, so a later
    # successful `using SeaPearl` by the user is still picked up.
    try
        Base.eval(Main, :(import SeaPearl))
        _SEAPEARL_LOADED[] = true
        return true
    catch
        return false
    end
end

# ─── Quantization / scaling helpers ──────────────────────────────────────────

"""
    _seapearl_flux_ceiling(A, b, max_value) -> Float64

Estimate the flux ceiling `max_value` of the quantization grid from the data:
twice the maximum of the (clipped) minimum-norm least-squares solution.
Fallback chain: flat flux-matched estimate `sum(b)/sum(A)`, then `1.0`.
A user-supplied `max_value` always wins.
"""
function _seapearl_flux_ceiling(A::AbstractMatrix{Float64}, b::AbstractVector{Float64},
                                max_value::Union{Nothing,Real})
    max_value !== nothing && return Float64(max_value)
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

# ─── Model construction ──────────────────────────────────────────────────────

"""
    _seapearl_build_model(A, b, sigma, delta_mv, n_levels, k_sigma, smooth_bins;
                          integer_scale) -> (model, qs, delta)

Build the SeaPearl CPModel for one relaxation attempt:

  * `n` branchable `IntVar`s `q_j ∈ 0..n_levels-1` (`"phi_1" ... "phi_n"`);
  * per response row `i`: a *weighted* sum `Σ_j c_ij·q_j ∈ [lo_i, hi_i]`
    with `c_ij = round(A_ij·Δ·s)`, expressed through `SumGreaterThan` /
    `SumLessThan` over `IntVarViewMul` views (negative coefficients via
    `IntVarViewOpposite`), where the window `[lo_i, hi_i]` covers
    `[b_i - k·σ_i, b_i + k·σ_i]` widened by the worst-case
    coefficient-rounding tolerance `0.5·n·(n_levels-1)` (so no truly feasible
    quantized spectrum is excluded);
  * smoothing: `|q_{j+1} - q_j| ≤ smooth_bins` via two `LessOrEqual`
    constraints on `IntVarViewOffset` views.

Returns the model, the vector of flux variables and the quantization step `Δ`.
"""
function _seapearl_build_model(A::Matrix{Float64}, b::Vector{Float64},
                               sigma::Vector{Float64}, mv::Float64,
                               n_levels::Int, k_sigma::Float64, smooth_bins::Int;
                               integer_scale::Float64=1e8)
    sp = _seapearl()
    m, n = size(A)
    delta = mv / (n_levels - 1)
    s = Float64(integer_scale)

    # Integer-scaled coefficients and rounding tolerance
    C = round.(Int, A .* (delta * s))
    c_max = isempty(C) ? 0 : maximum(abs.(C))
    max_pred = Float64(n) * (n_levels - 1) * c_max
    max_pred < 2e18 || throw(ArgumentError(
        "Integer coefficients overflow risk (max prediction $max_pred); " *
        "reduce `integer_scale` or `n_levels`."))

    tol = 0.5 * n * (n_levels - 1)   # worst-case coefficient rounding error

    trailer = sp.Trailer()
    model = sp.CPModel(trailer)
    qs = sp.IntVar[]
    for j in 1:n
        q = sp.IntVar(0, n_levels - 1, "phi_$j", trailer)
        sp.addVariable!(model, q)
        push!(qs, q)
    end

    # kσ-compatibility on every response row (weighted sum over mul-views)
    infeasible_flag = Ref(false)
    for i in 1:m
        lo = ceil(Int, (b[i] - k_sigma * sigma[i]) * s - tol)
        hi = floor(Int, (b[i] + k_sigma * sigma[i]) * s + tol)
        lo > hi && continue   # empty window — propagation will report infeasibility
        terms = sp.AbstractIntVar[]
        for j in 1:n
            c = C[i, j]
            c == 0 && continue
            if c > 0
                push!(terms, sp.IntVarViewMul(qs[j], c, "w_$(i)_$(j)"))
            else
                opp = sp.IntVarViewOpposite(qs[j], "n_$(i)_$(j)")
                push!(terms, sp.IntVarViewMul(opp, -c, "w_$(i)_$(j)_opp"))
            end
        end
        if isempty(terms)
            # Zero row: prediction is identically 0 — feasible iff 0 ∈ window
            (lo ≤ 0 ≤ hi) || (infeasible_flag[] = true)
            continue
        end
        sp.addConstraint!(model, sp.SumGreaterThan(terms, lo, trailer))
        sp.addConstraint!(model, sp.SumLessThan(terms, hi, trailer))
    end
    if infeasible_flag[]
        # Force an empty search tree: q_1 ≥ n_levels is impossible in 0..n_levels-1
        sp.addConstraint!(model, sp.SumGreaterThan([qs[1]], n_levels, trailer))
    end

    # Adjacent-bin smoothness: |q_{j+1} - q_j| ≤ smooth_bins
    smooth_bins > 0 || return model, qs, delta
    for j in 1:n-1
        up = sp.IntVarViewOffset(qs[j], smooth_bins, "off_$(j)_up")
        sp.addConstraint!(model, sp.LessOrEqual(qs[j+1], up, trailer))
        dn = sp.IntVarViewOffset(qs[j+1], smooth_bins, "off_$(j)_dn")
        sp.addConstraint!(model, sp.LessOrEqual(qs[j], dn, trailer))
    end
    return model, qs, delta
end

"""
    _seapearl_collect_solutions(model, qs, delta) -> Matrix{Float64}

Convert the raw SeaPearl solutions (`model.statistics.solutions`) into an
`(n_solutions × n)` matrix of flux spectra (`q·Δ` per bin).
"""
function _seapearl_collect_solutions(model, qs, delta::Float64)
    n = length(qs)
    sols = filter(s -> s !== nothing, model.statistics.solutions)
    out = Matrix{Float64}(undef, length(sols), n)
    for (k, sol) in enumerate(sols)
        for j in 1:n
            out[k, j] = Int(sol["phi_$j"]) * delta
        end
    end
    return out
end

# ─── Main solver ─────────────────────────────────────────────────────────────

"""
    solve_seapearl(A, b, x0; n_levels=16, k_sigma=2.0, noise_level=0.01,
                   smoothness=nothing, max_value=nothing, integer_scale=1e8,
                   strategy=:dfs, ilds_max_discrepancy=2,
                   max_solutions=256, time_limit_ms=60000, node_limit=nothing,
                   value_order=:minimum, learned_heuristic=nothing,
                   variable_heuristic=nothing, strategy_object=nothing,
                   relax_attempts=2, relax_step=0.01,
                   expand_attempts=2, expand_factor=4.0) -> UnfoldResult

CP-based spectrum unfolding with feasibility-set enumeration (SeaPearl.jl).

# Arguments
- `A::AbstractMatrix{T}`: response matrix (m × n)
- `b::AbstractVector{T}`: measurements (m,)
- `x0::AbstractVector{T}`: accepted for interface compatibility; **not used**
  by the CP formulation (it is data-driven — pass `max_value` to control the
  flux ceiling instead)
- `n_levels`: quantization levels per energy bin (≥ 2; `Δ = max_value/(n_levels-1)`)
- `k_sigma`: half-width of the compatibility window in σ units
- `noise_level`: relative measurement noise used to build σ (`σ_i = max(noise_level·b_i, σ_floor)`)
- `smoothness`: max adjacent-bin flux jump in flux units; `nothing` →
  `max(1, n_levels ÷ 3)` quantization steps
- `max_value`: flux ceiling of the quantization grid; `nothing` → estimated
  from the data (2 × max of the minimum-norm least-squares solution)
- `integer_scale`: scale factor mapping the float prediction onto the integer
  grid (coefficients `round(A_ij·Δ·integer_scale)`)
- `strategy`: `:dfs` or `:ilds` (or pass a SeaPearl strategy via `strategy_object`)
- `ilds_max_discrepancy`: max discrepancy for `:ilds`
- `max_solutions`: enumeration cap (solution limit handed to SeaPearl)
- `time_limit_ms`: per-attempt search time limit in ms (`nothing` → unlimited)
- `node_limit`: per-attempt node limit (`nothing` → unlimited)
- `value_order`: `:minimum` or `:maximum` — level visited first by the
  `BasicHeuristic` (ignored when `learned_heuristic`/`value_selection` is given)
- `learned_heuristic`: optional pre-trained SeaPearl `LearnedHeuristic`
  (CP + RL pipeline); mutually exclusive with `value_order`
- `variable_heuristic`: optional custom variable-selection object
  (default: `MinDomainVariableSelection()`)
- `strategy_object`: optional raw SeaPearl `SearchStrategy` (overrides `strategy`)
- `relax_attempts` / `relax_step`: if no feasible spectrum is found, the noise
  level is increased by `relax_step` per attempt (physically: admit larger
  measurement uncertainty) up to `relax_attempts` times
- `expand_attempts` / `expand_factor`: if still infeasible, the flux ceiling
  is multiplied by `expand_factor` per attempt (quantization was too coarse)

# Returns
`UnfoldResult` whose spectrum is the **minimum-χ² member of the enumerated
feasible set**. `extra` contains the CP-specific products:
- `"spectrum_lower"` / `"spectrum_upper"`: per-bin min/max over all enumerated
  feasible spectra (interval estimates of the null-space uncertainty)
- `"spectrum_mean_samples"`: per-bin mean over the feasible set
- `"n_solutions"`, `"exhaustive"`, `"status"`, `"relax_attempt"`,
  `"chi2_best"`, `"solutions_chi2"`, `"delta"`, `"max_value"`, `"n_levels"`,
  `"k_sigma"`, `"noise_level_effective"`, `"smoothness_bins"`,
  `"solutions_truncated"`
"""
function solve_seapearl(A::AbstractMatrix{T}, b::AbstractVector{T}, x0::AbstractVector{T};
                        n_levels::Integer=16,
                        k_sigma::Real=2.0,
                        noise_level::Real=0.01,
                        smoothness::Union{Nothing,Real}=nothing,
                        max_value::Union{Nothing,Real}=nothing,
                        integer_scale::Real=1e8,
                        strategy::Symbol=:dfs,
                        ilds_max_discrepancy::Integer=2,
                        max_solutions::Integer=256,
                        time_limit_ms::Union{Nothing,Integer}=60000,
                        node_limit::Union{Nothing,Integer}=nothing,
                        value_order::Symbol=:minimum,
                        learned_heuristic=nothing,
                        variable_heuristic=nothing,
                        strategy_object=nothing,
                        relax_attempts::Integer=2,
                        relax_step::Real=0.01,
                        expand_attempts::Integer=2,
                        expand_factor::Real=4.0) where T<:AbstractFloat
    m, n = size(A)
    length(b) == m || throw(ArgumentError("b length ($(length(b))) must match A rows ($m)"))
    length(x0) == n || throw(ArgumentError("x0 length ($(length(x0))) must match A columns ($n)"))
    n_levels ≥ 2 || throw(ArgumentError("n_levels must be ≥ 2, got $n_levels"))
    max_solutions ≥ 1 || throw(ArgumentError("max_solutions must be ≥ 1"))
    expand_factor > 1 || throw(ArgumentError("expand_factor must be > 1"))
    value_order in (:minimum, :maximum) || throw(ArgumentError(
        "value_order must be :minimum or :maximum, got :$value_order"))
    (learned_heuristic === nothing || value_order === :minimum) || throw(ArgumentError(
        "pass either `learned_heuristic` or `value_order`, not both"))

    if !_try_load_seapearl()
        return UnfoldResult(zeros(T, n), 0, false, T(0),
                            Dict{String,Any}("error" => "SeaPearl.jl not available"))
    end

    # σ estimate: relative statistical noise with an absolute floor for b_i ≈ 0
    Af = Float64.(A); bf = Float64.(b)
    σ_floor = 1e-9 * max(1.0, maximum(abs.(bf)))
    σ_base = max.(noise_level .* abs.(bf), σ_floor)

    mv0 = _seapearl_flux_ceiling(Af, bf, max_value)
    smooth_bins = smoothness === nothing ? max(1, n_levels ÷ 3) :
        max(0, round(Int, Float64(smoothness) / (mv0 / (n_levels - 1))))

    # Relaxation ladder: first widen the noise, then expand the flux ceiling
    attempts = Tuple{Float64,Float64,Int}[]   # (noise_eff, mv_mult, attempt_index)
    for t in 0:relax_attempts
        push!(attempts, (noise_level + t * relax_step, 1.0, t))
    end
    for t in 1:expand_attempts
        push!(attempts, (noise_level + relax_attempts * relax_step,
                         Float64(expand_factor)^t, relax_attempts + t))
    end

    sp = _seapearl()
    time_limit_s = time_limit_ms === nothing ? nothing : Float64(time_limit_ms) / 1000.0

    strategy_obj = strategy_object !== nothing ? strategy_object :
        strategy === :ilds ? sp.ILDSearch(Int(ilds_max_discrepancy)) :
        strategy === :dfs ? sp.DFSearch() :
        throw(ArgumentError("strategy must be :dfs or :ilds, got :$strategy"))

    var_heur = variable_heuristic !== nothing ? variable_heuristic :
        sp.MinDomainVariableSelection()

    spectra = Matrix{Float64}(undef, 0, n)
    status = :Infeasible
    noise_eff = Float64(noise_level)
    mv_eff = mv0
    attempt_used = 0
    sb_eff = smoothness === nothing ? max(1, n_levels ÷ 3) :
        max(0, round(Int, Float64(smoothness) / (mv0 / (n_levels - 1))))

    for (noise_t, mv_mult, t) in attempts
        mv_t = mv0 * mv_mult
        sb_t = smoothness === nothing ? max(1, n_levels ÷ 3) :
            max(0, round(Int, Float64(smoothness) / (mv_t / (n_levels - 1))))
        σ_t = max.(noise_t .* abs.(bf), σ_floor)

        model, qs, delta = _seapearl_build_model(Af, bf, σ_t, mv_t, Int(n_levels),
                                                 Float64(k_sigma), sb_t;
                                                 integer_scale=Float64(integer_scale))
        model.limit = sp.Limit(node_limit === nothing ? nothing : Int(node_limit),
                               Int(max_solutions),
                               time_limit_s === nothing ? nothing : Int(ceil(time_limit_s)),
                               nothing)

        # Value heuristic: BasicHeuristic by default, RL agent on demand
        value_heur = learned_heuristic !== nothing ? learned_heuristic :
            value_order === :maximum ?
                sp.BasicHeuristic((x; cpmodel=nothing) -> maximum(x.domain)) :
                sp.BasicHeuristic((x; cpmodel=nothing) -> minimum(x.domain))

        status = sp.solve!(model, strategy_obj;
                           variableHeuristic=var_heur, valueSelection=value_heur)

        spectra = _seapearl_collect_solutions(model, qs, delta)
        if size(spectra, 1) > 0
            noise_eff, mv_eff, attempt_used = noise_t, mv_t, t
            sb_eff = sb_t
            break
        end
    end

    n_solutions = size(spectra, 1)

    # ─── Infeasible even after relaxations: flux-matched flat fallback ──────
    if n_solutions == 0
        denom = sum(Af)
        fallback = ones(n) .* (denom > 0 ? Float64(sum(bf)) / denom : 0.0)
        residual = bf .- Af * fallback
        @warn "solve_seapearl: no feasible quantized spectrum found (status=:$(status)); " *
              "returning a flux-matched fallback spectrum. Increase `n_levels`, " *
              "relax `k_sigma`/`noise_level` or pass a larger `max_value`."
        return UnfoldResult(Vector{T}(fallback), 0, false,
                            T(norm(residual)),
                            Dict{String,Any}(
                                "status" => String(status),
                                "n_solutions" => 0,
                                "error" => "No feasible quantized spectrum found",
                                "n_levels" => Int(n_levels),
                                "k_sigma" => Float64(k_sigma),
                                "smoothness_bins" => sb_eff,
                            ))
    end

    # ─── χ² ranking and interval estimates over the feasible set ────────────
    pred = spectra * Af'                       # (n_solutions × m)
    chi2 = vec(sum(((pred' .- bf) ./ σ_base) .^ 2; dims=1))
    best = argmin(chi2)
    spectrum = spectra[best, :]
    lower = vec(minimum(spectra; dims=1))
    upper = vec(maximum(spectra; dims=1))
    mean_s = vec(mean(spectra; dims=1))
    exhaustive = (status == :Optimal)          # DFS finished the whole tree

    residual = bf .- Af * spectrum
    return UnfoldResult(Vector{T}(spectrum), Int(n_solutions), true,
                        T(norm(residual)),
                        Dict{String,Any}(
                            "spectrum_lower"      => lower,
                            "spectrum_upper"      => upper,
                            "spectrum_mean_samples" => mean_s,
                            "n_solutions"         => n_solutions,
                            "exhaustive"          => exhaustive,
                            "solutions_truncated" => !exhaustive && n_solutions == max_solutions,
                            "status"              => String(status),
                            "relax_attempt"       => attempt_used,
                            "chi2_best"           => chi2[best],
                            "solutions_chi2"      => chi2,
                            "delta"               => mv_eff / (n_levels - 1),
                            "max_value"           => mv_eff,
                            "n_levels"            => Int(n_levels),
                            "k_sigma"             => Float64(k_sigma),
                            "noise_level_effective" => noise_eff,
                            "smoothness_bins"     => sb_eff,
                        ))
end
