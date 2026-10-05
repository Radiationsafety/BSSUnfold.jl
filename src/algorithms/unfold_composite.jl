"""
Composite (ensemble) multi-method spectrum unfolding.

Port of `bssunfold/src/bssunfold/core/unfold_composite.py`.

This module implements an adaptive ensemble approach to neutron spectrum
unfolding. It classifies the unknown spectrum by hardness, selects a pool of
suitable individual methods, runs them, and combines their results with
confidence-weighted averaging. The ensemble reduces sensitivity to individual
method failures and delivers consistent performance across diverse spectra.

The Python original resolves method names against the Detector-level
`unfold_*` wrappers. The Julia port reuses the repository's existing
name-based solver dispatch (`BSSUnfold._dispatch_solver` with
`BSSUnfold.METHOD_DISPATCH`, the same idiom used by `solve_binned`), so the
members are the `solve_*` functions called on the `(A, b, x0)` system.

Python guards each member with a `SIGALRM` wall-clock timeout; Julia has no
equivalent in-process mechanism, so the timeout is not enforced — the
robustness it provides is preserved by running every member inside
`try/catch`, so a failing (or long-running) member is dropped and never
kills the ensemble.

References:
- Wolpert, "Stacked generalization" (1992)
- Reginatto et al., "Sequential Bayesian approach for neutron spectrum
  unfolding"
"""

"""
Default method pools per spectrum-hardness bin (matches the documented table).
"""
const DEFAULT_BIN_METHODS = Dict{String,Vector{String}}(
    "very_soft" => ["tsvd", "bayes", "cvxpy", "statreg", "lanczos"],
    "soft" => ["mlem", "landweber", "bayes_spline", "gravel", "qpsolvers"],
    "intermediate" => ["cvxpy", "qpsolvers", "hybrid_parametric", "parametric2"],
    "hard" => ["genetic", "interpret", "maeo_ensemble", "mystic", "mystic_hybrid", "cs"],
    "very_hard" => ["scip", "docplex", "epic", "cs", "interpret"],
)

"""
A curated, fast and robust pool used when no spectrum is supplied for
classification (e.g. when only readings are available).
"""
const GENERAL_METHODS = String["tsvd", "mlem", "cvxpy", "qpsolvers", "bayes_spline"]

"""
Optional base weights per method (`1.0` == equal contribution).
"""
const DEFAULT_ENSEMBLE_WEIGHTS = Dict{String,Float64}(
    "tsvd" => 1.0,
    "bayes" => 1.0,
    "cvxpy" => 1.0,
    "statreg" => 1.0,
    "lanczos" => 1.0,
    "mlem" => 1.0,
    "landweber" => 1.0,
    "bayes_spline" => 1.0,
    "gravel" => 1.0,
    "qpsolvers" => 1.0,
    "hybrid_parametric" => 1.0,
    "parametric2" => 1.0,
    "genetic" => 0.8,
    "interpret" => 0.8,
    "maeo_ensemble" => 0.8,
    "mystic" => 0.8,
    "mystic_hybrid" => 0.85,
    "cs" => 0.8,
    "scip" => 0.8,
    "docplex" => 0.8,
    "epic" => 0.8,
    "kaczmarz" => 1.0,
)

"""
    compute_spectrum_features(spectrum, energy)

Compute simple discriminating features of a spectrum.

# Arguments
- `spectrum`: spectrum values on the energy grid
- `energy`: energy grid (MeV)

# Returns
`Dict{String,Float64}` including `hardness_ratio` (mean energy in MeV),
`total_flux`, `entropy` and `peak`.
"""
function compute_spectrum_features(spectrum::AbstractVector{<:Real},
                                   energy::AbstractVector{<:Real})::Dict{String,Float64}
    s = collect(Float64, spectrum)
    e = collect(Float64, energy)
    total = sum(s) + 1e-30

    mean_energy = sum(s .* e) / total
    # Normalized entropy of the spectral shape.
    p = s ./ total
    p = p[p .> 0]
    entropy = -sum(p .* log.(p .+ 1e-30)) / log(length(s) + 1e-30)
    peak = isempty(s) ? 0.0 : maximum(s)

    return Dict{String,Float64}(
        "hardness_ratio" => mean_energy,
        "total_flux" => total,
        "entropy" => entropy,
        "peak" => peak,
    )
end

"""
    classify_spectrum_by_hardness(features)

Classify a spectrum into a hardness bin from its features.

Thresholds are applied to `features["hardness_ratio"]` (mean energy, MeV):
`very_soft < 0.1`, `soft 0.1-0.3`, `intermediate 0.3-0.5`, `hard 0.5-1.0`,
`very_hard > 1.0`.
"""
function classify_spectrum_by_hardness(features::AbstractDict)::String
    hr = Float64(get(features, "hardness_ratio", 0.5))
    if hr < 0.1
        return "very_soft"
    elseif hr < 0.3
        return "soft"
    elseif hr < 0.5
        return "intermediate"
    elseif hr < 1.0
        return "hard"
    end
    return "very_hard"
end

"""
    _confidence_weight(spectrum, others)

Confidence of a single solution relative to the other ensemble members.
"""
function _confidence_weight(spectrum::AbstractVector{<:Real},
                            others::Vector{<:AbstractVector})
    isempty(others) && return 1.0
    sims = filter(isfinite, [BSSUnfold.cosine_similarity(spectrum, o) for o in others])
    isempty(sims) && return 1.0
    return clamp(sum(sims) / length(sims), 0.0, 1.0)
end

"""
    solve_composite(A, b, x0; n_methods=5, timeout_per_method=30.0,
                    save_result=false, spectrum=nothing, energy=nothing,
                    method_names=nothing, ensemble_weights=nothing)

Run an adaptive ensemble of unfolding methods and combine the results with
confidence-weighted averaging.

# Arguments
- `A::AbstractMatrix{T}`: response matrix (m × n)
- `b::AbstractVector{T}`: measurements (m,)
- `x0::AbstractVector{T}`: initial spectrum handed to every member solver
- `n_methods::Integer`: maximum number of methods to combine
- `timeout_per_method::Real`: accepted for parity with Python; the timeout is
  **not enforced** in Julia (see the module docstring)
- `save_result::Bool`: accepted for parity with Python; there is no detector
  history at the solver level
- `spectrum`: reference/estimated spectrum used to select the method pool by
  hardness; if omitted the general robust pool is used
- `energy`: energy grid for `spectrum`; at the solver level it defaults to the
  repository's synthetic log grid `10.^(-9..2)` MeV (as in `solve_express`)
- `method_names`: explicit list of method short names to run (overrides
  selection)
- `ensemble_weights`: per-method base weights (defaults to
  `DEFAULT_ENSEMBLE_WEIGHTS`)

# Returns
`UnfoldResult` with the combined spectrum; `extra` carries
`successful_methods`, `consistency`, `weights`, `individual_spectra`,
`method_order` (the insertion order of the members, since a Julia `Dict` is
unordered — Python's dict is insertion-ordered), `candidates`, `hardness_bin`,
`status` and `message`.
"""
function solve_composite(A::AbstractMatrix{T}, b::AbstractVector{T}, x0::AbstractVector{T};
                         n_methods::Integer=5,
                         timeout_per_method::Real=30.0,
                         save_result::Bool=false,
                         spectrum::Union{Nothing,AbstractVector{T}}=nothing,
                         energy::Union{Nothing,AbstractVector{<:Real}}=nothing,
                         method_names::Union{Nothing,AbstractVector}=nothing,
                         ensemble_weights::Union{Nothing,AbstractDict}=nothing) where T<:AbstractFloat
    m, n = size(A)

    E = energy === nothing ? collect(10.0 .^ range(-9.0, 2.0, length=n)) :
                             collect(Float64, energy)
    weights = (ensemble_weights === nothing || isempty(ensemble_weights)) ?
              DEFAULT_ENSEMBLE_WEIGHTS : ensemble_weights

    # Select the candidate method pool.
    bin_name::Union{Nothing,String} = nothing
    if method_names !== nothing && !isempty(method_names)
        candidates = String[String(s) for s in method_names]
    elseif spectrum !== nothing
        features = compute_spectrum_features(collect(Float64, spectrum), E)
        bin_name = classify_spectrum_by_hardness(features)
        candidates = String[String(s) for s in
                            get(DEFAULT_BIN_METHODS, bin_name, GENERAL_METHODS)]
    else
        candidates = String[String(s) for s in GENERAL_METHODS]
    end
    candidates = candidates[1:min(Int(n_methods), length(candidates))]

    individual_spectra = Dict{String,Vector{T}}()
    ordered_names = String[]
    successful_methods = String[]
    messages = Dict{String,String}()

    for name in candidates
        func = BSSUnfold._dispatch_solver(name)
        if func === nothing
            messages[name] = "unknown method"
            continue
        end
        try
            result = func(A, b, copy(x0))
            spec = _member_spectrum(result, T, n)
            if spec === nothing
                messages[name] = "invalid output"
                continue
            end
            individual_spectra[name] = spec
            push!(ordered_names, name)
            push!(successful_methods, name)
        catch err
            # Julia has no per-method timeout: a failing member is dropped here,
            # which is the robustness Python's timeout path provides.
            messages[name] = "$(typeof(err)): $err"
        end
    end

    if isempty(ordered_names)
        throw(ErrorException("No method succeeded. Details: $messages"))
    end

    stacked = [individual_spectra[nm] for nm in ordered_names]

    # Confidence-weighted combination.
    combined = zeros(T, n)
    total_weight = zero(T)
    used_weights = Dict{String,Float64}()
    for (i, name) in enumerate(ordered_names)
        others = [stacked[j] for j in eachindex(stacked) if j != i]
        conf = _confidence_weight(stacked[i], others)
        base = Float64(get(weights, name, 1.0))
        w = base * conf
        combined .+= T(w) .* stacked[i]
        total_weight += T(w)
        used_weights[name] = w
    end
    if total_weight > 0
        combined ./= total_weight
    end

    # Consistency: mean pairwise cosine similarity among individual solutions.
    consistency = 0.0
    n_pairs = 0
    for i in eachindex(stacked), j in (i + 1):length(stacked)
        s = BSSUnfold.cosine_similarity(stacked[i], stacked[j])
        if isfinite(s)
            consistency += s
            n_pairs += 1
        end
    end
    consistency = n_pairs > 0 ? consistency / n_pairs : 0.0

    residual = norm(b .- A * combined)

    return UnfoldResult(vec(max.(combined, T(0))), length(successful_methods), true,
                        residual,
                        Dict{String,Any}(
                            "successful_methods" => successful_methods,
                            "consistency" => consistency,
                            "weights" => used_weights,
                            "individual_spectra" => individual_spectra,
                            "method_order" => ordered_names,
                            "candidates" => candidates,
                            "hardness_bin" => bin_name,
                            "messages" => messages,
                            "status" => "OK",
                            "message" => string("Combined ", length(successful_methods), "/",
                                                length(candidates), " methods with consistency ",
                                                @sprintf("%.3f", consistency)),
                        ))
end

"""
    _member_spectrum(result, ::Type{T}, n)

Extract a valid member spectrum (`nothing` marks an invalid one), mirroring
Python's `None` / NaN / non-positive-flux rejection.  The clamping to `>= 0`
reproduces what Python's framework already does inside `_standardize_output`
before `unfold_composite` sees the member result.
"""
function _member_spectrum(result::UnfoldResult, ::Type{T}, n::Int) where T<:AbstractFloat
    spec = max.(collect(T, vec(result.spectrum)), T(0))
    (length(spec) == n && all(isfinite, spec) && sum(spec) > 0) || return nothing
    return spec
end
function _member_spectrum(result::AbstractVector, ::Type{T}, n::Int) where T<:AbstractFloat
    return _member_spectrum(UnfoldResult(collect(T, vec(result)), 0, true, T(0)), T, n)
end
function _member_spectrum(result::Tuple, ::Type{T}, n::Int) where T<:AbstractFloat
    return _member_spectrum(result[1], T, n)
end
function _member_spectrum(::Any, ::Type{T}, n::Int) where T<:AbstractFloat
    return nothing
end
