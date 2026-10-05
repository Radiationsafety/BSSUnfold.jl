"""
Combined unfolding method applying multiple methods sequentially.

Port of `bssunfold/src/bssunfold/core/unfold_combined.py`.

This module provides `solve_combined`, which applies a pipeline of unfolding
methods sequentially, optionally feeding the result of each method as the
initial spectrum of the next.

Unlike the Python original (which drives the Detector-level `unfold_*`
wrappers), the Julia port orchestrates the repo solvers (`solve_*`) directly
on the `(A, b, x0)` system, reusing the existing name-based dispatch idiom
(`BSSUnfold._dispatch_solver` / `BSSUnfold.METHOD_DISPATCH`, as used by
`solve_binned`).
"""

"""
    solve_combined(A, b, x0; pipeline, ln_steps=nothing,
                   calculate_errors=false, verbose=true)

Apply a pipeline of unfolding methods sequentially.

# Arguments
- `A::AbstractMatrix{T}`: response matrix (m × n)
- `b::AbstractVector{T}`: measurements (m,)
- `x0::AbstractVector{T}`: initial spectrum for the first stage (n,)
- `pipeline`: required vector of stages. Each stage is a dict (or a named
  tuple) which may contain:
  - `"method"`: AbstractString — solver name resolved through `METHOD_DISPATCH`
    (e.g. `"cvxpy"`, `"landweber"`, `"mlem"`)
  - `"params"`: dict — keyword arguments for that solver
  - `"use_as_initial"`: Bool (optional, default `true`) — use the previous
    result as the initial spectrum of this stage
  - `"store_intermediate"`: Bool (optional, default `false`) — keep this
    stage's result in `extra["intermediate_results"]`
- `ln_steps`: accepted for interface parity with Python; the solver-level
  pipeline does not use it (the Julia framework folds the energy step into `A`)
- `calculate_errors::Bool`: Monte-Carlo uncertainty for the LAST stage only
- `verbose::Bool`: log each stage

# Returns
`UnfoldResult` of the last stage; `extra` carries `pipeline_info`
(`stages`, `params`), optionally `intermediate_results`, the last member's
own diagnostics and, when `calculate_errors`, the uncertainty keys.
"""
function solve_combined(A::AbstractMatrix{T}, b::AbstractVector{T}, x0::AbstractVector{T};
                        pipeline,
                        ln_steps::Union{Nothing,AbstractVector{T}}=nothing,
                        calculate_errors::Bool=false,
                        verbose::Bool=true) where T<:AbstractFloat
    m, n = size(A)
    isempty(pipeline) && throw(ArgumentError("pipeline must contain at least one stage"))

    current_spectrum::Union{Nothing,Vector{T}} = nothing
    intermediate_results = Dict{String,Any}()
    final_result::Union{Nothing,UnfoldResult} = nothing
    stage_methods = String[]
    stage_params_log = Dict{String,Any}[]
    stage_spectra = Vector{Vector{T}}()
    uncert = Dict{String,Any}()

    if verbose
        @info "Combined algorithm, methods = $(length(pipeline))"
    end

    for (i, stage) in enumerate(pipeline)
        method = String(_stage_get(stage, "method", ""))
        params = _stage_symbol_params(stage)
        raw_params = _stage_raw_params(stage)
        use_as_initial = Bool(_stage_get(stage, "use_as_initial", true))
        store_intermediate = Bool(_stage_get(stage, "store_intermediate", false))

        if verbose
            @info "Stage $i/$(length(pipeline)): $method"
        end

        # Chain the previous result as the initial spectrum of this stage
        # (Python injects `params["initial_spectrum"]`; here x0 is positional).
        seeded_from_previous = false
        x_stage = x0
        if current_spectrum !== nothing && use_as_initial
            x_stage = copy(current_spectrum)
            seeded_from_previous = true
        elseif haskey(params, :initial_spectrum)
            provided = pop!(params, :initial_spectrum)
            x_stage = collect(T, vec(float.(provided)))
        end
        if seeded_from_previous && verbose
            @info "Previous result used as initial spectrum"
        end

        fn = BSSUnfold._dispatch_solver(method)
        if fn === nothing
            throw(ArgumentError(
                "Method '$method' not found. " *
                "Available methods: $(sort(collect(keys(BSSUnfold.METHOD_DISPATCH))))"))
        end

        result = try
            _as_unfold_result(fn(A, b, vec(T.(x_stage)); params...), T)
        catch e
            @error "Error in method $method: $e"
            rethrow(e)
        end

        # Python's framework clamps each member output to >= 0 in
        # `_standardize_output`; do the same before chaining.
        x_sol = max.(collect(T, vec(result.spectrum)), T(0))
        result = UnfoldResult(x_sol, result.iterations, result.converged,
                              norm(b .- A * x_sol), result.extra)

        if !isempty(x_sol)
            current_spectrum = x_sol
            if verbose
                @info "  Spectrum norm: $(round(norm(current_spectrum), digits=6))"
            end
        end
        # Only calculate errors for the last stage if requested
        if i == length(pipeline) && calculate_errors
            mc = BSSUnfold.monte_carlo_uncertainty(
                (Aa, bb, xx) -> fn(Aa, bb, xx; params...),
                A, b, vec(T.(x_stage)), 0.01, 100)
            uncert["spectrum_uncert_mean"]   = mc.mean
            uncert["spectrum_uncert_std"]    = mc.std
            uncert["spectrum_uncert_median"] = mc.median
            uncert["spectrum_uncert_p5"]     = mc.p5
            uncert["spectrum_uncert_p95"]    = mc.p95
            uncert["spectrum_uncert_all"]    = mc.all
            uncert["montecarlo_samples"]     = 100
            uncert["noise_level"]            = 0.01
        end

        if store_intermediate
            intermediate_results["stage_$(i)_$(method)"] = result
        end

        final_result = result
        push!(stage_methods, method)
        push!(stage_params_log, raw_params)
        push!(stage_spectra, x_sol)
    end

    if verbose
        @info "Combined method finished"
    end

    final_result === nothing && throw(ErrorException("No stage produced a result"))

    spectrum = collect(T, vec(final_result.spectrum))
    residual = norm(b .- A * spectrum)
    extra = Dict{String,Any}()
    for (k, v) in final_result.extra
        extra[String(k)] = v
    end
    for (k, v) in uncert
        extra[String(k)] = v
    end
    extra["pipeline_info"] = Dict{String,Any}(
        "stages" => stage_methods,
        "params" => stage_params_log,
    )
    extra["stage_spectra"] = stage_spectra
    if !isempty(intermediate_results)
        extra["intermediate_results"] = intermediate_results
    end

    return UnfoldResult(spectrum, final_result.iterations, final_result.converged,
                        residual, extra)
end

"""
    _stage_get(stage, key, default)

`stage.get(key, default)` for a stage given either as an `AbstractDict`
(String or Symbol keys) or as a `NamedTuple`.
"""
function _stage_get(stage::AbstractDict, key::AbstractString, default)
    haskey(stage, key) && return stage[key]
    sk = Symbol(key)
    haskey(stage, sk) && return stage[sk]
    return default
end
function _stage_get(stage::NamedTuple, key::AbstractString, default)
    sk = Symbol(key)
    return haskey(stage, sk) ? getproperty(stage, sk) : default
end

"""
    _stage_symbol_params(stage)

The stage `params` dict with Symbol keys, ready for splatting into a solver.
"""
function _stage_symbol_params(stage)
    raw = _stage_get(stage, "params", Dict{String,Any}())
    return Dict{Symbol,Any}(Symbol(string(k)) => v for (k, v) in pairs(raw))
end

"""
    _stage_raw_params(stage)

The stage `params` as given by the caller (String keys), recorded verbatim in
`pipeline_info["params"]`, mirroring Python's `stage.get("params", {})`.
"""
function _stage_raw_params(stage)
    raw = _stage_get(stage, "params", Dict{String,Any}())
    return Dict{String,Any}(string(k) => v for (k, v) in pairs(raw))
end

"""
    _as_unfold_result(result, ::Type{T})

Adapt a solver return value to an `UnfoldResult` (mirrors the tolerance
already present in `solve_ensemble` for non-`UnfoldResult` solvers).
"""
function _as_unfold_result(result::UnfoldResult, ::Type{T}) where T<:AbstractFloat
    return result
end
function _as_unfold_result(result::AbstractVector, ::Type{T}) where T<:AbstractFloat
    return UnfoldResult(collect(T, vec(result)), 0, true, T(NaN))
end
function _as_unfold_result(result::Tuple, ::Type{T}) where T<:AbstractFloat
    return UnfoldResult(collect(T, vec(result[1])), length(result) > 1 ? Int(result[2]) : 0,
                        length(result) > 2 ? Bool(result[3]) : true, T(NaN))
end
