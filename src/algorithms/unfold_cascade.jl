"""
    Cascade multi-method spectrum unfolding.

Port of `bssunfold/src/bssunfold/core/unfold_cascade.py` (bssunfold 0.29.0).

Several unfolding methods are applied in sequence, each stage optionally
refining the previous one:

1. **Coarse-to-fine refinement** (`coarse`/`multi_resolution`): a stage runs on
   a column-sum coarsened response matrix and its solution is prolongated back
   onto the fine grid (see `coarsen_columns`/`split_coarse`).
2. **Prior information transfer** (`use_as_initial`, `use_as_prior`).
3. **Adaptive method selection** (`select_next_method`, `solve_adaptive_cascade`).
4. **Early stopping** (`quality_threshold` on the `overall_quality` metric).

Member methods are resolved by short name through the shared `METHOD_DISPATCH`
table (the idiom of `solve_binned`), i.e. `"mlem" -> BSSUnfold.solve_mlem`.

Python guards every stage with a `SIGALRM` wall-clock timeout (`_StageTimeout`
plus `_run_with_timeout`).  Julia has no equivalent and this port does not fake
one: a stage is simply run to completion, and the behaviour the timeout existed
to protect — a failing stage must not abort the cascade — is preserved by
running each stage inside `try`/`catch`, logging and continuing exactly as the
Python `except Exception` branch does.
"""

"""
    CascadeStage(method; params=Dict{Symbol,Any}(), use_as_initial=true,
                 use_as_prior=false, store_intermediate=false,
                 quality_threshold=nothing, max_iterations=nothing,
                 timeout=60.0, coarse=false, coarse_bins=nothing)

Configuration of a single cascade stage (port of the `CascadeStage` dataclass).

# Fields
- `method::String` — short unfolding method name (resolved through `METHOD_DISPATCH`)
- `params::Dict{Symbol,Any}` — keyword parameters forwarded to the member solver
- `use_as_initial::Bool` — use the previous stage's spectrum as the initial guess
- `use_as_prior::Bool` — use it as prior/reference spectrum (`bayes` family: `x0`)
- `store_intermediate::Bool` — keep this stage's result in `intermediate_results`
- `quality_threshold::Union{Nothing,Float64}` — stop the cascade once
  `overall_quality` reaches it
- `max_iterations::Union{Nothing,Int}` — upper bound on the internal iterations
  of iterative methods (a minimum with the stage's own `max_iterations`)
- `timeout::Float64` — kept for API parity; unused (see the module docstring)
- `coarse::Bool` — run the stage on a coarse energy grid and prolongate
- `coarse_bins::Union{Nothing,Int}` — coarse bins when `coarse`, else
  `max(8, n_energy_bins ÷ 8)`
"""
mutable struct CascadeStage{T<:AbstractFloat}
    method::String
    params::Dict{Symbol, Any}
    use_as_initial::Bool
    use_as_prior::Bool
    store_intermediate::Bool
    quality_threshold::Union{Nothing, T}
    max_iterations::Union{Nothing, Int}
    timeout::T
    coarse::Bool
    coarse_bins::Union{Nothing, Int}
end

function CascadeStage(method::AbstractString;
                      params::AbstractDict = Dict{Symbol, Any}(),
                      use_as_initial::Bool = true,
                      use_as_prior::Bool = false,
                      store_intermediate::Bool = false,
                      quality_threshold::Union{Nothing, <:Real} = nothing,
                      max_iterations::Union{Nothing, Integer} = nothing,
                      timeout::Real = 60.0,
                      coarse::Bool = false,
                      coarse_bins::Union{Nothing, Integer} = nothing)
    CascadeStage{Float64}(String(method),
                          Dict{Symbol, Any}(Symbol(k) => v for (k, v) in params),
                          use_as_initial, use_as_prior, store_intermediate,
                          quality_threshold === nothing ? nothing : Float64(quality_threshold),
                          max_iterations === nothing ? nothing : Int(max_iterations),
                          Float64(timeout), coarse,
                          coarse_bins === nothing ? nothing : Int(coarse_bins))
end

# Python's cascade shallow-copies stages (`copy.copy`) before toggling the
# coarse flag on the first one, so the caller's configuration is never mutated.
Base.copy(s::CascadeStage{T}) where {T<:AbstractFloat} =
    CascadeStage{T}(s.method, Dict{Symbol, Any}(s.params), s.use_as_initial,
                    s.use_as_prior, s.store_intermediate, s.quality_threshold,
                    s.max_iterations, s.timeout, s.coarse, s.coarse_bins)

"""
    CascadeResult

Typed container describing a cascade unfolding result (port of the
`CascadeResult` dataclass).  The same facts are returned as keys of
`UnfoldResult.extra` by `solve_cascade`, in line with the rest of the package;
the object itself is stored there under `"cascade_result"`.

# Fields
- `spectrum::Union{Nothing,Vector{Float64}}` — final spectrum
- `stages_run::Int` — number of stages that executed without raising
- `total_time::Float64` — wall-clock time of the cascade, seconds
- `intermediate_results::Dict{String,Any}` — `stage_<i>_<method>` payloads
- `quality_metrics::Dict{String,Float64}` — metrics of the final spectrum
- `method_sequence::Vector{String}` — methods of the leading `stages_run` stages
- `convergence_history::Vector{Dict{String,Any}}` — per-stage metrics
- `status::String` / `message::String`
"""
struct CascadeResult
    spectrum::Union{Nothing, Vector{Float64}}
    stages_run::Int
    total_time::Float64
    intermediate_results::Dict{String, Any}
    quality_metrics::Dict{String, Float64}
    method_sequence::Vector{String}
    convergence_history::Vector{Dict{String, Any}}
    status::String
    message::String

    function CascadeResult(spectrum::Union{Nothing, AbstractVector},
                           stages_run::Integer,
                           total_time::Real,
                           intermediate_results::AbstractDict,
                           quality_metrics::AbstractDict,
                           method_sequence::AbstractVector,
                           convergence_history::AbstractVector,
                           status::AbstractString,
                           message::AbstractString)
        new(spectrum === nothing ? nothing : vec(Float64.(spectrum)),
            Int(stages_run), Float64(total_time),
            Dict{String, Any}(String(k) => v for (k, v) in intermediate_results),
            Dict{String, Float64}(String(k) => Float64(v) for (k, v) in quality_metrics),
            String[String(m) for m in method_sequence],
            Dict{String, Any}[Dict{String, Any}(String(k) => v for (k, v) in h)
                              for h in convergence_history],
            String(status), String(message))
    end
end

# Raised (as the Python `status == "ERROR"` return) when no stage succeeded.
const _CASCADE_NO_STAGES = "solve_cascade: no successful stages"

"""
    compute_quality_metrics(spectrum, reconstructed_readings, measured_readings, energy)

Quality metrics of a spectrum solution (chi-square, log-spectrum smoothness,
flux conservation error, negativity count, hardness ratio, peak count and the
`overall_quality` composite used for early stopping).

# Returns
`Dict{String,Float64}` with keys `chi_square`, `smoothness`, `flux_error`,
`negativity_count`, `hardness_ratio`, `peak_count`, `overall_quality`.
"""
function compute_quality_metrics(spectrum::AbstractVector{T},
                                 reconstructed_readings::AbstractVector{T},
                                 measured_readings::AbstractVector{T},
                                 energy::AbstractVector) where T<:AbstractFloat
    eps = T(1e-10)

    residuals = (measured_readings .- reconstructed_readings) ./
                 (reconstructed_readings .+ eps)
    chi_square = Float64(sum(residuals .^ 2))

    log_spectrum = log.(spectrum .+ eps)
    smoothness = if length(log_spectrum) > 2
        second_deriv = diff(diff(log_spectrum))
        Float64(1.0 / (1.0 + std(second_deriv, corrected=false)))
    else
        1.0
    end

    total_flux_spec = sum(spectrum)
    total_flux_readings = sum(measured_readings)
    flux_error = Float64(abs(total_flux_spec - total_flux_readings) /
                         (total_flux_readings + eps))

    negativity_count = Float64(count(x -> x < 0, spectrum))

    hardness_ratio = if length(spectrum) > 10
        thermal_region = energy .< 0.5
        fast_region = energy .> 5.0
        thermal_fraction = sum(spectrum[thermal_region]) / (sum(spectrum) + eps)
        fast_fraction = sum(spectrum[fast_region]) / (sum(spectrum) + eps)
        Float64(fast_fraction / (thermal_fraction + eps))
    else
        0.0
    end

    peak_count = if length(spectrum) > 3
        mid = spectrum[2:end-1]
        Float64(count((mid .> spectrum[1:end-2]) .& (mid .> spectrum[3:end])))
    else
        0.0
    end

    return Dict{String, Float64}(
        "chi_square" => chi_square,
        "smoothness" => smoothness,
        "flux_error" => flux_error,
        "negativity_count" => negativity_count,
        "hardness_ratio" => hardness_ratio,
        "peak_count" => peak_count,
        "overall_quality" => smoothness / (1.0 + chi_square + flux_error * 10),
    )
end

"""
    select_next_method(current_metrics, available_methods, stage_number)

Adaptive method selection from the metrics of the current solution (port of
`select_next_method`): a rough spectrum prefers smoothing methods, a poor
chi-square prefers iterative ones, a flux mismatch prefers constrained
solvers; otherwise the general family.  Falls back to a rotation over
`["landweber", "mlem", "cvxpy", "bayes"]`.

# Returns
`String` — the selected method name.
"""
function select_next_method(current_metrics::AbstractDict,
                            available_methods::AbstractVector,
                            stage_number::Integer)
    smoothness = _metric(current_metrics, "smoothness", 0.5)
    chi_square = _metric(current_metrics, "chi_square", 10.0)
    flux_error = _metric(current_metrics, "flux_error", 1.0)

    preferred = if smoothness < 0.3
        ["tsvd", "statreg", "bayes", "tikhonov_tv"]
    elseif chi_square > 5.0
        ["mlem", "landweber", "cgls", "hybrid_gmres"]
    elseif flux_error > 0.2
        ["cvxpy", "qpsolvers", "gravel"]
    else
        ["bayes_spline", "parametric2", "hybrid_parametric"]
    end

    for method in preferred
        method in available_methods && return method
    end

    defaults = ["landweber", "mlem", "cvxpy", "bayes"]
    return defaults[(Int(stage_number) % length(defaults)) + 1]
end

"""
    create_default_cascade(spectrum_type="general")

Default cascade configurations (port of `create_default_cascade`) for
`"soft"`, `"hard"`, `"fast_refinement"` and — for any other value — the
general-purpose three-stage cascade.

# Returns
`Vector{CascadeStage{Float64}}`
"""
function create_default_cascade(spectrum_type::AbstractString = "general")
    if spectrum_type == "soft"
        # Optimized for thermal/soft spectra
        return CascadeStage{Float64}[
            CascadeStage("tsvd"; params=Dict(:truncation_rank => 10),
                         use_as_initial = false, store_intermediate = true),
            CascadeStage("landweber"; params=Dict(:max_iterations => 100),
                         use_as_initial = true, store_intermediate = false),
            CascadeStage("bayes_spline"; params=Dict(:spline_smooth => 0.5),
                         use_as_initial = true, use_as_prior = true,
                         store_intermediate = false),
        ]

    elseif spectrum_type == "hard"
        # Optimized for fast/hard spectra
        return CascadeStage{Float64}[
            CascadeStage("cvxpy"; params=Dict(:regularization => 1e-2, :norm => 1),
                         use_as_initial = false, store_intermediate = true),
            CascadeStage("mlem"; params=Dict(:max_iterations => 200),
                         use_as_initial = true, store_intermediate = false),
            CascadeStage("hybrid_parametric";
                         use_as_initial = true, store_intermediate = false),
        ]

    elseif spectrum_type == "fast_refinement"
        # Quick 2-stage refinement
        return CascadeStage{Float64}[
            CascadeStage("landweber"; params=Dict(:max_iterations => 50),
                         use_as_initial = false, timeout = 10.0),
            CascadeStage("cvxpy"; params=Dict(:regularization => 1e-2),
                         use_as_initial = true, timeout = 30.0),
        ]

    else
        # general — General-purpose 3-stage cascade
        return CascadeStage{Float64}[
            CascadeStage("tsvd"; params=Dict(:truncation_rank => 15),
                         use_as_initial = false, store_intermediate = true,
                         quality_threshold = 0.3),
            CascadeStage("mlem"; params=Dict(:max_iterations => 150),
                         use_as_initial = true, store_intermediate = false),
            CascadeStage("bayes_spline"; params=Dict(:spline_smooth => 0.3),
                         use_as_initial = true, use_as_prior = true,
                         store_intermediate = false),
        ]
    end
end

"""
    solve_cascade(A, b, x0; stages=nothing, E_MeV=nothing,
                  calculate_errors=false, verbose=true, save_result=false,
                  multi_resolution=false, coarse_bins=nothing)

Cascade unfolding with sequential method refinement: the methods named by
`stages` (a `Vector{<:CascadeStage}`; Python `cascade_stages`) are applied in
order, each stage able to use the previous spectrum as initial guess
(`use_as_initial`), as prior/reference (a `bayes`/`bayes_spline` stage takes it
as `x0`, any solver exposing a `reference_spectrum` keyword gets it as such),
and to stop the cascade early once `quality_threshold` is reached.
`stages === nothing` selects `create_default_cascade("general")`.

`multi_resolution` toggles `coarse` on the first stage, optionally with the
given `coarse_bins`.  `E_MeV` is the energy grid used by the quality metrics
(default `10 .^ range(-9, 2; length=n)`).  `calculate_errors` and `save_result`
are Python framework options kept for signature parity: they are not forwarded
to the member solvers here (that belongs to `run_unfolding`/`Detector`).

A stage whose method is unavailable, or which raises, is logged and skipped;
the cascade continues.  If no stage succeeds, an `ErrorException` is thrown
(Python returns `status == "ERROR"` with a `nothing` spectrum).

# Returns
`UnfoldResult`; `extra` carries the Python result dict: `stages_run`,
`total_time`, `intermediate_results`, `quality_metrics`, `method_sequence`,
`convergence_history`, `status`, `message` and `cascade_result`.
"""
function solve_cascade(A::AbstractMatrix{T}, b::AbstractVector{T}, x0::AbstractVector{T};
                       stages::Union{Nothing, AbstractVector{<:CascadeStage}} = nothing,
                       E_MeV::Union{Nothing, AbstractVector} = nothing,
                       calculate_errors::Bool = false,
                       verbose::Bool = true,
                       save_result::Bool = false,
                       multi_resolution::Bool = false,
                       coarse_bins::Union{Nothing, Integer} = nothing) where T<:AbstractFloat
    cascade_stages = stages === nothing ? create_default_cascade("general") : stages

    if multi_resolution && !isempty(cascade_stages)
        _staged = [copy(s) for s in cascade_stages]
        _staged[1].coarse = true
        if coarse_bins !== nothing
            _staged[1].coarse_bins = Int(coarse_bins)
        end
        cascade_stages = _staged
    end

    start_time = time()
    current::Union{Nothing, Vector{T}} = nothing
    intermediate_results = Dict{String, Any}()
    convergence_history = Dict{String, Any}[]
    stages_run = 0

    m, n_energy_bins = size(A)
    measured_readings = vec(b)
    energy = _cascade_energy_grid(T, E_MeV, n_energy_bins)

    coarse_cache = Dict{Int, Matrix{T}}()

    for (idx, stage) in enumerate(cascade_stages)
        stage_idx = idx - 1
        method_name = stage.method

        verbose && @info "Cascade Stage $(stage_idx + 1)/$(length(cascade_stages)): $method_name"

        # Resolve the response matrix (fine or coarse) used by this stage.
        A_target = A
        if stage.coarse
            n_coarse = (stage.coarse_bins === nothing || stage.coarse_bins == 0) ?
                       max(8, n_energy_bins ÷ 8) : Int(stage.coarse_bins)
            A_target = get!(coarse_cache, n_coarse) do
                T.(coarsen_columns(A, n_coarse))
            end
        end
        n_target = size(A_target, 2)

        unfold_func = BSSUnfold._dispatch_solver(method_name)
        if unfold_func === nothing
            verbose && @warn "Method $method_name not found, skipping"
            continue
        end

        params = Dict{Symbol, Any}(stage.params)

        x_stage::Vector{T} = copy(vec(x0))
        if current !== nothing
            if stage.use_as_initial
                if length(current) == n_target
                    x_stage = copy(current)
                    verbose && @info "  Using previous result as initial guess"
                else
                    verbose && @info "  (skipping initial guess: grid mismatch)"
                end
            end

            if stage.use_as_prior
                if length(current) == n_target
                    # The bayes family treats the initial spectrum as the prior;
                    # other methods either accept reference_spectrum or ignore it.
                    if method_name in ("bayes", "bayes_spline")
                        x_stage = copy(current)
                    elseif _accepts_keyword(unfold_func, A_target, measured_readings,
                                            x_stage, :reference_spectrum)
                        params[:reference_spectrum] = copy(current)
                    end
                    verbose && @info "  Using previous result as prior"
                else
                    verbose && @info "  (skipping prior: grid mismatch)"
                end
            end
        end

        # A coarse stage (or a grid mismatch) needs an initial guess on the
        # stage grid: bin totals of the cascade initial, as in solve_genetic.
        if length(x_stage) != n_target && n_target <= length(x_stage)
            x_stage = T.(vec(coarsen_columns(reshape(vec(Float64.(x_stage)), 1, :),
                                             n_target)))
        end

        if stage.max_iterations !== nothing
            cur = get(params, :max_iterations, nothing)
            params[:max_iterations] = cur === nothing ? stage.max_iterations :
                                      min(cur, stage.max_iterations)
        end

        try
            result = unfold_func(A_target, measured_readings, x_stage; params...)
            stages_run += 1

            spec = _cascade_spectrum(result, T)
            if spec !== nothing
                current = stage.coarse ? T.(split_coarse(spec, n_energy_bins)) : spec

                reconstructed_readings = A * current
                metrics = compute_quality_metrics(current, reconstructed_readings,
                                                 measured_readings, energy)

                entry = Dict{String, Any}("stage" => stage_idx, "method" => method_name)
                for (k, v) in metrics
                    entry[String(k)] = v
                end
                push!(convergence_history, entry)

                if verbose
                    @info @sprintf("  Chi²=%.3f, Smooth=%.3f, Flux err=%.3f",
                                   metrics["chi_square"], metrics["smoothness"],
                                   metrics["flux_error"])
                end

                if stage.store_intermediate
                    intermediate_results["stage_$(stage_idx)_$(method_name)"] =
                        Dict{String, Any}("spectrum" => copy(current),
                                          "metrics" => metrics,
                                          "result" => result)
                end

                if stage.quality_threshold !== nothing
                    overall_quality = get(metrics, "overall_quality", 0.0)
                    if overall_quality >= stage.quality_threshold
                        if verbose
                            @info @sprintf("  Quality threshold met (%.3f >= %s), stopping",
                                           overall_quality,
                                           string(stage.quality_threshold))
                        end
                        break
                    end
                end
            end

        catch err
            verbose && @error "  Error in $method_name: $err"
            continue
        end
    end

    total_time = time() - start_time

    if current === nothing
        throw(ErrorException(_CASCADE_NO_STAGES))
    end

    spectrum = vec(current)::Vector{T}
    reconstructed_readings = A * spectrum
    final_metrics = compute_quality_metrics(spectrum, reconstructed_readings,
                                           measured_readings, energy)
    method_sequence = String[stages.method
                             for stages in cascade_stages[1:min(stages_run,
                                                                length(cascade_stages))]]
    message = "Successfully completed $stages_run cascade stages"

    cascade_result = CascadeResult(spectrum, stages_run, total_time,
                                   intermediate_results, final_metrics,
                                   method_sequence, convergence_history,
                                   "OK", message)

    return UnfoldResult(Vector{T}(spectrum), stages_run, true,
                        T(norm(measured_readings .- reconstructed_readings)),
                        Dict{String, Any}(
                            "stages_run" => stages_run,
                            "total_time" => total_time,
                            "intermediate_results" => intermediate_results,
                            "quality_metrics" => final_metrics,
                            "method_sequence" => method_sequence,
                            "convergence_history" => convergence_history,
                            "status" => "OK",
                            "message" => message,
                            "cascade_result" => cascade_result,
                        ))
end

"""
    solve_adaptive_cascade(A, b, x0; max_stages=5, initial_method="tsvd",
                           calculate_errors=false, verbose=true,
                           save_result=false, multi_resolution=false,
                           coarse_bins=nothing, E_MeV=nothing)

Adaptive cascade unfolding with dynamic method selection (port of
`unfold_adaptive_cascade`): the first stage runs `initial_method`, every
further method is chosen by `select_next_method` from the metrics of the last
successful stage and never repeats a used method.  The cascade is re-run
incrementally so that each selection sees fresh metrics, and the fully grown
stage list is run once more at the end.

# Returns
`UnfoldResult` — see `solve_cascade`.
"""
function solve_adaptive_cascade(A::AbstractMatrix{T}, b::AbstractVector{T}, x0::AbstractVector{T};
                                max_stages::Integer = 5,
                                initial_method::AbstractString = "tsvd",
                                calculate_errors::Bool = false,
                                verbose::Bool = true,
                                save_result::Bool = false,
                                multi_resolution::Bool = false,
                                coarse_bins::Union{Nothing, Integer} = nothing,
                                E_MeV::Union{Nothing, AbstractVector} = nothing) where T<:AbstractFloat
    available_methods = String[
        "tsvd", "landweber", "mlem", "cvxpy", "qpsolvers",
        "statreg", "bayes", "bayes_spline", "cgls", "hybrid_gmres",
        "parametric2", "hybrid_parametric", "tikhonov_tv", "gravel",
    ]

    cascade_stages = CascadeStage{Float64}[]
    current_metrics = Dict{String, Any}(
        "smoothness" => 0.5, "chi_square" => 10.0, "flux_error" => 1.0)
    convergence_history = Dict{String, Any}[]

    for stage_idx in 0:(Int(max_stages) - 1)
        used = Set{String}(stages.method for stages in cascade_stages)
        remaining = [meth for meth in available_methods if !(meth in used)]
        if stage_idx == 0
            method = String(initial_method)
        else
            method = select_next_method(current_metrics, remaining, stage_idx)
            # Update metrics from the previous successful stage if available.
            if !isempty(convergence_history)
                last = convergence_history[end]
                current_metrics = Dict{String, Any}(String(k) => v for (k, v) in last)
            end
        end

        push!(cascade_stages,
              CascadeStage(method;
                           use_as_initial = stage_idx > 0,
                           use_as_prior = (stage_idx > 0 &&
                                           method in ("bayes", "bayes_spline")),
                           store_intermediate = true))

        verbose && @info "Adaptive stage $(stage_idx + 1): selected $method"

        # Run incrementally so the next selection can use fresh metrics.  A
        # cascade with no successful stage returns a `nothing` spectrum in
        # Python and throws here.
        result = try
            solve_cascade(A, b, x0; stages=cascade_stages,
                          calculate_errors=false, verbose=verbose,
                          save_result=save_result, E_MeV=E_MeV)
        catch err
            (err isa ErrorException && err.msg == _CASCADE_NO_STAGES) || rethrow()
            break
        end
        hist = result.extra["convergence_history"]::Vector{Dict{String, Any}}
        isempty(hist) || (convergence_history = hist)
    end

    return solve_cascade(A, b, x0; stages=cascade_stages,
                         calculate_errors=calculate_errors, verbose=verbose,
                         save_result=save_result, multi_resolution=multi_resolution,
                         coarse_bins=coarse_bins, E_MeV=E_MeV)
end

# ─── Internal helpers ────────────────────────────────────────────────────────

"""
    _cascade_spectrum(result, T)

Extract the stage spectrum from a member solver's output (Python
`result["spectrum"] is not None`), clamped to nonnegativity as the Python
wrappers do before the cascade consumes it.  `nothing` when absent.
"""
function _cascade_spectrum(result, ::Type{T}) where T<:AbstractFloat
    raw = if result isa UnfoldResult
        result.spectrum
    elseif result isa AbstractVector
        result
    elseif result isa AbstractDict
        get(result, "spectrum", nothing)
    elseif result isa Tuple && !isempty(result) && result[1] isa AbstractVector
        result[1]
    else
        nothing
    end
    raw === nothing && return nothing
    return max.(vec(collect(T, raw)), zero(T))
end

"""
    _cascade_energy_grid(T, E_MeV, n)

Energy grid used by the quality metrics; the package's standard synthetic grid
when the caller does not supply one.
"""
function _cascade_energy_grid(::Type{T}, E_MeV::Union{Nothing, AbstractVector},
                              n::Int) where T<:AbstractFloat
    E_MeV === nothing && return T.(collect(10.0 .^ range(-9, 2; length=n)))
    length(E_MeV) == n || throw(DimensionMismatch(
        "length(E_MeV) ($(length(E_MeV))) must match the number of energy bins ($n)"))
    return vec(collect(T, E_MeV))
end

"""
    _accepts_keyword(fn, A, b, x, key)

Python's `inspect.signature` check for a member method accepting `key`
(`reference_spectrum`).
"""
function _accepts_keyword(fn::Function, A::AbstractMatrix, b::AbstractVector,
                          x::AbstractVector, key::Symbol)
    try
        return hasmethod(fn, Tuple{typeof(A), typeof(b), typeof(x)}, (key,))
    catch
        return false
    end
end

"""
    _metric(metrics, key, default)

Metric lookup by name, tolerant of string and symbol keys.
"""
function _metric(metrics::AbstractDict, key::AbstractString, default::Float64)
    haskey(metrics, key) && return Float64(metrics[key])
    sk = Symbol(key)
    return haskey(metrics, sk) ? Float64(metrics[sk]) : default
end
