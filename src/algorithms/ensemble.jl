"""
    solve_ensemble(A, b, x0=nothing; methods=nothing, weights=nothing,
                   combination="weighted_average", trim_fraction=0.2)

Ensemble unfolding method: combines the results of several
base algorithms to reduce variance and to guard against specific
failures of individual methods.

Supported combination strategies:
- `"weighted_average"` — convex combination of spectra with weights
  inversely proportional to the residual norms (or given by `weights`);
- `"median"` — element-wise median of all solutions;
- `"trimmed_mean"` — mean after discarding a fraction of extreme values;
- `"best_residual"` — the single solution with the minimum residual.

`methods` is a vector of tuples `(solver_function, kwargs_dict)`; by
default the ensemble MLEM, Bayes, Landweber, CGLS and GRAVEL is used.
`kwargs_dict` may contain the key `"_name"` — the display name of the method.

# Returns
`UnfoldResult`; `extra` contains method names, residuals, weights and metadata.
"""
function solve_ensemble(A::AbstractMatrix{T}, b::AbstractVector{T},
                        x0::Union{Nothing,AbstractVector{T}}=nothing;
                        methods=nothing,
                        weights::Union{Nothing,AbstractVector{T}}=nothing,
                        combination::AbstractString="weighted_average",
                        trim_fraction::T=T(0.2)) where T<:AbstractFloat
    m, n = size(A)
    x0eff = x0 === nothing ? fill(T(0.5), n) : x0

    if methods === nothing
        methods = default_ensemble_methods(T)
    end

    if !(combination in ("weighted_average", "median", "trimmed_mean", "best_residual"))
        throw(ArgumentError("Unknown combination '$combination'. " *
                            "Choose from weighted_average, median, trimmed_mean, best_residual"))
    end

    spectra = Vector{Vector{T}}()
    residuals = Vector{T}()
    names = String[]

    for (idx, (solver, kwargs)) in enumerate(methods)
        name = String(get(kwargs, Symbol("_name"), "method_$(idx - 1)"))
        clean_kwargs = Dict{Symbol,Any}(k => v for (k, v) in kwargs if !(String(k) == "_name"))
        try
            result = solver(A, b, x0eff; clean_kwargs...)
            if result isa UnfoldResult
                x_sol = result.spectrum
            elseif result isa AbstractVector
                x_sol = collect(T, result)
            elseif result isa Tuple
                x_sol = collect(T, result[1])
            else
                error("unsupported solver output type $(typeof(result))")
            end
            x_sol = max.(vec(float.(x_sol)), T(0))
            length(x_sol) == n || error("solver returned wrong-length spectrum")
            push!(spectra, x_sol)
            push!(residuals, norm(A * x_sol .- b))
            push!(names, name)
        catch err
            @warn "Ensemble method $name failed: $err"
        end
    end

    isempty(spectra) && throw(ErrorException("All ensemble methods failed"))

    spectra_arr = hcat(spectra...)'
    effective_weights = weights

    if combination == "best_residual"
        best_idx = argmin(residuals)
        spectrum = spectra_arr[best_idx, :]
        info_str = "best=$(names[best_idx]) (res=$(residuals[best_idx]))"
    elseif combination == "median"
        spectrum = BSSUnfold.Statistics.median(spectra_arr, dims=1) |> vec
        info_str = "median of $(length(spectra)) methods"
    elseif combination == "trimmed_mean"
        k = max(1, floor(Int, trim_fraction * length(spectra)))
        sorted_spectra = sort(spectra_arr, dims=1)
        trimmed = k < length(spectra) ? sorted_spectra[k+1:end-k, :] : spectra_arr
        spectrum = BSSUnfold.Statistics.mean(trimmed, dims=1) |> vec
        info_str = "trimmed_mean (trim=$trim_fraction) of $(length(spectra)) methods"
    else
        if effective_weights === nothing
            effective_weights = _compute_weights_from_residuals(spectra, A, b)
        end
        w = float.(collect(effective_weights))
        w ./= sum(w)
        spectrum = w' * spectra_arr
        info_str = "weighted_average of $(length(spectra)) methods"
    end

    spectrum = vec(max.(collect(float.(spectrum)), T(0)))
    residual = norm(b .- A * spectrum)

    return UnfoldResult(spectrum, length(spectra), true, residual,
                        Dict{String,Any}(
                            "combination" => combination,
                            "n_methods" => length(spectra),
                            "method_names" => names,
                            "residuals" => residuals,
                            "weights" => effective_weights === nothing ? nothing : collect(float.(effective_weights)),
                            "info_str" => info_str,
                        ))
end

"""
    default_ensemble_methods(::Type{T})

Default ensemble: (MLEM, Bayes, Landweber, CGLS, GRAVEL) with
conservative parameters (`max_iterations=200, tolerance=1e-4`).
"""
function default_ensemble_methods(::Type{T}=Float64) where T<:AbstractFloat
    kw = Dict{Symbol,Any}(:max_iterations => 200, :tolerance => T(1e-4))
    return Tuple{Function,Dict{Symbol,Any}}[
        (BSSUnfold.solve_mlem,       merge(Dict{Symbol,Any}(Symbol("_name") => "MLEM"), kw)),
        (BSSUnfold.solve_bayes,      merge(Dict{Symbol,Any}(Symbol("_name") => "Bayes"), kw)),
        (BSSUnfold.solve_landweber,  merge(Dict{Symbol,Any}(Symbol("_name") => "Landweber"), kw)),
        (BSSUnfold.solve_cgls,       merge(Dict{Symbol,Any}(Symbol("_name") => "CGLS"), kw)),
        (BSSUnfold.solve_gravel,     merge(Dict{Symbol,Any}(Symbol("_name") => "GRAVEL"), kw)),
    ]
end

"""
    _compute_weights_from_residuals(spectra, A, b)

Inverse weights: `w_i = 1 / ||A x_i - b||`, normalized to one.
"""
function _compute_weights_from_residuals(spectra::Vector{Vector{T}},
                                         A::AbstractMatrix{T},
                                         b::AbstractVector{T}) where T<:AbstractFloat
    weights = T[]
    for x in spectra
        res = norm(A * x .- b)
        push!(weights, one(T) / (res + T(1e-30)))
    end
    weights ./= sum(weights)
    return weights
end
