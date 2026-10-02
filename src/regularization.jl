"""
Regularization parameter selection methods (port of regularization.py).
"""

"""
    lcurve_selection(A, b, x0; lambda_range, solve_func, max_iterations, kwargs...)

Select λ from the L-curve: balance between ||Ax-b|| and ||Lx||.
"""
function lcurve_selection(A::AbstractMatrix{T}, b::Vector{T}, x0::Vector{T};
                         lambda_range::AbstractVector{<:Real}=10.0 .^ range(-6, 2, length=30),
                         solve_func::Function=solve_tikhonov,
                         max_iterations::Integer=1000,
                         kwargs...) where T<:AbstractFloat
    residuals = Float64[]
    regularizers = Float64[]
    for λ in lambda_range
        res = solve_func(A, b, x0; regularization=T(λ), max_iterations=max_iterations, kwargs...)
        push!(residuals, log(res.residual_norm + eps(T)))
        push!(regularizers, log(norm(res.spectrum) + eps(T)))
    end
    # L-curve curvature (three-point formula)
    curvature = Float64[]
    for i in 2:length(residuals)-1
        dx1, dy1 = residuals[i] - residuals[i-1], regularizers[i] - regularizers[i-1]
        dx2, dy2 = residuals[i+1] - residuals[i], regularizers[i+1] - regularizers[i]
        κ = (dx1 * dy2 - dx2 * dy1) / ((dx1^2 + dy1^2)^1.5 + eps(T))
        push!(curvature, κ)
    end
    # Point of maximum curvature
    idx = argmax(curvature) + 1
    return lambda_range[idx], curvature
end


"""
    gcv_selection(A, b, x0; lambda_range, solve_func, kwargs...)

Generalized Cross-Validation: select λ minimizing GCV(λ).
"""
function gcv_selection(A::AbstractMatrix{T}, b::Vector{T}, x0::Vector{T};
                      lambda_range::AbstractVector{<:Real}=10.0 .^ range(-6, 2, length=30),
                      solve_func::Function=solve_tikhonov,
                      max_iterations::Integer=1000,
                      kwargs...) where T<:AbstractFloat
    m, n = size(A)
    gcv_values = Float64[]
    for λ in lambda_range
        res = solve_func(A, b, x0; regularization=T(λ), max_iterations=max_iterations, kwargs...)
        residual_norm = res.residual_norm
        # Effective number of parameters (approximation)
        dof = min(m, n) - count(!iszero, res.spectrum) / max(n, 1)
        gcv = residual_norm^2 / (1 - dof/m)^2
        push!(gcv_values, gcv)
    end
    idx = argmin(gcv_values)
    return lambda_range[idx], gcv_values
end


"""
    select_regularization_parameter(A, b, x0; method=:lcurve, kwargs...)

Select a regularization parameter with one of the methods: :lcurve, :gcv, :discrepancy.

# Returns
NamedTuple with fields `lambda`, `method`, `info`.
"""
function select_regularization_parameter(A::AbstractMatrix{T}, b::Vector{T}, x0::Vector{T};
                                        method::Symbol=:lcurve,
                                        kwargs...) where T<:AbstractFloat
    if method == :lcurve
        λ, info = lcurve_selection(A, b, x0; kwargs...)
    elseif method == :gcv
        λ, info = gcv_selection(A, b, x0; kwargs...)
    elseif method == :discrepancy
        # Discrepancy principle: choose λ such that ||Ax-b|| ≈ σ * sqrt(m)
        # where σ is a prior noise estimate
        noise_level = get(kwargs, :noise_level, T(0.01))
        target_residual = noise_level * sqrt(size(A, 1))
        # Binary search
        lo, hi = T(1e-6), T(1e2)
        for _ in 1:50
            mid = sqrt(lo * hi)
            res = solve_tikhonov(A, b, x0; regularization=mid, max_iterations=get(kwargs, :max_iterations, 1000))
            if res.residual_norm > target_residual
                hi = mid
            else
                lo = mid
            end
        end
        λ = sqrt(lo * hi)
        info = (target_residual=target_residual, final_residual=sqrt(lo * hi))
    else
        throw(ArgumentError("Unknown method: $method. Use :lcurve, :gcv, or :discrepancy"))
    end
    return (lambda=λ, method=method, info=info)
end


# ─── α selection from the Tikhonov family (port of regularization.py) ───────
#
# The Python selectors delegate to `pytikhonov` and fall back to the
# implementations below when it is missing. `pytikhonov` is not a dependency of
# the reference environment, so the fallbacks are what actually runs there and
# are therefore what gets ported.

"""
    _estimate_noise_variance(A, b)

Noise variance from the unregularized least-squares residual (`numpy.var`
semantics: mean squared deviation from the mean). Port of
`_estimate_noise_variance`.
"""
function _estimate_noise_variance(A::AbstractMatrix{T}, b::AbstractVector{T}) where T<:AbstractFloat
    x_ls = pinv(A) * b
    residual = b .- A * x_ls
    mu = sum(residual) / length(residual)
    return Float64(sum(abs2, residual .- mu) / length(residual))
end

_logspace(lo::Real, hi::Real, n::Integer) = 10.0 .^ range(Float64(lo), Float64(hi); length=n)

"""
    _tikhonov_alpha_path(A, b, alpha)

Unregularized-plus-α·I solution `x = max((A'A + α I)\\ A'b, 0)` and its residual
norm. `nothing` when the system cannot be solved (the Python
`LinAlgError` branch).
"""
function _tikhonov_alpha_path(A::AbstractMatrix{T}, b::AbstractVector{T},
                              alpha::T) where T<:AbstractFloat
    n = size(A, 2)
    P = Symmetric(A' * A + alpha * Matrix{T}(I, n, n), :U)
    x = try
        Vector{T}(P \ (A' * b))
    catch err
        return nothing
    end
    isfinite.(x) |> all || return nothing
    x = max.(x, T(0))
    return (x=x, residual=norm(A * x - b), normx=norm(x))
end

"""
    lcurve_alpha_selection(A, b; n_alphas=50, alpha_range=(1e-9, 1e2))

L-curve corner by maximum distance from the chord of the log-log curve.
Port of `_lcurve_fallback`.
"""
function lcurve_alpha_selection(A::AbstractMatrix{T}, b::AbstractVector{T};
                                n_alphas::Integer=50,
                                alpha_range::Tuple{Real,Real}=(1e-9, 1e2)) where T<:AbstractFloat
    alphas = _logspace(log10(alpha_range[1]), log10(alpha_range[2]), n_alphas)
    residuals = Float64[]
    norms = Float64[]
    for α in alphas
        path = _tikhonov_alpha_path(A, b, T(α))
        path === nothing || (push!(residuals, path.residual); push!(norms, path.normx))
    end
    length(residuals) < 3 && return 1.0
    log_res = log.(residuals)
    log_norm = log.(norms)
    p1 = (log_res[1], log_norm[1])
    p2 = (log_res[end], log_norm[end])
    chord = hypot(p2[1] - p1[1], p2[2] - p1[2])
    chord == 0 && return 1.0
    distances = [abs((p2[1] - p1[1]) * (p1[2] - log_norm[i]) -
                     (p2[2] - p1[2]) * (p1[1] - log_res[i])) / chord
                 for i in eachindex(log_res)]
    return Float64(alphas[argmax(distances)])
end

"""
    gcv_alpha_selection(A, b; n_alphas=50, alpha_range=(1e-9, 1e2))

Generalized cross-validation over the Tikhonov family, evaluated from a single
SVD. Port of `_gcv_fallback`.
"""
function gcv_alpha_selection(A::AbstractMatrix{T}, b::AbstractVector{T};
                             n_alphas::Integer=50,
                             alpha_range::Tuple{Real,Real}=(1e-9, 1e2)) where T<:AbstractFloat
    alphas = _logspace(log10(alpha_range[1]), log10(alpha_range[2]), n_alphas)
    m = size(A, 1)
    F = svd(A)
    s_sq = Float64.(F.S) .^ 2
    UTb = F.U' * b
    gcv_values = Float64[]
    for α in alphas
        filt = s_sq ./ (s_sq .+ α)
        residual_coeff = α ./ (s_sq .+ α)
        residual_sq = sum(abs2, residual_coeff .* UTb)
        push!(gcv_values, residual_sq / (m - sum(filt))^2)
    end
    (isempty(gcv_values) || all(isinf, gcv_values)) && return 1.0
    return Float64(alphas[argmin(gcv_values)])
end

"""
    discrepancy_alpha_selection(A, b; noise_var=nothing, n_alphas=50,
                                alpha_range=(1e-9, 1e2))

Discrepancy principle: the α whose residual is closest to `δ·√m`. Port of
`_dp_fallback`; `noise_var=nothing` estimates it from the data.
"""
function discrepancy_alpha_selection(A::AbstractMatrix{T}, b::AbstractVector{T};
                                     noise_var::Union{Nothing,Real}=nothing,
                                     n_alphas::Integer=50,
                                     alpha_range::Tuple{Real,Real}=(1e-9, 1e2)) where T<:AbstractFloat
    alphas = _logspace(log10(alpha_range[1]), log10(alpha_range[2]), n_alphas)
    var = noise_var === nothing ? _estimate_noise_variance(A, b) : Float64(noise_var)
    target = sqrt(var) * sqrt(length(b))
    residuals = Float64[]
    for α in alphas
        path = _tikhonov_alpha_path(A, b, T(α))
        push!(residuals, path === nothing ? Inf : path.residual)
    end
    return Float64(alphas[argmin(abs.(residuals .- target))])
end

"""
    cosine_similarity_selection(A, b, initial_spectrum; n_alphas=100,
                                alpha_range=(-9.0, 2.0))

α maximizing the cosine similarity between the non-negative Tikhonov solution
and a reference spectrum. Port of `cosine_similarity_selection`; note that its
`alpha_range` is already expressed in log10 units.
"""
function cosine_similarity_selection(A::AbstractMatrix{T}, b::AbstractVector{T},
                                     initial_spectrum::AbstractVector{T};
                                     n_alphas::Integer=100,
                                     alpha_range::Tuple{Real,Real}=(-9.0, 2.0),
                                     norm::Integer=2) where T<:AbstractFloat
    norm_init = sqrt(sum(abs2, initial_spectrum))
    norm_init == 0 && throw(ArgumentError("Initial spectrum has zero norm."))
    reference = initial_spectrum / norm_init
    alphas = _logspace(alpha_range[1], alpha_range[2], n_alphas)
    F = svd(A)
    s = Float64.(F.S)
    s_sq = s .^ 2
    UTb = F.U' * b
    similarities = Float64[]
    for α in alphas
        x = max.(F.Vt' * ((s ./ (s_sq .+ α)) .* UTb), 0.0)
        norm_x = sqrt(sum(abs2, x))
        # The Python original appends 0.0 and then still divides by the zero
        # norm, desynchronizing this list from `alphas`; that is only reachable
        # for a fully suppressed solution, so the intended branch is taken.
        push!(similarities, norm_x == 0 ? 0.0 : dot(x, reference) / norm_x)
    end
    return Float64(alphas[argmax(similarities)])
end

"""
    resolve_regularization_parameter(A, b, regularization_method="manual",
                                     regularization=1e-4, n_energy_bins=size(A, 2);
                                     initial_spectrum=nothing, norm=2,
                                     noise_var=nothing, verbose=true) -> Float64

Shared α resolution for the QP-based wrappers (`solve_qpsolvers`,
`solve_docplex`, `solve_scip`, `solve_commercial`): `'manual'`, `'cosine'` and
the automatic `'lcurve'`, `'gcv'`, `'dp'` selectors. Returns `regularization`
unchanged for `'manual'`. Port of `resolve_regularization_parameter`.
"""
function resolve_regularization_parameter(A::AbstractMatrix{T}, b::AbstractVector{T},
                                          regularization_method::AbstractString="manual",
                                          regularization::Real=T(1e-4),
                                          n_energy_bins::Integer=size(A, 2);
                                          initial_spectrum::Union{Nothing,AbstractVector{T}}=nothing,
                                          norm::Integer=2,
                                          noise_var::Union{Nothing,Real}=nothing,
                                          verbose::Bool=true) where T<:AbstractFloat
    if regularization_method == "manual"
        return Float64(regularization)
    end

    if regularization_method == "cosine"
        initial_spectrum === nothing && throw(ArgumentError(
            "For 'cosine' regularization method, initial_spectrum must be provided."))
        if norm != 2
            @warn "Cosine regularization selection assumes L2 norm, but norm=$norm was requested. Using L2 for selection."
        end
        reference = max.(initial_spectrum, T(0))
        if length(reference) != n_energy_bins
            throw(ArgumentError("Initial spectrum length ($(length(initial_spectrum))) " *
                                "must match number of energy bins ($n_energy_bins)"))
        end
        λ = cosine_similarity_selection(A, b, reference; norm=norm)
        verbose && @info "Selected regularization (method=cosine)" lambda=λ
        return λ
    end

    if norm != 2
        @warn "Automatic regularization selection methods assume L2 norm, but norm=$norm was requested. Using L2 for selection."
    end
    λ = try
        if regularization_method == "lcurve"
            lcurve_alpha_selection(A, b)
        elseif regularization_method == "gcv"
            gcv_alpha_selection(A, b)
        elseif regularization_method == "dp"
            discrepancy_alpha_selection(A, b; noise_var=noise_var)
        else
            throw(ArgumentError("Unknown regularization selection method: $regularization_method. " *
                                "Choose from 'lcurve', 'gcv', 'dp', 'cosine'."))
        end
    catch err
        err isa ArgumentError && occursin("Unknown regularization", err.msg) && rethrow(err)
        throw(ArgumentError("Regularization selection failed: $err. " *
                            "Consider using manual regularization."))
    end
    verbose && @info "Selected regularization (method=$regularization_method)" lambda=λ
    return λ
end
