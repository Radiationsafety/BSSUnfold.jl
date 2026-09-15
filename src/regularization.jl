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
