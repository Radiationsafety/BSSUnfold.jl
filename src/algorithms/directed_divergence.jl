"""
Directed divergence — unfolding for Bonner-sphere response matrices:
minimization of the Poisson I-divergence data term by multiplicative
updates; optionally a Tikhonov proximal step (1st or 2nd order)
after each update.
"""
function solve_directed_divergence(A::AbstractMatrix{T}, b::AbstractVector{T},
                                   x0::AbstractVector{T};
                                   max_iterations::Integer=200,
                                   tol_chi2::Real=one(T),
                                   tol_rel::Real=T(1e-6),
                                   relative_uncertainty::Real=T(0.05),
                                   sigma::Union{Nothing,AbstractVector{T}}=nothing,
                                   smoothness_order::Integer=0,
                                   smoothness_weight::Real=zero(T)) where T<:AbstractFloat
    n = size(A, 2)
    x = max.(copy(x0), T(1e-30))

    sigma_v = sigma === nothing ?
        max.(T(relative_uncertainty) .* max.(b, T(1e-30)), T(1e-30)) : sigma
    all(sigma_v .> 0) && length(sigma_v) == length(b) ||
        throw(ArgumentError("sigma must be positive and match b"))

    weights = 1 ./ sigma_v .^ 2
    denominator = max.(vec(sum(A, dims=1)), T(1e-30))

    penalty = nothing
    if smoothness_order != 0 && T(smoothness_weight) != 0
        L = _create_derivative_matrix(n, smoothness_order)
        penalty = Matrix(T(smoothness_weight) * (L' * L))
    end

    for iteration in 1:max_iterations
        predicted = max.(A * x, T(1e-30))
        chi2 = sum((predicted .- b) .^ 2 .* weights) / length(b)
        if chi2 <= T(tol_chi2)
            residual = b .- A * x
            return UnfoldResult(x, iteration, true, norm(residual))
        end

        update = A' * (b ./ predicted)
        new_x = x .* (update ./ denominator)
        if penalty !== nothing
            new_x = (Matrix{T}(I, n, n) + penalty) \ new_x
        end
        new_x = max.(new_x, T(1e-30))
        relative_change = maximum(abs.(new_x .- x) ./ max.(x, T(1e-30)))
        x = new_x
        if relative_change <= T(tol_rel)
            residual = b .- A * x
            return UnfoldResult(x, iteration, true, norm(residual))
        end
    end

    residual = b .- A * x
    return UnfoldResult(x, max_iterations, false, norm(residual))
end
