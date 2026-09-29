"""
STAY'SL — single-step linear Bayesian update (posterior mean):

    x = x0 + Cx A^T (Cb + A Cx A^T + regularization I)^-1 (b - A x0)

With diagonal defaults Cb = diag((relative_uncertainty * max(|b|,1e-12))^2)
and Cx = diag((prior_uncertainty * max(|x0|,1e-12))^2); the result is
clamped to be non-negative. One-shot method: iterations = 1, converged.
"""
function solve_staysl(A::AbstractMatrix{T}, b::AbstractVector{T}, x0::AbstractVector{T};
                      relative_uncertainty::Real=T(0.1),
                      prior_uncertainty::Real=one(T),
                      Cb=nothing,
                      Cx=nothing,
                      regularization::Real=T(1e-12)) where T<:AbstractFloat
    isempty(A) && throw(ArgumentError("Response matrix and measurements must be non-empty"))
    isempty(b) && throw(ArgumentError("Response matrix and measurements must be non-empty"))
    m, n = size(A)

    ru = T(relative_uncertainty); pu = T(prior_uncertainty); reg = T(regularization)
    if Cb === nothing
        b_safe = max.(abs.(b), T(1e-12))
        Cb_mat = Diagonal((ru .* b_safe) .^ 2)
    else
        Cb_mat = [T(x) for x in Cb]
    end
    if Cx === nothing
        x_safe = max.(abs.(x0), T(1e-12))
        Cx_mat = Diagonal((pu .* x_safe) .^ 2)
    else
        Cx_mat = [T(x) for x in Cx]
    end

    # Posterior mean: x0 + Cx A^T (Cb + A Cx A^T + reg I)^{-1} (b - A x0)
    bracket = Cb_mat + A * Cx_mat * A'
    y = (bracket + reg * I(m)) \ (b .- A * x0)
    spectrum = x0 .+ Cx_mat * (A' * y)

    spectrum = max.(spectrum, zero(T))
    residual = b .- A * spectrum
    return UnfoldResult(spectrum, 1, true, norm(residual))
end
