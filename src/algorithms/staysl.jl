"""STAY'SL Bayesian unfolding — faithful port of
`bssunfold.core.unfold_staysl.solve_staysl`.

A single-step linear Bayesian least-squares update that refines a prior
spectrum `x0` using the full measurement and prior covariance
information:

    Cb      = diag((relative_uncertainty · |b|)²)
    Cx      = diag((prior_uncertainty · |x0|)²)
    bracket = Cb + A Cx Aᵀ + regularization · I
    gain    = Cx Aᵀ bracket⁻¹
    x       = max.(x0 + gain (b − A x0), 0)

STAY'SL is a single-step method, so `iterations == 1` and
`converged == true` always.
"""
function solve_staysl(A::AbstractMatrix{T}, b::AbstractVector{T}, x0::AbstractVector{T};
                      relative_uncertainty::Real=T(0.1),
                      prior_uncertainty::Real=T(1.0),
                      Cb::Union{Nothing,AbstractMatrix{<:Real}}=nothing,
                      Cx::Union{Nothing,AbstractMatrix{<:Real}}=nothing,
                      regularization::Real=T(1e-12)) where T<:AbstractFloat
    A = Matrix{T}(A)
    b = Vector{T}(b)
    x0f = Vector{T}(x0)
    m, n = size(A)

    isempty(A) || isempty(b) &&
        throw(ArgumentError("Response matrix and measurements must be non-empty"))

    if Cb === nothing
        b_safe = max.(abs.(b), T(1e-12))
        Cb_m = Matrix{T}(Diagonal((T(relative_uncertainty) .* b_safe) .^ 2))
    else
        Cb_m = Matrix{T}(Cb)
    end
    if Cx === nothing
        x_safe = max.(abs.(x0f), T(1e-12))
        Cx_m = Matrix{T}(Diagonal((T(prior_uncertainty) .* x_safe) .^ 2))
    else
        Cx_m = Matrix{T}(Cx)
    end

    bracket = Cb_m .+ A * Cx_m * A' .+ T(regularization) .* Matrix{T}(I, m, m)
    gain = Cx_m * A' * inv(bracket)
    spectrum = max.(x0f .+ gain * (b .- A * x0f), T(0))

    residual = b .- A * spectrum
    return UnfoldResult(spectrum, 1, true, norm(residual))
end
