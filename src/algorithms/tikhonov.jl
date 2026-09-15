"""
Tikhonov regularization.

Minimizes: ||Ax - b||² + λ ||Lx||²

where L is the regularization operator (identity matrix by default).
Solved via normal equations: (AᵀA + λLᵀL) x = Aᵀb.
"""
function solve_tikhonov(A::AbstractMatrix{T}, b::AbstractVector{T}, x0::AbstractVector{T};
                       max_iterations::Integer=1,
                       tolerance::T=T(1e-8),
                       regularization::T=T(1e-3),
                       eps::T=T(1e-10)) where T<:AbstractFloat
    n = size(A, 2)
    λ = regularization
    # Solve the Tikhonov problem via the augmented least-squares system
    # [A; √λ·I] x ≈ [b; 0], which is numerically stable even when the
    # normal equations (AᵀA + λI) are severely ill-conditioned (rank-deficient A).
    augmented = vcat(A, sqrt(λ) * Matrix{T}(I, n, n))
    rhs = vcat(b, zeros(T, n))
    x = augmented \ rhs
    # Project to non-negative
    x = max.(x, T(0))

    residual = b .- A * x
    return UnfoldResult(x, 1, true, norm(residual))
end
