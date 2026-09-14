"""
Tikhonov regularization.

Минимизирует: ||Ax - b||² + λ ||Lx||²

где L — оператор регуляризации (по умолчанию единичная матрица).
Решается через нормальные уравнения: (AᵀA + λLᵀL) x = Aᵀb.
"""
function solve_tikhonov(A::AbstractMatrix{T}, b::AbstractVector{T}, x0::AbstractVector{T};
                       max_iterations::Integer=1,
                       tolerance::T=T(1e-8),
                       regularization::T=T(1e-3),
                       eps::T=T(1e-10)) where T<:AbstractFloat
    m, n = size(A)
    λ = regularization
    # Build regularized normal equations
    ATA = A' * A
    ATb = A' * b
    # Identity regularizer (Tikhonov 0th order)
    M = ATA + λ * Matrix{T}(I, n, n)

    # Solve via Cholesky (positive definite)
    x = cholesky(Symmetric(M)) \ ATb
    # Project to non-negative
    x = max.(x, T(0))

    residual = b .- A * x
    return UnfoldResult(x, 1, true, norm(residual))
end
