"""
Landweber iteration — метод простой итерации для Ax=b.

    x_{k+1} = x_k + ω Aᵀ (b - A x_k),   0 < ω < 2/||A||²

С проекцией на неотрицательный ортант (physical constraint).
"""
function solve_landweber(A::AbstractMatrix{T}, b::AbstractVector{T}, x0::AbstractVector{T};
                        max_iterations::Integer=1000,
                        tolerance::T=T(1e-6),
                        omega::T=T(0.0),
                        eps::T=T(1e-10)) where T<:AbstractFloat
    m, n = size(A)
    if omega <= 0
        # Эвристический шаг
        omega = T(1.0) / (opnorm(A)^2 + eps)
    end
    AT = Matrix(A')
    x = max.(copy(x0), T(0))
    converged = false
    iters = 0
    b_norm = norm(b) + eps

    @inbounds for k in 1:max_iterations
        iters = k
        r = b .- A * x
        x_new = x .+ omega .* (AT * r)
        @. x = max(x_new, T(0))
        if norm(r) / b_norm < tolerance
            converged = true
            break
        end
    end

    residual = b .- A * x
    return UnfoldResult(x, iters, converged, norm(residual))
end
