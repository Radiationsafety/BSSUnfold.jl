"""
FISTA — Fast Iterative Shrinkage-Thresholding Algorithm (Beck & Teboulle, 2009).

Решает: min_x (1/2)||Ax - b||² + λ ||x||_1

с проекцией на неотрицательность (soft-thresholding).

Отличается от ISTA ускорением O(1/k²) за счёт momentum.
"""
function solve_fista(A::AbstractMatrix{T}, b::AbstractVector{T}, x0::AbstractVector{T};
                    max_iterations::Integer=500,
                    tolerance::T=T(1e-6),
                    regularization::T=T(1e-3),
                    step_size::T=T(0.0),
                    eps::T=T(1e-10)) where T<:AbstractFloat
    m, n = size(A)

    # Шаг градиентного спуска: 1/L, где L = ||A||²
    L = step_size > 0 ? step_size : opnorm(A)^2
    step = T(1.0) / (L + eps)

    ATA = A' * A
    ATb = A' * b
    λ = regularization

    x = copy(x0)
    y = copy(x0)
    t = T(1)

    converged = false
    iters = 0

    @inbounds for k in 1:max_iterations
        iters = k
        # Gradient: Aᵀ(A y - b)
        grad = ATA * y .- ATb
        x_new = y .- step .* grad
        # Soft-thresholding + non-negative projection
        x_new = max.(x_new .- λ * step, T(0))  # ||x||_1 + nonneg

        t_new = (T(1) + sqrt(T(1) + 4t^2)) / 2
        y = x_new .+ ((t - 1) / t_new) .* (x_new .- x)

        diff = norm(x_new .- x) / (norm(x) + eps)
        x = x_new
        t = t_new

        if diff < tolerance
            converged = true
            break
        end
    end

    residual = b .- A * x
    return UnfoldResult(x, iters, converged, norm(residual))
end
