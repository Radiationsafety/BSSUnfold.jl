"""
Kaczmarz — row-action метод.

Итерирует по строкам A, обновляя x проекцией на гиперплоскость:

    for i = 1..m:
        x_{k+1} = x_k + (b_i - A[i,:]ᵀ x_k) / ||A[i,:]||² * A[i,:]

Цикл по строкам повторяется max_iterations раз.
Сходится для любой непротиворечивой системы Ax=b.
"""
function solve_kaczmarz(A::AbstractMatrix{T}, b::AbstractVector{T}, x0::AbstractVector{T};
                       max_iterations::Integer=100,
                       tolerance::T=T(1e-6),
                       eps::T=T(1e-10)) where T<:AbstractFloat
    m, n = size(A)
    x = max.(copy(x0), T(0))

    # Pre-compute squared row norms
    row_norm_sq = sum(A .^ 2, dims=2)[:]
    row_norm_sq = max.(row_norm_sq, eps)

    converged = false
    iters = 0

    @inbounds for k in 1:max_iterations
        iters = k
        x_prev = copy(x)
        for i in 1:m
            ai = view(A, i, :)
            factor = (b[i] - dot(ai, x)) / row_norm_sq[i]
            @. x = x + factor * ai
            x = max.(x, T(0))  # project to non-negative
        end
        if norm(x .- x_prev) / (norm(x_prev) + eps) < tolerance
            converged = true
            break
        end
    end

    residual = b .- A * x
    return UnfoldResult(x, iters, converged, norm(residual))
end
