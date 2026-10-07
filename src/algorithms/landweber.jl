"""
Landweber iteration — simple iteration method for Ax=b.

    x_{k+1} = x_k + ω Aᵀ (b - A x_k),   0 < ω < 2/||A||²

With projection onto the non-negative orthant (physical constraint).
"""
function solve_landweber(A::AbstractMatrix{T}, b::AbstractVector{T}, x0::AbstractVector{T};
                        max_iterations::Integer=1000,
                        tolerance::T=T(1e-6),
                        omega::T=T(0.0),
                        eps::T=T(1e-10)) where T<:AbstractFloat
    m, n = size(A)
    if omega <= 0
        # Heuristic step
        omega = T(1.0) / (opnorm(A)^2 + eps)
    end
    AT = Matrix(A')
    x = Vector{T}(max.(x0, T(0)))
    # Scratch buffers reused every iteration
    r = Vector{T}(undef, m)
    x_new = Vector{T}(undef, n)
    converged = false
    iters = 0
    b_norm = norm(b) + eps

    for k in 1:max_iterations
        iters = k
        mul!(r, A, x)
        @inbounds @simd for i in 1:m
            r[i] = b[i] - r[i]
        end
        norm_r = norm(r)
        copyto!(x_new, x)
        mul!(x_new, AT, r, omega, one(T))  # x_new = x + omega * Aᵀ r
        @inbounds @simd for j in 1:n
            x[j] = max(x_new[j], T(0))
        end
        if norm_r / b_norm < tolerance
            converged = true
            break
        end
    end

    residual = b .- A * x
    return UnfoldResult(x, iters, converged, norm(residual))
end
