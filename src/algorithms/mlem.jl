"""
MLEM (Maximum Likelihood Expectation Maximization).

Iterative algorithm:

    x_{k+1} = x_k ⊙ (Aᵀ (b ./ (A x_k)))

This is the standard algorithm for PET/SPECT and BSS unfolding.
Preserves non-negativity of x.
"""
function solve_mlem(A::AbstractMatrix{T}, b::AbstractVector{T}, x0::AbstractVector{T};
                   max_iterations::Integer=1000,
                   tolerance::T=T(1e-6),
                   eps::T=T(1e-10)) where T<:AbstractFloat
    m, n = size(A)
    x = Vector{T}(max.(x0, eps))
    AT = Matrix(A')  # materialised transpose for cache-friendly multiplications
    # Scratch buffers: allocated once, reused every iteration (in-place `mul!`)
    Ax = Vector{T}(undef, m)
    ratio = Vector{T}(undef, m)
    corr = Vector{T}(undef, n)
    x_new = Vector{T}(undef, n)
    converged = false
    iters = 0

    for k in 1:max_iterations
        iters = k
        mul!(Ax, A, x)
        @inbounds @simd for i in 1:m
            Ax[i] = max(Ax[i], eps)
            ratio[i] = b[i] / Ax[i]
        end
        mul!(corr, AT, ratio)
        norm_x = norm(x)
        s = T(0)
        @inbounds @simd for j in 1:n
            x_new[j] = x[j] * corr[j]
            d = x_new[j] - x[j]
            s += d * d
        end
        @inbounds @simd for j in 1:n
            x[j] = max(x_new[j], T(0))
        end
        if sqrt(s) / (norm_x + eps) < tolerance
            converged = true
            break
        end
    end

    residual = b .- A * x
    return UnfoldResult(x, iters, converged, norm(residual))
end
