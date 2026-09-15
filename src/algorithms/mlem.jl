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
    x = max.(copy(x0), eps)
    AT = Matrix(A')  # materialised transpose for cache-friendly multiplications
    converged = false
    iters = 0

    @inbounds for k in 1:max_iterations
        iters = k
        Ax = A * x
        @. Ax = max(Ax, eps)
        ratio = b ./ Ax
        correction = AT * ratio
        x_new = x .* correction
        diff = norm(x_new .- x) / (norm(x) + eps)
        @. x = max(x_new, 0)
        if diff < tolerance
            converged = true
            break
        end
    end

    residual = b .- A * x
    return UnfoldResult(x, iters, converged, norm(residual))
end
