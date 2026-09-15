"""
Sandii — iterative unfolding algorithm (Sandii, 1970).

Close to MLEM, but with an asymmetric update scheme:

    x_{k+1}[j] = x_k[j] * Σ_i (A[i,j] * b_i / (A x_k)_i)

(without normalization by Σ_i A[i,j], as in MLEM)
"""
function solve_sandii(A::AbstractMatrix{T}, b::AbstractVector{T}, x0::AbstractVector{T};
                     max_iterations::Integer=1000,
                     tolerance::T=T(1e-6),
                     eps::T=T(1e-10)) where T<:AbstractFloat
    m, n = size(A)
    x = max.(copy(x0), eps)
    AT = Matrix(A')

    converged = false
    iters = 0

    @inbounds for k in 1:max_iterations
        iters = k
        Ax = A * x
        Ax = max.(Ax, eps)
        ratio = b ./ Ax
        correction = AT * ratio
        # Sandii: without normalization by the column sum
        x_new = x .* correction
        # Renormalize to match total counts
        total = sum(x_new)
        target_total = sum(b)
        if total > eps
            x_new = x_new .* (target_total / total)
        end
        diff = norm(x_new .- x) / (norm(x) + eps)
        x = max.(x_new, T(0))
        if diff < tolerance
            converged = true
            break
        end
    end

    residual = b .- A * x
    return UnfoldResult(x, iters, converged, norm(residual))
end
