"""
Bunki — modified MLEM for BSS.

Difference from MLEM: a relaxation coefficient α is introduced
(usually 0.7–0.9):

    x_{k+1}[j] = x_k[j] * (1 + α * ((Aᵀ (b ./ (A x_k)))[j] - 1))

This gives faster convergence on ill-conditioned problems.
"""
function solve_bunki(A::AbstractMatrix{T}, b::AbstractVector{T}, x0::AbstractVector{T};
                    max_iterations::Integer=1000,
                    tolerance::T=T(1e-6),
                    alpha::T=T(0.8),
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
        # Bunki update: x_new = x * (1 + α * (correction - 1))
        # NB: in classical MLEM correction = Aᵀ(b/Ax); in Bunki:
        x_new = x .* (T(1) .+ alpha .* (correction .- T(1)))
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
