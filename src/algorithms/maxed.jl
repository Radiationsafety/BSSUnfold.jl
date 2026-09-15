"""
MAXED — Maximum Entropy Deconvolution.

Iterative algorithm maximizing the Shannon entropy
H(x) = -Σ x_j ln(x_j / x0_j) under the constraints A x = b.

Uses a modified Lagrange multiplier with
a dual formulation (Reginski et al., 1981).
"""
function solve_maxed(A::AbstractMatrix{T}, b::AbstractVector{T}, x0::AbstractVector{T};
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
        # Dual variables λ: λ_i = ln(Ax_i / b_i)
        # Update: x_j = x0_j * exp(-Σ_i A_ij * λ_i)
        # Simplified scheme here: use the residual as λ
        ratio = log.(Ax ./ max.(b, eps))
        # x_new = x0 .* exp.(-Aᵀ * ratio), but use a partial update for stability
        correction = AT * ratio
        x_new = x .* exp.(-correction .* T(0.5))
        x_new = max.(x_new, eps)
        diff = norm(x_new .- x) / (norm(x) + eps)
        x = x_new
        if diff < tolerance
            converged = true
            break
        end
    end

    residual = b .- A * x
    return UnfoldResult(x, iters, converged, norm(residual))
end
