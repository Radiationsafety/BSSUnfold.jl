"""
CGLS — Conjugate Gradient Least Squares.

Solves min ||Ax - b||² by CG applied to the normal equations
without explicitly forming them. Naturally acts as a regularization:
early iterations ≈ TSVD.

Algorithm: see Hansen "Discrete Inverse Problems", 2010, Algorithm 6.1.
"""
function solve_cgls(A::AbstractMatrix{T}, b::AbstractVector{T}, x0::AbstractVector{T};
                   max_iterations::Integer=200,
                   tolerance::T=T(1e-6),
                   eps::T=T(1e-10)) where T<:AbstractFloat
    m, n = size(A)

    x = max.(copy(x0), T(0))
    r = b .- A * x
    p = A' * r
    rs_old = dot(r, r)

    converged = false
    iters = 0

    @inbounds for k in 1:max_iterations
        iters = k
        Ap = A * p
        α = rs_old / (dot(Ap, Ap) + eps)
        @. x += α * p
        x = max.(x, T(0))
        @. r -= α * Ap
        rs_new = dot(r, r)
        if sqrt(rs_new) < tolerance * (norm(b) + eps)
            converged = true
            break
        end
        β = rs_new / (rs_old + eps)
        p = A' * r .+ β .* p
        rs_old = rs_new
    end

    residual = b .- A * x
    return UnfoldResult(x, iters, converged, norm(residual))
end
