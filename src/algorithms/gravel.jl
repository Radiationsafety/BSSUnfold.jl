"""
GRAVEL — weighted log-likelihood iterative algorithm.

    x_{k+1}[j] = x_k[j] * exp( Σ_i W[i,j] * ln(b_i / (A x_k)_i) / Σ_i W[i,j] )

where W[i,j] = b_i * A[i,j] * x_k[j] / (A x_k)_i

Port from bssunfold/src/bssunfold/core/unfold_gravel.py.
"""
function solve_gravel(A::AbstractMatrix{T}, b::AbstractVector{T}, x0::AbstractVector{T};
                     max_iterations::Integer=1000,
                     tolerance::T=T(1e-8),
                     regularization::T=T(0.0),
                     eps::T=T(1e-10)) where T<:AbstractFloat
    m, n = size(A)

    valid = b .> 0
    if !any(valid)
        throw(ArgumentError("All measurements are zero or negative"))
    end
    Av = A[valid, :]
    bv = b[valid]
    mv = sum(valid)

    x = copy(x0)

    J_prev = T(0)
    dJ_prev = T(1)
    converged = false
    iters = 0

    @inbounds for k in 1:max_iterations
        iters = k
        computed = Av * x
        computed_safe = max.(computed, eps)

        log_ratio = log.(bv ./ computed_safe)
        @inbounds for i in 1:mv
            if !(bv[i] > 0 && computed_safe[i] > 0 && computed[i] > 0)
                log_ratio[i] = 0
            end
        end

        numerator   = zeros(T, n)
        denominator = zeros(T, n)
        @inbounds for j in 1:n
            xj = max(x[j], T(0))
            s_num = T(0)
            s_den = T(0)
            for i in 1:mv
                Wij = bv[i] * Av[i, j] * xj / computed_safe[i]
                s_den += Wij
                s_num += Wij * log_ratio[i]
            end
            numerator[j] = s_num
            denominator[j] = s_den
        end

        for j in 1:n
            if denominator[j] > 0
                reg_term = regularization * log(x[j] + eps)
                update = exp((numerator[j] - reg_term) / denominator[j])
                x[j] *= update
            end
        end

        computed_final = Av * x
        chi_sq = sum((computed_final .- bv).^2 ./ max.(bv, eps))
        J = chi_sq / sum(computed_final)
        dJ = J_prev - J
        ddJ = abs(dJ - dJ_prev)
        J_prev = J
        dJ_prev = dJ
        if ddJ <= tolerance
            converged = true
            break
        end
    end

    residual = bv .- Av * x
    return UnfoldResult(max.(x, T(0)), iters, converged, norm(residual))
end
