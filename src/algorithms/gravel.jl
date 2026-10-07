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

    # Scratch buffers reused every iteration
    computed    = Vector{T}(undef, mv)
    w           = Vector{T}(undef, mv)
    log_ratio   = Vector{T}(undef, mv)
    tmpw        = Vector{T}(undef, mv)
    numerator   = Vector{T}(undef, n)
    denominator = Vector{T}(undef, n)

    for k in 1:max_iterations
        iters = k
        mul!(computed, Av, x)
        # w[i] = b_i / (A x)_i  (clamped); Σ_i W[i,:] = Aᵀ (b_i/(Ax)_i)
        @inbounds for i in 1:mv
            cs = max(computed[i], eps)
            w[i] = bv[i] / cs
            log_ratio[i] = computed[i] > 0 ? log(bv[i] / cs) : zero(T)
        end
        mul!(denominator, transpose(Av), w)
        @inbounds for i in 1:mv
            tmpw[i] = w[i] * log_ratio[i]
        end
        mul!(numerator, transpose(Av), tmpw)

        for j in 1:n
            xj = max(x[j], T(0))
            if xj * denominator[j] > 0
                reg_term = regularization * log(x[j] + eps)
                update = exp((xj * numerator[j] - reg_term) / (xj * denominator[j]))
                x[j] *= update
            end
        end

        mul!(computed, Av, x)
        chi_sq = T(0)
        sum_comp = T(0)
        @inbounds for i in 1:mv
            d = computed[i] - bv[i]
            chi_sq += d * d / max(bv[i], eps)
            sum_comp += computed[i]
        end
        J = chi_sq / sum_comp
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
