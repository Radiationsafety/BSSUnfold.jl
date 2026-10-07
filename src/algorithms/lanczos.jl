"""
Lanczos-hybrid unfolding (Golub-Kahan bidiagonalization + GCV).

Faithful port of `solve_lanczos` from bssunfold 0.28.0 (`unfold_lanczos.py`).
At each Krylov dimension k the projected bidiagonal problem
min ||B_k y - beta*e_1||^2 + lambda^2 ||y||^2 is solved, with lambda chosen
by GCV on the projected problem. No a-priori spectrum is required
(x0 is accepted for API compatibility only).
"""

# GCV selection of lambda on the projected bidiagonal problem (k+1 x k).
function _projected_gcv(B::AbstractMatrix{T}, bhat::AbstractVector{T}, m::Integer;
                        n_lambdas::Integer=200,
                        lambda_range::Tuple{T,T}=(T(1e-12), T(1e2))) where T<:AbstractFloat
    F = svd(B; full=false)
    c = F.U' * bhat
    orth_res = norm(bhat)^2 - sum(abs2, c)
    s2 = F.S .^ 2

    λs = 10 .^ range(log10(lambda_range[1]), log10(lambda_range[2]); length=Int(n_lambdas))
    gcv_min = typemax(T)
    lam_best = λs[1]
    @inbounds for lam in λs
        num = sum(abs2, c .* lam ./ (s2 .+ lam)) + orth_res
        den = (m - sum(s2 ./ (s2 .+ lam)))^2
        val = num / den
        if val < gcv_min
            gcv_min = val
            lam_best = lam
        end
    end
    return T(lam_best)
end

# Upper bidiagonal (k+1) x k matrix from Lanczos coefficients.
function _build_bidiagonal(alphas::AbstractVector{T}, betas::AbstractVector{T}, k::Integer) where T
    B = zeros(T, k + 1, k)
    @inbounds for i in 1:k
        B[i, i] = alphas[i]
        if i < k
            B[i + 1, i] = betas[i]
        end
    end
    return B
end

"""
    solve_lanczos(A, b, x0; max_iterations=nothing, kwargs...)

Lanczos-hybrid unfolding (Golub-Kahan bidiagonalization; at each Krylov
dimension the projected problem is solved with λ chosen by GCV). Port of
bssunfold 0.28.0 `unfold_lanczos.solve_lanczos`; `x0` is accepted for API
compatibility only.
"""
function solve_lanczos(A::AbstractMatrix{T}, b::AbstractVector{T}, x0::AbstractVector{T};
                       max_iterations::Union{Integer,Nothing}=nothing,
                       regularization::Real=T(1e-8),
                       noise_level::Union{Real,Nothing}=nothing) where T<:AbstractFloat
    m, n = size(A)
    length(b) == m || throw(DimensionMismatch("length(b) != size(A,1)"))
    reg = T(regularization)

    kmax = max_iterations === nothing ? min(m, n) : max(1, Int(max_iterations))

    β = norm(b)
    if β == 0.0
        return UnfoldResult(zeros(T, n), 0, true, T(0))
    end

    U = zeros(T, m, kmax + 1)
    U[:, 1] .= b ./ β
    V = zeros(T, n, kmax)
    alphas = zeros(T, kmax)
    betas = zeros(T, kmax)

    best_x = zeros(T, n)
    iterations = 0
    converged = false
    breakdown_tol = T(1e-14)

    @inbounds for k in 1:kmax
        u = view(U, :, k)
        v = k == 1 ? A' * u : A' * u .- betas[k - 1] * view(V, :, k - 1)
        alpha = norm(v)
        if alpha <= breakdown_tol
            converged = true
            break
        end
        v ./= alpha
        V[:, k] .= v

        u2 = A * v .- alpha * u
        new_beta = norm(u2)
        if new_beta <= breakdown_tol
            converged = true
        else
            U[:, k + 1] .= u2 ./ new_beta
        end

        alphas[k] = alpha
        betas[k] = new_beta

        B = _build_bidiagonal(view(alphas, 1:k), view(betas, 1:k), k)
        bhat = zeros(T, k + 1)
        bhat[1] = β

        lam = _projected_gcv(B, bhat, m)
        if lam <= 0 || !isfinite(lam)
            lam = reg
        end

        F = svd(B; full=false)
        c = F.U' * bhat
        s = F.S
        # NOTE: the Python reference multiplies by Vh directly (Vh @ w), not Vh'.
        # Mirrored here for exact agreement with bssunfold 0.28.0.
        y = F.Vt * (s .* c ./ (s .^ 2 .+ lam))
        x = view(V, :, 1:k) * y
        best_x = copy(x)
        iterations = k

        if noise_level !== nothing
            residual = norm(A * x .- b)
            if residual <= T(noise_level) * sqrt(T(m))
                converged = true
                break
            end
        end

        if converged
            break
        end
    end

    return UnfoldResult(best_x, iterations, converged, norm(b .- A * best_x))
end
