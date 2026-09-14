"""
Lanczos-hybrid unfolding method (Golub-Kahan bidiagonalization + GCV).

Порт из `bssunfold/src/bssunfold/core/unfold_lanczos.py`.

Выполняет bidiagonalization матрицы `A`, генерируя последовательность
Krylov-подпространств. На каждой итерации регуляризация выбирается
автоматически через GCV на маленькой проекционной задаче.

Ссылки:
- Hansen, "Discrete Inverse Problems: Insight and Algorithms", 2010
- Chung, Nagy, O'Leary, "A Weighted GCV Method for Lanczos Hybrid Regularization"
"""

"""
Выбор λ на проекционной задаче через Generalized Cross-Validation.
"""
function _projected_gcv(B::AbstractMatrix{T}, bhat::AbstractVector{T}, m::Integer;
                       n_lambdas::Integer=200,
                       lambda_range::Tuple{T,T}=(T(1e-12), T(1e2))) where T<:AbstractFloat
    F = svd(B)
    Ub, s = F.U, F.S
    c = Ub' * bhat
    orth_res = norm(bhat)^2 - sum(c .^ 2)
    s2 = s .^ 2

    λs = 10 .^ range(log10(lambda_range[1]), log10(lambda_range[2]), length=n_lambdas)
    gcv_values = similar(λs)
    @inbounds for i in eachindex(λs)
        lam = λs[i]
        num = sum((c .* lam ./ (s2 .+ lam)) .^ 2) + orth_res
        den = (m - sum(s2 ./ (s2 .+ lam)))^2
        gcv_values[i] = num / den
    end

    idx = argmin(gcv_values)
    return λs[idx]
end


# Построить верхнюю бидиагональную матрицу (k+1 × k)
function _build_bidiagonal(alphas::Vector{T}, betas::Vector{T}, k::Integer) where T
    B = zeros(T, k+1, k)
    @inbounds for i in 1:k
        B[i, i] = alphas[i]
        if i < k
            B[i+1, i] = betas[i]
        end
    end
    return B
end


"""
    solve_lanczos(A, b, x0; max_iterations, regularization, noise_level)

Lanczos-hybrid метод: Golub-Kahan bidiagonalization + GCV-регуляризация.

Не требует начального спектра (x0 не используется, но принят для API-совместимости).

# Аргументы
- `A::AbstractMatrix{T}`: response matrix (m × n)
- `b::AbstractVector{T}`: измерения (m,)
- `x0::AbstractVector{T}`: не используется (API)
- `max_iterations`: макс. размерность Krylov (по умолчанию min(m, n))
- `regularization`: fallback λ (если GCV вернёт дегенерат)
- `noise_level`: относительный уровень шума для early stopping (опц.)

# Возвращает
- `UnfoldResult{T}` со спектром
"""
function solve_lanczos(A::AbstractMatrix{T}, b::AbstractVector{T}, x0::AbstractVector{T};
                      max_iterations::Union{Integer,Nothing}=nothing,
                      regularization::T=T(1e-8),
                      noise_level::Union{T,Nothing}=nothing,
                      eps::T=T(1e-14)) where T<:AbstractFloat
    m, n = size(A)
    max_k = max_iterations === nothing ? min(m, n) : max(1, min(max_iterations, min(m, n)))

    β = norm(b)
    if β == 0.0
        return UnfoldResult(zeros(T, n), 0, true, T(0))
    end

    # Golub-Kahan bidiagonalization: U (m × k+1), V (n × k), B (k+1 × k) bidiagonal
    U = Matrix{T}(undef, m, max_k + 1)
    V = Matrix{T}(undef, n, max_k)
    alphas = Vector{T}(undef, max_k)
    betas = Vector{T}(undef, max_k)

    U[:, 1] .= b ./ β
    best_x = zeros(T, n)
    iterations = 0
    converged = false

    @inbounds for k in 1:max_k
        u = view(U, :, k)
        if k == 1
            v = A' * u
        else
            v = A' * u .- betas[k-1] .* view(V, :, k-1)
        end
        alpha = norm(v)
        if alpha ≤ eps
            converged = true
            break
        end
        v ./= alpha
        V[:, k] .= v

        u2 = A * v .- alpha .* u
        new_beta = norm(u2)

        alphas[k] = alpha
        betas[k] = new_beta

        if new_beta ≤ eps
            converged = true
            # финальная проекция на текущем k
            B = _build_bidiagonal(alphas, betas, k)
            bhat = zeros(T, k+1); bhat[1] = β
            lam = _projected_gcv(B, bhat, m)
            if !isfinite(lam) || lam ≤ 0
                lam = regularization
            end
            F = svd(B)
            c = F.U' * bhat
            y = F.Vt' * (F.S .* c ./ (F.S .^ 2 .+ lam))
            best_x = V[:, 1:k] * y
            iterations = k
            break
        else
            U[:, k+1] .= u2 ./ new_beta
        end

        # Построить B (k+1 × k) и решить проекционную задачу
        B = _build_bidiagonal(alphas, betas, k)
        bhat = zeros(T, k+1); bhat[1] = β
        lam = _projected_gcv(B, bhat, m)
        if !isfinite(lam) || lam ≤ 0
            lam = regularization
        end
        F = svd(B)
        c = F.U' * bhat
        y = F.Vt' * (F.S .* c ./ (F.S .^ 2 .+ lam))
        best_x = V[:, 1:k] * y
        iterations = k

        # Early stopping по принципу невязки
        if noise_level !== nothing
            residual = norm(A * best_x .- b)
            if residual ≤ noise_level * sqrt(T(m))
                converged = true
                break
            end
        end

        if converged
            break
        end
    end

    best_x = max.(best_x, T(0))
    residual = b .- A * best_x
    return UnfoldResult(best_x, iterations, converged, norm(residual))
end
