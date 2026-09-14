"""
Staysl — байесовский алгоритм с априорным спектром (Staysl, 1982).

Использует x0 как априорный спектр. Обновление:

    x_{k+1}[j] = x0[j] * (1 + Σ_i (A[i,j] * b_i / (A x_k)_i - A[i,j]) / n_total)

Аналог MAP-EM с априорным распределением Дирихле.
"""
function solve_staysl(A::AbstractMatrix{T}, b::AbstractVector{T}, x0::AbstractVector{T};
                     max_iterations::Integer=1000,
                     tolerance::T=T(1e-6),
                     eps::T=T(1e-10)) where T<:AbstractFloat
    m, n = size(A)
    x = max.(copy(x0), eps)
    x_prior = copy(x0)  # Staysl uses initial as prior
    AT = Matrix(A')

    converged = false
    iters = 0
    n_total = T(sum(A))

    @inbounds for k in 1:max_iterations
        iters = k
        Ax = A * x
        Ax = max.(Ax, eps)
        ratio = b ./ Ax
        correction = AT * ratio
        # Staysl update: x_new[j] = x0[j] * (1 + Σ_i A[i,j] (b_i/(Ax)_i - 1) / n_total)
        update_factor = T(1) .+ (correction .- vec(sum(A, dims=1))) ./ (n_total + eps)
        x_new = x_prior .* update_factor
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
