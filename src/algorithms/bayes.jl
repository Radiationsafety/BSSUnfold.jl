"""
    solve_bayes(A, b, x0; max_iterations=4000, tolerance=1e-3, eps=1e-300)

Байесовский итеративный алгоритм развёртки Д'Агостино.

Ответная матрица нормируется по столбцам (условные вероятности
P(D_j | E_i)), итерации выполняются в пространстве «эффективных отсчётов»
y = total_counts * prior, затем результат делится на суммы столбцов
для возврата в физические единицы:

    y_i <- y_i * Σ_j b_j * P_ji / (P y)_j

Бины с нулевой чувствительностью остаются на уровне априорного спектра.

# Возвращает
`UnfoldResult` со спектром в физических единицах.
"""
function solve_bayes(A::AbstractMatrix{T}, b::AbstractVector{T}, x0::AbstractVector{T};
                     max_iterations::Integer=4000,
                     tolerance::T=T(1e-3),
                     eps::T=T(1e-300)) where T<:AbstractFloat
    m, n = size(A)

    column_sums = vec(sum(A, dims=1))
    column_sums_safe = [s > 0 ? s : one(T) for s in column_sums]
    P = A ./ reshape(column_sums_safe, 1, n)

    zero_sens = column_sums .<= 0

    prior = if sum(x0) > 0
        x0 ./ sum(x0)
    else
        fill(one(T) / n, n)
    end

    total_counts = sum(b)
    y = total_counts .* prior
    y_new = similar(y)

    converged = false
    iters = 0

    for k in 1:max_iterations
        iters = k
        f_norm = P * y
        f_norm_safe = [f > 0 ? f : eps for f in f_norm]

        weight = (b ./ f_norm_safe) .* P
        correction = vec(sum(weight, dims=1))
        @. y_new = y * correction

        if any(zero_sens)
            for i in 1:n
                if zero_sens[i]
                    y_new[i] = prior[i] * total_counts
                end
            end
        end

        denom = max(one(T), norm(y))
        if norm(y_new .- y) / denom < tolerance
            y = y_new
            converged = true
            break
        end
        y, y_new = y_new, y
    end

    x = y ./ column_sums_safe
    residual = b .- A * x
    return UnfoldResult(x, iters, converged, norm(residual))
end
