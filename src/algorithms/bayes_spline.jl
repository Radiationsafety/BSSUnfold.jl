"""
    solve_bayes_spline(A, b, x0; max_iterations=4000, tolerance=1e-3,
                       spline_degree=3, spline_smooth=1e-2, eps=1e-300)

Байесовская итеративная развёртка Д'Агостино со сплайновой регуляризацией.

Итерации выполняются в пространстве эффективных отсчётов (как в
[`solve_bayes`](@ref)), но после каждого шага физический спектр
сглаживается квадратичным B-сплайном в log10-пространстве. Вместо
`scipy.interpolate.UnivariateSpline` ( penalized least squares) здесь
используется регуляризованная квадратичная аппроксимация: минимизируется

    Σ_k (s(t_k) - log_x_k)² + λ ∫ (s''(t))² dt

с естественным B-сплайном степени 3 на равномерном узловом векторе,
что даёт эквивалентный сглаживающий эффект без внешних зависимостей.

# Возвращает
`UnfoldResult` со сглаженным спектром в физических единицах.
"""
function solve_bayes_spline(A::AbstractMatrix{T}, b::AbstractVector{T}, x0::AbstractVector{T};
                            max_iterations::Integer=4000,
                            tolerance::T=T(1e-3),
                            spline_degree::Integer=3,
                            spline_smooth::T=T(1e-2),
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

    x_smooth = y ./ column_sums_safe
    converged = false
    iters = 0

    for k in 1:max_iterations
        iters = k
        y_old = copy(y)

        f_norm = P * y
        f_norm_safe = [f > 0 ? f : eps for f in f_norm]

        weight = (b ./ f_norm_safe) .* P
        correction = vec(sum(weight, dims=1))
        y = y .* correction

        if any(zero_sens)
            for i in 1:n
                if zero_sens[i]
                    y[i] = prior[i] * total_counts
                end
            end
        end

        x = y ./ column_sums_safe

        if n > spline_degree + 1
            log_x = log10.(max.(x, eps))
            log_x_smooth = _smooth_quadratic_spline(log_x, spline_smooth)
            x_smooth = T(10) .^ log_x_smooth
        else
            x_smooth = x
        end

        x_smooth = max.(x_smooth, T(0))
        y = x_smooth .* column_sums_safe

        denom = max(one(T), norm(y_old))
        if norm(y .- y_old) / denom < tolerance
            converged = true
            break
        end
    end

    residual = b .- A * x_smooth
    return UnfoldResult(x_smooth, iters, converged, norm(residual),
                        Dict{String,Any}("spline_degree" => spline_degree,
                                         "spline_smooth" => spline_smooth))
end

"""
    _smooth_quadratic_spline(values, lambda)

Сглаживание вектора дискретным квадратичным B-сплайном (Whittaker–Henderson
порядка 2 — равномерный квадратичный сплайн в лог-пространстве).

Решается penalized least squares

    z* = argmin ||z - values||² + p · ||D₂ z||²,

где D₂ — конечноразностная аппроксимация второй производной
(дискретный аналог ∫ (s'')² dt для квадратичного сплайна);
`lambda = 0` означает чистое сглаживание с вариационным параметром
`p = lambda * 100` (по умолчанию `spline_smooth = 1e-2` даёт штраф 1.0,
эквивалент умеренного сглаживания в диапазоне `s` у `UnivariateSpline`).
"""
function _smooth_quadratic_spline(values::AbstractVector{T},
                                  lambda::T) where T<:AbstractFloat
    n = length(values)
    B = _quadratic_bspline_basis(n, n)
    BtB = B' * B
    if lambda > 0
        D = _second_difference_matrix(n - 2, n)
        system = BtB + lambda * (D' * D) * T(100)
    else
        system = BtB + T(1e-10) * I
    end
    coefs = try
        system \ (B' * values)
    catch
        return copy(values)
    end
    smoothed = B * coefs
    for i in eachindex(smoothed)
        if !isfinite(smoothed[i])
            smoothed[i] = values[i]
        end
    end
    return smoothed
end

"""
    _quadratic_bspline_basis(n_points, n_coefs)

Матрица базиса `n_points × n_coefs` равномерного квадратичного B-сплайна
со склонированными граничными узлами; базис нормируется в сумме на
единицу в каждой точке (party hat partition of unity).
"""
function _quadratic_bspline_basis(n_points::Int, n_coefs::Int)
    degree = 3
    knots = _uniform_knots(n_coefs, degree)
    B = zeros(n_points, n_coefs)
    for ip in 0:(n_points - 1)
        t = ip / max(n_points - 1, 1)
        ip == n_points - 1 && (t = prevfloat(1.0))
        for j in 1:n_coefs
            B[ip + 1, j] = _bspline_basis_value(t, j - 1, degree, knots)
        end
        s = sum(view(B, ip + 1, :))
        s > 0 && (view(B, ip + 1, :) ./= s)
    end
    return B
end

"""
    _uniform_knots(n_coefs, degree)

Равномерный узловой вектор с клонированными граничными узлами:
`degree + 1` повторов на каждом краю (стиль `UnivariateSpline`).
"""
function _uniform_knots(n_coefs::Int, degree::Int)
    n_interior = n_coefs - degree - 1
    knots = Float64[]
    append!(knots, zeros(degree + 1))
    for i in 1:n_interior
        push!(knots, i / (n_interior + 1))
    end
    append!(knots, ones(degree + 1))
    return knots
end

"""
    _bspline_basis_value(t, i, degree, knots)

Рекуррентное вычисление базисного B-сплайна (Cox–de Boor) на сетке узлов.
"""
function _bspline_basis_value(t::Float64, i::Int, degree::Int,
                              knots::AbstractVector{Float64})::Float64
    if degree == 0
        return (knots[i+1] <= t < knots[i+2]) ? 1.0 : 0.0
    end
    left = knots[i+1]
    right = knots[i + degree + 2]
    denom1 = knots[i + degree + 1] - knots[i+1]
    denom2 = knots[i + degree + 2] - knots[i+2]
    term1 = 0.0
    term2 = 0.0
    if denom1 > 0
        term1 = (t - left) / denom1 * _bspline_basis_value(t, i, degree - 1, knots)
    end
    if denom2 > 0
        term2 = (right - t) / denom2 * _bspline_basis_value(t, i + 1, degree - 1, knots)
    end
    return term1 + term2
end

"""
    _second_difference_matrix(rows, cols)

Плотная матрица второй конечной разности размера `rows × cols`
(штраф на кривизну сплайна).
"""
function _second_difference_matrix(rows::Int, cols::Int)
    D = zeros(rows, cols)
    for i in 1:rows
        D[i, i] = 1.0
        D[i, i + 1] = -2.0
        D[i, i + 2] = 1.0
    end
    return D
end
