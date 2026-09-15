"""
    solve_express_full(A, b, E, x0; n_groups=6, interval_boundaries=nothing,
                   max_iterations=3, tol_iteration=0.05, relative_uncertainty=0.05)

Метод Express (коarse-to-fine) для показаний детекторов Боннера.

Спектр моделируется кусочно-экспоненциальной функцией: логарифм значения
линейно интерполируется между границами интервалов `boundaries`
(по умолчанию равномерно разбитый на `n_groups` интервалов диапазон `E`).
Параметры модели (логарифмы значений на границах) подбираются
нелинейным МНК (Гаусс–Ньютон с численным Якобианом — порт
`scipy.optimize.least_squares`), с весами `sigma = relative_uncertainty * b`.

Требует неотрицательные измерения и строго возрастающую сетку `E`.

# Возвращает
`UnfoldResult` с восстановленным спектром на сетке `E`.
"""
function solve_express_full(A::AbstractMatrix{T}, b::AbstractVector{T}, E::AbstractVector{T},
                        x0::Union{Nothing,AbstractVector{T}}=nothing;
                       n_groups::Integer=6,
                       interval_boundaries::Union{Nothing,AbstractVector{T}}=nothing,
                       max_iterations::Integer=3,
                       tol_iteration::T=T(0.05),
                       relative_uncertainty::T=T(0.05)) where T<:AbstractFloat
    m, n = size(A)

    if any(b .< 0) || any(.!isfinite.(b)) || any(diff(E) .<= 0)
        throw(ArgumentError("Express requires non-negative readings and increasing E"))
    end

    boundaries = if interval_boundaries === nothing
        if n_groups < 2
            throw(ArgumentError("n_groups must be at least 2"))
        end
        collect(range(E[1], E[end], length=n_groups + 1))
    else
        bnd = collect(float.(interval_boundaries))
        if length(bnd) < 2 || any(diff(bnd) .<= 0)
            throw(ArgumentError("interval_boundaries must be strictly increasing"))
        end
        if bnd[1] < E[1] || bnd[end] > E[end]
            throw(ArgumentError("interval_boundaries must lie within E"))
        end
        bnd
    end

    n_groups_eff = length(boundaries)

    initial = x0 === nothing ? ones(T, n) : max.(float.(x0), T(1e-30))
    if length(initial) != n
        throw(ArgumentError("x0 must match the energy grid"))
    end

    guess = _interp_log(centers_like(boundaries), E, log.(initial))
    sigma = max.(relative_uncertainty .* max.(b, T(1e-30)), T(1e-30))

    function model(log_values::AbstractVector{T})
        spectrum = _piecewise_exponential(E, boundaries, exp.(log_values))
        return A * spectrum
    end

    function residuals(p::AbstractVector{T})
        return (model(p) .- b) ./ sigma
    end

    p = copy(guess)
    max_nfev = max(1, max_iterations) * 100
    nfev = 0
    success = false

    lambda = T(1e-2)
    cost = dot(residuals(p), residuals(p))
    nfev += 1

    for _ in 1:max_nfev
        nfev += 1
        r0 = residuals(p)
        J = _numerical_jacobian(residuals, p)
        JtJ = J' * J
        Jtr = J' * r0

        stepped = false
        for _ in 1:20
            system = JtJ + lambda .* Matrix{T}(I, length(p), length(p))
            dp = try
                system \ (-Jtr)
            catch
                break
            end
            p_new = p .+ dp
            r_new = residuals(p_new)
            cost_new = dot(r_new, r_new)
            if isfinite(cost_new) && cost_new < cost
                p = p_new
                cost = cost_new
                lambda = max(lambda / 3, T(1e-12))
                stepped = true
                break
            else
                lambda *= 5
                lambda > T(1e10) && break
            end
        end
        if !stepped
            break
        end
        if norm(residuals(p)) < T(1e-10) * max(1.0, norm(b ./ sigma))
            success = true
            break
        end
    end

    spectrum = _piecewise_exponential(E, boundaries, exp.(p))
    relative_change = norm(model(p) .- b) / (norm(b) + T(1e-30))
    converged = success || relative_change <= tol_iteration

    residual = b .- A * spectrum
    return UnfoldResult(max.(spectrum, T(0)), nfev, converged, norm(residual),
                        Dict{String,Any}("n_groups" => n_groups_eff - 1,
                                         "relative_change" => relative_change))
end

centers_like(v::AbstractVector) = collect(float.(v))

"""
    solve_express(A, b, x0; kwargs...)

Универсальная сигнатура (A, b, x0): использует псевдо-равномерную лог-сетку
10^(-9..2) МэВ. Показания с округлением вниз до нуля (физика отсчётов ≥ 0).
"""
function solve_express(A::AbstractMatrix{T}, b::AbstractVector{T},
                       x0::AbstractVector{T};
                       kwargs...) where T<:AbstractFloat
    n = size(A, 2)
    E = collect(10.0 .^ range(-9.0, 2.0, length=n))
    b_eff = max.(Float64.(b), 0.0)
    return solve_express_full(A, b_eff, E, Float64.(x0);
                         kwargs...)
end

"""
    _piecewise_exponential(E, boundaries, values)

Оценка спектра, логарифм которого линеен между границами `boundaries`
(интерполяция в log-пространстве).
"""
function _piecewise_exponential(E::AbstractVector{T}, boundaries::AbstractVector{T},
                                values::AbstractVector{T}) where T<:AbstractFloat
    log_values = log.(max.(values, T(1e-300)))
    return exp.(_interp_linear(E, boundaries, log_values))
end

"""
    _interp_linear(xq, x, y)

Линейная интерполяция с клэмпированием на краях (порт `np.interp`).
"""
function _interp_linear(xq::AbstractVector{T}, x::AbstractVector{T},
                        y::AbstractVector{T}) where T<:AbstractFloat
    n = length(x)
    out = Vector{T}(undef, length(xq))
    for (k, xv) in enumerate(xq)
        if xv <= x[1]
            out[k] = y[1]
        elseif xv >= x[n]
            out[k] = y[n]
        else
            i = searchsortedlast(x, xv)
            i = clamp(i, 1, n - 1)
            t = (xv - x[i]) / (x[i+1] - x[i])
            out[k] = y[i] + t * (y[i+1] - y[i])
        end
    end
    return out
end

"""
    _interp_log(xq, x, log_y)

Интерполяция значений `log_y` (уже в log-пространстве) на точках `xq`
с линейной экстраполяцией за пределами диапазона.
"""
function _interp_log(xq::AbstractVector{T}, x::AbstractVector{T},
                     log_y::AbstractVector{T}) where T<:AbstractFloat
    n = length(x)
    out = Vector{T}(undef, length(xq))
    slope_lo = n > 1 ? (log_y[2] - log_y[1]) / max(x[2] - x[1], eps(T)) : zero(T)
    slope_hi = n > 1 ? (log_y[n] - log_y[n-1]) / max(x[n] - x[n-1], eps(T)) : zero(T)
    for (k, xv) in enumerate(xq)
        if xv <= x[1]
            out[k] = log_y[1] + slope_lo * (xv - x[1])
        elseif xv >= x[n]
            out[k] = log_y[n] + slope_hi * (xv - x[n])
        else
            i = searchsortedlast(x, xv)
            i = clamp(i, 1, n - 1)
            t = (xv - x[i]) / (x[i+1] - x[i])
            out[k] = log_y[i] + t * (log_y[i+1] - log_y[i])
        end
    end
    return out
end

"""
    _numerical_jacobian(f, p)

Численный якобиан функции `f` в точке `p` (центральные разности,
шаг по корню машинной точности).
"""
function _numerical_jacobian(f::Function, p::AbstractVector{T}) where T<:AbstractFloat
    f0 = f(p)
    J = Matrix{T}(undef, length(f0), length(p))
    for j in eachindex(p)
        h = sqrt(eps(T)) * max(abs(p[j]), one(T))
        p_plus = copy(p); p_plus[j] += h
        p_minus = copy(p); p_minus[j] -= h
        J[:, j] = (f(p_plus) .- f(p_minus)) ./ (2h)
    end
    return J
end
