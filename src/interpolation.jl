"""
Функции интерполяции спектров (порт из utils/interpolation.py).

Используется PCHIP-интерполяция (монотонный кубический сплайн Эрмита),
которая сохраняет монотонность и не создаёт осцилляций на логарифмических
сетках. Реализация совместима с `scipy.interpolate.PchipInterpolator`.
"""

"""
    _pchip_derivatives(u, y) -> d

Вычислить производные PCHIP в узлах (алгоритм Fritsch–Carlson; краевые
производные — односторонняя трёхточечная формула, как в SciPy).
"""
function _pchip_derivatives(u::Vector{Float64}, y::Vector{Float64})
    n = length(u)
    n >= 3 || throw(ArgumentError("PCHIP requires at least 3 nodes"))
    n == length(y) || throw(ArgumentError("x and y must have equal length"))

    hm = diff(u)
    all(hm .> 0) || throw(ArgumentError("x must be strictly increasing"))

    δ = (y[2:n] .- y[1:n-1]) ./ hm

    d = Vector{Float64}(undef, n)
    for i in 2:n-1
        if δ[i-1] * δ[i] <= 0
            d[i] = 0.0
        else
            w1 = 2 * hm[i] + hm[i-1]
            w2 = hm[i] + 2 * hm[i-1]
            d[i] = (w1 + w2) / (w1 / δ[i-1] + w2 / δ[i])
        end
    end

    function edge(h0, h1, m0, m1)
        dd = ((2 * h0 + h1) * m0 - h0 * m1) / (h0 + h1)
        if sign(dd) != sign(m0)
            dd = 0.0
        elseif sign(m0) != sign(m1) && abs(dd) > 3 * abs(m0)
            dd = 3 * m0
        end
        return dd
    end

    d[1] = edge(hm[1], hm[2], δ[1], δ[2])
    d[n] = edge(hm[n-1], hm[n-2], δ[n-1], δ[n-2])

    return d
end

"""
    _pchip_eval(u, y, d, u_dst)

Вычислить кубический сплайн Эрмита с узлами `u`, значениями `y` и
производными `d` в точках `u_dst`. Точки вне `u` экстраполируются
ближайшим интервалом (очистка выполняет вызывающий код).
"""
function _pchip_eval(u::Vector{Float64}, y::Vector{Float64},
                     d::Vector{Float64}, u_dst::AbstractVector{<:Real})
    n = length(u)
    out = Vector{Float64}(undef, length(u_dst))
    j = 1
    while j <= length(u_dst)
        i = searchsortedlast(u, u_dst[j])
        i = clamp(i, 1, n - 1)
        h = u[i+1] - u[i]
        s = (u_dst[j] - u[i]) / h
        h00 = 2 * s^3 - 3 * s^2 + 1
        h10 = s^3 - 2 * s^2 + s
        h01 = -2 * s^3 + 3 * s^2
        h11 = s^3 - s^2
        out[j] = h00 * y[i] + h10 * h * d[i] + h01 * y[i+1] + h11 * h * d[i+1]
        j += 1
    end
    return out
end

"""
    _handle_extrapolation(interp_vals, u_src, u_dst; fill_value=0.0,
                          replace_negative=true)

Заполнить вне-диапазонные точки `u_dst` значением `fill_value` и
заменить отрицательные значения нулями.
"""
function _handle_extrapolation(interp_vals::Vector{Float64}, u_src::Vector{Float64},
                               u_dst::AbstractVector{<:Real};
                               fill_value::Float64=0.0, replace_negative::Bool=true)
    lo, hi = extrema(u_src)
    for j in eachindex(u_dst)
        if u_dst[j] < lo || u_dst[j] > hi
            interp_vals[j] = fill_value
        end
    end
    if replace_negative
        interp_vals = map(v -> isfinite(v) ? max(v, 0.0) : 0.0, interp_vals)
    else
        interp_vals = map(v -> isnan(v) ? fill_value : v, interp_vals)
    end
    return interp_vals
end

"""
    interpolate_spectrum(spectrum, E_from, E_to; fill_value=0.0,
                         replace_negative=true)

Интерполяция спектра с сетки `E_from` на сетку `E_to` (PCHIP в лог-масштабе).

# Возвращает
`Vector{Float64}` со значениями на `E_to`.
"""
function interpolate_spectrum(spectrum::AbstractVector{<:Real},
                              E_from::AbstractVector{<:Real},
                              E_to::AbstractVector{<:Real};
                              fill_value::Real=0.0,
                              replace_negative::Bool=true)
    length(spectrum) == length(E_from) ||
        throw(ArgumentError("spectrum and E_from must have equal length"))
    all(E_from .> 0) || throw(ArgumentError("E_from values must be positive"))
    all(E_to .> 0) || throw(ArgumentError("E_to values must be positive"))
    issorted(E_from) || throw(ArgumentError("E_from must be ascending"))

    Emin = minimum(E_from)
    u_src = log10.(Float64.(E_from) ./ Emin)
    u_to = log10.(Float64.(E_to) ./ Emin)

    y_src = Float64.(spectrum)
    d = _pchip_derivatives(u_src, y_src)
    interp_vals = _pchip_eval(u_src, y_src, d, u_to)

    return _handle_extrapolation(interp_vals, u_src, u_to;
                                 fill_value=Float64(fill_value),
                                 replace_negative=replace_negative)
end

"""
    discretize_spectra(spectra, target_E_MeV; energy_key="E_MeV")

Привести словарь спектров (с ключом `energy_key`) к целевой сетке.

# Аргументы
- `spectra::Dict{String,Vector{Float64}}`: словарь со `"E_MeV"` и ключами спектров
- `target_E_MeV::Vector{Float64}`: целевая сетка энергий, МэВ

# Возвращает
`Dict{String,Vector{Float64}}` с ключом `"E_MeV"` = `target_E_MeV` и
интерполированными спектрами.
"""
function discretize_spectra(spectra::Dict{String,<:Vector{<:Real}},
                            target_E_MeV::Vector{Float64};
                            energy_key::AbstractString="E_MeV")
    haskey(spectra, energy_key) || throw(ArgumentError("spectra must contain $energy_key"))
    result = Dict{String,Vector{Float64}}(energy_key => copy(target_E_MeV))
    for (key, vals) in spectra
        key == energy_key && continue
        result[key] = interpolate_spectrum(vals, spectra[energy_key], target_E_MeV)
    end
    return result
end

"""
    resample_to_log_grid(spectrum, E_MeV; n_points=nothing, Emin=nothing, Emax=nothing)

Привести спектр к равномерной лог-сетке.

# Возвращает
Кортеж `(new_E_MeV, new_spectrum)`.
"""
function resample_to_log_grid(spectrum::AbstractVector{<:Real},
                              E_MeV::AbstractVector{<:Real};
                              n_points::Union{Nothing,Integer}=nothing,
                              Emin::Union{Nothing,Real}=nothing,
                              Emax::Union{Nothing,Real}=nothing)
    N = something(n_points, length(E_MeV))
    Emin2 = Emin === nothing ? minimum(E_MeV) : Float64(Emin)
    Emax2 = Emax === nothing ? maximum(E_MeV) : Float64(Emax)
    logE = range(log10(Emin2), log10(Emax2), length=Int(N))
    E_new = collect(10.0 .^ logE)
    new_spec = interpolate_spectrum(spectrum, E_MeV, E_new)
    return E_new, new_spec
end
