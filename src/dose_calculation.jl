"""
Расчёт дозовых мощностей (порт из dose_calculation.py).

Функции для расчёта доз от нейтронных спектров с использованием наборов
конверсионных коэффициентов (ICRP-116, ICRP-74, NRB99-2009).
"""

const DOSE_COEFFICIENTS_REGISTRY = Dict{String,Dict{String,Vector{Float64}}}()

"""
Доступные наборы конверсионных коэффициентов.
"""
const DOSE_COEFFICIENTS_NAMES = [
    "ICRP116",
    "ICRP74_effective",
    "NRB99_2009_effective",
    "ICRP74_operational",
]

function _build_dose_registry!()
    isempty(DOSE_COEFFICIENTS_REGISTRY) || return DOSE_COEFFICIENTS_REGISTRY
    registry = DOSE_COEFFICIENTS_REGISTRY
    registry["ICRP116"] = ICRP116_COEFF_EFFECTIVE_DOSE
    registry["ICRP74_effective"] = ICRP74_COEFF_EFFECTIVE_DOSE
    registry["NRB99_2009_effective"] = NRB99_2009_COEFF_EFFECTIVE_DOSE
    registry["ICRP74_operational"] = ICRP74_COEFF_OPERATIONAL_QUANTITIES
    return registry
end

"""
    get_icrp116_coefficients()

Получить конверсионные коэффициенты ICRP-116 (эффективная доза).
"""
function get_icrp116_coefficients()
    return ICRP116_COEFF_EFFECTIVE_DOSE
end

"""
    get_coefficients(name)

Получить набор конверсионных коэффициентов по имени.

# Аргументы
- `name::AbstractString`:
  - `"ICRP116"` — эффективная доза ICRP-116 (AP, PA, LLAT, RLAT, ROT, ISO)
  - `"ICRP74_effective"` — эффективная доза ICRP-74
  - `"NRB99_2009_effective"` — NRB99-2009
  - `"ICRP74_operational"` — операционные величины ICRP-74
    (ADE, PDE0, PDE45, PDE60, PDE75)

# Возвращает
`Dict{String,Vector{Float64}}` с ключом `"E_MeV"` и ключами геометрий/величин.
"""
function get_coefficients(name::AbstractString)
    registry = _build_dose_registry!()
    haskey(registry, name) || throw(ArgumentError(
        "Unknown dose coefficient name: '$name'. Available: $DOSE_COEFFICIENTS_NAMES"))
    return registry[name]
end

"""
    interpolate_coefficients(cc, E_target; fill_value=0.0)

Интерполировать конверсионные коэффициенты `cc` (словарь с `"E_MeV"` и
ключами геометрий) на целевую сетку `E_target`.

Используется линейная интерполяция; для энергий вне диапазона исходной
сетки значение равно `fill_value` (по умолчанию 0.0), как в Python-порте.
"""
function interpolate_coefficients(cc::Dict{String,<:Vector{<:Real}},
                                  E_target::Vector{Float64};
                                  fill_value::Real=0.0)
    haskey(cc, "E_MeV") || throw(ArgumentError("cc must contain E_MeV"))
    E_source = Float64.(cc["E_MeV"])
    issorted(E_source) || throw(ArgumentError("cc E_MeV must be ascending"))
    fill = Float64(fill_value)
    result = Dict{String,Vector{Float64}}(
        "E_MeV" => Float64.(copy(E_target)))
    for (key, values) in cc
        key == "E_MeV" && continue
        v = Float64.(values)
        length(v) == length(E_source) ||
            throw(ArgumentError("cc[$key] length must match cc[E_MeV]"))
        interp = similar(E_target, Float64)
        for (i, e) in enumerate(E_target)
            interp[i] = _linear_interp(E_source, v, e)
        end
        interp[E_target .< E_source[1]] .= fill
        interp[E_target .> E_source[end]] .= fill
        result[key] = interp
    end
    return result
end

# Линейная интерполяция со схожением на краях (аналог np.interp)
function _linear_interp(x_src::Vector{Float64}, y_src::Vector{Float64}, x::Real)
    n = length(x_src)
    n == length(y_src) || throw(ArgumentError("length mismatch"))
    x < x_src[1] && return y_src[1]
    x > x_src[end] && return y_src[end]
    j = searchsortedlast(x_src, x)
    j == n && return y_src[n]
    if x_src[j] == x_src[j+1]
        return y_src[j+1]
    end
    t = (x - x_src[j]) / (x_src[j+1] - x_src[j])
    return y_src[j] + t * (y_src[j+1] - y_src[j])
end

"""
    calculate_dose_rates(spectrum; cc=default, dlnE=0.2)

Рассчитать дозовые мощности из развёрнутого спектра, используя
конверсионные коэффициенты.

# Аргументы
- `spectrum::AbstractVector{<:Real}`: нейтронный спектр (флюенс в bins)
- `cc::Dict`: словарь коэффициентов (по умолчанию ICRP-116); должен
  содержать `"E_MeV"` и один или несколько ключей геометрий (AP, PA, ISO, ...)
- `dlnE::Real`: шаг в лог-энергии для интегрирования (default: 0.2)

# Возвращает
`Dict{String,Float64}` — дозовые мощности для каждой геометрии в pSv/s.
"""
function calculate_dose_rates(spectrum::AbstractVector{<:Real};
                              cc::Union{Nothing,Dict{String,<:Vector{<:Real}}}=nothing,
                              dlnE::Real=0.2)
    if cc === nothing
        cc = get_icrp116_coefficients()
    end
    isempty(cc) && return Dict{String,Float64}()

    ln10_dlnE = log(10.0) * Float64(dlnE)
    spec = Float64.(spectrum)
    n_spec = length(spec)

    CC = Vector{String}()
    for g in keys(cc)
        g != "E_MeV" && push!(CC, g)
    end
    isempty(CC) && return Dict{String,Float64}()

    cc_matrix = Matrix{Float64}(undef, length(CC), n_spec)
    for (idx, geom) in enumerate(CC)
        k = Float64.(cc[geom])
        min_len = min(length(k), n_spec)
        cc_matrix[idx, 1:min_len] .= k[1:min_len]
        if min_len < n_spec
            cc_matrix[idx, min_len+1:end] .= 0.0
        end
    end

    doses = cc_matrix * spec
    doses .*= ln10_dlnE

    return Dict{String,Float64}(CC[i] => doses[i] for i in eachindex(CC))
end
