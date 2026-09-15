"""
NSDUAZ unfolding method (порт из unfold_nsduaz.py).

NSDUAZ ("Neutron Spectrometry and Dosimetry from the Universidad Autonoma
de Zacatecas"; Ortiz-Rodriguez & Vega-Carrillo, 2012) — развёртка по
Боннер-сферам на базе итерационного алгоритма SPUNIT (Doroshenko et al.,
1977; та же итерация, что и в BUNKI).  Отличительная черта — автоматический
выбор начального спектра из *каталога* стандартных нейтронных спектров:
экспериментальные скорости счёта нормируются на показание сферы
20.32 см и сравниваются (статистический тест) с предсказаниями каждого
спектра каталога.  Запись каталога, лучше всего воспроизводящая измеренный
относительный рисунок показаний, используется как начальный спектр для
итерации SPUNIT, которая выполняется до относительного изменения решения
ниже ~1%.

Реализовано:
- `solve_nsduaz` — итерация SPUNIT (обёртка над `solve_bunki`) с
  NSDUAZ-умолчанием порога сходимости;
- `select_catalogue_initial` — выбор начального спектра из каталога
  (статистический тест по отношениям показаний к опорной сфере);
- `builtin_catalogue` — встроенный мини-каталог аналитических стандартных
  спектров (241Am/9Be, 252Cf, thermal + 1/E + fission reactor-like);
- `unfold_nsduaz` — обёртка уровня Detector (см. detector.jl).
"""

# ─── Аналитические стандартные спектры ──────────────────────────────────────

"""
    _watt_spectrum(E_MeV; a=1.025, b=2.926)

Аналитический ватт-спектр деления (например 252Cf):
`exp(-E/a) * sinh(sqrt(b*E))`, нормированный на единицу суммы.
"""
function _watt_spectrum(E_MeV::AbstractVector{<:Real}; a::Real=1.025, b::Real=2.926)
    E = max.(Float64.(E_MeV), 1e-9)
    w = @. exp(-E / a) * sinh(sqrt(b * E))
    total = sum(w)
    return total > 0 ? w ./ total : fill(1.0 / length(w), length(w))
end

"""
    _ambe_spectrum(E_MeV)

Аналитическая форма 241Am/9Be(alpha,n): эвапораторный континуум +
пик на 4.2 МэВ, нормированная на единицу суммы.
"""
function _ambe_spectrum(E_MeV::AbstractVector{<:Real})
    E = max.(Float64.(E_MeV), 1e-9)
    continuum = @. exp(-E / 2.0)
    peak = @. exp(-0.5 * ((E - 4.2) / 1.2)^2)
    spec = continuum .+ 3.5 .* peak
    total = sum(spec)
    return total > 0 ? spec ./ total : fill(1.0 / length(spec), length(spec))
end

"""
    _reactor_spectrum(E_MeV)

Аналитический реактороподобный спектр: тепловая максвеллиана + 1/E +
быстрый ватт-спектр деления, нормированная на единицу суммы.
"""
function _reactor_spectrum(E_MeV::AbstractVector{<:Real})
    E = max.(Float64.(E_MeV), 1e-9)
    kT = 0.0253e-6  # 0.0253 эВ в МэВ
    thermal = @. (E / kT) * exp(-E / kT)
    epithermal = map(e -> e > 1e-6 ? 1.0 / max(e, 1e-9) : 0.0, E)
    fast = _watt_spectrum(E)
    spec = 1e-3 .* thermal .+ 0.1 .* epithermal .+ fast
    total = sum(spec)
    return total > 0 ? spec ./ total : fill(1.0 / length(spec), length(spec))
end

"""
    builtin_catalogue(E_MeV) -> Dict{String,Vector{Float64}}

Построить встроенный мини-каталог аналитических стандартных спектров
на энергетической сетке `E_MeV`: ключи `"ambe"`, `"cf252"`, `"reactor"`.
"""
function builtin_catalogue(E_MeV::AbstractVector{<:Real})
    E = collect(Float64, E_MeV)
    return Dict{String,Vector{Float64}}(
        "ambe"    => _ambe_spectrum(E),
        "cf252"   => _watt_spectrum(E),
        "reactor" => _reactor_spectrum(E),
    )
end

# ─── Поиск опорной сферы ────────────────────────────────────────────────────

function _find_reference_index(detector_names::Vector{String}, A::AbstractMatrix{<:Real})
    for (i, name) in enumerate(detector_names)
        lowered = lowercase(name)
        if occursin("20.32", lowered) || occursin("20in", lowered) ||
           occursin("8in", lowered) || occursin("8 in", lowered)
            return i
        end
    end
    # Fallback: детектор с наибольшей интегральной чувствительностью.
    return argmax(vec(sum(abs.(A), dims=2)))
end

# ─── Выбор начального спектра из каталога ───────────────────────────────────

"""
    select_catalogue_initial(readings, detector_names, sensitivities;
                             catalogue=nothing, reference_name=nothing,
                             E_MeV=nothing) -> (spectrum, label)

Выбрать начальный спектр из каталога с помощью статистического теста.

Экспериментальные показания нормируются на показание опорной сферы
(20.32 см по умолчанию) и сравниваются с относительным рисунком показаний,
предсказываемым каждым спектром каталога, свёрнутым с ответной матрицей.
Выбирается запись, минимизирующая взвешенный хи-квадрат относительных
отношений, и пересчитывается так, чтобы её предсказанное показание опорной
сферы совпадало с измеренным.

# Аргументы
- `readings::Dict{String,<:Real}`: показания детекторов
- `detector_names::Vector{String}`: имена доступных детекторов
- `sensitivities::Dict{String,Vector{<:Real}}`: чувствительности
- `catalogue`: Dict(label => спектр на сетке детектора); `nothing` —
  использовать `builtin_catalogue`
- `reference_name`: имя опорного детектора; `nothing` — авто-поиск сферы 20.32 см
- `E_MeV`: сетка для построения встроенного каталога; `nothing` —
  репрезентативная лог-сетка по длине чувствительности

# Возвращает
`(initial_spectrum, catalogue_label)`.
"""
function select_catalogue_initial(readings::Dict{String,<:Real},
                                 detector_names::Vector{String},
                                 sensitivities::Dict{String,<:Vector{<:Real}};
                                 catalogue::Union{Nothing,Dict{String,<:Vector{<:Real}}}=nothing,
                                 reference_name::Union{Nothing,String}=nothing,
                                 E_MeV::Union{Nothing,Vector{Float64}}=nothing)
    selected = [name for name in detector_names if haskey(readings, name)]
    isempty(selected) && throw(ArgumentError("No detector readings available for catalogue selection"))
    b = Float64[readings[name] for name in selected]
    A = Matrix(hcat([Float64.(sensitivities[name]) for name in selected]...)')

    if reference_name !== nothing
        reference_name in readings || throw(ArgumentError(
            "reference_name '$(reference_name)' is not present in readings"))
        ref_idx = findfirst(==(reference_name), selected)
        ref_idx === nothing && throw(ArgumentError(
            "reference_name '$(reference_name)' is not among available detectors"))
    else
        ref_idx = _find_reference_index(selected, A)
    end

    if catalogue === nothing
        n_bins = size(A, 2)
        grid = if E_MeV !== nothing && length(E_MeV) == n_bins
            E_MeV
        else
            collect(range(1e-9, 1e2, length=n_bins))  # лог-равномерный репрезентативный аналог
        end
        catalogue = builtin_catalogue(grid)
    end

    b_ref = b[ref_idx]
    b_ref > 0 || throw(ArgumentError("Reference sphere reading must be strictly positive"))
    r_ratio = b ./ b_ref

    best_label = nothing
    best_chi = Inf
    best_scale = 1.0
    best_spec = nothing

    for (label, spec) in catalogue
        length(spec) == size(A, 2) || throw(ArgumentError(
            "Catalogue spectrum '$(label)' has length $(length(spec)), expected $(size(A, 2))"))
        any(>(0), spec) || continue
        c = A * max.(spec, 0.0)
        c_ref = c[ref_idx]
        c_ref <= 0 && continue
        s_ratio = c ./ c_ref
        denom = max.(s_ratio, 1e-12)
        chi = sum(((r_ratio .- s_ratio) ./ denom) .^ 2)
        if chi < best_chi
            best_chi = chi
            best_label = label
            best_scale = b_ref / c_ref
            best_spec = spec
        end
    end

    best_spec === nothing && throw(ArgumentError("Catalogue is empty or has no usable spectrum"))
    return max.(best_scale .* best_spec, 0.0), best_label
end

# ─── Основной солвер ────────────────────────────────────────────────────────

"""
    solve_nsduaz(A, b, x0; smoothing=0.1, max_iterations=1000, tolerance=0.01, alpha=0.8)

Решить задачу развёртки итерацией NSDUAZ (SPUNIT).

Это итерация SPUNIT с NSDUAZ-умолчанием порога сходимости (~1% относительного
изменения).  Начальный спектр `x0` обычно получается через
[`select_catalogue_initial`](@ref) (или задан пользователем).
Тонкая обёртка над [`solve_bunki`](@ref); параметр `alpha` — коэффициент
релаксации SPUNIT (в Python-оригинале соответствовал `smoothing`).

# Аргументы
- `A::AbstractMatrix{T}`: ответная матрица (m × n)
- `b::AbstractVector{T}`: измерения (m,)
- `x0::AbstractVector{T}`: начальный спектр (n,)
- `max_iterations`: макс. число итераций (default 1000)
- `tolerance`: порог относительного изменения для ранней остановки (default 0.01)
- `alpha`: коэффициент релаксации SPUNIT (default 0.8)

# Возвращает
- `UnfoldResult{T}` со спектром
"""
function solve_nsduaz(A::AbstractMatrix{T}, b::AbstractVector{T}, x0::AbstractVector{T};
                     max_iterations::Integer=1000,
                     tolerance::T=T(0.01),
                     alpha::T=T(0.8)) where T<:AbstractFloat
    return solve_bunki(A, b, x0;
                       max_iterations=max_iterations,
                       tolerance=tolerance,
                       alpha=alpha)
end
