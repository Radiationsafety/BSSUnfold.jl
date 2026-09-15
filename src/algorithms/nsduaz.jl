"""
NSDUAZ (Neutron Spectrometry and Dosimetry — Universidad Autonoma de
Zacatecas; Ortiz-Rodríguez & Vega-Carrillo, 2012).

Развёртка Боннера на базе итерации SPUNIT (Doroshenko / BUNKI).
Здесь `solve_nsduaz` — тонкая обёртка над `BSSUnfold.solve_bunki` с
NSDUAZ-умолчаниями (tolerance ~1% относительного изменения решения,
smoothing = 0.1).  Дополнительно перенесены: `NSDUAZ_BUILTIN_CATALOGUE`
(аналитические стандартные спектры 241Am/9Be, 252Cf/Watt,
термал+1/E+фиссион), `nsduaz_reference_index` (поиск эталонной сферы
20.32 см) и `nsduaz_select_catalogue_initial` — выбор начального
спектра из каталога статистическим тестом по отношениям счётов.
"""

function _nsduaz_watt_spectrum(E::Vector{Float64}, a::Float64=1.025, b::Float64=2.926)
    w = @. exp(-max(E, 1e-9) / a) * sinh(sqrt(b * max(E, 1e-9)))
    total = sum(w)
    return total > 0 ? w ./ total : fill(1.0 / length(w), length(w))
end

function _nsduaz_ambe_spectrum(E::Vector{Float64})
    continuum = @. exp(-max(E, 1e-9) / 2.0)
    peak = @. exp(-0.5 * ((max(E, 1e-9) - 4.2) / 1.2)^2)
    spec = continuum .+ 3.5 .* peak
    total = sum(spec)
    return total > 0 ? spec ./ total : fill(1.0 / length(spec), length(spec))
end

function _nsduaz_reactor_spectrum(E::Vector{Float64})
    kT = 0.0253e-6
    thermal = @. (max(E, 1e-9) / kT) * exp(-max(E, 1e-9) / kT)
    epithermal = [E_i > 1e-6 ? 1.0 / max(E_i, 1e-9) : 0.0 for E_i in E]
    fast = _nsduaz_watt_spectrum(E)
    spec = 1e-3 .* thermal .+ 0.1 .* epithermal .+ fast
    total = sum(spec)
    return total > 0 ? spec ./ total : fill(1.0 / length(spec), length(spec))
end

"""
    nsduaz_builtin_catalogue(E_MeV) -> Dict{String,Vector{Float64}}

Встроенный мини-каталог аналитических стандартных спектров:
`"ambe"` — 241Am/9Be, `"cf252"` — Watt (252Cf), `"reactor"` —
термал-Maxwell + 1/E + быстрый фиссионный вклад.
"""
function nsduaz_builtin_catalogue(E_MeV::AbstractVector{<:Real})
    E = Float64.(collect(E_MeV))
    return Dict{String,Vector{Float64}}(
        "ambe" => _nsduaz_ambe_spectrum(E),
        "cf252" => _nsduaz_watt_spectrum(E),
        "reactor" => _nsduaz_reactor_spectrum(E),
    )
end

"""
    nsduaz_reference_index(detector_names, A) -> Int

Индекс эталонной сферы (20.32 см / 8in-конвенции UTA/IAEA); fallback —
строка ответа с наибольшей суммой |A|.
"""
function nsduaz_reference_index(detector_names::Vector{String}, A::AbstractMatrix{T}) where T<:AbstractFloat
    for (i, name) in enumerate(detector_names)
        lowered = lowercase(name)
        if occursin("20.32", lowered) || occursin("20in", lowered) ||
           occursin("8in", lowered) || occursin("8 in", lowered)
            return i
        end
    end
    return argmax(sum(abs.(A), dims=2)[:])
end

"""
    nsduaz_select_catalogue_initial(readings, detector_names, sensitivities;
                                    catalogue=nothing, reference_name=nothing,
                                    E_MeV=nothing) -> (Vector{Float64}, String)

Выбор начального спектра из каталога: показания нормируются на
показание эталонной сферы и сравниваются с относительной картиной
счётов, предсказанной каждым каталог-спектром; выбор минимизирует
взвешенный хи-квадрат отношений и масштабируется под измеренное
показание эталона.  Возвращает `(initial_spectrum, catalogue_label)`.
"""
function nsduaz_select_catalogue_initial(readings::Dict{String,Float64},
                                         detector_names::Vector{String},
                                         sensitivities::Dict{String,Vector{Float64}};
                                         catalogue::Union{Nothing,Dict{String,Vector{Float64}}}=nothing,
                                         reference_name::Union{Nothing,String}=nothing,
                                         E_MeV::Union{Nothing,Vector{Float64}}=nothing)
    selected = String[name for name in detector_names if haskey(readings, name)]
    isempty(selected) && throw(ArgumentError("No detector readings available for catalogue selection"))
    b = [readings[name] for name in selected]
    A = Matrix{Float64}(hcat([sensitivities[name] for name in selected]...)')

    if reference_name !== nothing
        reference_name in readings || throw(ArgumentError("reference_name '$(reference_name)' is not present in readings"))
        ref_idx = findfirst(==(reference_name), selected)
    else
        ref_idx = nsduaz_reference_index(selected, A)
    end

    cat = catalogue
    if cat === nothing
        nb = size(A, 2)
        if E_MeV !== nothing && length(E_MeV) == nb
            cat = nsduaz_builtin_catalogue(E_MeV)
        else
            E_rep = 10.0 .^ range(log10(1e-9), log10(1e2), length=nb)
            cat = nsduaz_builtin_catalogue(E_rep)
        end
    end

    b_ref = b[ref_idx]
    b_ref > 0 || throw(ArgumentError("Reference sphere reading must be strictly positive"))
    r_ratio = b ./ b_ref

    best_label = ""
    best_chi = Inf
    best_scale = 1.0
    best_spec = Vector{Float64}(undef, 0)

    for (label, spec) in cat
        length(spec) == size(A, 2) ||
            throw(ArgumentError("Catalogue spectrum '$label' has length $(length(spec)), expected $(size(A, 2))"))
        any(spec .> 0) || continue
        c = A * max.(spec, 0.0)
        c_ref = c[ref_idx]
        c_ref > 0 || continue
        s_ratio = c ./ c_ref
        chi = sum(((r_ratio .- s_ratio) ./ max.(s_ratio, 1e-12)) .^ 2)
        if chi < best_chi
            best_chi = chi
            best_label = label
            best_scale = b_ref / c_ref
            best_spec = Float64.(spec)
        end
    end

    isempty(best_spec) && throw(ArgumentError("Catalogue is empty or has no usable spectrum"))
    return max.(best_scale .* best_spec, 0.0), best_label
end

"""
    solve_nsduaz(A, b, x0; smoothing=0.1, max_iterations=1000, tolerance=0.01,
                 lethargy_weights=nothing) -> UnfoldResult

SPUNIT-итерация (обёртка над `BSSUnfold.solve_bunki`) с NSDUAZ-умолчаниями:
tolerance ~1% относительного изменения решения, three-point smoothing
фактор 0.1 (передаётся как параметр релаксации `alpha`).  Начальный
спектр обычно получают через `nsduaz_select_catalogue_initial` или
задают вручную.  Если `A` — не летаргично-взвешенная матрица, передайте
`lethargy_weights` (длины корзин): столбцы `A` будут взвешены.
"""
function solve_nsduaz(A::AbstractMatrix{T}, b::AbstractVector{T}, x0::AbstractVector{T};
                      smoothing::Real=0.1,
                      max_iterations::Integer=1000,
                      tolerance::Real=0.01,
                      lethargy_weights::Union{Nothing,AbstractVector}=nothing) where T<:AbstractFloat
    if lethargy_weights !== nothing
        w = Vector{Float64}(lethargy_weights)
        A = Matrix{Float64}(A) .* reshape(w, 1, :)
        b = Vector{Float64}(b)
        x0 = Vector{Float64}(x0)
    end
    return BSSUnfold.solve_bunki(A, b, x0;
                                 max_iterations=max_iterations,
                                 tolerance=tolerance,
                                 alpha=smoothing)
end
