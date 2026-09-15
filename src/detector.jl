"""
Тип `Detector` — спектрометр Боннера с набором сфер.

Это упрощённая версия `bssunfold/core/detector.py` (6684 LOC),
содержащая только необходимый API для развёртки. Response functions
должны быть загружены отдельно (см. `load_response_functions`).
"""

"""
    Detector

Спектрометр Боннера с набором сфер.

# Поля
- `config::DetectorConfig` — конфигурация
- `results_history::Vector{Dict{String,Any}}` — история развёрток (для сравнения)
"""
mutable struct Detector
    config::DetectorConfig
    results_history::Vector{Dict{String,Any}}

    function Detector(config::DetectorConfig)
        new(config, Dict{String,Any}[])
    end
end

# Конструктор из сырых данных
function Detector(detector_names::Vector{String},
                 E_MeV::Vector{Float64},
                 sensitivities::Dict{String,Vector{T}},
                 cc_icrp116::Dict{String,Vector{T}}) where T<:AbstractFloat
    config = DetectorConfig(detector_names, E_MeV,
                           Dict(k => Float64.(v) for (k, v) in sensitivities),
                           Dict(k => Float64.(v) for (k, v) in cc_icrp116))
    Detector(config)
end

"""
    _convert_rf_to_matrix_variable_step(rf::Dict{String,Vector{Float64}};
                                         Emin=1e-9)

Порт `Detector._convert_rf_to_matrix_variable_step`: превратить сырые
функции отклика в матрицу `A` с поправкой на переменный шаг энергий:
`A[:, i] = rf[:, i] .* (log шаг × ln(10))`.

Реальные функции отклика заданы на бин-серединах и умножаются на ширину
бина в лог-энергии, чтобы интегралы по спектру были корректны.
"""
function _convert_rf_to_matrix_variable_step(rf::Dict{String,Vector{Float64}};
                                             Emin::Real=1e-9)
    haskey(rf, "E_MeV") || throw(ArgumentError("rf dict must contain E_MeV"))
    energies = collect(Float64, rf["E_MeV"])
    sphere_names = String[k for k in keys(rf) if k != "E_MeV"]
    n = length(energies)
    n >= 2 || throw(ArgumentError("At least 2 energy bins are required"))
    for namev in sphere_names
        length(rf[namev]) == n || throw(ArgumentError(
            "Column $namev length must match E_MeV length"))
    end

    log_energies = log10.(energies ./ Emin)
    log_steps = zeros(n)
    log_steps[1] = log_energies[2] - log_energies[1]
    log_steps[end] = log_energies[end] - log_energies[end-1]
    for i in 2:n-1
        log_steps[i] = (log_energies[i+1] - log_energies[i-1]) / 2
    end
    ln_steps = log_steps .* log(10)

    Amat = zeros(n, length(sphere_names))
    for (j, namev) in enumerate(sphere_names)
        Amat[:, j] .= rf[namev] .* ln_steps
    end
    return Amat, energies, sphere_names, log_steps
end

# ─── Конструкторы Detector ───────────────────────────────────────────────────

"""
    Detector(; cc_type="ICRP116")

Спектрометр по умолчанию: реальные функции отклика GSF и конверсионные
коэффициенты `cc_type`.
"""
function Detector(; cc_type::AbstractString="ICRP116")
    return Detector(RF_GSF; cc_type=cc_type)
end

"""
    Detector(rf::Dict{String,Vector{Float64}}; cc_type="ICRP116",
             Emin=1e-9, apply_log_step=true)

Спектрометр из набора реальных функций отклика (`RF_GSF`, `RF_PTB`,
`RF_LANL`, ...). Ответные функции масштабируются на лог-шаг энергий
(как в Python-порте `bssunfold`), дозовые коэффициенты интерполируются
на энергетическую сетку.
"""
function Detector(rf::Dict{String,Vector{Float64}};
                  cc_type::AbstractString="ICRP116",
                  Emin::Real=1e-9)
    Amat, E_MeV, sphere_names, log_steps =
        _convert_rf_to_matrix_variable_step(rf; Emin=Emin)
    sensitivities = Dict{String,Vector{Float64}}(
        sphere_names[j] => Vector{Float64}(Amat[:, j]) for j in eachindex(sphere_names))
    cc_raw = get_coefficients(cc_type)
    cc_interp = interpolate_coefficients(cc_raw, E_MeV)
    config = DetectorConfig(sphere_names, E_MeV, sensitivities, cc_interp, cc_raw, String(cc_type))
    return Detector(config)
end

# ─── Набор дозовых коэффициентов ─────────────────────────────────────────────

"""
    set_dose_coefficients!(d::Detector, name)

Сменить набор конверсионных коэффициентов (`"ICRP116"`, `"ICRP74_effective"`,
`"NRB99_2009_effective"`, `"ICRP74_operational"`), переинтерполировав его
на энергетическую сетку детектора.
"""
function set_dose_coefficients!(d::Detector, name::AbstractString)
    cc_raw = get_coefficients(name)
    d.config.cc_type = String(name)
    d.config.cc_raw = cc_raw
    d.config.cc_icrp116 = interpolate_coefficients(cc_raw, d.config.E_MeV)
    return name
end

"""
    get_effective_readings_for_spectra(d::Detector, spectra::Dict{String,Vector{Float64}})

Вычислить «эффективные показания» детектора для заданного спектра:
интерполяция на сетку детектора (PCHIP в лог-масштабе) и
`reading[i] = max(0, Σ φ(E) · A[:, i])`.

# Возвращает
`Dict{String,Float64}` — показания по каждой сфере.
"""
function get_effective_readings_for_spectra(d::Detector,
                                            spectra::Dict{String,<:Vector{<:Real}})
    haskey(spectra, "E_MeV") || throw(ArgumentError("spectra must contain E_MeV"))
    E_src = collect(Float64, spectra["E_MeV"])
    spec_names = String[k for k in keys(spectra) if k != "E_MeV"]
    isempty(spec_names) && throw(ArgumentError("spectra must contain at least one spectrum"))
    # В Python-порте используется один спектр: "Phi" или первая неэнергетическая колонка
    need_interp = !(length(E_src) == length(d.config.E_MeV) &&
                    isapprox(E_src, d.config.E_MeV; rtol=1e-12, atol=0.0))
    if need_interp
        φ = discretize_spectra(spectra, d.config.E_MeV)
        spectrum_values = Float64.(φ[spec_names[1]])
    else
        spectrum_values = Float64.(spectra[spec_names[1]])
    end
    return get_effective_readings_for_spectra(d, d.config.E_MeV, spectrum_values)
end

"""
    get_effective_readings_for_spectra(d, E_MeV, spectrum)

Вариант с явной сеткой эпергий и значениями спектра.
"""
function get_effective_readings_for_spectra(d::Detector,
                                            E_MeV::AbstractVector{<:Real},
                                            spectrum::AbstractVector{<:Real})
    if !(length(E_MeV) == length(d.config.E_MeV) &&
         isapprox(collect(Float64, E_MeV), d.config.E_MeV; rtol=1e-12, atol=0.0))
        spectrum = interpolate_spectrum(spectrum, E_MeV, d.config.E_MeV)
    end
    length(spectrum) == length(d.config.E_MeV) || throw(ArgumentError(
        "Spectrum length ($(length(spectrum))) must match energy grid length"))

    effective_readings = Dict{String,Float64}()
    for name in d.config.detector_names
        reading = dot(Float64.(spectrum), d.config.sensitivities[name])
        effective_readings[name] = max(0.0, reading)
    end
    return effective_readings
end

"""
    upper_bounds(E_MeV, max_neutron_energy)

Массив верхних границ для QP-солвера: активные бины (`E_MeV <= cutoff`)
получают `+Inf`, остальные — `0.0`.
"""
function upper_bounds(E_MeV::AbstractVector{<:Real}, max_neutron_energy::Union{Nothing,Real})
    ub = fill(Inf, length(E_MeV))
    if max_neutron_energy !== nothing
        ub[E_MeV .> Float64(max_neutron_energy)] .= 0.0
    end
    return ub
end

"""
    max_energy_mask(E_MeV, max_neutron_energy)

Булева маска бинов с `E_MeV <= max_neutron_energy`.
"""
function max_energy_mask(E_MeV::AbstractVector{<:Real}, max_neutron_energy::Union{Nothing,Real})
    max_neutron_energy === nothing && return ones(Bool, length(E_MeV))
    return E_MeV .<= Float64(max_neutron_energy)
end

"""
    detector_upper_bounds(d::Detector, max_neutron_energy)
    detector_max_energy_mask(d::Detector, max_neutron_energy)

Варианты `upper_bounds` / `max_energy_mask` для детектора.
"""
detector_upper_bounds(d::Detector, max_energy::Union{Nothing,Real}) =
    upper_bounds(d.config.E_MeV, max_energy)
detector_max_energy_mask(d::Detector, max_energy::Union{Nothing,Real}) =
    max_energy_mask(d.config.E_MeV, max_energy)

# ─── Свойства ────────────────────────────────────────────────────────────────

"""
    subdetector(d::Detector, mask::AbstractVector{Bool})

Подмножество детектора, ограниченное энергетическими бинами `mask`
(соответствие строк `d.config.E_MeV`).
"""
function subdetector(d::Detector, mask::AbstractVector{Bool})
    length(mask) == length(d.config.E_MeV) ||
        throw(ArgumentError("mask length must match energy grid"))
    sub_names = d.config.detector_names
    sub_E = d.config.E_MeV[mask]
    sub_sens = Dict{String,Vector{Float64}}(
        name => d.config.sensitivities[name][mask] for name in sub_names)
    config = DetectorConfig(sub_names, sub_E, sub_sens,
                            d.config.cc_icrp116, d.config.cc_raw, d.config.cc_type)
    return Detector(config)
end

# ─── Свойства ────────────────────────────────────────────────────────────────

# ─── Свойства ─────────────────────────────────────────────────────────────────
Base.length(d::Detector) = length(d.config.detector_names)
n_energy_bins(d::Detector) = d.config.n_energy_bins
energy_grid(d::Detector) = d.config.E_MeV
detector_names(d::Detector) = d.config.detector_names

"""
    save_result!(d::Detector, result::Dict)

Сохранить результат развёртки в историю.
"""
function save_result!(d::Detector, result::Dict{String,Any})
    push!(d.results_history, result)
    return length(d.results_history)
end

"""
    unfold_mlem(d::Detector, readings; kwargs...)

Запустить MLEM развёртку на этом детекторе. Все kwargs (max_iterations,
tolerance, ...) передаются в solve_mlem.
"""
function unfold_mlem(d::Detector, readings::Dict{String,T}; kwargs...) where T<:AbstractFloat
    # Разделяем kwargs на solve_kwargs (для solve_mlem) и framework-опции
    framework_keys = (:initial_spectrum, :default_initial, :method_name,
                     :calculate_errors, :noise_level, :n_montecarlo,
                     :random_state, :save_result)
    framework = Dict{Symbol,Any}()
    solve_kwargs = Dict{Symbol,Any}()
    for (k, v) in pairs(kwargs)
        if k in framework_keys
            framework[k] = v
        else
            solve_kwargs[k] = v
        end
    end
    run_unfolding(solve_mlem,
                  d.config.detector_names, d.config.n_energy_bins,
                  d.config.E_MeV, d.config.sensitivities, d.config.cc_icrp116,
                  readings;
                  method_name=get(framework, :method_name, "MLEM"),
                  default_initial=get(framework, :default_initial,
                                     ones(Float64, d.config.n_energy_bins) * 0.5),
                  solve_kwargs=NamedTuple(solve_kwargs),
                  calculate_errors=get(framework, :calculate_errors, false),
                  noise_level=get(framework, :noise_level, T(0.01)),
                  n_montecarlo=get(framework, :n_montecarlo, 100),
                  random_state=get(framework, :random_state, nothing),
                  save_result=get(framework, :save_result, nothing),
                  initial_spectrum=get(framework, :initial_spectrum, nothing))
end

# Generic generator для остальных unfold_* методов
for (m, fn, method_label) in [(:unfold_gravel,       :solve_gravel,       "GRAVEL"),
                              (:unfold_landweber,    :solve_landweber,    "Landweber"),
                              (:unfold_maxed,        :solve_maxed,        "MAXED"),
                              (:unfold_tikhonov,     :solve_tikhonov,     "Tikhonov"),
                              (:unfold_tsvd,         :solve_tsvd,          "TSVD"),
                              (:unfold_sandii,       :solve_sandii,       "Sandii"),
                              (:unfold_bunki,        :solve_bunki,        "Bunki"),
                              (:unfold_kaczmarz,     :solve_kaczmarz,      "Kaczmarz"),
                              (:unfold_cgls,         :solve_cgls,          "CGLS"),
                              (:unfold_fista,        :solve_fista,         "FISTA"),
                              (:unfold_bsrem,        :solve_bsrem,         "BSREM"),
                              (:unfold_osem,          :solve_osem,          "OSEM"),
                              (:unfold_staysl,       :solve_staysl,        "Staysl"),
                              (:unfold_doroshenko,   :solve_doroshenko,    "Doroshenko"),
                              (:unfold_lanczos,              :solve_lanczos,              "Lanczos"),
                              (:unfold_iterative_refinement, :solve_iterative_refinement, "IterativeRefinement"),
                              (:unfold_randomized_kaczmarz,  :solve_randomized_kaczmarz,  "RandomizedKaczmarz"),
                              (:unfold_cvxpy,                :solve_cvxpy,                "CVXPY"),
                              (:unfold_qpsolvers,            :solve_qpsolvers,            "QPsolvers"),
                              # Новые методы (порт bssunfold)
                              (:unfold_amaxed,               :solve_amaxed,               "AMAXED"),
                              (:unfold_amaxed_regularization,:solve_amaxed_regularization,"AMAXED_Reg"),
                              (:unfold_imaxed,               :solve_imaxed,               "IMAXED"),
                              (:unfold_sart,                 :solve_sart,                 "SART"),
                              (:unfold_mapem,                :solve_mapem,                "MAPEM"),
                              (:unfold_mlem_stop,            :solve_mlem_stop,            "MLEM_Stop"),
                              (:unfold_bunkiut,              :solve_bunkiut,              "BunkiUT"),
                              (:unfold_rebunki,              :solve_rebunki,              "Rebunki"),
                              (:unfold_directed_divergence,  :solve_directed_divergence,  "DirectedDivergence"),
                              (:unfold_ferdor,               :solve_ferdor,               "Ferdor"),
                              (:unfold_scipy_direct,         :solve_scipy_direct,         "SciPy_Direct"),
                              (:unfold_tikhonov_tv,          :solve_tikhonov_tv,          "Tikhonov_TV"),
                              (:unfold_tikhonov_legendre,    :solve_tikhonov_legendre,    "Tikhonov_Legendre"),
                              (:unfold_statreg,              :solve_statreg,              "StatReg"),
                              (:unfold_reconst,              :solve_reconst,              "RECONST"),
                              (:unfold_bayes,                :solve_bayes,                "Bayes"),
                              (:unfold_bayes_spline,         :solve_bayes_spline,         "Bayes_Spline"),
                              (:unfold_eki,                  :solve_eki,                  "EKI"),
                              (:unfold_express,              :solve_express,              "EXPRESS"),
                              (:unfold_crystal_ball,         :solve_crystal_ball,         "CrystalBall"),
                              (:unfold_ensemble,             :solve_ensemble,             "Ensemble"),
                              (:unfold_cs,                   :solve_cs,                   "CS"),
                              (:unfold_binned,               :solve_binned,               "Binned"),
                              (:unfold_gks,                  :solve_gks,                  "GKS"),
                              (:unfold_maeo,                 :solve_maeo,                 "MAEO"),
                              (:unfold_nsduaz,               :solve_nsduaz,               "NSDUAZ"),
                              (:unfold_nnksvd,               :solve_nnksvd,               "NNKSVD"),
                              (:unfold_nspline,              :solve_nspline,              "N-SPLINE"),
                              (:unfold_hybrid_gmres,         :solve_hybrid_gmres,         "HybridGMRES"),
                              (:unfold_hybrid_parametric,    :solve_hybrid_parametric,    "HybridParametric"),
                              (:unfold_parametric,           :solve_parametric,           "Parametric"),
                              (:unfold_parametric2,          :solve_parametric2,          "Parametric2")]
    @eval function $(m)(d::Detector, readings::Dict{String,T}; kwargs...) where T<:AbstractFloat
        framework_keys = (:initial_spectrum, :default_initial, :method_name,
                         :calculate_errors, :noise_level, :n_montecarlo,
                         :random_state, :save_result)
        framework = Dict{Symbol,Any}()
        solve_kwargs = Dict{Symbol,Any}()
        for (k, v) in pairs(kwargs)
            if k in framework_keys
                framework[k] = v
            else
                solve_kwargs[k] = v
            end
        end
        run_unfolding($(fn),
                      d.config.detector_names, d.config.n_energy_bins,
                      d.config.E_MeV, d.config.sensitivities, d.config.cc_icrp116,
                      readings;
                      method_name=get(framework, :method_name, $(method_label)),
                      default_initial=get(framework, :default_initial,
                                         ones(Float64, d.config.n_energy_bins) * 0.5),
                      solve_kwargs=NamedTuple(solve_kwargs),
                      calculate_errors=get(framework, :calculate_errors, false),
                      noise_level=get(framework, :noise_level, T(0.01)),
                      n_montecarlo=get(framework, :n_montecarlo, 100),
                      random_state=get(framework, :random_state, nothing),
                      save_result=get(framework, :save_result, nothing),
                      initial_spectrum=get(framework, :initial_spectrum, nothing))
    end
end
