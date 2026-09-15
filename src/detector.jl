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

"""
    unfold_nsduaz(d::Detector, readings; kwargs...)

Запустить NSDUAZ развёртку на этом детекторе.

Специфичные kwargs: `catalogue` (Dict(label => спектр)), `use_catalogue`
(default `true` — выбирать начальный спектр из каталога, когда
`initial_spectrum` не задан), `reference_name` (опорная сфера).
Остальные kwargs передаются в `solve_nsduaz` (max_iterations, tolerance,
alpha) или во фреймворк (calculate_errors, ...).
"""
function unfold_nsduaz(d::Detector, readings::Dict{String,T}; kwargs...) where T<:AbstractFloat
    framework_keys = (:initial_spectrum, :default_initial, :method_name,
                     :calculate_errors, :noise_level, :n_montecarlo,
                     :random_state, :save_result,
                     :catalogue, :use_catalogue, :reference_name)
    framework = Dict{Symbol,Any}()
    solve_kwargs = Dict{Symbol,Any}()
    for (k, v) in pairs(kwargs)
        if k in framework_keys
            framework[k] = v
        else
            solve_kwargs[k] = v
        end
    end

    initial_spectrum = get(framework, :initial_spectrum, nothing)
    use_catalogue = get(framework, :use_catalogue, true)
    catalogue = get(framework, :catalogue, nothing)
    reference_name = get(framework, :reference_name, nothing)

    cat_label = nothing
    if initial_spectrum === nothing && use_catalogue
        initial_spectrum, cat_label = select_catalogue_initial(
            readings, d.config.detector_names, d.config.sensitivities;
            catalogue=catalogue, reference_name=reference_name,
            E_MeV=d.config.E_MeV)
    end

    output = run_unfolding(solve_nsduaz,
                  d.config.detector_names, d.config.n_energy_bins,
                  d.config.E_MeV, d.config.sensitivities, d.config.cc_icrp116,
                  readings;
                  method_name=get(framework, :method_name, "NSDUAZ"),
                  default_initial=ones(Float64, d.config.n_energy_bins),
                  solve_kwargs=NamedTuple(solve_kwargs),
                  calculate_errors=get(framework, :calculate_errors, false),
                  noise_level=get(framework, :noise_level, T(0.01)),
                  n_montecarlo=get(framework, :n_montecarlo, 100),
                  random_state=get(framework, :random_state, nothing),
                  save_result=get(framework, :save_result, nothing),
                  initial_spectrum=initial_spectrum)
    if cat_label !== nothing
        output["catalogue"] = cat_label
    end
    return output
end

"""
    unfold_nspline(d::Detector, readings; kwargs...)

Запустить N-сплайн развёртку (Исламгулов & Ларцев, 2008) на этом детекторе.

Энергетическая сетка `E_MeV` подставляется из конфигурации детектора
автоматически (можно переопределить kwargs[:E_MeV]).  Специфичные kwargs:
`knots`, `sigma_rel`, `continuity`, `max_iterations`, `tol`, `step_theta`,
`smoothing`, `n_segments` — см. `solve_nspline`.
"""
function unfold_nspline(d::Detector, readings::Dict{String,T}; kwargs...) where T<:AbstractFloat
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
    if !haskey(solve_kwargs, :E_MeV)
        solve_kwargs[:E_MeV] = d.config.E_MeV
    end
    run_unfolding(solve_nspline,
                  d.config.detector_names, d.config.n_energy_bins,
                  d.config.E_MeV, d.config.sensitivities, d.config.cc_icrp116,
                  readings;
                  method_name=get(framework, :method_name, "NSpline"),
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
                              (:unfold_mcmc,                 :solve_mcmc,                 "MCMC"),
                              (:unfold_genetic,              :solve_genetic,              "Genetic"),
                              (:unfold_qubo,                 :solve_qubo,                 "QUBO-Annealing")]      
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
