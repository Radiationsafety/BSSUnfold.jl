"""
The `Detector` type — a Bonner sphere spectrometer with a set of spheres.

This is a simplified version of `bssunfold/core/detector.py` (6684 LOC),
containing only the API necessary for unfolding. Response functions
must be loaded separately (see `load_response_functions`).
"""

"""
    Detector

Bonner sphere spectrometer with a set of spheres.

# Fields
- `config::DetectorConfig` — configuration
- `results_history::Vector{Dict{String,Any}}` — history of unfoldings (for comparison)
"""
mutable struct Detector
    config::DetectorConfig
    results_history::Vector{Dict{String,Any}}

    function Detector(config::DetectorConfig)
        new(config, Dict{String,Any}[])
    end
end

# Constructor from raw data
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

Port of `Detector._convert_rf_to_matrix_variable_step`: turn raw response
functions into a matrix `A` corrected for variable energy step:
`A[:, i] = rf[:, i] .* (log step × ln(10))`.

Response functions are defined at bin midpoints and multiplied by the
bin width in log energy so spectrum integrals are correct.
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

# ─── Detector constructors ───────────────────────────────────────────────────

"""
    Detector(; cc_type="ICRP116")

Default spectrometer: real GSF response functions and conversion
coefficients of type `cc_type`.
"""
function Detector(; cc_type::AbstractString="ICRP116")
    return Detector(RF_GSF; cc_type=cc_type)
end

"""
    Detector(rf::Dict{String,Vector{Float64}}; cc_type="ICRP116",
             Emin=1e-9, apply_log_step=true)

Spectrometer from a set of real response functions (`RF_GSF`, `RF_PTB`,
`RF_LANL`, ...). Response functions are scaled by the log-step of energies
(as in the Python port of `bssunfold`), dose coefficients are interpolated
onto the energy grid.
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

# ─── Set of dose coefficients ────────────────────────────────────────────────

"""
    set_dose_coefficients!(d::Detector, name)

Switch the set of conversion coefficients (`"ICRP116"`, `"ICRP74_effective"`,
`"NRB99_2009_effective"`, `"ICRP74_operational"`), re-interpolating it onto
the detector energy grid.
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

Compute the "effective readings" of the detector for a given spectrum:
interpolation onto the detector grid (PCHIP in log scale) and
`reading[i] = max(0, Σ φ(E) · A[:, i])`.

# Returns
`Dict{String,Float64}` — readings of each sphere.
"""
function get_effective_readings_for_spectra(d::Detector,
                                            spectra::Dict{String,<:Vector{<:Real}})
    haskey(spectra, "E_MeV") || throw(ArgumentError("spectra must contain E_MeV"))
    E_src = collect(Float64, spectra["E_MeV"])
    spec_names = String[k for k in keys(spectra) if k != "E_MeV"]
    isempty(spec_names) && throw(ArgumentError("spectra must contain at least one spectrum"))
    # The Python port uses one spectrum: "Phi" or the first non-energy column
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

Variant with explicit energy grid and spectrum values.
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

Array of upper bounds for the QP solver: active bins (`E_MeV <= cutoff`)
get `+Inf`, the rest get `0.0`.
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

Boolean mask of bins with `E_MeV <= max_neutron_energy`.
"""
function max_energy_mask(E_MeV::AbstractVector{<:Real}, max_neutron_energy::Union{Nothing,Real})
    max_neutron_energy === nothing && return ones(Bool, length(E_MeV))
    return E_MeV .<= Float64(max_neutron_energy)
end

"""
    detector_upper_bounds(d::Detector, max_neutron_energy)
    detector_max_energy_mask(d::Detector, max_neutron_energy)

Variants of `upper_bounds` / `max_energy_mask` for a detector.
"""
detector_upper_bounds(d::Detector, max_energy::Union{Nothing,Real}) =
    upper_bounds(d.config.E_MeV, max_energy)
detector_max_energy_mask(d::Detector, max_energy::Union{Nothing,Real}) =
    max_energy_mask(d.config.E_MeV, max_energy)

# ─── Properties ──────────────────────────────────────────────────────────────

"""
    subdetector(d::Detector, mask::AbstractVector{Bool})

Subset of the detector restricted to the energy bins in `mask`
(matching rows of `d.config.E_MeV`).
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

# ─── Properties ────────────────────────────────────────────────────────────────

# ─── Properties ─────────────────────────────────────────────────────────────────
Base.length(d::Detector) = length(d.config.detector_names)
n_energy_bins(d::Detector) = d.config.n_energy_bins
energy_grid(d::Detector) = d.config.E_MeV
detector_names(d::Detector) = d.config.detector_names

"""
    save_result!(d::Detector, result::Dict)

Save an unfolding result to the history.
"""
function save_result!(d::Detector, result::Dict{String,Any})
    push!(d.results_history, result)
    return length(d.results_history)
end

"""
    unfold_mlem(d::Detector, readings; kwargs...)

Run an MLEM unfolding on this detector. All kwargs (max_iterations,
tolerance, ...) are forwarded to solve_mlem.
"""
function unfold_mlem(d::Detector, readings::Dict{String,T}; kwargs...) where T<:AbstractFloat
    # Split kwargs into solve_kwargs (for solve_mlem) and framework options
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

Run an NSDUAZ unfolding on this detector.

Specific kwargs: `catalogue` (Dict(label => spectrum)), `use_catalogue`
(default `true` — pick the initial spectrum from the catalogue when
`initial_spectrum` is not given), `reference_name` (reference sphere).
Other kwargs are forwarded to `solve_nsduaz` (max_iterations, tolerance,
alpha) or to the framework (calculate_errors, ...).
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

Run an N-spline unfolding (Islamgulov & Lartsev, 2008) on this detector.

The energy grid `E_MeV` is taken from the detector configuration
automatically (can be overridden via kwargs[:E_MeV]).  Specific kwargs:
`knots`, `sigma_rel`, `continuity`, `max_iterations`, `tol`, `step_theta`,
`smoothing`, `n_segments` — see `solve_nspline`.
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

# Generic generator for the remaining unfold_* methods
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
                              # New methods (port of bssunfold)
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
                              (:unfold_nnksvd,               :solve_nnksvd,               "NNKSVD"),
                              (:unfold_hybrid_gmres,         :solve_hybrid_gmres,         "HybridGMRES"),
                              (:unfold_hybrid_parametric,    :solve_hybrid_parametric,    "HybridParametric"),
                              (:unfold_parametric,           :solve_parametric,           "Parametric"),
                              (:unfold_parametric2,          :solve_parametric2,          "Parametric2"),
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
