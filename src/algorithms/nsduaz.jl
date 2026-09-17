"""
NSDUAZ unfolding method (port from unfold_nsduaz.py).

NSDUAZ ("Neutron Spectrometry and Dosimetry from the Universidad Autonoma
de Zacatecas"; Ortiz-Rodriguez & Vega-Carrillo, 2012) — unfolding over
Bonner spheres based on the SPUNIT iterative algorithm (Doroshenko et al.,
1977; the same iteration as in BUNKI).  A distinctive feature — automatic
selection of the initial spectrum from a *catalogue* of standard neutron spectra:
experimental count rates are normalized to the reading of the
20.32 cm sphere and compared (statistical test) with the predictions of each
spectrum of the catalogue.  The catalogue entry that best reproduces the measured
relative pattern of readings is used as the initial spectrum for
the SPUNIT iteration, which runs until the relative change of the solution
is below ~1%.

Implemented:
- `solve_nsduaz` — the SPUNIT iteration (a wrapper over `solve_bunki`) with
  the NSDUAZ default convergence threshold;
- `select_catalogue_initial` — selection of the initial spectrum from the catalogue
  (statistical test on reading ratios to the reference sphere);
- `builtin_catalogue` — built-in mini-catalogue of analytical standard
  spectra (241Am/9Be, 252Cf, thermal + 1/E + fission reactor-like);
- `unfold_nsduaz` — Detector-level wrapper (see detector.jl).
"""

# ─── Analytical standard spectra ──────────────────────────────────────────────

"""
    _watt_spectrum(E_MeV; a=1.025, b=2.926)

Analytic watt fission spectrum (e.g. 252Cf):
`exp(-E/a) * sinh(sqrt(b*E))`, normalized to a unit sum.
"""
function _watt_spectrum(E_MeV::AbstractVector{<:Real}; a::Real=1.025, b::Real=2.926)
    E = max.(Float64.(E_MeV), 1e-9)
    w = @. exp(-E / a) * sinh(sqrt(b * E))
    total = sum(w)
    return total > 0 ? w ./ total : fill(1.0 / length(w), length(w))
end

"""
    _ambe_spectrum(E_MeV)

Analytic form of 241Am/9Be(alpha,n): evaporator continuum +
a peak at 4.2 MeV, normalized to a unit sum.
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

Analytic reactor-like spectrum: thermal Maxwellian + 1/E +
fast watt fission spectrum, normalized to a unit sum.
"""
function _reactor_spectrum(E_MeV::AbstractVector{<:Real})
    E = max.(Float64.(E_MeV), 1e-9)
    kT = 0.0253e-6  # 0.0253 eV in MeV
    thermal = @. (E / kT) * exp(-E / kT)
    epithermal = map(e -> e > 1e-6 ? 1.0 / max(e, 1e-9) : 0.0, E)
    fast = _watt_spectrum(E)
    spec = 1e-3 .* thermal .+ 0.1 .* epithermal .+ fast
    total = sum(spec)
    return total > 0 ? spec ./ total : fill(1.0 / length(spec), length(spec))
end

"""
    builtin_catalogue(E_MeV) -> Dict{String,Vector{Float64}}

Build the built-in mini-catalogue of analytical standard spectra
on the energy grid `E_MeV`: keys `"ambe"`, `"cf252"`, `"reactor"`.
"""
function builtin_catalogue(E_MeV::AbstractVector{<:Real})
    E = collect(Float64, E_MeV)
    return Dict{String,Vector{Float64}}(
        "ambe"    => _ambe_spectrum(E),
        "cf252"   => _watt_spectrum(E),
        "reactor" => _reactor_spectrum(E),
    )
end

# ─── Reference sphere search ────────────────────────────────────────────────

function _find_reference_index(detector_names::Vector{String}, A::AbstractMatrix{<:Real})
    # NB: names like "18in"/"18inPb" CONTAIN the substring "8in", so a plain
    # substring test would wrongly select an 18-inch sphere depending on the
    # detector ordering (Python relies on its own name order where "8in"
    # happens to come first).  Anchor the diameter: a match requires that
    # the "8in"/"8 in" token is not preceded by another digit.
    function _diameter8(name::AbstractString)
        lowered = lowercase(name)
        for pat in ("8in", "8 in")
            idx = findfirst(pat, lowered)
            while idx !== nothing
                start = first(idx)
                (start == 1 || !(lowered[start-1] in ('0', '1', '2', '3', '4',
                                                 '5', '6', '7', '8', '9'))) &&
                    return true
                idx = findnext(pat, lowered, start + 1)
            end
        end
        return false
    end
    for (i, name) in enumerate(detector_names)
        lowered = lowercase(name)
        if occursin("20.32", lowered) || occursin("20in", lowered) ||
           _diameter8(name)
            return i
        end
    end
    # Fallback: detector with the largest integral sensitivity.
    return argmax(vec(sum(abs.(A), dims=2)))
end

# ─── Selection of the initial spectrum from the catalogue ───────────────────

"""
    select_catalogue_initial(readings, detector_names, sensitivities;
                             catalogue=nothing, reference_name=nothing,
                             E_MeV=nothing) -> (spectrum, label)

Select the initial spectrum from the catalogue using a statistical test.

Experimental readings are normalized to the reading of the reference sphere
(20.32 cm by default) and compared with the relative pattern of readings
predicted by each spectrum of the catalogue, convolved with the response matrix.
The entry is selected minimizing the weighted chi-square of the relative
ratios, and rescaled so that its predicted reading of the reference
sphere matches the measured one.

# Arguments
- `readings::Dict{String,<:Real}`: detector readings
- `detector_names::Vector{String}`: names of the available detectors
- `sensitivities::Dict{String,Vector{<:Real}}`: sensitivities
- `catalogue`: Dict(label => spectrum on the detector grid); `nothing` —
  use `builtin_catalogue`
- `reference_name`: name of the reference detector; `nothing` — auto-search of the 20.32 cm sphere
- `E_MeV`: grid for building the built-in catalogue; `nothing` —
  a representative log-grid over the length of the sensitivity

# Returns
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
            # log-uniform representative grid (port of numpy.logspace)
            collect(10.0 .^ range(log10(1e-9), log10(1e2), length=n_bins))
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

# ─── Main solver ────────────────────────────────────────────────────────────

"""
    solve_nsduaz(A, b, x0; smoothing=0.1, max_iterations=1000, tolerance=0.01)

Solve the unfolding problem with the NSDUAZ (SPUNIT) iteration.

This is the SPUNIT iteration (a thin wrapper over [`solve_bunki`](@ref))
with the NSDUAZ default convergence threshold (~1% relative change).
The initial spectrum `x0` is usually obtained via
[`select_catalogue_initial`](@ref) (or supplied by the user).

# Arguments
- `A::AbstractMatrix{T}`: response matrix (m × n)
- `b::AbstractVector{T}`: measurements (m,)
- `x0::AbstractVector{T}`: initial spectrum (n,)
- `smoothing`: three-point smoothing factor (default 0.1)
- `max_iterations`: max number of iterations (default 1000)
- `tolerance`: threshold of relative change for early stopping (default 0.01)

# Returns
- `UnfoldResult{T}` with the spectrum
"""
function solve_nsduaz(A::AbstractMatrix{T}, b::AbstractVector{T}, x0::AbstractVector{T};
                      smoothing::Real=T(0.1),
                      max_iterations::Integer=1000,
                      tolerance::Real=T(0.01)) where T<:AbstractFloat
    return solve_bunki(A, b, x0;
                       smoothing=smoothing,
                       max_iterations=max_iterations,
                       tolerance=tolerance)
end
