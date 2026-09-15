"""
Data types for BSSUnfold.jl.
"""

"""
    UnfoldResult{T<:AbstractFloat}

Unfolding result: spectrum, number of iterations, convergence flag, residual norm.

# Fields
- `spectrum::Vector{T}` — reconstructed spectrum (nonnegative)
- `iterations::Int` — number of iterations actually performed
- `converged::Bool` — whether convergence was reached
- `residual_norm::T` — `||b - A*x||₂`
- `extra::Dict{String,Any}` — additional algorithm-specific metadata
"""
struct UnfoldResult{T<:AbstractFloat}
    spectrum::Vector{T}
    iterations::Int
    converged::Bool
    residual_norm::T
    extra::Dict{String,Any}
end

# Constructor without extra
function UnfoldResult(spectrum::Vector{T}, iterations::Int, converged::Bool,
                     residual_norm::T) where T<:AbstractFloat
    UnfoldResult{T}(spectrum, iterations, converged, residual_norm, Dict{String,Any}())
end

# Constructor accepting a tuple (as in Python)
function UnfoldResult(t::Tuple{Vector{T}, Int, Bool, T}) where T<:AbstractFloat
    UnfoldResult(t[1], t[2], t[3], t[4])
end

function Base.show(io::IO, r::UnfoldResult)
    print(io, "UnfoldResult(spectrum length=$(length(r.spectrum)), " *
              "iterations=$(r.iterations), converged=$(r.converged), " *
              "residual_norm=$(round(r.residual_norm, digits=6)))")
end


"""
    DetectorConfig

Configuration of a Bonner sphere spectrometer: sphere names, energy grid,
response functions, dose conversion coefficients.

# Fields
- `detector_names::Vector{String}` — sphere names (e.g. `["0_in", "2_in", ...]`)
- `E_MeV::Vector{Float64}` — energy grid, MeV
- `sensitivities::Dict{String,Vector{Float64}}` — response function of each sphere
- `cc_icrp116::Dict{String,Vector{Float64}}` — conversion coefficients
  interpolated onto `E_MeV` (e.g. ICRP-116: AP, PA, ..., ISO)
- `cc_raw::Dict{String,Vector{Float64}}` — original (non-interpolated) set
- `cc_type::String` — name of the coefficient set
"""
mutable struct DetectorConfig
    detector_names::Vector{String}
    E_MeV::Vector{Float64}
    sensitivities::Dict{String,Vector{Float64}}
    cc_icrp116::Dict{String,Vector{Float64}}
    cc_raw::Dict{String,Vector{Float64}}
    cc_type::String
    n_energy_bins::Int
end

function DetectorConfig(detector_names::Vector{String},
                       E_MeV::Vector{Float64},
                       sensitivities::Dict{String,Vector{Float64}},
                       cc_icrp116::Dict{String,Vector{Float64}},
                       cc_raw::Dict{String,Vector{Float64}},
                       cc_type::String)
    n = length(E_MeV)
    @assert all(length(s) == n for s in values(sensitivities)) "Sensitivity length must match E_MeV"
    @assert all(length(c) == n for c in values(cc_icrp116)) "ICRP-116 length must match E_MeV"
    DetectorConfig(detector_names, E_MeV, sensitivities, cc_icrp116, cc_raw, cc_type, n)
end

# Legacy 4-argument constructor: cc_icrp116 is passed as the raw set,
# interpolated onto E_MeV; cc_raw = original coefficients.
function DetectorConfig(detector_names::Vector{String},
                       E_MeV::Vector{Float64},
                       sensitivities::Dict{String,Vector{Float64}},
                       cc_raw::Dict{String,Vector{Float64}}; cc_type::String="ICRP116")
    n = length(E_MeV)
    @assert all(length(s) == n for s in values(sensitivities)) "Sensitivity length must match E_MeV"
    cc_interp = haskey(cc_raw, "E_MeV") ?
        interpolate_coefficients(cc_raw, E_MeV) : cc_raw
    DetectorConfig(detector_names, E_MeV, sensitivities, cc_interp, cc_raw, cc_type)
end
