"""
Типы данных BSSUnfold.jl.
"""

"""
    UnfoldResult{T<:AbstractFloat}

Результат развёртки: спектр, число итераций, флаг сходимости, норма остатка.

# Поля
- `spectrum::Vector{T}` — восстановленный спектр (неотрицательный)
- `iterations::Int` — фактически выполненное число итераций
- `converged::Bool` — достигнута ли сходимость
- `residual_norm::T` — `||b - A*x||₂`
- `extra::Dict{String,Any}` — дополнительные алгоритм-специфичные метаданные
"""
struct UnfoldResult{T<:AbstractFloat}
    spectrum::Vector{T}
    iterations::Int
    converged::Bool
    residual_norm::T
    extra::Dict{String,Any}
end

# Конструктор без extra
function UnfoldResult(spectrum::Vector{T}, iterations::Int, converged::Bool,
                     residual_norm::T) where T<:AbstractFloat
    UnfoldResult{T}(spectrum, iterations, converged, residual_norm, Dict{String,Any}())
end

# Конструктор, принимающий кортеж (как в Python)
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

Конфигурация спектрометра Боннера: имена сфер, энергетическая сетка,
ответные функции, коэффициенты пересчёта в дозу.

# Поля
- `detector_names::Vector{String}` — имена сфер (например, `["0_in", "2_in", ...]`)
- `E_MeV::Vector{Float64}` — энергетическая сетка, МэВ
- `sensitivities::Dict{String,Vector{Float64}}` — ответные функции каждой сферы
- `cc_icrp116::Dict{String,Vector{Float64}}` — коэффициенты ICRP-116 для дозы
"""
struct DetectorConfig
    detector_names::Vector{String}
    E_MeV::Vector{Float64}
    sensitivities::Dict{String,Vector{Float64}}
    cc_icrp116::Dict{String,Vector{Float64}}
    n_energy_bins::Int
end

function DetectorConfig(detector_names::Vector{String},
                       E_MeV::Vector{Float64},
                       sensitivities::Dict{String,Vector{Float64}},
                       cc_icrp116::Dict{String,Vector{Float64}})
    n = length(E_MeV)
    @assert all(length(s) == n for s in values(sensitivities)) "Sensitivity length must match E_MeV"
    @assert all(length(c) == n for c in values(cc_icrp116)) "ICRP-116 length must match E_MeV"
    DetectorConfig(detector_names, E_MeV, sensitivities, cc_icrp116, n)
end
