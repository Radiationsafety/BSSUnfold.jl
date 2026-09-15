"""
Утилитарные функции: валидация, построение системы, нормализация.
"""

"""
    validate_system(A, b; x0=nothing, max_iterations=1000, tolerance=1e-6)

Проверить размерности и типы ответной матрицы и измерений.

# Исключения
- `ArgumentError` если размеры не согласованы или параметры некорректны.
"""
function validate_system(A::AbstractMatrix{T}, b::AbstractVector{T};
                        x0::Union{Nothing,AbstractVector{T}}=nothing,
                        max_iterations::Integer=1000,
                        tolerance::Real=T(1e-6)) where T<:AbstractFloat
    m, n = size(A)
    if length(b) != m
        throw(ArgumentError("Length of b ($(length(b))) must match number of rows in A ($m)"))
    end
    if x0 !== nothing && length(x0) != n
        throw(ArgumentError("Length of x0 ($(length(x0))) must match number of columns in A ($n)"))
    end
    if max_iterations <= 0
        throw(ArgumentError("max_iterations must be positive, got $max_iterations"))
    end
    if tolerance <= 0
        throw(ArgumentError("tolerance must be positive, got $tolerance"))
    end
    return A, b, x0
end


"""
    build_system(readings::Dict{String,T}, detector_names, sensitivities)

Построить (A, b) из словаря показаний и ответных функций.
Возвращает также список имён детекторов, которые присутствуют в readings.
"""
function build_system(readings::Dict{String,T},
                      detector_names::Vector{String},
                      sensitivities::Dict{String,Vector{T}}) where T<:AbstractFloat
    selected = filter(name -> haskey(readings, name), detector_names)
    if isempty(selected)
        throw(ArgumentError("No detector names match readings keys"))
    end
    b = T[readings[name] for name in selected]
    A = Matrix(hcat([sensitivities[name] for name in selected]...)')
    return A, b, selected
end


"""
    normalize_initial(initial_spectrum, default_initial, n_energy_bins)

Нормализовать начальный спектр: если `nothing` — вернуть default, иначе проверить размер.
Отрицательные значения обнуляются (физический спектр ≥ 0).
"""
function normalize_initial(initial_spectrum::Union{Nothing,AbstractVector{T}},
                          default_initial::Vector{T},
                          n_energy_bins::Int) where T<:AbstractFloat
    if initial_spectrum === nothing
        return copy(default_initial)
    end
    x = collect(T, initial_spectrum)
    if length(x) != n_energy_bins
        throw(ArgumentError("Initial spectrum length ($(length(x))) must match n_energy_bins ($n_energy_bins)"))
    end
    return max.(x, T(0))
end


"""
    standardize_output(spectrum, A, b, E_MeV, selected, cc_icrp116, method, extra)

Создать стандартизованный выходной словарь с дозовыми коэффициентами.
`extra` может быть Dict{String,Any} или Dict{String,Integer} и т.п.
"""
function standardize_output(spectrum::Vector{T},
                           A::AbstractMatrix{T},
                           b::AbstractVector{T},
                           E_MeV::Vector{Float64},
                           selected::Vector{String},
                           cc_icrp116::Dict{String,Vector{T}},
                           method::AbstractString,
                           extra::Union{Nothing,Dict}=nothing) where T<:AbstractFloat
    spectrum_nonneg = max.(spectrum, T(0))
    computed_readings = A * spectrum_nonneg
    residual = b .- computed_readings

    output = Dict{String,Any}(
        "energy"            => copy(E_MeV),
        "spectrum"          => copy(spectrum_nonneg),
        "spectrum_absolute" => copy(spectrum_nonneg),
        "effective_readings" => Dict(name => Float64(v) for (name, v) in zip(selected, computed_readings)),
        "residual"          => copy(residual),
        "residual_norm"     => Float64(norm(residual)),
        "method"            => method,
    )

    # Дозовые коэффициенты
    if !isempty(cc_icrp116)
        dose_dict = Dict{String,Float64}()
        for (name, cc) in cc_icrp116
            dose_dict[name] = Float64(dot(spectrum_nonneg, cc))
        end
        output["doserates"] = dose_dict
    end

    if extra !== nothing
        merge!(output, Dict{String,Any}(k => v for (k, v) in extra))
    end
    return output
end
