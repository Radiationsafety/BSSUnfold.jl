"""
    solve_binned(A, b, bin_lookup; x0=nothing)

Бин-адаптивная развёртка: для каждого энергетического бина выбирается
лучший метод из предвычисленной эталонной таблицы (benchmark lookup),
итоговый спектр собирается по-бинно из «победивших» методов.

Опирается на эмпирическое наблюдение, что разные алгоритмы развёртки
лучше работают в разных энергетических областях (grid-search бенчмарк
67 методов на 271 эталонных спектрах по 41 метрике качества).

Для каждого кандидата из `bin_lookup["unique_methods"]` решатель
находится через `getfield(BSSUnfold, Symbol(...))` по таблице
`METHOD_DISPATCH`; недоступные методы пропускаются с `@warn`.
Бин без доступных методов заполняется медианой успешных спектров.

# Возвращает
`UnfoldResult`; в `extra` — `method_map` (индекс метода на каждый бин),
`successful_methods`, `individual_spectra`, `errors`, `n_bins`.
"""
function solve_binned(A::AbstractMatrix{T}, b::AbstractVector{T},
                      bin_lookup::AbstractDict;
                      x0::Union{Nothing,AbstractVector{T}}=nothing) where T<:AbstractFloat
    m, n_bins = size(A)

    bin_to_methods = bin_lookup["bin_to_methods"]
    bin_to_methods = Dict{Int,Any}(Int(k) => v for (k, v) in bin_to_methods)

    candidate_names = String[]
    if haskey(bin_lookup, "unique_methods") && !isempty(bin_lookup["unique_methods"])
        candidate_names = String.(bin_lookup["unique_methods"])
    else
        seen = Set{String}()
        for ranking in values(bin_to_methods)
            for entry in ranking
                push!(seen, entry[1])
            end
        end
        candidate_names = sort!(collect(seen))
    end

    x0eff = x0 === nothing ? fill(one(T) / n_bins, n_bins) : x0

    spectra = Dict{String,Vector{T}}()
    successes = String[]
    errors = Dict{String,String}()

    for name in candidate_names
        fn = _dispatch_solver(name)
        if fn === nothing
            errors[name] = "method not available in BSSUnfold"
            continue
        end
        try
            result = fn(A, b, x0eff)
            spec = result isa UnfoldResult ? collect(T, result.spectrum) :
                   collect(T, float.(result))
            if length(spec) == n_bins && all(isfinite, spec) && sum(spec) > 0
                spectra[name] = max.(spec, T(0))
                push!(successes, name)
            else
                errors[name] = "invalid output"
            end
        catch err
            errors[name] = sprint(showerror, err)
        end
    end

    assembled = zeros(T, n_bins)
    method_map = fill(-1, n_bins)
    name_to_idx = Dict(name => i for (i, name) in enumerate(candidate_names))

    for b_idx in 1:n_bins
        ranking = get(bin_to_methods, b_idx, [])
        picked = false
        for entry in ranking
            method_name = entry[1]
            if haskey(spectra, method_name)
                assembled[b_idx] = spectra[method_name][b_idx]
                method_map[b_idx] = name_to_idx[method_name]
                picked = true
                break
            end
        end
        if !picked
            vals = T[spectrum[b_idx] for spectrum in values(spectra)]
            assembled[b_idx] = isempty(vals) ? zero(T) : BSSUnfold.Statistics.median(vals)
        end
    end

    residual = norm(b .- A * assembled)
    return UnfoldResult(assembled, length(successes), !isempty(successes), residual,
                        Dict{String,Any}(
                            "method_map" => method_map,
                            "candidate_methods" => candidate_names,
                            "successful_methods" => successes,
                            "individual_spectra" => spectra,
                            "errors" => errors,
                            "n_bins" => n_bins,
                        ))
end

"""
    _dispatch_solver(name)

Найти функцию-решатель для короткого имени метода через
`METHOD_DISPATCH` и `getfield(BSSUnfold, Symbol(fname))`.  Если метод
не определён в пакете — `@warn` и `nothing`.
"""
function _dispatch_solver(name::AbstractString)
    fname = get(METHOD_DISPATCH, name, "solve_$(name)")
    sym = Symbol(fname)
    if isdefined(BSSUnfold, sym)
        fn = getfield(BSSUnfold, sym)
        fn isa Function && return fn
    end
    @warn "solve_binned: метод '$name' ($fname) недоступен в BSSUnfold, пропускается"
    return nothing
end

"""
    METHOD_DISPATCH

Таблица распределения коротких имён методов бенчмарка → имена
функций-решателей BSSUnfold.  Имена без реализованных аналогов
пропускаются во время исполнения с предупреждением.
"""
const METHOD_DISPATCH = Dict{String,String}(
    "tsvd" => "solve_tsvd",
    "bayes" => "solve_bayes",
    "cvxpy" => "solve_cvxpy",
    "statreg" => "solve_statreg",
    "lanczos" => "solve_lanczos",
    "mlem" => "solve_mlem",
    "landweber" => "solve_landweber",
    "bayes_spline" => "solve_bayes_spline",
    "gravel" => "solve_gravel",
    "qpsolvers" => "solve_qpsolvers",
    "hybrid_parametric" => "solve_hybrid_parametric",
    "parametric2" => "solve_parametric2",
    "genetic" => "solve_genetic",
    "interpret" => "solve_interpret",
    "maeo_ensemble" => "solve_maeo",
    "mystic" => "solve_mystic",
    "mystic_hybrid" => "solve_mystic_hybrid",
    "cs" => "solve_cs",
    "scip" => "solve_scip",
    "docplex" => "solve_docplex",
    "epic" => "solve_epic",
    "kaczmarz" => "solve_kaczmarz",
    "sart" => "solve_sart",
    "osem" => "solve_osem",
    "bsrem" => "solve_bsrem",
    "mapem" => "solve_mapem",
    "ferdor" => "solve_ferdor",
    "rebunki" => "solve_rebunki",
    "nsduaz" => "solve_nsduaz",
    "doroshenko" => "solve_doroshenko",
    "sandii" => "solve_sandii",
    "bunki" => "solve_bunki",
    "bunkiut" => "solve_bunkiut",
    "reconst" => "solve_reconst",
    "amaxed" => "solve_amaxed",
    "amaxed_regularization" => "solve_amaxed_regularization",
    "imaxed" => "solve_imaxed",
    "maxed" => "solve_maxed",
    "mlem_odl" => "solve_mlem_odl",
    "mlem_stop" => "solve_mlem_stop",
    "cgls" => "solve_cgls",
    "gks" => "solve_gks",
    "hybrid_gmres" => "solve_hybrid_gmres",
    "tikhonov_legendre" => "solve_tikhonov_legendre",
    "tikhonov_tv" => "solve_tikhonov_tv",
    "fista" => "solve_fista",
    "crystal_ball" => "solve_crystal_ball",
    "rfsp_jul" => "solve_rfsp_jul",
    "staysl" => "solve_staysl",
    "parametric" => "solve_parametric",
    "parametric_cvxpy" => "solve_parametric",
    "parametric_qpsolvers" => "solve_parametric",
    "parametric_combined" => "solve_parametric",
    "lmfit" => "solve_lmfit",
    "lmfit_ic" => "solve_lmfit",
    "scipy_direct_method" => "solve_scipy_direct_method",
    "qubo" => "solve_qubo",
    "zfit" => "solve_zfit",
    "mcmc" => "solve_mcmc",
    "bayesian_parametric" => "solve_bayesian_parametric",
    "eki" => "solve_eki",
    "maeo" => "solve_maeo",
    "odl_pdhg" => "solve_odl_pdhg",
    "odl_douglas_rachford" => "solve_odl_douglas_rachford",
    "combined" => "solve_ensemble",
    "cascade" => "solve_cascade",
    "composite" => "solve_composite",
)

const _DEFAULT_LOOKUP = joinpath(@__DIR__, "..", "data", "bin_lookup.json")

"""
    load_bin_lookup(path::AbstractString=_DEFAULT_LOOKUP)

Загрузить предвычисленную таблицу распределения методов по бинам из
JSON-файла.  По умолчанию используется встроенная таблица
`src/data/bin_lookup.json`.

# Возвращает
`Dict{String,Any}` с ключами `"bin_to_methods"`
(`Dict{Int,Vector{Tuple{String,Float64}}}`), `"unique_methods"`
(`Vector{String}`), `"n_bins"` (`Int`).
"""
function load_bin_lookup(path::AbstractString=_DEFAULT_LOOKUP)
    isfile(path) || throw(ArgumentError(
        "Bin lookup not found at $path. " *
        "Run the benchmark builder to generate it."))
    raw = BSSUnfold.JSON.parsefile(path)

    bin_to_methods = Dict{Int,Vector{Tuple{String,Float64}}}()
    for (k, ranking) in raw["bin_to_methods"]
        entries = Tuple{String,Float64}[]
        for entry in ranking
            push!(entries, (String(entry[1]), Float64(entry[2])))
        end
        bin_to_methods[parse(Int, k)] = entries
    end

    return Dict{String,Any}(
        "bin_to_methods" => bin_to_methods,
        "unique_methods" => String[String(u) for u in raw["unique_methods"]],
        "n_bins" => Int(raw["n_bins"]),
    )
end

"""
    save_bin_lookup(lookup, path)

Сохранить таблицу распределения методов по бинам в JSON-файл
(ключи бинов сериализуются как строки, как в Python-порте).
"""
function save_bin_lookup(lookup::AbstractDict, path::AbstractString)
    mkpath(dirname(abspath(path)))
    serializable = Dict{String,Any}(
        "bin_to_methods" => Dict{String,Any}(
            string(k) => Any[[name, score] for (name, score) in ranking]
            for (k, ranking) in lookup["bin_to_methods"]),
        "unique_methods" => collect(String, lookup["unique_methods"]),
        "n_bins" => Int(lookup["n_bins"]),
    )
    open(path, "w") do io
        BSSUnfold.JSON.print(io, serializable, 2)
    end
    return path
end

"""
    build_bin_lookup(ref_spectra, method_spectra; n_bins=60, top_k=5)

Построить таблицу распределения методов по бинам из результатов
бенчмарка: для каждого бина вычисляется средняя абсолютная ошибка
каждого метода относительно эталонных спектров; на каждый бин
сохраняются `top_k` лучших методов.

# Аргументы
- `ref_spectra::Dict{String,Vector{Float64}}` — эталонные спектры
  (ключ → вектор длины `n_bins`);
- `method_spectra::Dict{String,Dict{String,Vector{Float64}}}` —
  развернутые спектры: имя метода → (ключ эталона → спектр).

# Возвращает
`Dict{String,Any}` с `"bin_to_methods"` (bin → [(method, score), ...]),
`"unique_methods"` и `"n_bins"`.
"""
function build_bin_lookup(ref_spectra::Dict{String,Vector{Float64}},
                           method_spectra::Dict{String,Dict{String,Vector{Float64}}};
                           n_bins::Integer=60,
                           top_k::Integer=5)
    methods = sort(collect(keys(method_spectra)))
    bin_errors = fill(NaN, length(methods), n_bins)

    for (m_idx, m_name) in enumerate(methods)
        per_bin_accum = zeros(Float64, n_bins)
        count = 0
        for (key, ref_spec) in ref_spectra
            data = method_spectra[m_name]
            if haskey(data, key)
                unf = data[key]
                if length(unf) == n_bins && all(isfinite, unf)
                    per_bin_accum .+= abs.(unf .- ref_spec)
                    count += 1
                end
            end
        end
        if count > 0
            bin_errors[m_idx, :] .= per_bin_accum ./ count
        end
    end

    bin_to_methods = Dict{Int,Vector{Tuple{String,Float64}}}()
    all_method_names = Set{String}()

    for b_idx in 1:n_bins
        col = bin_errors[:, b_idx]
        valid = [i for i in eachindex(col) if isfinite(col[i])]
        if isempty(valid)
            bin_to_methods[b_idx] = Tuple{String,Float64}[]
            continue
        end
        order = valid[sortperm(col[valid])]
        top = Tuple{String,Float64}[(methods[i], Float64(col[i])) for i in order[1:min(top_k, length(order))]]
        bin_to_methods[b_idx] = top
        union!(all_method_names, first.(top))
    end

    return Dict{String,Any}(
        "bin_to_methods" => bin_to_methods,
        "unique_methods" => sort!(collect(all_method_names)),
        "n_bins" => n_bins,
    )
end
