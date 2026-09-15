"""
Bayesian MCMC unfolding method (порт из unfold_mcmc.py, pymc → Turing.jl).

Полный байесовский подход к развёртке нейтронных спектров с помощью
Markov Chain Monte Carlo, а именно сэмплера NUTS (No-U-Turn Sampler) —
адаптивного варианта Hamiltonian Monte Carlo.

Спектр моделируется в лог-масштабе со сглаживающим (Орнштейн-Уленбек)
приором, заякоренным на data-driven центре (начальный спектр `x0` или
неотрицательное решение методом наименьших квадратов).  Это удерживает
сильно недоопределённую задачу развёртки в узде: спектр остаётся
положительным, гладким и ограниченным в нуль-пространстве ответной
матрицы, а апостериорное среднее совпадает с детерминированными солверами
(например `solve_cvxpy`) на референсной базе спектров МАГАТЭ.

Байесовская структура даёт:
- полные апостериорные распределения для каждого энергетического бина;
- квантификацию неопределённости через доверительные интервалы (HPD);
- автоматическую регуляризацию через спецификацию приора;
- иерархическое моделирование шума правдоподобия (use_hierarchical=true).

Зависимость: Turing.jl (ленивая загрузка при первом вызове; без неё
функция выдаёт предупреждение и возвращает нулевой спектр).

Модель (аналог pymc-модели Python-оригинала):

    s ~ HalfNormal(lambda_prior)                    # амплитуда отклонений
    z ~ MvNormal(0, I)                              # белое латентное поле
    theta = mu_prior + s * (L_corr * z)             # OU-коррелированное поле
    spectrum = exp(theta)                           # положительный спектр
    sigma = sigma_prior * |b|                       # относительный шум
    b ~ MvNormal(A * spectrum, Diagonal(sigma^2))   # правдоподобие

где `C_ou[i, j] = exp(-|i - j| / lengthscale)` — OU-корреляция
(нецентрированная параметризация через Cholesky-фактор L_corr).

Апостериорные выборки спектра восстанавливаются из выборок `s` и `z`
(`spectrum = exp(mu_prior + s * L_corr * z)`), что не зависит от записи
детерминированных переменных в цепь в конкретных версиях Turing.
"""

const _TURING_LOADED = Ref(false)

"""
    _try_load_turing() -> Bool

Ленивая загрузка Turing.jl при первом вызове `solve_mcmc` и определение
байесовской модели (макрос `Turing.@model` требует загруженного Turing,
поэтому раскрывается в рантайме через `@eval`).

Turing загружается в `Main` текущей сессии (аналогично Requires.jl):
пакет не может `using` не-прямую зависимость из своего пространства имён,
но может загрузить её в окружение пользователя. Если Turing уже загружена
пользователем (`using Turing`) — просто переиспользуем её.

Возвращает `true`, если Turing доступна (в `Main.Turing`).
"""
function _try_load_turing()
    if _TURING_LOADED[]
        return true
    end
    try
        Base.eval(Main, :(using Turing))
    catch err
        @warn "Turing.jl не удалось загрузить; solve_mcmc недоступен. " *
              "Установите через: Pkg.add(\"Turing\")" exception=err
        return false
    end

    # Определяем байесовскую модель в Main (макрос @model раскрывается
    # с уже загруженным Turing; его реэкспорты Normal/MvNormal/etc.
    # доступны в Main).
    if !isdefined(Main, :_bssunfold_bayesian_model)
        try
            Base.eval(Main, quote
                Turing.@model function _bssunfold_bayesian_model(
                        A, b_abs, mu_prior, L_corr, lambda_prior,
                        sigma_prior, use_hierarchical, n_energy)
                    # Пространственная амплитуда лог-отклонений от центра приора
                    s ~ truncated(Normal(0.0, lambda_prior), 0.0, Inf)
                    # Белёное латентное поле; theta = mu + s*(L_corr*z) —
                    # нецентрированный MvNormal с OU-ковариацией.
                    z ~ MvNormal(zeros(n_energy), I)
                    theta = mu_prior .+ s .* (L_corr * z)
                    spectrum = exp.(theta)

                    # Шум правдоподобия: фиксированный относительный масштаб
                    # или оцениваемый иерархически.
                    if use_hierarchical === true
                        rel_noise ~ truncated(Normal(0.0, sigma_prior), 0.0, Inf)
                        sigma = rel_noise .* b_abs
                    else
                        sigma = sigma_prior .* b_abs
                    end

                    # Прямая модель (конвенция пакета: b = A * spectrum)
                    b ~ MvNormal(A * spectrum, Diagonal(sigma .^ 2))
                end
            end)
        catch err
            @warn "Не удалось определить байесовскую модель Turing" exception=err
            return false
        end
    end
    _TURING_LOADED[] = true
    return true
end

# ─── Математика приора ──────────────────────────────────────────────────────

"""
    _ou_correlation_cholesky(n_bins, lengthscale) -> Matrix{Float64}

Cholesky-фактор корреляционной матрицы Орнштейна-Уленбека.

OU-корреляция `C[i, j] = exp(-|i - j| / lengthscale)` даёт гладкие,
стационарные приорные выборки с ограниченной амплитудой (в отличие от
чистого случайного блуждания), что удерживает апостериорное распределение
сильно недоопределённой развёртки управляемым для NUTS.
"""
function _ou_correlation_cholesky(n_bins::Integer, lengthscale::Real)
    ls = max(Float64(lengthscale), 1e-9)
    corr = [exp(-abs(i - j) / ls) for i in 0:(n_bins - 1), j in 0:(n_bins - 1)]
    corr += 1e-9 * Matrix{Float64}(I, n_bins, n_bins)
    return cholesky(Symmetric(corr)).L
end

"""
    _prior_center(A, b, initial_spectrum, n_energy) -> Vector{Float64}

Data-driven центр лог-приора спектра.

Использует пользовательский `initial_spectrum`, если он задан (и содержит
хотя бы одно положительное значение), иначе неотрицательное
МНК-решение `A @ x = b`.  Возвращается в лог-масштабе
`log(max(x, eps))`, так что приор спектра `f = exp(theta)` заякорен
около спектра, согласованного с измерениями.
"""
function _prior_center(A::AbstractMatrix{<:Real}, b::AbstractVector{<:Real},
                      initial_spectrum::Union{Nothing,AbstractVector{<:Real}},
                      n_energy::Integer)
    local center::Vector{Float64}
    if initial_spectrum !== nothing
        c = max.(Float64.(collect(initial_spectrum)), 0.0)
        center = (length(c) == n_energy && any(>(0), c)) ? c : zeros(n_energy)
    else
        center = max.(qr(A, ColumnNorm()) \ b, 0.0)
    end
    return log.(max.(center, 1e-6))
end

# ─── HPD-интервал и диагностика сходимости ──────────────────────────────────

"""
    _hpd_interval(samples::AbstractMatrix, prob=0.95) -> (lower, upper)

Кратчайший (highest posterior density) интервал по столбцам выборок.

Вычисляется чистой Julia по оси выборок (ось 1), минуя различия
семантики `az.hdi` между версиями ArviZ.

# Возвращает
`(lower, upper)` — границы HPD-интервала, каждая длиной n_energy.
"""
function _hpd_interval(samples::AbstractMatrix{<:Real}, prob::Real=0.95)
    n_total, n_energy = size(samples)
    n_keep = max(ceil(Int, prob * n_total), 1)
    lower = Vector{Float64}(undef, n_energy)
    upper = Vector{Float64}(undef, n_energy)
    for j in 1:n_energy
        sorted = sort(samples[:, j])
        if n_keep >= n_total
            lower[j], upper[j] = sorted[1], sorted[end]
            continue
        end
        widths = sorted[n_keep:end] .- sorted[1:(n_total - n_keep + 1)]
        best_idx = argmin(widths)
        lower[j] = sorted[best_idx]
        upper[j] = sorted[best_idx + n_keep - 1]
    end
    return lower, upper
end

"""
    _split_rhat(samples::AbstractMatrix, n_chains::Int) -> Vector{Float64}

Split-R̂ диагностика сходимости Гельмана-Рубина по столбцам
(порт rhat из ArviZ в минимальном объёме).

`samples` — (n_total, n_energy) с цепями, соединёнными по строкам
(цепь за цепью).  Каждая цепь дополнительно делится пополам, что даёт
`2 * n_chains` подсегментов.
"""
function _split_rhat(samples::AbstractMatrix{<:Real}, n_chains::Integer)
    n_total, n_energy = size(samples)
    (n_chains >= 1 && n_total >= 8) || return fill(NaN, n_energy)
    d = n_total ÷ n_chains
    d < 4 && return fill(NaN, n_energy)
    half = d ÷ 2
    segments = Vector{Matrix{eltype(samples)}}()
    for ci in 1:n_chains
        base = (ci - 1) * d
        push!(segments, samples[(base + 1):(base + half), :])
        push!(segments, samples[(base + half + 1):(base + 2 * half), :])
    end
    k = length(segments)
    means = reduce(hcat, vec(mean(s, dims=1)) for s in segments)   # (n, k)
    vars_ = reduce(hcat, vec(var(s, dims=1)) for s in segments)    # (n, k)
    W = vec(mean(vars_, dims=2))                                   # внутрицепная
    B = half * vec(var(means, dims=2; corrected=true))             # межцепная
    var_hat = @. (half - 1) / half * W + B / half
    return sqrt.(max.(var_hat ./ max.(W, 1e-300), 0.0))
end

# ─── Основной солвер ────────────────────────────────────────────────────────

"""
    _extract_posterior_samples(chain, n_energy) -> (s_vec, z_mat)

Версионно-независимое извлечение выборок параметров `s` и `z` из цепи:

- MCMCChains (Turing <= 0.4x): `chain[:s]` → (draws, chains),
  `chain[:z]` → (draws, chains, n_energy);
- FlexiChains/VNChain (Turing >= 0.49): `chain[:s]` → DimMatrix
  (draws, chains), `chain[:z]` → DimMatrix (draws, chains) с векторным
  eltype; fallback — покомпонентные ключи `Symbol("z[j]")`.

Возвращает `(s_vec, z_mat)`: вектор амплитуд (n_total,) и матрицу
латентного поля (n_total × n_energy), цепи идут подряд по строкам.
"""
function _extract_posterior_samples(chain, n_energy)
    s_arr = parent(chain[:s])
    ndims(s_arr) == 1 && (s_arr = reshape(s_arr, :, 1))
    s_vec = vec(Float64.(s_arr))                     # (n_total,), цепи подряд

    local z_arr
    try
        z_arr = parent(chain[:z])
    catch
        z_arr = nothing
    end

    z_mat = if z_arr !== nothing && ndims(z_arr) == 3
        reshape(z_arr, size(z_arr, 1) * size(z_arr, 2), size(z_arr, 3))
    elseif z_arr !== nothing && ndims(z_arr) == 2 && eltype(z_arr) <: AbstractVector
        # FlexiChains: каждый элемент — вектор длины n_energy
        rows = [Float64.(collect(z_arr[i, c]))
                for c in 1:size(z_arr, 2) for i in 1:size(z_arr, 1)]
        reduce(vcat, r' for r in rows)
    else
        # Fallback: покомпонентные ключи "z[j]" (MCMCChains)
        cols = [vec(Float64.(parent(chain[Symbol("z[$j]")]))) for j in 1:n_energy]
        reduce(hcat, cols)
    end

    size(z_mat, 1) == length(s_vec) || error(
        "MCMC: рассогласование числа выборок s ($(length(s_vec))) и z ($(size(z_mat, 1)))")
    size(z_mat, 2) == n_energy || error(
        "MCMC: ожидалось $n_energy колонок 'z', получено $(size(z_mat, 2))")
    return s_vec, z_mat
end

"""
    _turing_sampling_pipeline(A, b_abs, mu_prior, L_corr, lambda_prior,
                              sigma_prior, use_hierarchical, n_energy,
                              n_samples, chains, target_accept) -> Matrix

Полный пайплайн Turing: конструкция модели → NUTS-сэмплирование →
извлечение апостериорных выборок спектра (n_total × n_energy).
Вызывается ТОЛЬКО через `Base.invokelatest` (Turing и модель загружаются
динамически; методы из нового world age недоступны из старого кадра).
"""
function _turing_sampling_pipeline(A, b_abs, mu_prior, L_corr, lambda_prior,
                                  sigma_prior, use_hierarchical, n_energy,
                                  n_samples, chains, target_accept)
    Turing = Main.Turing
    model = Main._bssunfold_bayesian_model(
        A, b_abs, mu_prior, L_corr, lambda_prior, sigma_prior,
        use_hierarchical, n_energy)

    # Сэмплирование NUTS
    chain = if chains > 1
        Turing.sample(model, Turing.NUTS(target_accept), Turing.MCMCThreads(),
                      n_samples, chains; progress=false)
    else
        Turing.sample(model, Turing.NUTS(target_accept), n_samples;
                      progress=false)
    end

    s_vec, z_mat = _extract_posterior_samples(chain, n_energy)

    # theta = mu_prior + s * (L_corr * z); spectrum = exp(theta)
    return exp.(mu_prior' .+ s_vec .* (z_mat * L_corr'))  # (n_total, n)
end

"""
    solve_mcmc(A, b, x0; sigma_prior=0.05, lambda_prior=0.5, lengthscale=3.0,
               n_samples=1000, tune=500, chains=2, target_accept=0.95,
               use_hierarchical=false, random_state=nothing) -> UnfoldResult

Решить задачу развёртки байесовским MCMC с сэмплером NUTS (Turing.jl).

Спектр моделируется в лог-масштабе со сглаживающим OU-приором,
заякоренным на центре `x0` (или МНК-решении, если `x0` тривиален).
Сэмплер NUTS генерирует выборки из апостериорного p(f|b), из которых
вычисляются статистики (среднее, медиана, std, HPD-интервалы, R̂).

# Аргументы
- `A::AbstractMatrix{T}`: ответная матрица (n_detectors × n_energy)
- `b::AbstractVector{T}`: измеренные показания (n_detectors,)
- `x0::AbstractVector{T}`: центр приора (n_energy,); тривиальный `x0`
  заменяется неотрицательным МНК-решением
- `sigma_prior`: относительный масштаб шума измерений (default 0.05)
- `lambda_prior`: масштаб амплитуды `s` лог-отклонений (default 0.5)
- `lengthscale`: OU-длина корреляции приора, в бинах (default 3.0)
- `n_samples`: число MCMC-выборок на цепь (default 1000)
- `tune`: число адаптационных выборок на цепь (default 500)
- `chains`: число независимых цепей (default 2)
- `target_accept`: целевой уровень принятия NUTS (default 0.95)
- `use_hierarchical`: оценивать масштаб шума из данных (default false)
- `random_state`: seed для воспроизводимости

# Возвращает
`UnfoldResult` со спектром — апостериорным средним; в `extra`:
`samples`, `mean`, `median`, `std`, `hpd_lower`, `hpd_upper`,
`rhat`, `rhat_max`.

Если Turing.jl не установлена, возвращает нулевой спектр с предупреждением
(установить: `Pkg.add("Turing")`).
"""
function solve_mcmc(A::AbstractMatrix{T}, b::AbstractVector{T}, x0::AbstractVector{T};
                   sigma_prior::Real=0.05,
                   lambda_prior::Real=0.5,
                   lengthscale::Real=3.0,
                   n_samples::Integer=1000,
                   tune::Integer=500,
                   chains::Integer=2,
                   target_accept::Real=0.95,
                   use_hierarchical::Bool=false,
                   random_state::Union{Integer,Nothing}=nothing) where T<:AbstractFloat
    m, n_energy = size(A)
    length(b) == m || throw(ArgumentError("b length ($(length(b))) must match A rows ($m)"))

    if !_try_load_turing()
        @warn "Turing.jl не установлена. Установите через: Pkg.add(\"Turing\"). " *
              "Возвращаю нулевой спектр."
        return UnfoldResult(zeros(T, n_energy), 0, false, T(0),
                            Dict{String,Any}("error" => "Turing.jl not available"))
    end

    # Центр приора и Cholesky-фактор OU-корреляции
    mu_prior = _prior_center(A, b, x0, n_energy)
    L_corr = _ou_correlation_cholesky(n_energy, lengthscale)
    b_abs = abs.(Float64.(b)) .+ 1e-6

    random_state !== nothing && Random.seed!(Int(random_state))

    # Весь пайплайн Turing (конструкция модели, сэмплирование, извлечение
    # выборок) выполняется через invokelatest: Turing и модель определены
    # ТОЛЬКО ЧТО через eval, и их методы недоступны из текущего world age.
    local samples
    try
        samples = Base.invokelatest(
            _turing_sampling_pipeline,
            Float64.(Matrix(A)), b_abs, mu_prior, L_corr,
            Float64(lambda_prior), Float64(sigma_prior),
            Bool(use_hierarchical), Int(n_energy),
            Int(n_samples), Int(chains), Float64(target_accept))
    catch err
        error("MCMC sampling failed: ", sprint(showerror, err))
    end

    mean_spec = vec(mean(samples, dims=1))
    median_spec = vec(median(samples, dims=1))
    std_spec = vec(std(samples, dims=1))
    hpd_lower, hpd_upper = _hpd_interval(samples, 0.95)
    rhat = _split_rhat(samples, chains)
    rhat_max = isempty(filter(isfinite, rhat)) ? NaN : maximum(filter(isfinite, rhat))

    residual = b .- A * mean_spec
    return UnfoldResult(
        Vector{T}(max.(mean_spec, 0.0)), Int(n_samples), true, T.(norm(residual)),
        Dict{String,Any}(
            "samples" => samples,
            "mean" => mean_spec,
            "median" => median_spec,
            "std" => std_spec,
            "hpd_lower" => hpd_lower,
            "hpd_upper" => hpd_upper,
            "rhat" => rhat,
            "rhat_max" => rhat_max,
            "n_chains" => Int(chains),
            "n_samples" => Int(n_samples),
        ))
end
