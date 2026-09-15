"""
N-spline unfolding method (порт из unfold_nspline.py).

Реализация подхода Р. Ф. Исламгулова и В. Д. Ларцева, «Восстановление
спектров нейтронов по активационным измерениям в виде N-сплайнов»,
Атомная энергия 104(5), 295-302 (май 2008) — РФЯЦ-ВНИИТФ им. Е. И. Забабахина.

Метод решает систему активационных интегралов

    Q_i = ∫ sigma_i(E) phi(E) dE,   i = 1..N                (ур. 1)

параметризацией искомого спектра phi(E) специализированным «нейтронным»
сплайном (N-сплайном) с базисными функциями

    N_k(E) = exp(a_k + q_k ln E + r_k E),  E_k <= E <= E_{k+1},
    k = 1..M                                                 (ур. 2)

т.е. кусочными функциями, логарифм которых линеен как по ln E, так и по E.
Это семейство содержит классические модельные спектры (1/E, максвелловская
эвапораторная exp(-E/T), делительный sqrt(E) exp(-bE), ...) как частные
случаи, поэтому базис близок к полному для реакторных и ускорительных
спектров, и весь спектр описывают лишь 3M параметров.

Реализованы три компонента статьи:

1. `build_continuity_matrix` / `fit_nspline` — сам N-сплайн.  Непрерывность
   C0/C1 в внутренних узлах (ур. 3-4) накладывается блочной матрицей D
   (ур. 5), а поточечная аппроксимация табулированного спектра (ур. 6-7)
   сводится к взвешенной линейной МНК в лог-домене при линейных
   ограничениях-равенствах D X = 0, X = (a, q, r)^T, через KKT-систему.

2. `solve_nspline_full` — цикл минимизации направленной дивергенции
   (обобщённый алгоритм MIRD Ларцева; Тараско, Препринт ФЭИ № 1446, 1983).
   С нормированными измеренными активациями p_i = Q_i / sum(Q) функционал

       H = Σ_i [pN_i ln(pN_i / p_i) - pN_i + p_i] >= 0   (ур. 8-9)

   (pN_i — нормированные расчётные активации) уменьшается
   сохраняющей флюенс градиентной итерацией

       phi_{n+1}(E) = phi_n(E) [1 - dmu_n (R_n(E) - Rbar_n)],

   где Rbar_n — взвешенное по флюенсу среднее R_n (сохраняет флюенс), а шаг
   dmu_n стартует с консервативного значения 0.1 / sup|R_n - Rbar_n| и
   уменьшается вдвое (backtracking), пока H не убудет.  После *каждой*
   итерации текущий спектр сглаживается пере-подборкой N-сплайна —
   ключевой регуляризационный приём статьи: итерация фактически действует
   на 3M параметров сплайна вместо n значений бинов.

3. Критерии остановки и контроль качества статьи: итерации останавливаются,
   когда H достигает уровня, соответствующего погрешностям измерений,
       H <= H_target = 0.5 * mean_i (dQ_i / Q_i)^2,
   или когда относительное убывание H за итерацию падает ниже `tol`.
   Пригодность восстановленного спектра измеряется среднеквадратичным
   остатком nev = sqrt(1/(N-1) Σ_i ((Qr_i - Q_i)/dQ_i)^2), приемлемым при
   nev <= 1 + 2/sqrt(N).

Наборы узлов для реакторов БАРС-5, ИГРИК (канал и поверхность) и ЯГУАР
заданы в `NSPLINE_KNOT_PRESETS`; `auto_knots` строит лог-равномерную сетку
по умолчанию.
"""

const _PHI_FLOOR = 1e-300  # абсолютный пол для положительных значений спектра
const _LOG_CLIP = 50.0     # клиппинг ln(pN/p) для подавления выбросов

"""
    NSPLINE_KNOT_PRESETS::Dict{String,Vector{Float64}}

Наборы узлов (ур. 2, МэВ) из статьи («Восстановление спектров реакторов
БАРС-5, ИГРИК, ЯГУАР»).
"""
const NSPLINE_KNOT_PRESETS = Dict{String,Vector{Float64}}(
    # Канал реактора БАРС-5
    "BARS5_channel" => [
        1e-10, 1.3e-7, 3.83e-7, 8e-6, 2e-5, 3e-5, 7.3e-5,
        3.2e-3, 0.38, 0.95, 7.0, 17.0, 20.0,
    ],
    # Канал реактора ИГРИК
    "IGRIK_channel" => [
        1e-10, 2e-8, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 1.5e-4,
        3e-4, 6e-4, 6e-3, 0.27, 1.0, 2.7, 7.0, 13.0, 20.0,
    ],
    # Поверхность реактора ИГРИК
    "IGRIK_surface" => [
        1e-10, 2e-8, 1e-7, 2e-7, 3e-6, 5e-6, 2.5e-4, 0.6,
        0.8, 1.5, 2.7, 7.0, 11.5, 14.0, 20.0,
    ],
    # Канал реактора ЯГУАР
    "YAGUAR_channel" => [
        1e-10, 2e-8, 1e-7, 6e-7, 1e-6, 3e-6, 1e-5, 4.3e-5,
        1.8e-4, 6.3e-4, 5e-3, 0.6, 0.8, 1.0, 2.5, 7.0, 11.0,
        13.0, 20.0,
    ],
)

# ─── Утилиты узлов ──────────────────────────────────────────────────────────

"""
    auto_knots(E_MeV; n_segments=12) -> Vector{Float64}

Построить лог-равномерную сетку узлов, покрывающую диапазон `E_MeV`
(`n_segments + 1` узлов от min(E) до max(E)).
"""
function auto_knots(E_MeV::AbstractVector{<:Real}; n_segments::Integer=12)
    Epos = filter(>(0), collect(Float64, E_MeV))
    length(Epos) >= 2 || throw(ArgumentError(
        "auto_knots requires at least two positive energy points, got $(length(Epos))"))
    emin, emax = extrema(Epos)
    (isfinite(emin) && isfinite(emax) && emin < emax) || throw(ArgumentError(
        "auto_knots requires finite min(E) < max(E), got [$emin, $emax]"))
    n_segments >= 1 || throw(ArgumentError("n_segments must be >= 1, got $n_segments"))
    # geomspace: лог-равномерная сетка
    return collect(emin .* (emax / emin) .^ range(0.0, 1.0, length=n_segments + 1))
end

function _resolve_knots(knots::Union{Nothing,String,AbstractVector{<:Real}},
                       E_MeV::Vector{Float64},
                       n_segments::Union{Nothing,Integer})
    Epos = filter(>(0), E_MeV)
    emin = isempty(Epos) ? 1e-10 : minimum(Epos)
    emax = maximum(E_MeV)

    local kn::Vector{Float64}
    local src::String
    if knots === nothing
        ns = n_segments === nothing ?
             min(12, max(4, length(E_MeV) ÷ 4)) : Int(n_segments)
        kn = auto_knots(E_MeV; n_segments=ns)
        src = "auto"
    elseif knots isa String
        haskey(NSPLINE_KNOT_PRESETS, knots) || throw(ArgumentError(
            "Unknown N-spline knot preset '$(knots)'. Available presets: " *
            join(sorted_keys(NSPLINE_KNOT_PRESETS), ", ")))
        kn = copy(NSPLINE_KNOT_PRESETS[knots])
        src = "preset:$(knots)"
    else
        kn = collect(Float64, knots)
        src = "user"
    end

    length(kn) >= 2 || throw(ArgumentError("N-spline needs at least 2 knots, got $(length(kn))"))
    all(>(0), diff(kn)) || throw(ArgumentError("N-spline knots must be strictly increasing"))

    # Клиппинг узлов к диапазону сетки и расширение внешних узлов так,
    # чтобы домен сплайна покрывал всю сетку.
    kn = clamp.(kn, emin, emax)
    kn = sort(unique(kn))
    kn[1] = min(kn[1], emin)
    kn[end] = max(kn[end], emax)
    if length(kn) < 2
        kn = [emin, emax]
    end
    return kn, src
end

sorted_keys(d::Dict{String,Vector{Float64}}) = sort!(collect(keys(d)))

"""
    _segment_indices(E, knots) -> Vector{Int}

Отобразить энергетические точки на индексы сегментов сплайна 0..M-1
(в Julia — 1..M).
"""
function _segment_indices(E::AbstractVector{<:Real}, knots::Vector{Float64})
    # searchsortedright: индекс последнего узла <= E, клиппинг к [1, M]
    k = searchsortedlast.(Ref(knots), E)  # knots[i] <= E
    return clamp.(k, 1, length(knots) - 1)
end

# ─── Определение N-сплайна (ур. 2-5) ────────────────────────────────────────

"""
    build_continuity_matrix(knots; continuity="C0C1") -> Matrix{Float64}

Построить матрицу непрерывности сплайна D (ур. 5).

Вектор параметров N-сплайна: `X = (a, q, r)^T` с `a = (a_1..a_M)`,
`q = (q_1..q_M)`, `r = (r_1..r_M)`.  Условия непрерывности в внутренних
узлах (ур. 3-4):

    C0: a_k - a_{k+1} + u (q_k - q_{k+1}) + E (r_k - r_{k+1}) = 0
    C1: (q_k - q_{k+1}) + E (r_k - r_{k+1}) = 0,   u = ln E

собираются в `D = [[A, B, C], [0, A, C]]`, `D X = 0`.

`continuity`: `"C0C1"` (по умолчанию) — непрерывность значения и производной;
`"C0"` — только значения; `"none"` — без непрерывности.
"""
function build_continuity_matrix(knots::AbstractVector{<:Real};
                                continuity::AbstractString="C0C1")
    kn = Float64.(collect(knots))
    M = length(kn) - 1
    M >= 1 || throw(ArgumentError("knots must contain at least 2 values"))
    cont = uppercase(replace(continuity, " " => ""))
    cont in ("C0C1", "C0", "NONE") || throw(ArgumentError(
        "continuity must be one of 'C0C1', 'C0', 'none', got '$(continuity)'"))

    n_int = M - 1  # внутренние узлы
    (cont == "NONE" || n_int == 0) && return zeros(0, 3 * M)

    rows_c0 = cont == "C0C1"
    n_rows = rows_c0 ? 2 * n_int : n_int
    D = zeros(n_rows, 3 * M)
    for k in 1:n_int
        Ek = kn[k + 1]
        uk = log(Ek)
        # C0-строка: a_k - a_{k+1} + u(q_k - q_{k+1}) + E(r_k - r_{k+1}) = 0
        D[k, k] = -1.0
        D[k, k + 1] = 1.0
        D[k, M + k] = -uk
        D[k, M + k + 1] = uk
        D[k, 2M + k] = -Ek
        D[k, 2M + k + 1] = Ek
        if rows_c0
            # C1-строка: (q_k - q_{k+1}) + E(r_k - r_{k+1}) = 0
            row = n_int + k
            D[row, M + k] = -1.0
            D[row, M + k + 1] = 1.0
            D[row, 2M + k] = -Ek
            D[row, 2M + k + 1] = Ek
        end
    end
    return D
end

"""
    nspline_eval(E, a, q, r, knots) -> Vector{Float64}

Вычислить N-сплайн `N(E) = exp(a_k + q_k ln E + r_k E)`.

# Аргументы
- `E`: энергии (МэВ), строго положительные
- `a, q, r`: векторы длины M (число сегментов)
- `knots`: M+1 значений узлов
"""
function nspline_eval(E::AbstractVector{<:Real},
                     a::AbstractVector{<:Real},
                     q::AbstractVector{<:Real},
                     r::AbstractVector{<:Real},
                     knots::AbstractVector{<:Real})
    any(<=(0), E) && throw(ArgumentError("nspline_eval requires strictly positive energies"))
    M = length(knots) - 1
    (length(a) == length(q) == length(r) == M) || throw(ArgumentError(
        "a, q, r must all have length M=$(M) segments, got " *
        "$(length(a)), $(length(q)), $(length(r))"))
    kseg = _segment_indices(E, Float64.(collect(knots)))
    Ef = Float64.(collect(E))
    return [exp(a[k] + q[k] * log(Ef[i]) + r[k] * Ef[i]) for (i, k) in enumerate(kseg)]
end

"""
    directed_divergence(p_calc, p_meas) -> Float64

Направленная (типа Кульбака-Лейблера) дивергенция ур. (8-9):

    H = Σ_i [pN_i ln(pN_i / p_i) - pN_i + p_i] >= 0,

H = 0 тогда и только тогда, когда расчётные активации равны измеренным.
"""
function directed_divergence(p_calc::AbstractVector{<:Real},
                            p_meas::AbstractVector{<:Real})
    pN = max.(Float64.(p_calc), 1e-300)
    p = max.(Float64.(p_meas), 1e-300)
    return sum(@. pN * log(pN / p) - pN + p)
end

# ─── Поточечная N-сплайн аппроксимация (ур. 6-7) ────────────────────────────

"""
    fit_nspline(E, phi; knots=nothing, rel_err=nothing, continuity="C0C1",
                n_segments=nothing) -> (N_E, info)

Аппроксимация поточечного спектра N-сплайном (ур. 2, 5-7).

Решает взвешенную МНК-задачу в лог-домене с ограничениями непрерывности:

    min_X Σ_j w_j^2 (a_kj + u_j q_kj + E_j r_kj - ln phi_j)^2
    s.t.  D X = 0,   w_j = 1 / eps_j,

через KKT-систему (множители Лагранжа)

    [[G^T W G, D^T], [D, 0]] [X; lam] = [G^T W Y; 0].

# Возвращает
`(N_E, info)`, где `N_E` — подобранный сплайн на `E`, а `info` содержит
`knots`, `knots_source`, `a`/`q`/`r`, `log_rms_residual`, `continuity`.
"""
function fit_nspline(E::AbstractVector{<:Real},
                    phi::AbstractVector{<:Real};
                    knots::Union{Nothing,String,AbstractVector{<:Real}}=nothing,
                    rel_err::Union{Nothing,AbstractVector{<:Real}}=nothing,
                    continuity::AbstractString="C0C1",
                    n_segments::Union{Nothing,Integer}=nothing)
    E_arr = Float64.(collect(E))
    phi_arr = Float64.(collect(phi))
    length(E_arr) == length(phi_arr) || throw(ArgumentError(
        "E and phi length mismatch: $(length(E_arr)) vs $(length(phi_arr))"))
    all(>(0), E_arr) || throw(ArgumentError("fit_nspline requires strictly positive energies"))
    length(phi_arr) >= 3 || throw(ArgumentError("fit_nspline requires at least 3 spectrum points"))

    kn, src = _resolve_knots(knots, E_arr, n_segments)
    M = length(kn) - 1
    n = length(E_arr)

    # Точка -> сегмент и лог-доменная матрица проектирования G (n x 3M).
    kseg = _segment_indices(E_arr, kn)
    u = log.(E_arr)
    G = zeros(n, 3M)
    for i in 1:n
        k = kseg[i]
        G[i, k] = 1.0
        G[i, M + k] = u[i]
        G[i, 2M + k] = E_arr[i]
    end

    # Пол пола для микроскопических/нулевых бинов с ослаблением веса.
    phi_max = maximum(phi_arr)
    tiny = max(_PHI_FLOOR, 1e-12 * phi_max)
    floored = phi_arr .< tiny
    y = log.([f ? tiny : v for (f, v) in zip(floored, phi_arr)])

    w = rel_err === nothing ? ones(n) :
        1.0 ./ max.(Float64.(collect(rel_err)), 1e-12)
    w = [f ? 1e-3 * wi : wi for (f, wi) in zip(floored, w)]  # сильный относительный штраф веса

    D = build_continuity_matrix(kn; continuity=continuity)
    nc = size(D, 1)

    # Взвешенные нормальные уравнения + KKT-блок ограничений.
    # Рида не добавляем: pivoted-QR решение возвращает решение
    # минимальной нормы для рангово-дефицитных систем (пустые сегменты).
    Gw = G .* w
    yw = y .* w
    H_norm = Gw' * Gw
    KKT = zeros(3M + nc, 3M + nc)
    KKT[1:3M, 1:3M] .= H_norm
    if nc > 0
        KKT[1:3M, 3M+1:end] .= D'
        KKT[3M+1:end, 1:3M] .= D
    end
    rhs = vcat(Gw' * yw, zeros(nc))

    sol = qr(KKT, ColumnNorm()) \ rhs
    X = sol[1:3M]

    N_E = exp.(G * X)
    resid = w .* (G * X .- y)
    rms = sqrt(mean(resid .^ 2)) / max(mean(w), 1e-300)

    info = Dict{String,Any}(
        "knots" => kn,
        "knots_source" => src,
        "continuity" => continuity,
        "a" => X[1:M],
        "q" => X[M+1:2M],
        "r" => X[2M+1:3M],
        "log_rms_residual" => rms,
    )
    return N_E, info
end

# ─── Трапеция ───────────────────────────────────────────────────────────────

function _trapz(y::AbstractVector{<:Real}, x::AbstractVector{<:Real})
    n = length(y)
    n == length(x) || throw(ArgumentError("y and x must have the same length"))
    n < 2 && return 0.0
    s = 0.0
    @inbounds for i in 1:(n - 1)
        s += (y[i] + y[i + 1]) * (x[i + 1] - x[i])
    end
    return s / 2.0
end

# ─── Развёртка направленной дивергенции с поитерационным N-сплайн
#     сглаживанием ──────────────────────────────────────────────────────────

"""
    solve_nspline_full(A, b, x0, E_MeV; knots=nothing, sigma_rel=nothing,
                       continuity="C0C1", max_iterations=200, tol=1e-3,
                       step_theta=0.1, smoothing=true, n_segments=nothing) -> Dict

Полная N-сплайн развёртка с диагностикой (Исламгулов & Ларцев, 2008).

Итеративно минимизирует направленную дивергенцию H между измеренными и
расчётными нормированными активациями, сглаживая спектр подгонкой N-сплайна
на каждой итерации (регуляризация статьи).  Использует критерии остановки
статьи (H на уровне погрешности измерений или остановившееся относительное
убывание) и возвращает статистику остатка `nev` с границей приемлемости
`nev <= 1 + 2/sqrt(N)`.

# Аргументы
- `A::AbstractMatrix`: ответная матрица активационных детекторов (m, n)
- `b::AbstractVector`: измеренные показания / активационные интегралы (m,)
- `x0`: начальная догадка спектра (n,); `nothing` — плоский спектр
- `E_MeV`: энергетическая сетка (МэВ), строго положительная (обязательный)
- `knots`: имя пресета (`NSPLINE_KNOT_PRESETS`), явная последовательность
  или `nothing` (авто лог-сетка)
- `sigma_rel`: относительные погрешности измерений dQ_i/Q_i (m,);
  `nothing` — 0.1 для каждого детектора
- `continuity`: `"C0C1"` (по умолчанию), `"C0"` или `"none"`
- `max_iterations`: бюджет итераций (по умолчанию 200)
- `tol`: порог относительного убывания H (по умолчанию 1e-3)
- `step_theta`: консервативный начальный фактор шага: dmu = step_theta /
  sup|R-Rbar| (значение статьи 0.1); backtracking делит пополам, пока H растёт
- `smoothing`: пере-подбирать N-сплайн после каждой итерации
  (по умолчанию true, процедура статьи; `false` сводит к обычному циклу MIRD)
- `n_segments`: число сегментов при `knots=nothing`

# Возвращает
`Dict{String,Any}` с ключами `spectrum`, `iterations`, `converged`,
`stop_reason`, `H`, `H_history`, `H_target`, `nev`, `nev_limit`,
`acceptable`, `Qr`, `relative_residuals`, `fluence`, `mean_energy`,
`knots`, `knots_source`, `continuity`, `params`.
"""
function solve_nspline_full(A::AbstractMatrix{<:Real},
                           b::AbstractVector{<:Real},
                           x0::Union{Nothing,AbstractVector{<:Real}},
                           E_MeV::AbstractVector{<:Real};
                           knots::Union{Nothing,String,AbstractVector{<:Real}}=nothing,
                           sigma_rel::Union{Nothing,AbstractVector{<:Real}}=nothing,
                           continuity::AbstractString="C0C1",
                           max_iterations::Integer=200,
                           tol::Real=1e-3,
                           step_theta::Real=0.1,
                           smoothing::Bool=true,
                           n_segments::Union{Nothing,Integer}=nothing)
    A_arr = Float64.(Matrix(A))
    b_arr = Float64.(collect(b))
    m, n = size(A_arr)
    length(b_arr) == m || throw(ArgumentError(
        "b length ($(length(b_arr))) does not match A rows ($m)"))
    m >= 1 || throw(ArgumentError("At least one measurement is required"))
    E = Float64.(collect(E_MeV))
    length(E) == n || throw(ArgumentError(
        "E_MeV length ($(length(E))) does not match A columns ($n)"))
    all(>(0), E) || throw(ArgumentError("E_MeV must contain strictly positive energies"))
    max_iterations >= 1 || throw(ArgumentError("max_iterations must be >= 1, got $max_iterations"))
    0 < step_theta <= 1 || throw(ArgumentError("step_theta must be in (0, 1], got $step_theta"))
    tol > 0 || throw(ArgumentError("tol must be positive, got $tol"))

    # Детекторы с положительными показаниями (нулевые измерения не несут
    # информации для минимизации дивергенции).
    valid = b_arr .> 0
    any(valid) || throw(ArgumentError(
        "solve_nspline requires at least one positive measurement"))
    A_v = A_arr[valid, :]
    b_v = b_arr[valid]

    sigma_v = sigma_rel === nothing ? fill(0.1, length(b_v)) :
              max.(Float64.(collect(sigma_rel))[valid], 1e-12)

    kn, knot_src = _resolve_knots(knots, E, n_segments)

    # Нормированные измеренные активации и H-цель статьи: ожидаемая
    # направленная дивергенция, когда все расчётные активации отстоят
    # на 1 сигму от измерений, E[H] ~ 0.5 Σ_i p_i delta_i^2.
    p = b_v ./ sum(b_v)
    H_target = 0.5 * sum(p .* sigma_v .^ 2)

    # Начальный спектр: пересчёт x0 к измеренному полному отклику, затем
    # N-сплайн сглаживание (в духе статьи — сплайн MC-спектра как
    # начальная аппроксимация).
    x = x0 === nothing ? ones(n) :
        [isfinite(v) ? max(v, 0.0) : 0.0 for v in Float64.(collect(x0))]
    length(x) == n || throw(ArgumentError("x0 length ($(length(x))) does not match A columns ($n)"))
    sum(x) <= 0 && (x = ones(n))

    Qc0 = A_v * x
    scale = sum(b_v) / max(sum(Qc0), 1e-300)
    x = max.(x .* scale, _PHI_FLOOR)

    # Поточечные относительные ошибки для поитерационных сглаживаний
    # (вес w = 1/eps статьи, ур. 7): бины с низкой суммарной
    # чувствительностью детекторов несут меньше информации и получают
    # пропорционально большие предполагаемые ошибки (пуассоновская
    # sqrt-масштабировка), чтобы не тянуть сплайн.
    sens = vec(sum(A_v, dims=1))
    sens_max = isempty(sens) ? 0.0 : maximum(sens)
    smooth_rel_err = (smoothing && sens_max > 0) ?
        sqrt.(clamp.(sens_max ./ max.(sens, 1e-300), 1.0, 1e12)) : nothing

    if smoothing
        x, fit_info = fit_nspline(E, x; knots=kn, rel_err=smooth_rel_err,
                                  continuity=continuity)
        x = max.(x, _PHI_FLOOR)
        # Сохраняем масштаб активаций после shape-only сплайн-подборки.
        x .*= sum(b_v) / max(sum(A_v * x), 1e-300)
    else
        fit_info = Dict{String,Any}()
    end

    b_total = sum(b_v)
    eps_scale = 1e-12 * max(b_total, 1e-300)

    _gauge(xx) = xx .* (b_total / max(sum(A_v * xx), 1e-300))

    function _state(xx)
        xx = max.(xx, _PHI_FLOOR)
        Qc_ = max.(A_v * xx, eps_scale)
        pN_ = Qc_ ./ max(sum(Qc_), 1e-300)
        H_ = directed_divergence(pN_, p)
        return xx, Qc_, pN_, H_
    end

    x, _Qc, pN, H = _state(_gauge(x))
    H_history = [H]
    converged = false
    stop_reason = "max_iterations"
    iterations = 0

    if H <= H_target
        converged = true
        stop_reason = "H_target (initial)"
    end

    for iteration in 1:max_iterations
        iterations = iteration

        # Градиент H по спектру (с точностью до константы 1/sum(Q)):
        # R(E) = Σ_i (p_i / Q_i) sigma_i(E) ln(pN_i / p_i).
        ln_ratio = clamp.(log.(pN ./ p), -_LOG_CLIP, _LOG_CLIP)
        R = (A_v' * ln_ratio) ./ sum(b_v)
        x_sum = sum(x)
        Rbar = dot(x, R) / max(x_sum, 1e-300)
        g = R .- Rbar
        g_max = maximum(abs.(g))
        if !isfinite(g_max) || g_max <= 0.0
            stop_reason = "stalled_gradient"
            iterations -= 1
            break
        end

        # Консервативный шаг статьи (dmu0 = 0.1 / sup|R - Rbar|) с
        # backtracking-делением пополам, пока H не перестанет расти.
        mu = step_theta / g_max
        accepted = false
        local x_new, Qc_new, pN_new, H_new
        for _bt in 1:60
            x_trial = x .* (1.0 .- mu .* g)
            if smoothing
                x_trial, _ = fit_nspline(E, x_trial; knots=kn,
                                         rel_err=smooth_rel_err,
                                         continuity=continuity)
            end
            # Фиксация масштаба активаций после shape-only обновления.
            x_new, Qc_new, pN_new, H_new = _state(_gauge(x_trial))
            if isfinite(H_new) && H_new <= H + 1e-4 * max(H, 1e-300)
                accepted = true
                break
            end
            mu *= 0.5
        end
        if !accepted
            stop_reason = "no_further_reduction"
            iterations -= 1
            break
        end

        H_prev = H
        x, _Qc, pN, H = x_new, Qc_new, pN_new, H_new
        push!(H_history, H)

        # Критерии остановки статьи.
        if H <= H_target
            converged = true
            stop_reason = "H_target"
            break
        end
        if abs(H_prev - H) <= tol * max(H_prev, 1e-300)
            converged = true
            stop_reason = "relative_change"
            break
        end
    end

    # Статистика приемлемости статьи: nev = RMS((Qr - Q)/dQ),
    # приемлемо при nev <= 1 + 2/sqrt(N).
    Qr_full = A_arr * x
    rel_res = zeros(m)
    denom = max.(sigma_v .* b_v, 1e-300)
    rel_res[valid] .= (Qr_full[valid] .- b_v) ./ denom
    cnt = count(valid)
    div = cnt > 1 ? cnt - 1 : cnt
    nev = sqrt(sum(rel_res[valid] .^ 2) / max(div, 1))
    nev_limit = 1.0 + 2.0 / sqrt(cnt)
    acceptable = nev <= nev_limit

    fluence = _trapz(x, E)
    mean_energy = fluence > 0 ? _trapz(E .* x, E) / fluence : NaN

    # Финальная сплайн-параметризация восстановленного спектра.
    M = length(kn) - 1
    if haskey(fit_info, "a")
        params = Dict{String,Any}("a" => fit_info["a"], "q" => fit_info["q"],
                                  "r" => fit_info["r"])
    else
        kseg = _segment_indices(E, kn)
        G = zeros(n, 3M)
        for i in 1:n
            k = kseg[i]
            G[i, k] = 1.0
            G[i, M + k] = log(E[i])
            G[i, 2M + k] = E[i]
        end
        Xl = qr(G, ColumnNorm()) \ log.(max.(x, _PHI_FLOOR))
        params = Dict{String,Any}("a" => Xl[1:M], "q" => Xl[M+1:2M],
                                  "r" => Xl[2M+1:3M])
    end

    return Dict{String,Any}(
        "spectrum" => x,
        "iterations" => iterations,
        "converged" => converged,
        "stop_reason" => stop_reason,
        "H" => H,
        "H_history" => H_history,
        "H_target" => H_target,
        "nev" => nev,
        "nev_limit" => nev_limit,
        "acceptable" => acceptable,
        "Qr" => Qr_full,
        "relative_residuals" => rel_res,
        "fluence" => fluence,
        "mean_energy" => mean_energy,
        "knots" => kn,
        "knots_source" => knot_src,
        "continuity" => continuity,
        "params" => params,
    )
end

"""
    solve_nspline(A, b, x0; E_MeV, knots=nothing, sigma_rel=nothing,
                  continuity="C0C1", max_iterations=200, tol=1e-3,
                  step_theta=0.1, smoothing=true, n_segments=nothing) -> UnfoldResult

Стандартный-API солвер N-сплайн метода: тонкая обёртка над
[`solve_nspline_full`](@ref), возвращающая `UnfoldResult` со спектром,
числом итераций и флагом сходимости.

# Аргументы
- `A::AbstractMatrix{T}`: ответная матрица (m × n)
- `b::AbstractVector{T}`: измерения (m,)
- `x0::AbstractVector{T}`: начальная догадка спектра (n,)
- `E_MeV`: энергетическая сетка (МэВ), строго положительная — **обязательный**
  keyword (обёртка `unfold_nspline` подставляет сетку детектора автоматически)
- остальные ключевые аргументы идентичны `solve_nspline_full`
"""
function solve_nspline(A::AbstractMatrix{T}, b::AbstractVector{T}, x0::AbstractVector{T};
                      E_MeV::Union{Nothing,AbstractVector{<:Real}}=nothing,
                      knots::Union{Nothing,String,AbstractVector{<:Real}}=nothing,
                      sigma_rel::Union{Nothing,AbstractVector{<:Real}}=nothing,
                      continuity::AbstractString="C0C1",
                      max_iterations::Integer=200,
                      tol::Real=1e-3,
                      step_theta::Real=0.1,
                      smoothing::Bool=true,
                      n_segments::Union{Nothing,Integer}=nothing) where T<:AbstractFloat
    E_MeV === nothing && throw(ArgumentError(
        "E_MeV (energy grid in MeV) is required for solve_nspline"))
    result = solve_nspline_full(A, b, x0, E_MeV;
                                knots=knots, sigma_rel=sigma_rel,
                                continuity=continuity,
                                max_iterations=max_iterations, tol=tol,
                                step_theta=step_theta, smoothing=smoothing,
                                n_segments=n_segments)
    spectrum = Vector{T}(result["spectrum"])
    res = b .- A * spectrum
    return UnfoldResult(spectrum, result["iterations"], result["converged"],
                        norm(res),
                        Dict{String,Any}(
                            "stop_reason" => result["stop_reason"],
                            "H" => result["H"],
                            "H_target" => result["H_target"],
                            "nev" => result["nev"],
                            "nev_limit" => result["nev_limit"],
                            "acceptable" => result["acceptable"],
                            "fluence" => result["fluence"],
                            "mean_energy" => result["mean_energy"],
                            "knots_source" => result["knots_source"],
                        ))
end
