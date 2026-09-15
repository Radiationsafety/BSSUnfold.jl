"""
N-spline unfolding (Islamgulov & Lartsev, Atomic Energy 104(5), 295-302, 2008).

Спектр параметризуется «нейтронным» (N-) сплайном
`N_k(E) = exp(a_k + q_k ln E + r_k E)`, E в [E_k, E_{k+1}] (Eq. 2) —
семейство, содержащее 1/E, максвеллову, фиссионную и двухкомпонентные
модели.  C0/C1-непрерывность во внутренних узлах (Eqs. 3-4) задаётся
блок-матрицей D (Eq. 5); кусочная аппроксимация табличного спектра
(Eqs. 6-7) — взвешенный LS в лог-домене с линейными равенствами,
решается через KKT-систему.  Развёртка `solve_nspline_full` — цикл
минимизации направленной дивергенции (MIRD) с N-сплайн-сглаживанием
после каждой итерации, критериями останова по H_target и относительному
уменьшению H, стат ``nev`` и порогом `1 + 2/sqrt(N)`.

Пресеты узлов (MeV) для реакторов BARS-5, IGRIK, YAGUAR —
`NSPLINE_KNOT_PRESETS`; `auto_knots` строит лог-равномерную сетку.
"""
const NSPLINE_KNOT_PRESETS = Dict{String,Tuple{Float64,Vararg{Float64}}}(
    "BARS5_channel" => (1e-10, 1.3e-7, 3.83e-7, 8e-6, 2e-5, 3e-5, 7.3e-5,
                        3.2e-3, 0.38, 0.95, 7.0, 17.0, 20.0),
    "IGRIK_channel" => (1e-10, 2e-8, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 1.5e-4,
                        3e-4, 6e-4, 6e-3, 0.27, 1.0, 2.7, 7.0, 13.0, 20.0),
    "IGRIK_surface" => (1e-10, 2e-8, 1e-7, 2e-7, 3e-6, 5e-6, 2.5e-4, 0.6,
                        0.8, 1.5, 2.7, 7.0, 11.5, 14.0, 20.0),
    "YAGUAR_channel" => (1e-10, 2e-8, 1e-7, 6e-7, 1e-6, 3e-6, 1e-5, 4.3e-5,
                         1.8e-4, 6.3e-4, 5e-3, 0.6, 0.8, 1.0, 2.5, 7.0, 11.0,
                         13.0, 20.0),
)

const NSPLINE_PHI_FLOOR = 1e-300
const NSPLINE_LOG_CLIP = 50.0

"""
    nspline_auto_knots(E_MeV; n_segments=12) -> Vector{Float64}

Лог-равномерная сетка из `n_segments + 1` узлов от min(E) до max(E)
(требуются хотя бы две положительные энергии).
"""
function nspline_auto_knots(E::AbstractVector{<:Real}; n_segments::Integer=12)
    Epos = filter(>(0), collect(Float64.(E)))
    length(Epos) >= 2 || throw(ArgumentError("auto_knots requires at least two positive energy points, got $(length(Epos))"))
    emin, emax = extrema(Epos)
    isfinite(emin) && isfinite(emax) && emin < emax ||
        throw(ArgumentError("auto_knots requires finite min(E) < max(E), got [$emin, $emax]"))
    Int(n_segments) >= 1 || throw(ArgumentError("n_segments must be >= 1, got $n_segments"))
    return collect(10.0 .^ range(log10(emin), log10(emax), length=Int(n_segments) + 1))
end

const auto_knots = nspline_auto_knots

function _nspline_resolve_knots(knots, E::Vector{Float64}, n_segments::Union{Nothing,Integer})
    emin =isempty([e for e in E if e > 0]) ? 1e-10 : minimum(filter(>(0), E))
    emax = isempty(E) ? 1e2 : maximum(E)

    if knots === nothing
        ns = n_segments === nothing ? min(12, max(4, length(E) ÷ 4)) : Int(n_segments)
        return nspline_auto_knots(E; n_segments=ns), "auto"
    end

    if knots isa AbstractString
        key = strip(String(knots))
        haskey(NSPLINE_KNOT_PRESETS, key) ||
            throw(ArgumentError("Unknown N-spline knot preset '$key'. Available presets: $(sort(collect(keys(NSPLINE_KNOT_PRESETS))))"))
        src = "preset:$key"
        kn = collect(Float64.(NSPLINE_KNOT_PRESETS[key]))
    else
        src = "user"
        kn = collect(Float64.(knots))
    end

    length(kn) >= 2 || throw(ArgumentError("N-spline needs at least 2 knots, got $(length(kn))"))
    all(diff(kn) .> 0) || throw(ArgumentError("N-spline knots must be strictly increasing"))

    kn = clamp.(kn, emin, emax)
    kn = sort(unique(kn))
    kn[1] > emin && (kn[1] = emin)
    kn[end] < emax && (kn[end] = emax)
    length(kn) < 2 && (kn = [emin, emax])
    return kn, src
end

function _nspline_segment_indices(E::Vector{Float64}, knots::Vector{Float64})
    k = [searchsortedlast(knots, e) for e in E]
    return clamp.(k, 1, length(knots) - 1)
end

"""
    nspline_build_continuity_matrix(knots; continuity="C0C1") -> Matrix{Float64}

Матрица непрерывности D (Eq. 5) для параметров `X = (a, q, r)`,
`(rows, 3M)`: C0 — непрерывность значения, C1 — производной
во внутренних узлах.  `continuity` принимает `"C0C1"` (по умолчанию),
`"C0"`, `"fehlen"`/`"none"`.
"""
function nspline_build_continuity_matrix(knots::AbstractVector{<:Real}; continuity::String="C0C1")
    kn = collect(Float64.(knots))
    M = length(kn) - 1
    M >= 1 || throw(ArgumentError("knots must contain at least 2 values"))
    cont = uppercase(replace(String(continuity), " " => ""))
    cont in ("C0C1", "C0", "NONE") ||
        throw(ArgumentError("continuity must be one of 'C0C1', 'C0', 'none', got '$continuity'"))

    n_int = M - 1
    (cont == "NONE" || n_int == 0) && return zeros(0, 3 * M)

    rows_c0 = cont == "C0C1"
    n_rows = rows_c0 ? 2 * n_int : n_int
    D = zeros(n_rows, 3 * M)
    for k in 1:n_int
        Ek = kn[k+1]
        uk = log(Ek)
        D[k, k] = -1.0
        D[k, k+1] = 1.0
        D[k, M + k] = -uk
        D[k, M + k + 1] = uk
        D[k, 2M + k] = -Ek
        D[k, 2M + k + 1] = Ek
        if rows_c0
            row = n_int + k
            D[row, M + k] = -1.0
            D[row, M + k + 1] = 1.0
            D[row, 2M + k] = -Ek
            D[row, 2M + k + 1] = Ek
        end
    end
    return D
end

const build_continuity_matrix = nspline_build_continuity_matrix

"""
    nspline_eval(E, a, q, r, knots) -> Vector{Float64}

Значения N-сплайна `N(E) = exp(a_k + q_k ln E + r_k E)` на сетке E
(padausk segment mapping).  E должно быть строго положительно.
"""
function nspline_eval(E::AbstractVector{<:Real}, a::AbstractVector{<:Real},
                      q::AbstractVector{<:Real}, r::AbstractVector{<:Real},
                      knots::AbstractVector{<:Real})
    E_arr = Float64.(collect(E))
    all(>(0), E_arr) || throw(ArgumentError("nspline_eval requires strictly positive energies"))
    a_arr = Float64.(collect(a)); q_arr = Float64.(collect(q)); r_arr = Float64.(collect(r))
    M = length(knots) - 1
    length(a_arr) == length(q_arr) == length(r_arr) == M ||
        throw(ArgumentError("a, q, r must all have length M=$M segments, got $(length(a_arr)), $(length(q_arr)), $(length(r_arr))"))
    kseg = _nspline_segment_indices(E_arr, collect(Float64.(knots)))
    return @. exp(a_arr[kseg] + q_arr[kseg] * log(clamp(E_arr, 1e-300, Inf)) + r_arr[kseg] * E_arr)
end

"""
    directed_divergence(p_calc, p_meas) -> Float64

Направленная (Kullback-Leibler) дивергенция Eq. 8-9:
`H = sum pN ln(pN/p) - pN + p >= 0`; H = 0 <=> рассчитанные активации
равны измеренным.
"""
function directed_divergence(p_calc::AbstractVector{<:Real}, p_meas::AbstractVector{<:Real})
    pN = max.(Float64.(collect(p_calc)), 1e-300)
    p = max.(Float64.(collect(p_meas)), 1e-300)
    return sum(@. pN * log(pN / p) - pN + p)
end

"""
    nspline_fit(E, phi; knots=nothing, rel_err=nothing, continuity="C0C1",
                n_segments=nothing) -> (N_E, info::Dict{String,Any})

Кусочная аппроксимация спектра N-сплайном (Eqs. 2, 5-7): взвешенный
LS в лог-домене с ограничениями непрерывности через KKT-систему
`[[G^T W G, D^T], [D, 0]] X = [G^T W Y; 0]`.  `knots` — None (авто),
имя пресета или явная сетка; `rel_err` — относительные ошибки корзин
(веса w = 1/eps).  Возвращает сплайн-значения `N_E` и info с
параметрами `a/q/r`, узлами и RMS остатка.
"""
function nspline_fit(E::AbstractVector{<:Real}, phi::AbstractVector{<:Real};
                     knots=nothing, rel_err=nothing, continuity::String="C0C1",
                     n_segments::Union{Nothing,Integer}=nothing)
    E_arr = Float64.(collect(E))
    phi_arr = Float64.(collect(phi))
    length(E_arr) == length(phi_arr) ||
        throw(ArgumentError("E and phi length mismatch: $(length(E_arr)) vs $(length(phi_arr))"))
    all(>(0), E_arr) || throw(ArgumentError("fit_nspline requires strictly positive energies"))
    length(phi_arr) >= 3 || throw(ArgumentError("fit_nspline requires at least 3 spectrum points"))

    kn, src = _nspline_resolve_knots(knots, E_arr, n_segments)
    M = length(kn) - 1
    nel = length(E_arr)

    kseg = _nspline_segment_indices(E_arr, kn)
    u = log.(E_arr)
    G = zeros(nel, 3M)
    for j in 1:nel
        k = kseg[j]
        G[j, k] = 1.0
        G[j, M + k] = u[j]
        G[j, 2M + k] = E_arr[j]
    end

    phi_max = maximum(phi_arr)
    tiny = max(NSPLINE_PHI_FLOOR, 1e-12 * phi_max)
    floored = phi_arr .< tiny
    y = log.(max.(phi_arr, tiny))

    w = rel_err === nothing ? ones(nel) : 1.0 ./ max.(Float64.(collect(rel_err)), 1e-12)
    w = [floored[j] ? 1e-3 * w[j] : w[j] for j in 1:nel]

    Dmat = nspline_build_continuity_matrix(kn; continuity=continuity)
    nc = size(Dmat, 1)

    Gw = G .* w
    yw = y .* w
    H_norm = Gw' * Gw
    KKT = zeros(3M + nc, 3M + nc)
    KKT[1:3M, 1:3M] .= H_norm
    if nc > 0
        KKT[1:3M, 3M+1:end] .= Dmat'
        KKT[3M+1:end, 1:3M] .= Dmat
    end
    rhs = vcat(Gw' * yw, zeros(nc))

    sol = pinv(KKT) * rhs
    X = sol[1:3M]

    N_E = exp.(G * X)
    resid = w .* (G * X .- y)
    rms = sqrt(sum(abs2, resid) / nel) / max(sum(w) / nel, 1e-300)

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

const fit_nspline = nspline_fit

function _nspline_trapz(x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    s = 0.0
    for i in 1:(length(x)-1)
        s += (x[i+1] - x[i]) * (y[i] + y[i+1]) / 2
    end
    return s
end

"""
    solve_nspline_full(A, b, x0; E_MeV, knots=nothing, sigma_rel=nothing,
                       continuity="C0C1", max_iterations=200, tol=1e-3,
                       step_theta=0.1, smoothing=true, n_segments=nothing)
      -> Dict{String,Any}

Полная N-spline развёртка с диагностикой: минимизация направленной
дивергенции H между нормированными измеренными и расчётными
активациями с N-сплайн сглаживанием после каждой итерации
(консервативный шаг `dmu = step_theta / sup|R - Rbar|` с бэктрекингом;
гейдж суммарной активности `sum(A x) = sum(b)`).  Останов по
`H <= 0.5 mean (dQ/Q)^2` или стагнации относительного уменьшения H.
Возвращает Dict: spectrum, iterations, converged, stop_reason, H,
H_history, H_target, nev, nev_limit, acceptable, Qr, relative_residuals,
fluence, mean_energy, knots, knots_source, continuity, params.
"""
function solve_nspline_full(A::AbstractMatrix, b::AbstractVector, x0::Union{Nothing,AbstractVector}=nothing;
                            E_MeV::Union{Nothing,AbstractVector}=nothing,
                            knots=nothing,
                            sigma_rel::Union{Nothing,AbstractVector}=nothing,
                            continuity::String="C0C1",
                            max_iterations::Integer=200,
                            tol::Real=1e-3,
                            step_theta::Real=0.1,
                            smoothing::Bool=true,
                            n_segments::Union{Nothing,Integer}=nothing)
    A_arr = Matrix{Float64}(A)
    b_arr = Vector{Float64}(b)
    m, n = size(A_arr)
    length(b_arr) == m || throw(ArgumentError("b length ($(length(b_arr))) does not match A rows ($m)"))
    E_arr = E_MeV === nothing ? collect(range(1e-9, 1e2, length=n)) : Float64.(collect(E_MeV))
    length(E_arr) == n || throw(ArgumentError("E_MeV length ($(length(E_arr))) does not match A columns ($n)"))
    all(>(0), E_arr) || throw(ArgumentError("E_MeV must contain strictly positive energies"))
    max_iterations >= 1 || throw(ArgumentError("max_iterations must be >= 1, got $max_iterations"))
    0 < step_theta <= 1 || throw(ArgumentError("step_theta must be in (0, 1], got $step_theta"))
    tol > 0 || throw(ArgumentError("tol must be positive, got $tol"))

    valid = findall(>(0), b_arr)
    !isempty(valid) || throw(ArgumentError("solve_nspline requires at least one positive measurement"))
    A_v = A_arr[valid, :]
    b_v = b_arr[valid]

    if sigma_rel === nothing
        sigma_v = fill(0.1, length(b_v))
    else
        sv_all = max.(Float64.(collect(sigma_rel)), 1e-12)
        sigma_v = [sv_all[j] for j in valid]
    end

    kn, knot_src = _nspline_resolve_knots(knots, E_arr, n_segments)

    p_norm = b_v ./ sum(b_v)
    H_target = 0.5 * sum(p_norm .* sigma_v .^ 2)

    x = x0 === nothing ? ones(n) : Vector{Float64}(x0)
    length(x) == n || throw(ArgumentError("x0 length ($(length(x))) does not match A columns ($n)"))
    x = [isfinite(v) ? v : 0.0 for v in x]
    x = max.(x, 0.0)
    sum(x) <= 0 && (x = ones(n))

    Qc0 = A_v * x
    scale = sum(b_v) / max(sum(Qc0), 1e-300)
    x = max.(x .* scale, NSPLINE_PHI_FLOOR)

    sens = vec(sum(A_v, dims=1))
    sens_max = isempty(sens) ? 0.0 : maximum(sens)
    smooth_rel_err = nothing
    if smoothing && sens_max > 0
        smooth_rel_err = sqrt.(clamp.(sens ./ max.(sens, 1e-300), 1.0, 1e12))
    end

    if smoothing
        x, fit_info = nspline_fit(E_arr, x; knots=kn, rel_err=smooth_rel_err,
                                  continuity=continuity)
        x = max.(x, NSPLINE_PHI_FLOOR)
        x = x .* (sum(b_v) / max(sum(A_v * x), 1e-300))
    else
        fit_info = Dict{String,Any}()
    end

    b_total = sum(b_v)
    eps_scale_v = 1e-12 * max(b_total, 1e-300)

    gauge(vv) = vv .* (b_total / max(sum(A_v * vv), 1e-300))
    function state(vv)
        xx = max.(vv, NSPLINE_PHI_FLOOR)
        Qc = max.(A_v * xx, eps_scale_v)
        pN = Qc ./ max(sum(Qc), 1e-300)
        H = directed_divergence(pN, p_norm)
        return xx, Qc, pN, H
    end

    x, _Qc, pN, H = state(gauge(x))
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

        ln_ratio = clamp.(log.(pN ./ p_norm), -NSPLINE_LOG_CLIP, NSPLINE_LOG_CLIP)
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

        mu = step_theta / g_max
        accepted = false
        x_new = x
        _Qc_new = _Qc
        pN_new = pN
        H_new = H
        for _ in 1:60
            x_trial = x .* (1.0 .- mu .* g)
            if smoothing
                x_trial, _ = nspline_fit(E_arr, x_trial; knots=kn, rel_err=smooth_rel_err,
                                         continuity=continuity)
            end
            x_new, _Qc_new, pN_new, H_new = state(gauge(x_trial))
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
        x, _Qc, pN, H = x_new, _Qc_new, pN_new, H_new
        push!(H_history, H)

        if H <= H_target
            converged = true
            stop_reason = "H_target"
            break
        end
        abs(H_prev - H) <= tol * max(H_prev, 1e-300) && (converged = true; stop_reason = "relative_change"; break)
    end

    Qr_full = A_arr * x
    rel_res = zeros(m)
    denom = max.(sigma_v .* b_v, 1e-300)
    for (k, j) in enumerate(valid)
        rel_res[j] = (Qr_full[j] - b_v[k]) / denom[k]
    end
    cnt = length(valid)
    div = cnt > 1 ? cnt - 1 : cnt
    nev = sqrt(sum(abs2, [rel_res[j] for j in valid]) / max(div, 1))
    nev_limit = 1.0 + 2.0 / sqrt(cnt)
    acceptable = nev <= nev_limit

    fluence = _nspline_trapz(E_arr, x)
    mean_energy = fluence > 0 ? _nspline_trapz(x .* E_arr, E_arr) ./ fluence : NaN

    M = length(kn) - 1
    if !haskey(fit_info, "a")
        kseg = _nspline_segment_indices(E_arr, kn)
        G = zeros(n, 3M)
        for j in 1:n
            G[j, kseg[j]] = 1.0
            G[j, M + kseg[j]] = log(E_arr[j])
            G[j, 2M + kseg[j]] = E_arr[j]
        end
        Xl = G \ log.(max.(x, NSPLINE_PHI_FLOOR))
        params = Dict{String,Any}("a" => Xl[1:M], "q" => Xl[M+1:2M], "r" => Xl[2M+1:3M])
    else
        params = Dict{String,Any}("a" => fit_info["a"], "q" => fit_info["q"], "r" => fit_info["r"])
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
    solve_nspline(A, b, x0=nothing; E_MeV=nothing, knots=nothing,
                  sigma_rel=nothing, continuity="C0C1", max_iterations=200,
                  tol=1e-3, step_theta=0.1, smoothing=true, n_segments=nothing)
      -> UnfoldResult

Стандартная обёртка `solve_nspline_full` в API пакета.  Если `E_MeV
не передан, используется равномерная псевдосетка 1e-9..1e2 MeV.
"""
function solve_nspline(A::AbstractMatrix, b::AbstractVector, x0::Union{Nothing,AbstractVector}=nothing;
                       E_MeV::Union{Nothing,AbstractVector}=nothing,
                       knots=nothing,
                       sigma_rel::Union{Nothing,AbstractVector}=nothing,
                       continuity::String="C0C1",
                       max_iterations::Integer=200,
                       tol::Real=1e-3,
                       step_theta::Real=0.1,
                       smoothing::Bool=true,
                       n_segments::Union{Nothing,Integer}=nothing)
    out = solve_nspline_full(A, b, x0; E_MeV=E_MeV, knots=knots, sigma_rel=sigma_rel,
                             continuity=continuity, max_iterations=max_iterations,
                             tol=tol, step_theta=step_theta, smoothing=smoothing,
                             n_segments=n_segments)
    x = out["spectrum"]
    res = UnfoldResult(x, out["iterations"], out["converged"], norm(Matrix{Float64}(A) * x .- b), out)
    return res
end
