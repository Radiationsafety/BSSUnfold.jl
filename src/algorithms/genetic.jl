"""
Meta-heuristic (evolutionary) unfolding methods (порт из unfold_genetic.py).

Популяционные метаэвристики для развёртки, вдохновлённые опубликованными
генетическими/эволюционными работами:

- Shahabinejad & Sohrabpour, Rad. Phys. Chem. 136 (2017): PSO с хаотическим
  инерционным весом и стоимостью
  `||b - A x||^2 / ||b||^2 + lambda * ||x||^2`;
- Suman & Sarkar, BARC/2013/E/005 и Indian J. Pure Appl. Phys. 50 (2012):
  генетический алгоритм со сглаживанием вторыми разностями
  `sum((x_{j-1} - 2 x_j + x_{j+1})^2)`;
- Woo et al., Prog. Nucl. Sci. Technol. 6 (2019): многоцелевая формулировка,
  максимизирующая также энтропию Шеннона;
- Mukherjee, Radiat. Prot. Dosim. 110 (2004): ANDI-03, GA для данных
  активационных детекторов без предварительной догадки спектра.

`solve_genetic` предоставляет селектор `solver`, отображающийся на разные
метаэвристики. В Python-оригинале движки брались из пакета mealpy
(PSO, GA, DE, ES, EP, ABC, GWO, CMA-ES) плюс самописные numpy-движки
(nsga2, TGASU-GA). В Julia-порте вместо mealpy используются **нативные
встроенные движки** (без внешних зависимостей):

- `:pso` — глобально-лучший PSO с линейно убывающим инерционным весом
  (w: 0.9 → 0.4, c1 = c2 = 2.0, как в C_PSO mealpy / SDPSO);
- `:ga` — TGASU-стиль GA (турнирная селекция, одноточечное или
  арифметическое скрещивание, случайная или итеративная мутация,
  элитизм 10%) — порт `_run_numpy_ga`;
- `:de` — DE/rand/1/bin (wf = 0.7, cr = 0.9) — аналог mealpy OriginalDE;
- `:gwo` — Grey Wolf Optimizer — аналог mealpy OriginalGWO;
- `:nsga2` — реальный (real-coded) NSGA-II с двумя целями — относительная
  ошибка отклика и отрицательная энтропия Шеннона (Deb et al., 2002;
  Woo et al., 2019) — порт `_run_nsga2`.

Численная стратегия (как в оригинале): задача развёртки сильно некорректна
(бинов много больше, чем детекторов), поэтому оптимизатор

- ищет в **лог-пространстве** (`y = log(x)`);
- засеивается **тёплым стартом Ландвебера** (или пользовательским
  `initial_spectrum`);
- ограничен `log(seed) ± half_range` декадами;
- минимизирует **масштабно-согласованную цель**, где остаток,
  регуляризация и гладкость безразмерны и сравнимы.
"""

# ─── Derivative matrix (аналог create_derivative_matrix) ────────────────────

"""
    _create_derivative_matrix(n, order) -> Matrix{Float64}

Матрица конечных разностей порядка 1 или 2 размера (n-1 или n-2) × n
(аналог `_matrix_utils.create_derivative_matrix` Python-оригинала).
"""
function _create_derivative_matrix(n::Integer, order::Integer)
    order in (1, 2) || throw(ArgumentError("Unsupported derivative order: $order"))
    rows = n - order
    rows >= 1 || throw(ArgumentError("n must exceed order, got n=$n"))
    L = zeros(rows, n)
    if order == 1
        for i in 1:rows
            L[i, i] = -1.0
            L[i, i + 1] = 1.0
        end
    else
        for i in 1:rows
            L[i, i] = 1.0
            L[i, i + 1] = -2.0
            L[i, i + 2] = 1.0
        end
    end
    return L
end

# ─── Seed и лог-границы ─────────────────────────────────────────────────────

"""
    _genetic_seed(A, b, x0) -> Vector{Float64}

Построить сид-спектр для инициализации популяции.

Если нетривиальная начальная догадка `x0` задана — используется она.
Иначе вычисляется тёплый старт Ландвебера (с нулевой догадки), дающий
метаэвристике гладкую, физически правдоподобную стартовую точку.
"""
function _genetic_seed(A::AbstractMatrix{<:Real}, b::AbstractVector{<:Real},
                      x0::Union{Nothing,AbstractVector{<:Real}})
    n = size(A, 2)
    if x0 !== nothing && any(>(0), x0)
        return max.(Float64.(collect(x0)), 1e-12)
    end
    try
        lw = solve_landweber(A, b, zeros(n); max_iterations=500)
        return max.(lw.spectrum, 1e-12)
    catch
        # Fallback: плоский спектр, масштабированный к показаниям.
        x_scale = norm(b) / max(norm(A), eps(Float64))
        return fill(max(x_scale / sqrt(n), 1e-12), n)
    end
end

"""
    _genetic_log_bounds(seed, half_range) -> (lb, ub)

Лог-пространственные границы вокруг сида: `log(seed) ± half_range` декад.
"""
function _genetic_log_bounds(seed::Vector{Float64}, half_range::Real)
    y0 = log.(max.(seed, 1e-300))
    h = half_range * log(10.0)
    return y0 .- h, y0 .+ h
end

# ─── Масштабно-согласованная функция цели (порт _build_fitness) ─────────────

"""
    _build_genetic_fitness(A, b, alpha, norm, L, smoothness_weight, entropy_weight)

Построить целевую функцию развёртки в лог-пространстве.

Оптимизатор ищет `y` с `x = exp(y)`. Все члены нормированы на свои
естественные масштабы, так что остаток, регуляризация и гладкость
безразмерны и сравнимы:

    f(y) = ||b - A exp(y)||^2 / ||b||^2
         + alpha * ||exp(y)||_norm / x_scale^p
         + smoothness_weight * ||L exp(y)||^2 / x_scale^2
         - entropy_weight * H(exp(y))

где `x_scale = ||b|| / ||A||` — характерная величина спектра,
воспроизводящего показания.
"""
function _build_genetic_fitness(A::AbstractMatrix{Float64}, b::Vector{Float64},
                               alpha::Float64, norm_type::Int,
                               L::Union{Nothing,Matrix{Float64}},
                               smoothness_weight::Float64, entropy_weight::Float64)
    denom = max(dot(b, b), 1.0)
    A_fro = max(norm(A), eps(Float64))
    x_scale = sqrt(denom) / A_fro
    x_scale2 = x_scale * x_scale

    return function fitness(y::AbstractVector{Float64})
        x = exp.(y)
        residual = A * x .- b
        value = dot(residual, residual) / denom
        if alpha > 0
            if norm_type == 2
                value += alpha * dot(x, x) / x_scale2
            elseif norm_type == 1
                value += alpha * sum(abs.(x)) / x_scale
            end
        end
        if L !== nothing && smoothness_weight > 0
            Lx = L * x
            value += smoothness_weight * dot(Lx, Lx) / x_scale2
        end
        if entropy_weight > 0
            total = sum(x)
            if total > 0
                p_ = x ./ total
                logp = log.(max.(p_, 1e-300))
                value -= entropy_weight * dot(p_, logp)
            end
        end
        return value
    end
end

# ─── Движок GA (TGASU-стиль, порт _run_numpy_ga) ────────────────────────────

function _run_tgasu_ga(fitness::Function, seed::Vector{Float64},
                      lb::Vector{Float64}, ub::Vector{Float64};
                      epoch::Int, pop_size::Int,
                      crossover::Symbol, mutation::Symbol,
                      pc::Float64, pm::Float64,
                      rng::AbstractRNG,
                      extra::Union{Nothing,Vector{Float64}}=nothing)
    n = length(seed)
    y0 = log.(max.(seed, 1e-300))
    pop = [collect(lb .+ rand(rng, n) .* (ub .- lb)) for _ in 1:pop_size]
    pop[1] = copy(y0)
    if extra !== nothing && pop_size >= 2
        pop[2] = clamp.(log.(max.(extra, 1e-300)), lb, ub)
    end
    elite = max(1, pop_size ÷ 10)
    scale0 = 0.3
    fvals = [fitness(y) for y in pop]

    for gen in 1:epoch
        order = sortperm(fvals)
        elites = [copy(pop[i]) for i in order[1:elite]]
        scale = scale0 * (1.0 - (gen - 1) / max(epoch, 1))

        # Двойной турнир (4 кандидата) по текущим значениям цели
        function tournament()
            a, b2 = rand(rng, 1:pop_size), rand(rng, 1:pop_size)
            w1 = fvals[a] <= fvals[b2] ? a : b2
            c, d = rand(rng, 1:pop_size), rand(rng, 1:pop_size)
            w2 = fvals[c] <= fvals[d] ? c : d
            return w1, w2
        end

        offspring = Vector{Vector{Float64}}()
        while length(offspring) < pop_size - elite
            w1, w2 = tournament()
            p1, p2 = pop[w1], pop[w2]

            c1, c2 = copy(p1), copy(p2)
            if rand(rng) < pc
                if crossover === :arithmetic
                    beta = rand(rng)
                    c1 = beta .* p1 .+ (1 - beta) .* p2
                    c2 = (1 - beta) .* p1 .+ beta .* p2
                else  # :single
                    point = rand(rng, 1:(n - 1))
                    c1 = vcat(p1[1:point], p2[(point + 1):end])
                    c2 = vcat(p2[1:point], p1[(point + 1):end])
                end
            end
            for child in (c1, c2)
                length(offspring) >= pop_size - elite && break
                mask = rand(rng, n) .< pm
                if any(mask)
                    if mutation === :iterative
                        # phi_new = phi_old * (1 + scale * beta), beta ∈ [-1, 1]
                        beta_m = 2 .* rand(rng, n) .- 1
                        child .+= log.(max.(1.0 .+ scale .* beta_m, 1e-12))
                    else  # :random
                        child .+= scale .* randn(rng, n)
                    end
                    child .= clamp.(child, lb, ub)
                end
                push!(offspring, child)
            end
        end

        pop = vcat(elites, offspring[1:(pop_size - elite)])
        fvals = vcat(fvals[order[1:elite]],
                     [fitness(y) for y in offspring[1:(pop_size - elite)]])
    end

    # Лучший индивид финальной популяции
    vals = [fitness(y) for y in pop]
    best_i = argmin(vals)
    return max.(exp.(pop[best_i]), 0.0), vals[best_i]
end

# ─── Движок PSO (аналог mealpy C_PSO: w: 0.9→0.4, c1 = c2 = 2.0) ────────────

function _run_pso(fitness::Function, seed::Vector{Float64},
                 lb::Vector{Float64}, ub::Vector{Float64};
                 epoch::Int, pop_size::Int,
                 w_min::Float64=0.4, w_max::Float64=0.9,
                 c1::Float64=2.0, c2::Float64=2.0,
                 rng::AbstractRNG,
                 extra::Union{Nothing,Vector{Float64}}=nothing)
    n = length(seed)
    y0 = log.(max.(seed, 1e-300))
    pos = [collect(lb .+ rand(rng, n) .* (ub .- lb)) for _ in 1:pop_size]
    pos[1] = copy(y0)
    if extra !== nothing && pop_size >= 2
        pos[2] = clamp.(log.(max.(extra, 1e-300)), lb, ub)
    end
    vel = [zeros(n) for _ in 1:pop_size]
    fvals = [fitness(y) for y in pos]
    pbest = [copy(p) for p in pos]
    pbest_f = copy(fvals)
    g_i = argmin(pbest_f)
    gbest = copy(pbest[g_i])
    gbest_f = pbest_f[g_i]

    for gen in 1:epoch
        w = w_max - (w_max - w_min) * (gen - 1) / max(epoch - 1, 1)
        for i in 1:pop_size
            r1 = rand(rng, n)
            r2 = rand(rng, n)
            vel[i] = @. w * vel[i] + c1 * r1 * (pbest[i] - pos[i]) +
                       c2 * r2 * (gbest - pos[i])
            pos[i] .= clamp.(pos[i] .+ vel[i], lb, ub)
            f = fitness(pos[i])
            if f < pbest_f[i]
                pbest_f[i] = f
                pbest[i] = copy(pos[i])
                if f < gbest_f
                    gbest_f = f
                    gbest = copy(pos[i])
                end
            end
        end
    end
    return max.(exp.(gbest), 0.0), gbest_f
end

# ─── Движок DE (DE/rand/1/bin, аналог mealpy OriginalDE) ────────────────────

function _run_de(fitness::Function, seed::Vector{Float64},
                lb::Vector{Float64}, ub::Vector{Float64};
                epoch::Int, pop_size::Int,
                wf::Float64=0.7, cr::Float64=0.9,
                rng::AbstractRNG,
                extra::Union{Nothing,Vector{Float64}}=nothing)
    n = length(seed)
    y0 = log.(max.(seed, 1e-300))
    pop = [collect(lb .+ rand(rng, n) .* (ub .- lb)) for _ in 1:pop_size]
    pop[1] = copy(y0)
    if extra !== nothing && pop_size >= 2
        pop[2] = clamp.(log.(max.(extra, 1e-300)), lb, ub)
    end
    fvals = [fitness(y) for y in pop]

    for _ in 1:epoch
        for i in 1:pop_size
            # Выбор трёх различных индексов ≠ i
            candidates = collect(1:pop_size)
            deleteat!(candidates, findall(==(i), candidates))
            r1 = splice!(candidates, rand(rng, 1:length(candidates)))[1]
            r2 = splice!(candidates, rand(rng, 1:length(candidates)))[1]
            r3 = splice!(candidates, rand(rng, 1:length(candidates)))[1]
            mutant = pop[r1] .+ wf .* (pop[r2] .- pop[r3])
            # Бинарное скрещивание
            jrand = rand(rng, 1:n)
            trial = copy(pop[i])
            for j in 1:n
                if rand(rng) < cr || j == jrand
                    trial[j] = mutant[j]
                end
            end
            trial .= clamp.(trial, lb, ub)
            f_trial = fitness(trial)
            if f_trial <= fvals[i]
                pop[i] = trial
                fvals[i] = f_trial
            end
        end
    end
    best_i = argmin(fvals)
    return max.(exp.(pop[best_i]), 0.0), fvals[best_i]
end

# ─── Движок GWO (аналог mealpy OriginalGWO) ─────────────────────────────────

function _run_gwo(fitness::Function, seed::Vector{Float64},
                 lb::Vector{Float64}, ub::Vector{Float64};
                 epoch::Int, pop_size::Int, rng::AbstractRNG,
                 extra::Union{Nothing,Vector{Float64}}=nothing)
    n = length(seed)
    y0 = log.(max.(seed, 1e-300))
    pop = [collect(lb .+ rand(rng, n) .* (ub .- lb)) for _ in 1:pop_size]
    pop[1] = copy(y0)
    if extra !== nothing && pop_size >= 2
        pop[2] = clamp.(log.(max.(extra, 1e-300)), lb, ub)
    end
    fvals = [fitness(y) for y in pop]
    order = sortperm(fvals)
    alpha_pos, beta_pos, delta_pos = pop[order[1]], pop[order[2]], pop[order[min(3, pop_size)]]
    alpha_f, beta_f, delta_f = fvals[order[1]], fvals[order[2]], fvals[order[min(3, pop_size)]]

    for gen in 1:epoch
        a = 2.0 - 2.0 * (gen - 1) / max(epoch - 1, 1)  # линейно 2 → 0
        for i in 1:pop_size
            x = copy(pop[i])
            newx = zeros(n)
            for leader_pos in (alpha_pos, beta_pos, delta_pos)
                A_w = 2 .* rand(rng, n) .- 1
                C_w = 2 .* rand(rng, n)
                D_w = abs.(C_w .* leader_pos .- x)
                X1 = leader_pos .- A_w .* D_w
                newx .+= X1
            end
            newx ./= 3.0
            newx .= clamp.(newx, lb, ub)
            f = fitness(newx)
            pop[i] = newx
            fvals[i] = f
            # Обновление иерархии альфа/бета/дельта
            if f < alpha_f
                delta_f, delta_pos = beta_f, copy(beta_pos)
                beta_f, beta_pos = alpha_f, copy(alpha_pos)
                alpha_f, alpha_pos = f, copy(newx)
            elseif f < beta_f
                delta_f, delta_pos = beta_f, copy(beta_pos)
                beta_f, beta_pos = f, copy(newx)
            elseif f < delta_f
                delta_f, delta_pos = f, copy(newx)
            end
        end
    end
    return max.(exp.(alpha_pos), 0.0), alpha_f
end

# ─── NSGA-II (порт _run_nsga2) ──────────────────────────────────────────────

"""
    _fast_non_dominated_sort(fvals) -> Vector{Vector{Int}}

Фронты Парето популяции (минимизация); `fronts[1]` — недоминируемый фронт.
"""
function _fast_non_dominated_sort(fvals::AbstractMatrix{<:Real})
    N = size(fvals, 1)
    dominates = falses(N, N)
    for i in 1:N, j in 1:N
        if i != j
            dominates[i, j] = all(fvals[i, k] <= fvals[j, k] for k in 1:size(fvals, 2)) &&
                             any(fvals[i, k] < fvals[j, k] for k in 1:size(fvals, 2))
        end
    end
    fronts = Vector{Vector{Int}}()
    remaining = collect(1:N)
    while !isempty(remaining)
        front = Int[i for i in remaining if
                    !any(dominates[j, i] for j in remaining if j != i)]
        push!(fronts, front)
        filter!(i -> i ∉ front, remaining)
    end
    return fronts
end

"""
    _crowding_distance(fvals, front) -> Vector{Float64}

Краудинг-дистанции индивидов фронта (Deb et al., 2002).
"""
function _crowding_distance(fvals::AbstractMatrix{<:Real}, front::Vector{Int})
    m = length(front)
    dist = fill(Inf, m)
    m <= 2 && return dist
    n_obj = size(fvals, 2)
    for obj in 1:n_obj
        order = sortperm(front, by=i -> fvals[i, obj])
        dist[order[1]] = Inf
        dist[order[end]] = Inf
        fmin = fvals[front[order[1]], obj]
        fmax = fvals[front[order[end]], obj]
        spread = fmax - fmin
        spread <= 0 && continue
        for idx in 2:(m - 1)
            i_prev = front[order[idx - 1]]
            i_next = front[order[idx + 1]]
            dist[order[idx]] += (fvals[i_next, obj] - fvals[i_prev, obj]) / spread
        end
    end
    return dist
end

"""
    _sbx_crossover(p1, p2, lb, ub, rng; eta_c=15.0) -> (c1, c2)

Simulated Binary Crossover с клиппингом к границам.
"""
function _sbx_crossover(p1::Vector{Float64}, p2::Vector{Float64},
                       lb::Vector{Float64}, ub::Vector{Float64},
                       rng::AbstractRNG; eta_c::Float64=15.0)
    n = length(p1)
    mask = rand(rng, n) .< 0.5
    u = rand(rng, n)
    beta = [u_i <= 0.5 ? (2 * u_i)^(1 / (eta_c + 1)) :
            (1 / (2 * (1 - u_i)))^(1 / (eta_c + 1)) for u_i in u]
    sum_p = p1 .+ p2
    diff = p1 .- p2
    c1 = @. 0.5 * (sum_p - beta * diff)
    c2 = @. 0.5 * (sum_p + beta * diff)
    c1 = [mask_i ? c : p for (mask_i, c, p) in zip(mask, c1, p1)]
    c2 = [mask_i ? c : p for (mask_i, c, p) in zip(mask, c2, p2)]
    return clamp.(c1, lb, ub), clamp.(c2, lb, ub)
end

"""
    _polynomial_mutation(p, lb, ub, rng; eta_m=20.0) -> Vector{Float64}

Полиномиальная мутация с клиппингом к границам.
"""
function _polynomial_mutation(p::Vector{Float64}, lb::Vector{Float64},
                             ub::Vector{Float64}, rng::AbstractRNG;
                             eta_m::Float64=20.0)
    n = length(p)
    pm_prob = 1.0 / n
    out = copy(p)
    for j in 1:n
        rand(rng) < pm_prob || continue
        u = rand(rng)
        delta = u <= 0.5 ? (2 * u)^(1 / (eta_m + 1)) - 1.0 :
                1.0 - (2 * (1 - u))^(1 / (eta_m + 1))
        out[j] = p[j] + delta * (ub[j] - lb[j])
    end
    return clamp.(out, lb, ub)
end

"""
    _select_knee(f0) -> Int

Индекс «колена» фронта Парето (ближайшая к идеальной точке).
"""
function _select_knee(f0::AbstractMatrix{<:Real})
    ideal = vec(minimum(f0, dims=1))
    spread = vec(maximum(f0, dims=1)) .- ideal
    spread = max.(spread, 1.0)
    normed = (f0 .- ideal') ./ spread'
    return argmin(vec(sum(abs2, normed, dims=2)))
end

function _nsga2_objectives(A::AbstractMatrix{Float64}, b::Vector{Float64},
                          pop::Vector{Vector{Float64}}, entropy_weight::Float64)
    denom = max(dot(b, b), 1.0)
    N = length(pop)
    fvals = Matrix{Float64}(undef, N, 2)
    for i in 1:N
        x = exp.(pop[i])
        resid = A * x .- b
        fvals[i, 1] = dot(resid, resid) / denom
        total = max(sum(x), 1e-300)
        p_ = x ./ total
        ent = -sum(p_ .* log.(max.(p_, 1e-300)))
        fvals[i, 2] = -entropy_weight * ent
    end
    return fvals
end

function _run_nsga2(A::AbstractMatrix{Float64}, b::Vector{Float64},
                   seed::Vector{Float64}, lb::Vector{Float64}, ub::Vector{Float64};
                   epoch::Int, pop_size::Int, rng::AbstractRNG,
                   pareto_select::Symbol, entropy_weight::Float64,
                   extra::Union{Nothing,Vector{Float64}}=nothing)
    n = length(seed)
    pop = [collect(lb .+ rand(rng, n) .* (ub .- lb)) for _ in 1:pop_size]
    pop[1] = log.(max.(seed, 1e-300))
    if extra !== nothing && pop_size >= 2
        pop[2] = clamp.(log.(max.(extra, 1e-300)), lb, ub)
    end

    for _ in 1:epoch
        fvals = _nsga2_objectives(A, b, pop, entropy_weight)
        fronts = _fast_non_dominated_sort(fvals)
        rank = fill(0, pop_size)
        for (r, front) in enumerate(fronts), i in front
            rank[i] = r
        end
        crowding = fill(0.0, pop_size)
        for front in fronts
            cd = _crowding_distance(fvals, front)
            for (idx, i) in enumerate(front)
                crowding[i] = cd[idx]
            end
        end

        # Бинарный турнир по (rank, -crowding)
        better(i1::Int, i2::Int) = rank[i1] < rank[i2] ||
                                   (rank[i1] == rank[i2] && crowding[i1] > crowding[i2])
        selected = Vector{Vector{Float64}}(undef, pop_size)
        for k in 1:pop_size
            c1, c2 = rand(rng, 1:pop_size), rand(rng, 1:pop_size)
            w1 = better(c1, c2) ? c1 : c2
            c3, c4 = rand(rng, 1:pop_size), rand(rng, 1:pop_size)
            w2 = better(c3, c4) ? c3 : c4
            selected[k] = copy(pop[better(w1, w2) ? w1 : w2])
        end

        # SBX + полиномиальная мутация
        offspring = Vector{Vector{Float64}}(undef, pop_size)
        for i in 1:2:pop_size
            c1, c2 = _sbx_crossover(selected[i], selected[min(i + 1, pop_size)], lb, ub, rng)
            offspring[i] = _polynomial_mutation(c1, lb, ub, rng)
            offspring[min(i + 1, pop_size)] = _polynomial_mutation(c2, lb, ub, rng)
        end

        # Слияние родителей и потомков, усечение до pop_size по фронтам
        combined = vcat(pop, offspring)
        cf = _nsga2_objectives(A, b, combined, entropy_weight)
        cfronts = filter!(!isempty, _fast_non_dominated_sort(cf))
        next_pop = Int[]
        for front in cfronts
            if length(next_pop) + length(front) <= pop_size
                append!(next_pop, front)
            else
                cd = _crowding_distance(cf, front)
                order = sortperm(cd; rev=true)
                needed = pop_size - length(next_pop)
                append!(next_pop, front[order[1:needed]])
                break
            end
        end
        pop = [combined[i] for i in next_pop]
    end

    fvals = _nsga2_objectives(A, b, pop, entropy_weight)
    fronts = _fast_non_dominated_sort(fvals)
    front0 = fronts[1]
    f0 = fvals[front0, :]
    local idx::Int
    if pareto_select === :min_residual
        idx = front0[argmin(f0[:, 1])]
    elseif pareto_select === :max_entropy
        idx = front0[argmin(f0[:, 2])]
    else  # :knee
        idx = front0[_select_knee(f0)]
    end
    spectrum = max.(exp.(pop[idx]), 0.0)
    diagnostics = Dict{String,Any}(
        "pareto_front_size" => length(front0),
        "pareto_min_residual" => minimum(f0[:, 1]),
        "pareto_max_entropy" => maximum(-f0[:, 2]),
        "pareto_select" => String(pareto_select),
    )
    return spectrum, diagnostics
end

# ─── Пост-обработка: сглаживатели (порт _apply_smoother) ────────────────────

"""
    gaussian_filter1d_nearest(x, sigma) -> Vector{Float64}

1D гауссов фильтр с mode="nearest" (аналог scipy.ndimage.gaussian_filter1d,
truncate = 4.0 по умолчанию).
"""
function gaussian_filter1d_nearest(x::AbstractVector{<:Real}, sigma::Real)
    n = length(x)
    sigma <= 0 && return collect(Float64, x)
    radius = ceil(Int, 4.0 * sigma)
    half = collect(-radius:radius)
    kernel = exp.(-0.5 .* (half ./ sigma) .^ 2)
    kernel ./= sum(kernel)
    out = Vector{Float64}(undef, n)
    for i in 1:n
        s = 0.0
        for (k, off) in enumerate(half)
            j = clamp(i + off, 1, n)   # mode="nearest"
            s += kernel[k] * x[j]
        end
        out[i] = s
    end
    return out
end

"""
    apply_smoother(x, smoother; sigma=2.0, smoothing_weight=1.0) -> Vector{Float64}

Двухстадийное сглаживание для подавления осцилляций (Suman & Sarkar, 2012;
Gaussian + мультипликативная коррекция смещения — схема TGASU,
Shahabinejad et al., 2016).  Сглаженный спектр клиппируется к
неотрицательным значениям и пересчитывается с сохранением полного флюенса.

`smoother`: `"none"`, `"gaussian"`, `"mbc"`, `"gaussian_mbc"` или
`"second_difference"`.
"""
function apply_smoother(x::AbstractVector{<:Real}, smoother::AbstractString;
                       sigma::Real=2.0, smoothing_weight::Real=1.0)
    name = lowercase(replace(strip(smoother), "-" => "_"))
    aliases = Dict{String,String}(
        "" => "none", "no" => "none", "off" => "none",
        "gauss" => "gaussian",
        "gaussian_multiplicative_bias_correction" => "gaussian_mbc",
        "gauss_mbc" => "gaussian_mbc", "mbc" => "gaussian_mbc",
        "2nd_difference" => "second_difference", "seconddifference" => "second_difference",
        "d2" => "second_difference",
    )
    name = get(aliases, name, name)
    name in ("none", "gaussian", "mbc", "gaussian_mbc", "second_difference") ||
        (name = "none")

    x_arr = Float64.(collect(x))
    name == "none" && return x_arr

    s = if name == "gaussian"
        gaussian_filter1d_nearest(x_arr, sigma)
    elseif name == "mbc"
        sm = gaussian_filter1d_nearest(x_arr, sigma)
        bias = x_arr ./ max.(sm, 1e-300)
        correction = gaussian_filter1d_nearest(bias, sigma)
        x_arr .* correction
    elseif name == "gaussian_mbc"
        sm = gaussian_filter1d_nearest(x_arr, sigma)
        bias = x_arr ./ max.(sm, 1e-300)
        correction = gaussian_filter1d_nearest(bias, sigma)
        sm .* correction
    else  # second_difference
        L = _create_derivative_matrix(length(x_arr), 2)
        M = Matrix{Float64}(I, length(x_arr), length(x_arr)) +
            smoothing_weight .* (L' * L)
        try
            M \ x_arr
        catch
            copy(x_arr)
        end
    end

    s = max.(s, 0.0)
    total_in = sum(x_arr)
    total_out = sum(s)
    if total_out > 0 && total_in > 0
        s .*= total_in / total_out
    end
    return s
end

# ─── Coarse/fine сетки (порт _multires) ─────────────────────────────────────

"""
    coarsen_columns(A, n_coarse) -> Matrix{Float64}

Слить соседние столбцы ответной матрицы в `n_coarse` бинов (суммирование):
`A_coarse[i, k] = Σ_{j ∈ bin k} A[i, j]`.
"""
function coarsen_columns(A::AbstractMatrix{<:Real}, n_coarse::Integer)
    m, n = size(A)
    0 < n_coarse <= n || throw(ArgumentError(
        "n_coarse must satisfy 0 < n_coarse <= $n"))
    edges = floor.(Int, collect(range(0, n, length=n_coarse + 1)))
    A_coarse = zeros(m, n_coarse)
    for k in 1:n_coarse
        lo, hi = edges[k], edges[k + 1]
        hi > lo && (A_coarse[:, k] .= vec(sum(A[:, (lo + 1):hi], dims=2)))
    end
    return A_coarse
end

"""
    split_coarse(x_coarse, n) -> Vector{Float64}

Распределить суммарные значения грубых бинов обратно на тонкую сетку
(равномерно внутри каждого грубого бина, с сохранением флюенса).
"""
function split_coarse(x_coarse::AbstractVector{<:Real}, n::Integer)
    n_coarse = length(x_coarse)
    edges = floor.(Int, collect(range(0, n, length=n_coarse + 1)))
    x = zeros(n)
    for k in 1:n_coarse
        lo, hi = edges[k], edges[k + 1]
        width = hi - lo
        if width > 0
            x[(lo + 1):hi] .= x_coarse[k] / width
        end
    end
    return x
end

# ─── Валидация параметров ───────────────────────────────────────────────────

const _SUPPORTED_SOLVERS = (:pso, :ga, :de, :gwo, :nsga2)
const _SOLVER_ALIASES = Dict{Symbol,Symbol}(
    :particle_swarm => :pso,
    :genetic => :ga, :genetic_algorithm => :ga,
    :differential_evolution => :de,
    :grey_wolf => :gwo, :gray_wolf => :gwo,
    :non_dominated_sorting_genetic_algorithm_ii => :nsga2,
    :pareto => :nsga2, :multi_objective => :nsga2,
)

function _normalize_genetic_solver(solver::Symbol)
    name = get(_SOLVER_ALIASES, solver, solver)
    name in _SUPPORTED_SOLVERS || throw(ArgumentError(
        "Unsupported solver: $solver. Supported solvers: " *
        join(_SUPPORTED_SOLVERS, ", ") *
        ". (В Python-оригинале mealpy также предоставлял es, ep, abc, cmaes; " *
        "в Julia-порте эти движки пока не портированы.)"))
    return name
end

# ─── Основной солвер ────────────────────────────────────────────────────────

"""
    solve_genetic(A, b, x0; solver=:pso, epoch=100, pop_size=50,
                  regularization=1e-2, norm=2, smoothness_order=2,
                  smoothness_weight=1.0, entropy_weight=0.0, n_runs=1,
                  half_range=2.0, two_step=false, n_coarse=nothing,
                  smoother="none", sigma_smooth=2.0, crossover=:single,
                  mutation=:random, pareto_select=:knee,
                  random_state=nothing) -> UnfoldResult

Решить задачу развёртки метаэвристическим оптимизатором.

Оптимизатор ищет в лог-пространстве (`y = log(x)`), популяция засеивается
тёплым стартом Ландвебера (или `x0`) и ограничивается
`log(seed) ± half_range` декадами. Все члены цели масштабно-согласованы
(безразмерны), что не даёт оптимизатору выдавать шумной произвольной
спектральной заготовки.

# Аргументы
- `A::AbstractMatrix{T}`: ответная матрица (m × n)
- `b::AbstractVector{T}`: измерения (m,)
- `x0::Union{Nothing,AbstractVector{T}}`: начальная догадка; `nothing` —
  тёплый старт Ландвебера
- `solver::Symbol`: `:pso`, `:ga`, `:de`, `:gwo` или `:nsga2` (default `:pso`);
  длинные псевдонимы (`:differential_evolution`, `:pareto`, ...) поддержаны
- `epoch`: число поколений/итераций (default 100; в Python 500 — уменьшено
  для разумного времени выполнения; увеличьте при необходимости)
- `pop_size`: размер популяции (default 50)
- `regularization`: вес тихоновской регуляризации α (default 1e-2)
- `norm`: норма регуляризации (1 для L1, 2 для L2), default 2
- `smoothness_order`: порядок сглаживания (0, 1 или 2)
- `smoothness_weight`: вес члена сглаживания (default 1.0)
- `entropy_weight`: вес отрицательной энтропии Шеннона (0 — выключено)
- `n_runs`: число независимых запусков; результаты усредняются (default 1)
- `half_range`: полуширина лог-границ в декадах вокруг сида (default 2.0)
- `two_step`: TGASU-стиль двухшаговая схема: сначала задача решается на
  грубой сетке, затем результат интерполируется для засева полной
  популяции (default false)
- `n_coarse`: число грубых бинов для `two_step`; `nothing` — `max(8, n ÷ 4)`
- `smoother`: пост-сглаживатель `"none"`, `"gaussian"`, `"mbc"`,
  `"gaussian_mbc"` или `"second_difference"` (default `"none"`)
- `sigma_smooth`: сигма гауссова фильтра (default 2.0)
- `crossover`: `:single` или `:arithmetic` (TGASU); только для `:ga`
- `mutation`: `:random` или `:iterative` (TGASU); только для `:ga`
- `pareto_select`: выбор с фронта Парето для `:nsga2`: `:knee`,
  `:min_residual` или `:max_entropy` (default `:knee`)
- `random_state`: seed для воспроизводимости

# Возвращает
`UnfoldResult` со спектром; в `extra` — `solver`, `fitness` (лучшее значение
цели), `diagnostics` (для `:nsga2`).
"""
function solve_genetic(A::AbstractMatrix{T}, b::AbstractVector{T},
                      x0::Union{Nothing,AbstractVector{T}};
                      solver::Symbol=:pso,
                      epoch::Integer=100,
                      pop_size::Integer=50,
                      regularization::Real=1e-2,
                      norm::Integer=2,
                      smoothness_order::Integer=2,
                      smoothness_weight::Real=1.0,
                      entropy_weight::Real=0.0,
                      n_runs::Integer=1,
                      half_range::Real=2.0,
                      two_step::Bool=false,
                      n_coarse::Union{Nothing,Integer}=nothing,
                      smoother::AbstractString="none",
                      sigma_smooth::Real=2.0,
                      crossover::Symbol=:single,
                      mutation::Symbol=:random,
                      pareto_select::Symbol=:knee,
                      random_state::Union{Integer,Nothing}=nothing) where T<:AbstractFloat
    m, n = size(A)
    length(b) == m || throw(ArgumentError("b length ($(length(b))) must match A rows ($m)"))
    solver_name = _normalize_genetic_solver(solver)
    norm in (1, 2) || throw(ArgumentError("Unsupported norm type: $norm"))
    smoothness_order in (0, 1, 2) || throw(ArgumentError(
        "Unsupported smoothness order: $smoothness_order"))
    crossover in (:single, :arithmetic) || throw(ArgumentError(
        "Unsupported crossover operator: $crossover. Use :single or :arithmetic."))
    mutation in (:random, :iterative) || throw(ArgumentError(
        "Unsupported mutation operator: $mutation. Use :random or :iterative."))
    pareto_select in (:knee, :min_residual, :max_entropy) || throw(ArgumentError(
        "Unsupported pareto_select: $pareto_select. " *
        "Use :knee, :min_residual or :max_entropy."))

    rng_master = random_state === nothing ? MersenneTwister() :
                 MersenneTwister(Int(random_state))

    # Two-step схема (TGASU-стиль): грубая сетка → интерполяция → полный запуск
    local extra_starting::Union{Nothing,Vector{Float64}} = nothing
    if two_step
        n_coarse_ = n_coarse === nothing ? max(8, n ÷ 4) : Int(n_coarse)
        n_coarse_ >= n && (n_coarse_ = max(1, n ÷ 2))
        A_coarse = coarsen_columns(A, n_coarse_)
        coarse_x0 = (x0 !== nothing && any(>(0), x0)) ?
                    vec(coarsen_columns(reshape(Float64.(x0), 1, :), n_coarse_)) :
                    ones(n_coarse_)
        coarse = solve_genetic(A_coarse, b, coarse_x0;
                               solver=solver, epoch=max(20, epoch ÷ 2),
                               pop_size=pop_size,
                               regularization=regularization, norm=norm,
                               smoothness_order=smoothness_order,
                               smoothness_weight=smoothness_weight,
                               entropy_weight=entropy_weight, n_runs=n_runs,
                               half_range=half_range, two_step=false,
                               smoother="none", sigma_smooth=sigma_smooth,
                               crossover=crossover, mutation=mutation,
                               pareto_select=pareto_select,
                               random_state=random_state)
        extra_starting = split_coarse(max.(coarse.spectrum, 0.0), n)
    end

    L = smoothness_order in (1, 2) ?
        _create_derivative_matrix(n, smoothness_order) : nothing

    A_f = Float64.(Matrix(A))
    b_f = Float64.(collect(b))
    fitness = _build_genetic_fitness(A_f, b_f, Float64(regularization), Int(norm),
                                     L, Float64(smoothness_weight),
                                     Float64(entropy_weight))
    seed = _genetic_seed(A_f, b_f, x0)
    lb, ub = _genetic_log_bounds(seed, half_range)

    spectra = Vector{Vector{Float64}}()
    local best_fitness::Float64 = Inf
    local diagnostics::Dict{String,Any} = Dict{String,Any}()

    runs = max(1, Int(n_runs))
    for run in 1:runs
        rng = MersenneTwister(rand(rng_master, 1:typemax(Int32)))
        if solver_name === :nsga2
            spec, diag = _run_nsga2(A_f, b_f, seed, lb, ub;
                                    epoch=Int(epoch), pop_size=Int(pop_size),
                                    rng=rng, pareto_select=pareto_select,
                                    entropy_weight=entropy_weight > 0 ?
                                        Float64(entropy_weight) : 1.0,
                                    extra=extra_starting)
            diagnostics = diag
            best_fitness = min(best_fitness, diag["pareto_min_residual"])
            push!(spectra, spec)
        elseif solver_name === :ga
            spec, fbest = _run_tgasu_ga(fitness, seed, lb, ub;
                                        epoch=Int(epoch), pop_size=Int(pop_size),
                                        crossover=crossover, mutation=mutation,
                                        pc=0.9, pm=0.05, rng=rng,
                                        extra=extra_starting)
            best_fitness = min(best_fitness, fbest)
            push!(spectra, spec)
        elseif solver_name === :pso
            spec, fbest = _run_pso(fitness, seed, lb, ub;
                                   epoch=Int(epoch), pop_size=Int(pop_size),
                                   rng=rng, extra=extra_starting)
            best_fitness = min(best_fitness, fbest)
            push!(spectra, spec)
        elseif solver_name === :de
            spec, fbest = _run_de(fitness, seed, lb, ub;
                                  epoch=Int(epoch), pop_size=Int(pop_size),
                                  rng=rng, extra=extra_starting)
            best_fitness = min(best_fitness, fbest)
            push!(spectra, spec)
        else  # :gwo
            spec, fbest = _run_gwo(fitness, seed, lb, ub;
                                   epoch=Int(epoch), pop_size=Int(pop_size),
                                   rng=rng, extra=extra_starting)
            best_fitness = min(best_fitness, fbest)
            push!(spectra, spec)
        end
    end

    spectrum = runs > 1 ? vec(mean(reduce(hcat, spectra), dims=2)) : spectra[1]
    spectrum = max.(spectrum, 0.0)
    smoother_name = lowercase(replace(strip(smoother), "-" => "_"))
    if smoother_name ∉ ("", "none", "no", "off")
        spectrum = apply_smoother(spectrum, smoother; sigma=sigma_smooth,
                                  smoothing_weight=smoothness_weight)
    end

    residual = b_f .- A_f * spectrum
    return UnfoldResult(
        Vector{T}(spectrum), Int(epoch), isfinite(best_fitness), sqrt(sum(abs2, residual)),
        Dict{String,Any}(
            "solver" => String(solver_name),
            "fitness" => best_fitness,
            "diagnostics" => diagnostics,
        ))
end
