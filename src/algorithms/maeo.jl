"""
MAEO (Multi-algorithm Evolutionary Operation) unfolding.

Port of bssunfold.core.unfold_maeo (pymoo-based).  Four islands
(nsga3/ctaea/agemoea2/spea2) optimize the spectrum in log-space over 2-3
objectives: chi2, lambda_smooth * ||D2 phi||^2 and optional prior deviation.
Each island run mimics the corresponding pymoo algorithm: NSGA-III (pop =
Das-Dennis niches, random mating, SBX eta=30 prob=1.0, PM 0.9/20,
reference-direction niching survival), CTAEA (single-child SBX eta=30,
mutation prob 1/n_var), AGE-MOEA-II (SBX 0.9/eta=15, PM 0.9/20, NSGA-II
style survival) and SPEA2 (strength + kNN-density survival).  Island runs
ignore warm starting (as pymoo 0.6 does), reseed to seed + cycle before
every run and perform n_gen_per_cycle - 1 mating rounds after the initial
evaluation.  Island quality is hypervolume; the last
convergence_assist_ratio fraction of cycles runs only the best island.  The
solution is the knee point (minimum normalized distance to the ideal point)
of the combined Pareto front; non-negative least squares on failure.
"""
const MAEO_ALGORITHMS = ("nsga3", "ctaea", "agemoea2", "spea2")
# Legacy island flavor factors; kept for name validation and reporting.
const MAEO_ISLAND_PARAM = Dict{String,Float64}(
    "nsga3" => 1.0, "ctaea" => 1.3, "agemoea2" => 0.7, "spea2" => 1.6)

# pymoo defaults used by solve_maeo's islands
const MAEO_SBX_ETA = 15.0
const MAEO_SBX_PROB_VAR = 0.5
const MAEO_PM_ETA = 20.0
const MAEO_PM_PROB = 0.9

struct _MaeoIsland
    pop_fraction::Symbol          # :niches (Das-Dennis count) or :pop_size
    sbx_prob::Float64             # pymoo Crossover-level binary probability
    sbx_prob_var::Float64
    sbx_eta::Float64
    children::Int
    pm_gate::Float64              # Mutation-level binary probability (PM default 0.9)
    pm_prob_var::Float64          # per-variable mutation probability: pymoo default min(0.5, 1/n_var)
    pm_eta::Float64
    mating::Symbol                # :random | :rank | :spea
    survival::Symbol              # :niching | :spea2 | :rank_cd
end

function _maeo_island_cfg(name::AbstractString, n_energy::Int)
    pm_var = min(0.5, 1.0 / n_energy)
    if name == "nsga3"
        _MaeoIsland(:niches, 1.0, 0.5, 30.0, 2, MAEO_PM_PROB, pm_var, MAEO_PM_ETA, :random, :niching)
    elseif name == "ctaea"
        _MaeoIsland(:niches, 1.0, 0.5, 30.0, 1, MAEO_PM_PROB, pm_var, MAEO_PM_ETA, :rank, :niching)
    elseif name == "agemoea2"
        _MaeoIsland(:pop_size, 0.9, 0.5, MAEO_SBX_ETA, 2, MAEO_PM_PROB, pm_var, MAEO_PM_ETA, :rank, :rank_cd)
    elseif name == "spea2"
        _MaeoIsland(:pop_size, 0.9, MAEO_SBX_PROB_VAR, MAEO_SBX_ETA, 2, MAEO_PM_PROB, pm_var, MAEO_PM_ETA, :spea, :spea2)
    else
        throw(ArgumentError("Unknown algorithm '$name'. Available: $(collect(keys(MAEO_ISLAND_PARAM)))"))
    end
end

function _maeo_derivative2(n::Int)
    D2 = zeros(n - 2, n)
    for i in 1:(n-2)
        D2[i, i] = 1.0
        D2[i, i+1] = -2.0
        D2[i, i+2] = 1.0
    end
    return D2
end

function _maeo_objectives(x::Vector{Float64}, A::Matrix{Float64}, b::Vector{Float64},
                          D2::Matrix{Float64}, lambda_smooth::Float64,
                          prior_spectrum::Union{Nothing,Vector{Float64}})
    phi = exp.(clamp.(x, -50.0, 50.0))
    b_norm_sq = dot(b, b)
    b_norm_sq < 1e-20 && (b_norm_sq = 1.0)
    residual = b .- A * phi
    obj_data = dot(residual, residual) / b_norm_sq
    smoothness = dot(D2 * phi, D2 * phi)
    objectives = [obj_data, lambda_smooth * smoothness]
    if prior_spectrum !== nothing
        prior_norm_sq = dot(prior_spectrum, prior_spectrum)
        prior_norm_sq < 1e-20 && (prior_norm_sq = 1.0)
        push!(objectives, dot(phi .- prior_spectrum, phi .- prior_spectrum) / prior_norm_sq)
    end
    violated = any(o -> !isfinite(o), objectives)
    if violated
        objectives = fill(1e10, length(objectives))
    end
    return objectives, violated
end

function _maeo_das_dennis(m::Int, H::Int)
    pts = Vector{Vector{Float64}}()
    function rec!(cur::Vector{Float64}, m_left::Int, h_left::Int)
        if m_left == 1
            push!(pts, vcat(cur, h_left / H))
            return
        end
        for k in 0:h_left
            rec!(vcat(cur, k / H), m_left - 1, h_left - k)
        end
    end
    rec!(Float64[], m, H)
    pts
end

function _maeo_hypervolume(front::Matrix{Float64}, ref_point::Vector{Float64})
    n_obj = size(front, 2)
    if n_obj == 2
        idx = sortperm(front[:, 1])
        fs = front[idx, :]
        hv = 0.0
        last_f2 = ref_point[2]
        for i in 1:size(fs, 1)
            if fs[i, 2] < last_f2
                hv += (ref_point[1] - fs[i, 1]) * (last_f2 - fs[i, 2])
                last_f2 = fs[i, 2]
            end
        end
        return hv
    end
    if n_obj == 3
        # exact O(n^2 log n) sweep along the first objective
        idx = sortperm(front[:, 1])
        fs = front[idx, :]
        hv = 0.0
        for i in 1:size(fs, 1)
            hi = i < size(fs, 1) ? fs[i+1, 1] : ref_point[1]
            hi <= fs[i, 1] && continue
            keep = [j for j in 1:i if fs[j, 2] < ref_point[2] && fs[j, 3] < ref_point[3]]
            isempty(keep) && continue
            f23 = permutedims(reduce(hcat, [fs[j, 2:3] for j in keep]))
            hv += (hi - fs[i, 1]) * _maeo_hypervolume(f23, ref_point[2:3])
        end
        return hv
    end
    minimum_vals = minimum(front, dims=1)[:]
    return prod(ref_point .- minimum_vals) * 0.5
end

function _maeo_front_ranks(objectives::Vector{Vector{Float64}})
    n = length(objectives)
    ranks = fill(-1, n)
    cur = collect(1:n)
    r = 0
    while !isempty(cur)
        nd = Int[]
        for i in cur
            dominated = false
            for j in cur
                i == j && continue
                if all(objectives[j] .<= objectives[i]) &&
                   any(objectives[j] .< objectives[i])
                    dominated = true
                    break
                end
            end
            dominated || push!(nd, i)
        end
        isempty(nd) && append!(nd, cur)
        for i in nd
            ranks[i] = r
        end
        cur = setdiff(cur, nd)
        r += 1
    end
    return ranks
end

function _maeo_crowding(objectives::Vector{Vector{Float64}}, ranks::Vector{Int})
    n = length(objectives)
    cd = zeros(n)
    n_obj = length(objectives[1])
    for r in Set(ranks)
        idx = findall(==(r), ranks)
        if length(idx) <= 2
            for i in idx
                cd[i] = Inf
            end
            continue
        end
        for obj in 1:n_obj
            vals = [objectives[i][obj] for i in idx]
            order = sortperm(vals)
            cd[idx[order[1]]] = Inf
            cd[idx[order[end]]] = Inf
            span = vals[order[end]] - vals[order[1]]
            span > 0 || continue
            for p in 2:(length(order)-1)
                i = idx[order[p]]
                cd[i] += (vals[order[p+1]] - vals[order[p-1]]) / span
            end
        end
    end
    return cd
end

# pymoo PolynomialMutation (mut_pm): per-variable bernoulli mask, index polynomial
function _maeo_polynomial_mutation(rng::AbstractRNG, x::Vector{Float64},
                                   lo::Vector{Float64}, hi::Vector{Float64},
                                   prob::Float64, eta::Float64)
    y = copy(x)
    for i in eachindex(x)
        rand(rng) < prob || continue
        span = hi[i] - lo[i]
        span <= 0 && continue
        delta1 = (x[i] - lo[i]) / span
        delta2 = (hi[i] - x[i]) / span
        r = rand(rng)
        dq = if r <= 0.5
            (2r + (1 - 2r) * (1 - delta1)^(eta + 1))^(1 / (eta + 1)) - 1
        else
            1 - (2(1 - r) + (2r - 1) * (1 - delta2)^(eta + 1))^(1 / (eta + 1))
        end
        y[i] = clamp(x[i] + dq * span, lo[i], hi[i])
    end
    return y
end

# pymoo SBX (cross_sbx): exact betaq formulas, single rand shared by both children
function _maeo_sbx_pair(rng::AbstractRNG, p1::Vector{Float64}, p2::Vector{Float64},
                        eta::Float64, prob::Float64, prob_var::Float64,
                        lo::Vector{Float64}, hi::Vector{Float64})
    c1 = copy(p1)
    c2 = copy(p2)
    rand(rng) < prob || return (c1, c2)
    for i in eachindex(p1)
        rand(rng) < prob_var || continue
        a = p1[i]
        b = p2[i]
        abs(a - b) <= 1e-14 && continue
        y1 = min(a, b)
        y2 = max(a, b)
        delta = y2 - y1
        r = rand(rng)
        beta = 1 + 2 * (y1 - lo[i]) / delta
        alpha = 2 - beta^(-(eta + 1))
        betaq = r <= 1 / alpha ? (r * alpha)^(1 / (eta + 1)) : (1 / (2 - r * alpha))^(1 / (eta + 1))
        beta2 = 1 + 2 * (hi[i] - y2) / delta
        alpha2 = 2 - beta2^(-(eta + 1))
        betaq2 = r <= 1 / alpha2 ? (r * alpha2)^(1 / (eta + 1)) : (1 / (2 - r * alpha2))^(1 / (eta + 1))
        half = 0.5 * (y1 + y2)
        c1[i] = clamp(half - 0.5 * betaq * delta, lo[i], hi[i])
        c2[i] = clamp(half + 0.5 * betaq2 * delta, lo[i], hi[i])
    end
    for i in eachindex(p1)
        ((c1[i] > c2[i]) != (rand(rng) < 0.5)) && ((c1[i], c2[i]) = (c2[i], c1[i]))
    end
    return c1, c2
end

function _maeo_pick_parent(rng::AbstractRNG, mode::Symbol, popn::Int,
                           ranks::Vector{Int}, cd::Vector{Float64},
                           score::Vector{Float64})
    i = rand(rng, 1:popn)
    j = rand(rng, 1:popn)
    if mode === :random
        return rand(rng, Bool) ? i : j
    elseif mode === :rank
        ri, rj = ranks[i], ranks[j]
        if ri < rj
            return i
        elseif rj < ri
            return j
        else
            ci, cj = cd[i], cd[j]
            (isinf(ci) || ci >= cj) ? i : j
        end
    else
        score[i] <= score[j] ? i : j
    end
end

# pymoo PM: the mutated vector is computed for the child but adopted only with the
# Mutation-level binary probability (cfg.pm_gate).
function _maeo_mutate_child(rng::AbstractRNG, c::Vector{Float64},
                            lo::Vector{Float64}, hi::Vector{Float64}, cfg::_MaeoIsland)
    y = _maeo_polynomial_mutation(rng, c, lo, hi, cfg.pm_prob_var, cfg.pm_eta)
    rand(rng) <= cfg.pm_gate ? y : c
end

function _maeo_make_offspring(rng::AbstractRNG, pop, ranks, cd, score, cfg::_MaeoIsland,
                              lo, hi, n_off::Int)
    popn = length(pop)
    off = Vector{Vector{Float64}}()
    n_matings = ceil(Int, n_off / cfg.children)
    for _ in 1:n_matings
        i1 = _maeo_pick_parent(rng, cfg.mating, popn, ranks, cd, score)
        i2 = _maeo_pick_parent(rng, cfg.mating, popn, ranks, cd, score)
        c1, c2 = _maeo_sbx_pair(rng, pop[i1], pop[i2], cfg.sbx_eta, cfg.sbx_prob,
                                cfg.sbx_prob_var, lo, hi)
        if cfg.children == 1
            push!(off, _maeo_mutate_child(rng, c1, lo, hi, cfg))
        else
            push!(off, _maeo_mutate_child(rng, c1, lo, hi, cfg))
            push!(off, _maeo_mutate_child(rng, c2, lo, hi, cfg))
        end
    end
    return off[1:n_off]
end

# SPEA2 strength + kNN density fitness (smaller is better); distances on the
# raw objective cloud (pymoo normalizes them; approximation).
function _maeo_spea2_score(F::Vector{Vector{Float64}})
    n = length(F)
    S = zeros(Int, n)
    R = zeros(Int, n)
    dists = [[norm(F[i] - F[j]) for j in 1:n] for i in 1:n]
    for i in 1:n, j in 1:n
        i == j && continue
        if all(F[i] .<= F[j]) && any(F[i] .< F[j])
            S[i] += 1
            R[j] += S[i]
        end
    end
    k = floor(Int, sqrt(n))
    D = zeros(n)
    for i in 1:n
        sd = sort(dists[i])
        D[i] = 1 / (sd[k+1] + 2)
    end
    score = R .+ D
    return score, R
end

function _maeo_survive_spea2(F::Vector{Vector{Float64}}, N::Int)
    score, R = _maeo_spea2_score(F)
    nondom = [i for i in eachindex(F) if R[i] == 0]
    if length(nondom) >= N
        survivors = copy(nondom)
        while length(survivors) > N
            worst, wdist = 0, Inf
            for idx in eachindex(survivors)
                i = survivors[idx]
                dmin = Inf
                for jdx in eachindex(survivors)
                    jdx == idx && continue
                    d = norm(F[i] - F[survivors[jdx]])
                    d < dmin && (dmin = d)
                end
                if dmin < wdist
                    worst, wdist = idx, dmin
                end
            end
            deleteat!(survivors, worst)
        end
        return survivors, score
    end
    rest = sort([i for i in eachindex(F) if R[i] > 0]; by = i -> score[i])
    return vcat(nondom, rest[1:min(N - length(nondom), length(rest))]), score
end

# NSGA-III reference-direction niching survival (approximate association).
function _maeo_survive_niching(rng::AbstractRNG, F::Vector{Vector{Float64}},
                               ranks::Vector{Int}, ref_dirs::Vector{Vector{Float64}}, N::Int)
    front0 = findall(==(0), ranks)
    m = length(F[1])
    ideal = [minimum(F[i][k] for i in front0) for k in 1:m]
    nadir = [maximum(F[i][k] for i in front0) for k in 1:m]
    spanv = [(nadir[k] - ideal[k]) < 1e-12 ? 1e-12 : nadir[k] - ideal[k] for k in 1:m]
    dirs = [v / norm(v) for v in ref_dirs]
    n_dirs = length(dirs)
    r_needed = maximum(ranks)
    cumulative = 0
    for r in sort!(unique(ranks))
        cumulative += count(==(r), ranks)
        if cumulative >= N
            r_needed = r
            break
        end
    end
    base = [i for i in eachindex(F) if ranks[i] < r_needed]
    last = [i for i in eachindex(F) if ranks[i] == r_needed]
    assign = zeros(Int, length(F))
    dists = fill(Inf, length(F))
    for i in vcat(base, last)
        n = [(F[i][k] - ideal[k]) / spanv[k] for k in 1:m]
        bd, bi = Inf, 1
        for (d, v) in enumerate(dirs)
            proj = dot(n, v)
            pd = norm(n .- proj .* v)
            if pd < bd
                bd, bi = pd, d
            end
        end
        assign[i] = bi
        dists[i] = bd
    end
    selected = copy(base)
    counts = zeros(Int, n_dirs)
    for i in base
        counts[assign[i]] += 1
    end
    remaining = Set(last)
    while length(selected) < N && !isempty(remaining)
        act = [i for i in last if i in remaining]
        minc = minimum(counts[assign[i]] for i in act)
        tied = unique([assign[i] for i in act if counts[assign[i]] == minc])
        d = rand(rng, tied)
        cand = [i for i in act if assign[i] == d]
        pick = counts[d] == 0 ? cand[argmin(dists[i] for i in cand)] : rand(rng, cand)
        push!(selected, pick)
        counts[d] += 1
        delete!(remaining, pick)
    end
    return selected[1:min(N, length(selected))]
end

function _maeo_survive_rank_cd(F::Vector{Vector{Float64}}, N::Int)
    ranks = _maeo_front_ranks(F)
    cd = _maeo_crowding(F, ranks)
    order = sort(eachindex(F); by = i -> (ranks[i], isinf(cd[i]) ? -Inf : -cd[i]))
    return order[1:min(N, length(order))], ranks, cd
end

# per-generation mating scores for the island (SPEA2 fitness or rank/crowding)
function _maeo_island_state(cfg::_MaeoIsland, pop_F::Vector{Vector{Float64}}, pop_v::Vector{Bool})
    n = length(pop_F)
    ranks = zeros(Int, n)
    cd = fill(Inf, n)
    score = fill(Inf, n)
    feasible = [i for i in eachindex(pop_F) if !pop_v[i]]
    isempty(feasible) && return (ranks, cd, score)
    sub = [pop_F[i] for i in feasible]
    if cfg.survival === :spea2
        sc, _ = _maeo_spea2_score(sub)
        fill!(ranks, 0)
        fill!(cd, 0.0)
        for (k, i) in enumerate(feasible)
            score[i] = sc[k]
        end
    else
        rk = _maeo_front_ranks(sub)
        cv = _maeo_crowding(sub, rk)
        for (k, i) in enumerate(feasible)
            ranks[i] = rk[k]
            cd[i] = cv[k]
        end
    end
    return (ranks, cd, score)
end

function _maeo_knee_select(objectives::Vector{Vector{Float64}}, solutions::Vector{Vector{Float64}})
    isempty(objectives) && return nothing, nothing
    mins = minimum(reduce(hcat, objectives), dims=2)[:]
    maxs = maximum(reduce(hcat, objectives), dims=2)[:]
    rng_ = maxs .- mins
    rng_[rng_ .< 1e-10] .= 1.0
    best_i = 1
    best_d = Inf
    for (i, o) in enumerate(objectives)
        d = norm((o .- mins) ./ rng_)
        if d < best_d
            best_d = d
            best_i = i
        end
    end
    return solutions[best_i], objectives[best_i]
end

"""
    solve_maeo(A, b, x0=nothing; E_MeV=nothing, n_cycles=20, n_gen_per_cycle=10,
               pop_size=100, algorithms=nothing, lambda_smooth=0.01,
               prior_spectrum=nothing, initial_spectrum=nothing,
               convergence_assist_ratio=0.2, seed=nothing, verbose=false)
      -> UnfoldResult

MAEO ensemble: several "islands" (one per algorithm from
`algorithms`, by default `["nsga3", "ctaea", "agemoea2", "spea2"]`)
optimize the spectrum in log-space over 2-3 objectives; after each
migration cycle island performance is evaluated by hypervolume,
and in the convergence phase (the last `convergence_assist_ratio` cycles)
only the best island runs.  The final solution is the knee-point
of the combined Pareto front; on failure - fallback to non-negative
LS.  `initial_spectrum` provides the warm start; the positional `x0`
mirrors Python's `E_MeV` slot (metadata only) - with no `initial_spectrum`
the seed is `sum(b)/n_det/mean(A)` in every bin, as in Python.
"""
function solve_maeo(A::AbstractMatrix, b::AbstractVector, x0::Union{Nothing,AbstractVector}=nothing;
                    E_MeV::Union{Nothing,AbstractVector}=nothing,
                    n_cycles::Integer=20,
                    n_gen_per_cycle::Integer=10,
                    pop_size::Integer=100,
                    algorithms::Union{Nothing,AbstractVector}=nothing,
                    lambda_smooth::Real=0.01,
                    prior_spectrum::Union{Nothing,AbstractVector}=nothing,
                    initial_spectrum::Union{Nothing,AbstractVector}=nothing,
                    convergence_assist_ratio::Real=0.2,
                    seed::Union{Nothing,Integer}=nothing,
                    verbose::Bool=false)
    AF = Matrix{Float64}(A)
    bf = Vector{Float64}(b)
    _, n_energy = size(AF)
    n_detectors = size(AF, 1)
    algo_names = algorithms === nothing ? collect(MAEO_ALGORITHMS) : String.(algorithms)
    for algo in algo_names
        haskey(MAEO_ISLAND_PARAM, algo) ||
            throw(ArgumentError("Unknown algorithm '$algo'. Available: $(collect(keys(MAEO_ISLAND_PARAM)))"))
    end
    n_islands = length(algo_names)
    D2 = n_energy >= 3 ? _maeo_derivative2(n_energy) : zeros(0, n_energy)
    lam_smooth = Float64(lambda_smooth)
    prior = prior_spectrum === nothing ? nothing : Vector{Float64}(prior_spectrum)

    effective_initial = initial_spectrum === nothing ? nothing : Vector{Float64}(initial_spectrum)
    rng = MersenneTwister(seed === nothing ? 1234 : Int(seed))

    n_migration_cycles = clamp(floor(Int, n_cycles * (1 - Float64(convergence_assist_ratio))), 0, n_cycles)

    if effective_initial !== nothing
        seed_spec = max.(effective_initial, 1e-10)
        seed_log = log.(seed_spec)
    else
        base = sum(bf) / n_detectors / (sum(AF) / max(length(AF), 1))
        seed_log = fill(log(max(base, 1e-10)), n_energy)
    end
    lower_bounds = seed_log .- 3.0
    upper_bounds = seed_log .+ 3.0

    n_obj = prior === nothing ? 2 : 3
    ref_dirs = _maeo_das_dennis(n_obj, 12)
    n_ref = length(ref_dirs)
    n_rounds = max(Int(n_gen_per_cycle) - 1, 0)

    hv_history = Dict{String,Vector{Float64}}(a => Float64[] for a in algo_names)
    pop_history = Dict{String,Vector{Int}}(a => Int[] for a in algo_names)
    all_pareto_solutions = Vector{Vector{Float64}}()
    all_pareto_objectives = Vector{Vector{Float64}}()
    n_evals = 0

    best_island_idx = 1

    for cycle in 0:(n_cycles-1)
        is_convergence_phase = cycle >= n_migration_cycles
        island_hvs = Dict{String,Float64}()

        for (algo_idx, algo_name) in enumerate(algo_names)
            if is_convergence_phase && algo_idx != best_island_idx
                continue
            end
            cfg = _maeo_island_cfg(algo_name, n_energy)
            island_size = cfg.pop_fraction === :niches ? n_ref : Int(pop_size)
            # mirror Python: minimize(..., seed=seed+cycle) reseeds before every island run
            island_rng = seed === nothing ? rng : MersenneTwister(Int(seed) + cycle)

            # pymoo 0.6 ignores the warm-started `algorithm.pop`: every island run
            # starts from a fresh uniform sample with the cycle's seed.
            pop = [lower_bounds .+ rand(island_rng, n_energy) .* (upper_bounds .- lower_bounds)
                   for _ in 1:island_size]
            pop_objs = map(x -> _maeo_objectives(x, AF, bf, D2, lam_smooth, prior), pop)
            n_evals += length(pop)
            pop_F = Vector{Vector{Float64}}([o[1] for o in pop_objs])
            pop_v = [o[2] for o in pop_objs]

            ranks, cd, score = _maeo_island_state(cfg, pop_F, pop_v)

            for gen in 1:n_rounds
                offspring = _maeo_make_offspring(island_rng, pop, ranks, cd, score, cfg,
                                                 lower_bounds, upper_bounds, island_size)
                off_objs = map(x -> _maeo_objectives(x, AF, bf, D2, lam_smooth, prior), offspring)
                n_evals += length(offspring)

                combined = vcat(pop, offspring)
                comb_objs = vcat(pop_objs, off_objs)
                comb_F = Vector{Vector{Float64}}([o[1] for o in comb_objs])
                comb_v = [o[2] for o in comb_objs]

                if cfg.survival === :spea2
                    surv, sc = _maeo_survive_spea2(comb_F, island_size)
                elseif cfg.survival === :niching
                    surv = _maeo_survive_niching(island_rng, comb_F, _maeo_front_ranks(comb_F),
                                                 ref_dirs, island_size)
                else
                    surv, _, _ = _maeo_survive_rank_cd(comb_F, island_size)
                end
                pop = [combined[i] for i in surv]
                pop_objs = [comb_objs[i] for i in surv]
                pop_F = Vector{Vector{Float64}}([o[1] for o in pop_objs])
                pop_v = [o[2] for o in pop_objs]
                for i in eachindex(pop)
                    all(isfinite, pop[i]) ||
                        (pop[i] = lower_bounds .+ rand(island_rng, n_energy) .* (upper_bounds .- lower_bounds);
                         pop_objs[i] = _maeo_objectives(pop[i], AF, bf, D2, lam_smooth, prior);
                         pop_F[i] = pop_objs[i][1])
                end
                ranks, cd, score = _maeo_island_state(cfg, pop_F, pop_v)
            end

            cands = [i for i in eachindex(pop) if !pop_v[i]]
            if !isempty(cands)
                front_idx = Vector{Int}()
                if cfg.survival === :spea2
                    sc, R = _maeo_spea2_score([pop_F[i] for i in cands])
                    for (k, i) in enumerate(cands)
                        R[k] == 0 && push!(front_idx, i)
                    end
                else
                    ranks_f = _maeo_front_ranks([pop_F[i] for i in cands])
                    for (k, i) in enumerate(cands)
                        ranks_f[k] == 0 && push!(front_idx, i)
                    end
                end
                front = permutedims(reduce(hcat, [pop_F[i] for i in front_idx]))
                ref_point = maximum(front, dims=1)[:] .* 1.1 .+ 0.1
                hv = _maeo_hypervolume(front, ref_point)
                island_hvs[algo_name] = hv
                push!(hv_history[algo_name], hv)
                push!(pop_history[algo_name], length(pop))
                for i in front_idx
                    push!(all_pareto_objectives, pop_F[i])
                    push!(all_pareto_solutions, pop[i])
                end
            else
                island_hvs[algo_name] = 0.0
                push!(hv_history[algo_name], 0.0)
                push!(pop_history[algo_name], 0)
            end
        end

        if !is_convergence_phase && length(island_hvs) > 1
            ran = [a for a in algo_names if haskey(island_hvs, a)]
            best_island_idx = findfirst(==(ran[argmax([island_hvs[a] for a in ran])]), algo_names)
        end
        verbose && println("MAEO cycle $(cycle+1)/$n_cycles, best island: $(algo_names[best_island_idx])")
    end

    best_solution_log, best_objectives = _maeo_knee_select(all_pareto_objectives, all_pareto_solutions)
    if best_solution_log === nothing
        fallback = max.(AF \ bf, 0.0)
        extra = Dict{String,Any}(
            "n_cycles_run" => n_cycles,
            "best_algorithm" => algo_names[best_island_idx],
            "hypervolume_history" => hv_history,
            "population_history" => pop_history,
            "algorithms_used" => algo_names,
            "n_evaluations" => n_evals,
            "fallback" => true,
        )
        return UnfoldResult(fallback, n_cycles, false, norm(bf .- AF * fallback), extra)
    end

    best_solution = max.(exp.(clamp.(best_solution_log, -50.0, 50.0)), 0.0)
    residual = bf .- AF * best_solution
    extra = Dict{String,Any}(
        "n_cycles_run" => n_cycles,
        "best_algorithm" => algo_names[best_island_idx],
        "hypervolume_history" => hv_history,
        "population_history" => pop_history,
        "pareto_front" => all_pareto_objectives,
        "objectives" => best_objectives,
        "algorithms_used" => algo_names,
        "n_islands" => n_islands,
        "n_evaluations" => n_evals,
        "convergence_assist_ratio" => Float64(convergence_assist_ratio),
    )
    return UnfoldResult(best_solution, n_cycles, true, norm(residual), extra)
end

"""
    solve_maeo_ensemble(A, b, x0=nothing; ...; migration_method="hypervolume",
                        parallel=false) -> UnfoldResult

Variants of `solve_maeo` with explicit ensemble control: the migration strategy
(`"hypervolume"` or `"uniform"`) and the parallel-island flag are
saved in `extra`.  Defaults match Python: `n_cycles=25, n_gen_per_cycle=8`.
"""
function solve_maeo_ensemble(A::AbstractMatrix, b::AbstractVector, x0::Union{Nothing,AbstractVector}=nothing;
                             E_MeV::Union{Nothing,AbstractVector}=nothing,
                             n_cycles::Integer=25,
                             n_gen_per_cycle::Integer=8,
                             pop_size::Integer=100,
                             algorithms=nothing,
                             lambda_smooth::Real=0.01,
                             prior_spectrum=nothing,
                             initial_spectrum=nothing,
                             convergence_assist_ratio::Real=0.2,
                             migration_method::String="hypervolume",
                             seed::Union{Nothing,Integer}=nothing,
                             verbose::Bool=false,
                             parallel::Bool=false)
    r = solve_maeo(A, b, x0; E_MeV=E_MeV, n_cycles=n_cycles,
                   n_gen_per_cycle=n_gen_per_cycle, pop_size=pop_size,
                   algorithms=algorithms, lambda_smooth=lambda_smooth,
                   prior_spectrum=prior_spectrum, initial_spectrum=initial_spectrum,
                   convergence_assist_ratio=convergence_assist_ratio,
                   seed=seed, verbose=verbose)
    r.extra["migration_method"] = migration_method
    r.extra["parallel_enabled"] = parallel
    return r
end
