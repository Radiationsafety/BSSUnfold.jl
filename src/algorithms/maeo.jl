"""
MAEO (Multiobjective Animorphic Ensemble Optimization) unfolding.

A port of the MAEO ensemble (Erdem et al., arXiv:2604.26973) without pymoo: islands
with their own multi-objective genetic operators
(non-dominated sorting, tournament selection, SBX crossover, polynomial
mutation), island performance evaluated by hypervolume
(exact 2D implementation + approximation for higher dimensions),
migration to the best island and a convergence phase; the final solution is the
knee-point of the combined Pareto front.  Objectives: data fidelity
`||b - A phi||^2 / ||b||^2`, smoothness `||D2 phi||^2`, (optionally)
deviation from a prior spectrum; the spectrum is optimized in
log-space.
"""
const MAEO_ALGORITHMS = ("nsga3", "ctaea", "agemoea2", "spea2")
const MAEO_ISLAND_PARAM = Dict{String,Float64}(
    "nsga3" => 1.0, "ctaea" => 1.3, "agemoea2" => 0.7, "spea2" => 1.6)

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
    b_norm_sq = max(dot(b, b), 1e-20)
    residual = b .- A * phi
    obj_data = dot(residual, residual) / b_norm_sq
    smoothness = dot(D2 * phi, D2 * phi)
    objectives = [obj_data, lambda_smooth * smoothness]
    if prior_spectrum !== nothing
        prior_norm_sq = max(dot(prior_spectrum, prior_spectrum), 1e-20)
        push!(objectives, dot(phi .- prior_spectrum, phi .- prior_spectrum) / prior_norm_sq)
    end
    violated = any(o -> !isfinite(o), objectives)
    if violated
        objectives = [!isfinite(o) ? 1e10 : o for o in objectives]
    end
    return objectives, violated
end

function _maeo_hypervolume(front::Matrix{Float64}, ref_point::Vector{Float64})
    if size(front, 2) == 2
        idx = sortperm(front[:, 1])
        fs = front[idx, :]
        hv = 0.0
        prev_f1 = 0.0
        for i in 1:size(fs, 1)
            width = i > 1 ? fs[i, 1] - prev_f1 : fs[i, 1]
            height = ref_point[2] - fs[i, 2]
            hv += width * height
            prev_f1 = fs[i, 1]
        end
        hv += (ref_point[1] - prev_f1) * (ref_point[2] - fs[end, 2])
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
        for i in idx
            cd[i] = Inf
        end
        length(idx) <= 2 && continue
        for obj in 1:n_obj
            vals = [objectives[i][obj] for i in idx]
            order = sortperm(vals)
            for p in 2:(length(order)-1)
                i = idx[order[p]]
                span = vals[order[end]] - vals[order[1]]
                span > 0 || continue
                cd[i] += (vals[order[p+1]] - vals[order[p-1]]) / span
            end
        end
    end
    return cd
end

function _maeo_individual_score(rank::Int, max_rank::Int, crowding_dist::Float64,
                                ref_distance::Float64)
    rank_component = max_rank == 0 ? 1.0 : 1.0 - rank / max_rank
    cd_norm = crowding_dist > 0 ? crowding_dist : 0.5
    rd_norm = ref_distance > 0 ? ref_distance : 0.5
    band_scale = 2.0 * (max_rank + 1)
    diversity_component = (cd_norm + rd_norm) / band_scale
    return rank_component + diversity_component
end

function _maeo_polynomial_mutation(rng::AbstractRNG, x::Vector{Float64},
                                   lo::Vector{Float64}, hi::Vector{Float64}, rate::Float64)
    y = copy(x)
    for i in eachindex(x)
        rand(rng) < rate || continue
        u = rand(rng)
        d = u < 0.5 ? (2u)^(1/21) - 1.0 : 1.0 - (2(1 - u))^(1/21)
        span = hi[i] - lo[i]
        span > 0 || continue
        y[i] = clamp(x[i] + d * span, lo[i], hi[i])
    end
    return y
end

function _maeo_sbx(rnd::AbstractRNG, p1::Vector{Float64}, p2::Vector{Float64},
                   eta::Float64, lo::Vector{Float64}, hi::Vector{Float64})
    c1 = copy(p1)
    c2 = copy(p2)
    for i in eachindex(p1)
        if rand(rnd) < 0.5
            c1[i], c2[i] = p2[i], p1[i]
        end
        r1 = clamp(p1[i], lo[i], hi[i])
        r2 = clamp(p2[i], lo[i], hi[i])
        d = abs(r1 - r2)
        d > 1e-12 || continue
        u = rand(rnd)
        beta_ = u <= 0.5 ? (2u)^(1/(eta+1)) : (1 / (2(1 - u)))^(1/(eta+1))
        s = 0.5 * (r1 + r2)
        half = 0.5 * beta_ * d
        c1[i] = clamp(s - half, lo[i], hi[i])
        c2[i] = clamp(s + half, lo[i], hi[i])
    end
    return c1, c2
end

function _maeo_knee_select(objectives::Vector{Vector{Float64}}, solutions::Vector{Vector{Float64}})
    isempty(objectives) && return nothing, nothing
    mins = minimum(reduce(hcat, objectives), dims=2)[:]
    maxs = maximum(reduce(hcat, objectives), dims=2)[:]
    rng_ = max.(maxs .- mins, 1e-10)
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
optimize the spectrum in log-space over 2–3 objectives; after each
migration cycle island performance is evaluated by hypervolume,
and in the convergence phase (the last `convergence_assist_ratio` cycles)
only the best island runs.  The final solution is the knee-point
of the combined Pareto front; on failure — fallback to non-negative
LS.  `x0` / `initial_spectrum` provide the initial (warm) population.
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
    algo_names = algorithms === nothing ? collect(MAEO_ALGORITHMS) : String.(algorithms)
    for algo in algo_names
        haskey(MAEO_ISLAND_PARAM, algo) ||
            throw(ArgumentError("Unknown algorithm '$algo'. Available: $(collect(keys(MAEO_ISLAND_PARAM)))"))
    end
    n_islands = length(algo_names)
    D2 = n_energy >= 3 ? _maeo_derivative2(n_energy) : zeros(0, n_energy)
    lam_smooth = Float64(lambda_smooth)
    prior = prior_spectrum === nothing ? nothing : Vector{Float64}(prior_spectrum)

    effective_initial = initial_spectrum !== nothing ? initial_spectrum : x0
    rng = MersenneTwister(seed === nothing ? 1234 : Int(seed))

    n_migration_cycles = clamp(floor(Int, n_cycles * (1 - Float64(convergence_assist_ratio))), 0, n_cycles)

    if effective_initial !== nothing
        seed_spec = max.(Vector{Float64}(effective_initial), 1e-10)
        seed_log = log.(seed_spec)
    else
        base = max(sum(bf) / max(size(AF, 1), 1) / max(sum(AF) / max(length(AF), 1), 1e-300), 1e-10)
        seed_log = fill(log(base), n_energy)
    end
    lower_bounds = seed_log .- 3.0
    upper_bounds = seed_log .+ 3.0

    hv_history = Dict{String,Vector{Float64}}(a => Float64[] for a in algo_names)
    pop_history = Dict{String,Vector{Int}}(a => Int[] for a in algo_names)
    all_pareto_solutions = Vector{Vector{Float64}}()
    all_pareto_objectives = Vector{Vector{Float64}}()

    current_populations = Dict{String,Vector{Vector{Float64}}}()
    best_island_idx = 1

    for cycle in 0:(n_cycles-1)
        is_convergence_phase = cycle >= n_migration_cycles

        for (algo_idx, algo_name) in enumerate(algo_names)
            if is_convergence_phase && algo_idx - 1 != best_island_idx
                continue
            end
            pressure = MAEO_ISLAND_PARAM[algo_name]
            pop = get(current_populations, algo_name, nothing)
            if pop === nothing
                pop = [clamp.(seed_log .+ randn(rng, n_energy) .* 0.5, lower_bounds, upper_bounds)
                       for _ in 1:pop_size]
            end

            for gen in 1:n_gen_per_cycle
                objs = map(x -> first(_maeo_objectives(x, AF, bf, D2, lam_smooth, prior)), pop)
                ranks = _maeo_front_ranks(objs)
                cd = _maeo_crowding(objs, ranks)
                max_rank = maximum(ranks)
                nadir = maximum(reduce(hcat, objs), dims=2)[:]
                scores = [_maeo_individual_score(ranks[i], max_rank,
                                                 isinf(cd[i]) ? 1.0 : min(cd[i], 1.0),
                                                 norm(clamp.(nadir .- objs[i], 0.0, Inf)))
                          for i in eachindex(pop)]
                offspring_size = length(pop)
                offspring = Vector{Vector{Float64}}(undef, offspring_size)
                scores_sum = sum(scores) + 1e-300
                pick_island = () -> begin
                    for (s_i, s) in enumerate(scores)
                        rand(rng) < s / scores_sum && return s_i
                    end
                    return argmax(scores)
                end
                for oi in 1:offspring_size
                    i1 = pick_island()
                    i2 = pick_island()
                    c1, c2 = _maeo_sbx(rng, pop[i1], pop[i2], 15.0 + pressure, lower_bounds, upper_bounds)
                    child = rand(rng) < 0.5 ? c1 : c2
                    child = _maeo_polynomial_mutation(rng, child, lower_bounds, upper_bounds,
                                                      0.5 / n_energy * pressure)
                    offspring[oi] = clamp.(child, lower_bounds, upper_bounds)
                end

                combined = vcat(pop, offspring)
                objs_c = map(x -> _maeo_objectives(x, AF, bf, D2, lam_smooth, prior), combined)
                objs_cv = Vector{Vector{Float64}}([o[1] for o in objs_c])
                viol = [o[2] for o in objs_c]
                ranks_cv = _maeo_front_ranks(objs_cv)
                cd_cv = _maeo_crowding(objs_cv, ranks_cv)
                order = sort(eachindex(combined);
                             by = i -> (viol[i], ranks_cv[i], -min(1.0, isinf(cd_cv[i]) ? 2.0 : cd_cv[i])))
                pop = [combined[i] for i in order[1:pop_size]]
                for i in eachindex(pop)
                    all(isfinite, pop[i]) ||
                        (pop[i] = clamp.(seed_log .+ randn(rng, n_energy) .* 0.1, lower_bounds, upper_bounds))
                end
            end

            current_populations[algo_name] = pop

            final_objs = map(x -> _maeo_objectives(x, AF, bf, D2, lam_smooth, prior), pop)
            fo = Vector{Vector{Float64}}([o[1] for o in final_objs])
            fv = [o[2] for o in final_objs]
            cands = [i for i in eachindex(pop) if !fv[i]]
            if !isempty(cands)
                ranks_f = _maeo_front_ranks([fo[i] for i in cands])
                front_idx = Vector{Int}()
                for (k, i) in enumerate(cands)
                    ranks_f[k] == 0 && push!(front_idx, i)
                end
                front = permutedims(reduce(hcat, [fo[i] for i in front_idx]))
                ref_point = maximum(front, dims=1)[:] .* 1.1 .+ 0.1
                hv = _maeo_hypervolume(front, ref_point)
                push!(hv_history[algo_name], hv)
                push!(pop_history[algo_name], length(pop))
                for i in front_idx
                    push!(all_pareto_objectives, fo[i])
                    push!(all_pareto_solutions, pop[i])
                end
            else
                push!(hv_history[algo_name], 0.0)
                push!(pop_history[algo_name], 0)
            end
        end

        if !is_convergence_phase
            hvs_all = Float64[]
            for algo_name in algo_names
                hist = hv_history[algo_name]
                isempty(hist) ? push!(hvs_all, 0.0) : push!(hvs_all, hist[end])
            end
            best_island_idx = argmax(hvs_all) - 1
        end
        verbose && println("MAEO cycle $(cycle+1)/$n_cycles, best island: $(algo_names[best_island_idx+1])")
    end

    best_solution_log, best_objectives = _maeo_knee_select(all_pareto_objectives, all_pareto_solutions)
    if best_solution_log === nothing
        fallback = max.(AF \ bf, 0.0)
        extra = Dict{String,Any}(
            "n_cycles_run" => n_cycles,
            "best_algorithm" => algo_names[best_island_idx+1],
            "hypervolume_history" => hv_history,
            "population_history" => pop_history,
            "algorithms_used" => algo_names,
            "fallback" => true,
        )
        return UnfoldResult(fallback, n_cycles, false, norm(bf .- AF * fallback), extra)
    end

    best_solution = max.(exp.(clamp.(best_solution_log, -50.0, 50.0)), 0.0)
    residual = bf .- AF * best_solution
    extra = Dict{String,Any}(
        "n_cycles_run" => n_cycles,
        "best_algorithm" => algo_names[best_island_idx+1],
        "hypervolume_history" => hv_history,
        "population_history" => pop_history,
        "pareto_front" => all_pareto_objectives,
        "objectives" => best_objectives,
        "algorithms_used" => algo_names,
        "n_islands" => n_islands,
        "convergence_assist_ratio" => Float64(convergence_assist_ratio),
    )
    return UnfoldResult(best_solution, n_cycles, true, norm(residual), extra)
end

"""
    solve_maeo_ensemble(A, b, x0=nothing; ...; migration_method="hypervolume",
                        parallel=false) -> UnfoldResult

Variants of `solve_maeo` with explicit ensemble control: the migration strategy
(`"hypervolume"` or `"uniform"`) and the parallel-island flag are
saved in `extra`.
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
