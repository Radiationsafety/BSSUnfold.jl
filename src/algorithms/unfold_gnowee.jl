"""
Gnowee-based unfolding (ported from `unfold_gnowee.py` + the vendored
`_gnowee.py` hybrid metaheuristic, bssunfold 0.29.0).

Gnowee (Bevins & Parsons, UC Berkeley / Slaybaugh Lab) combines four
complementary heuristics to balance diversification and intensification:

* Lévy flights (Cuckoo Search) — heavy-tailed random walks sampled with the
  Mantegna (1994) algorithm;
* crossover — golden-ratio weighted recombination of the best parent with an
  elite partner (Walton 2011 / Storn 1997);
* mutation — DE-style differential perturbation (Yang 2010 / Storn 1997);
* scatter search — path-relinking of two elite parents (Egea 2009).

The population is updated with elitism, a Metropolis-Hastings acceptance
fallback and stall-driven restarts.  The optimizer is *vendored* in this file
(Python imports no external `gnowee` package): `_gnowee_*` functions below are
the port of `bssunfold/core/_gnowee.py` restricted to the continuous-variable
heuristics, as in the original.

The numerical strategy mirrors `solve_genetic` (same helpers, same objective):

* search in log space (`y = log(x)`), so positivity is enforced;
* seed the population with a Landweber warm start (or `x0`);
* bound the search to `log(seed) ± half_range` decades;
* minimize a scale-consistent objective (relative residual, Tikhonov
  regularization and second-difference smoothness are dimensionless).

Randomness: Python draws from a single `numpy.random.default_rng(random_state)`
(PCG64) stream; this port threads one `MersenneTwister` seeded from
`random_state`, following the `solve_genetic` / `solve_qubo` convention.  The
streams therefore differ (statistical agreement, not bit reproduction).
"""

# ─── Special-function helper (no SpecialFunctions dependency) ───────────────

"""
    _gnowee_gamma(x) -> Float64

Gamma function via the Lanczos approximation (g = 7), used by the Lévy
sampling constants in place of `math.gamma`.
"""
function _gnowee_gamma(x::Real)
    xx = Float64(x)
    if xx < 0.5
        return pi / (sin(pi * xx) * _gnowee_gamma(1.0 - xx))
    end
    c = (0.99999999999980993, 676.5203681218851, -1259.1392167224028,
         771.32342877765313, -176.61502916414526, 12.507343278486836,
         -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7)
    xx -= 1.0
    a = c[1]
    t = xx + 7.5
    @inbounds for i in 2:9
        a += c[i] / (xx + i - 1)
    end
    return sqrt(2pi) * t^(xx + 0.5) * exp(-t) * a
end

"""
    _gnowee_polyval(p, x) -> Float64

Horner evaluation of the coefficient tuple `p` (highest degree first), the
analog of `numpy.polyval`.
"""
function _gnowee_polyval(p, x::Real)
    out = Float64(p[1])
    @inbounds for i in 2:length(p)
        out = out * x + p[i]
    end
    return out
end

# ─── Sampling primitives (port of Sampling.levy / Sampling.tlf) ─────────────

"""
    _gnowee_levy(rng, nc, nr=0; alpha=1.5, gam=1.0, n=1)

Symmetric Lévy stable samples via the Mantegna algorithm (port of `levy`).
Returns a `Vector{Float64}` of length `nc` when `nr == 0`, otherwise an
`(nr, nc)` matrix.
"""
function _gnowee_levy(rng::AbstractRNG, nc::Integer, nr::Integer=0;
                     alpha::Real=1.5, gam::Real=1.0, n::Integer=1)
    a = Float64(alpha)
    0.3 < a < 1.99 || throw(ArgumentError("alpha must be in (0.3, 1.99), got $a"))
    g = Float64(gam)
    g >= 0.0 || throw(ArgumentError("gamma must be non-negative"))
    nn = Int(n)
    nn >= 1 || throw(ArgumentError("n must be positive"))

    nc_i, nr_i = Int(nc), Int(nr)
    invalpha = 1.0 / a
    sigx = ((_gnowee_gamma(1.0 + a) * sin(pi * a / 2.0)) /
            (_gnowee_gamma((1.0 + a) / 2.0) * a * 2.0^((a - 1.0) / 2.0)))^invalpha

    if nr_i != 0
        v = sigx .* randn(rng, nn, nr_i, nc_i) ./
            (abs.(randn(rng, nn, nr_i, nc_i)) .^ invalpha)
    else
        v = sigx .* randn(rng, nn, nc_i) ./ (abs.(randn(rng, nn, nc_i)) .^ invalpha)
    end

    kappa = (a * _gnowee_gamma((a + 1.0) / (2.0 * a))) / _gnowee_gamma(invalpha) *
            ((a * _gnowee_gamma((a + 1.0) / 2.0)) /
             (_gnowee_gamma(1.0 + a) * sin(pi * a / 2.0)))^invalpha
    # Mantegna's polynomial fit for the temperature parameter c(alpha)
    p = (-17.7767, 113.3855, -281.5879, 337.5439, -193.5494, 44.8754)
    c = _gnowee_polyval(p, a)
    w = ((kappa - 1.0) .* exp.(-abs.(v) / c) .+ 1.0) .* v

    z = nn > 1 ? (1.0 / nn^invalpha) .* sum(w; dims=1) : w
    flat = vec(g^invalpha .* z)
    return nr_i != 0 ? reshape(flat, nr_i, nc_i) : reshape(flat, nc_i)
end

"""
    _gnowee_tlf(rng, num_row=1, num_col=1; alpha=1.5, gam=1.0, cut_point=10.0)

Truncated Lévy flight on (0, 1) (port of `tlf`): Lévy samples divided by
`cut_point`, with values above 1.0 resampled (up to `max_resamples` rounds)
and finally clipped.  Not used by the continuous-variable loop, ported for
parity with the vendored module.
"""
function _gnowee_tlf(rng::AbstractRNG, num_row::Integer=1, num_col::Integer=1;
                    alpha::Real=1.5, gam::Real=1.0, cut_point::Real=10.0,
                    max_resamples::Integer=50)
    nr, nc = Int(num_row), Int(num_col)
    z = reshape(abs.(_gnowee_levy(rng, nr, nc; alpha=alpha, gam=gam) ./ cut_point), nr, nc)
    mask = z .> 1.0
    n_attempts = 0
    while any(mask) && n_attempts < Int(max_resamples)
        n_bad = count(mask)
        n_bad == 0 && break
        replacements = vec(abs.(_gnowee_levy(rng, n_bad, 1; alpha=alpha, gam=gam) ./ cut_point))
        idxs = findall(mask)
        @inbounds for (k, I) in enumerate(idxs)
            z[I] = replacements[k]
        end
        mask = z .> 1.0
        n_attempts += 1
    end
    return min.(z, 1.0)
end

# ─── Boundary helpers (port of GnoweeHeuristics.simple/rejection_bounds) ────

"""
    _gnowee_simple_bounds(child, lb, ub)

Clip `child` to `[lb, ub]`.
"""
function _gnowee_simple_bounds(child::AbstractVector{Float64},
                              lb::AbstractVector{Float64},
                              ub::AbstractVector{Float64})
    return clamp.(child, lb, ub)
end

"""
    _gnowee_rejection_bounds(parent, child, step_size, lb, ub; max_reductions=5)

Iteratively halve the step of every out-of-bounds component until the child is
feasible; components still outside fall back to the parent value.
"""
function _gnowee_rejection_bounds(parent::AbstractVector{Float64},
                                 child::AbstractVector{Float64},
                                 step_size::AbstractVector{Float64},
                                 lb::AbstractVector{Float64},
                                 ub::AbstractVector{Float64};
                                 max_reductions::Integer=5)
    out = copy(child)
    step = copy(step_size)
    oob = (out .< lb) .| (out .> ub)
    for _ in 1:Int(max_reductions)
        any(oob) || break
        step[oob] .*= 0.5
        out[oob] .-= step[oob]
        oob = (out .< lb) .| (out .> ub)
    end
    out[oob] = parent[oob]
    return out
end

# ─── Population bookkeeping (port of Parent / Event / GnoweeSettings) ───────

mutable struct _GnoweeParent
    variables::Vector{Float64}
    fitness::Float64
    change_count::Int
    stall_count::Int
    _GnoweeParent(variables::Vector{Float64}) = new(variables, 1.0e15, 0, 0)
end

mutable struct _GnoweeEvent
    generation::Int
    evaluations::Int
    fitness::Float64
    design::Vector{Float64}
end

mutable struct _GnoweeSettings
    population::Int
    init_sampling::String
    frac_mutation::Float64
    frac_elite::Float64
    frac_levy::Float64
    alpha::Float64
    gamma::Float64
    n::Int
    scaling_factor::Float64
    penalty::Float64
    max_gens::Int
    max_fevals::Int
    conv_tol::Float64
    stall_limit::Int
    opt_conv_tol::Float64
    optimum::Float64
    verbose::Bool
end

function _GnoweeSettings(; population::Integer=25, init_sampling::AbstractString="lhc",
                        frac_mutation::Real=0.2, frac_elite::Real=0.2,
                        frac_levy::Real=1.0, alpha::Real=1.5, gamma::Real=1.0,
                        n::Integer=1, scaling_factor::Real=10.0, penalty::Real=0.0,
                        max_gens::Integer=200, max_fevals::Integer=5_000,
                        conv_tol::Real=1.0e-6, stall_limit::Integer=200,
                        opt_conv_tol::Real=1.0e-2, optimum::Real=0.0,
                        verbose::Bool=false)
    _GnoweeSettings(Int(population), String(init_sampling), Float64(frac_mutation),
                    Float64(frac_elite), Float64(frac_levy), Float64(alpha),
                    Float64(gamma), Int(n), Float64(scaling_factor), Float64(penalty),
                    Int(max_gens), Int(max_fevals), Float64(conv_tol),
                    Int(stall_limit), Float64(opt_conv_tol), Float64(optimum),
                    Bool(verbose))
end

mutable struct _GnoweeHeuristics{R<:AbstractRNG}
    lb::Vector{Float64}
    ub::Vector{Float64}
    objective::Any
    s::_GnoweeSettings
    rng::R
end

function _GnoweeHeuristics(lb::AbstractVector{Float64}, ub::AbstractVector{Float64},
                          objective, s::_GnoweeSettings, rng::AbstractRNG)
    length(lb) == length(ub) || throw(ArgumentError("lb and ub must have the same shape"))
    0.0 <= s.frac_mutation <= 1.0 || throw(ArgumentError("frac_mutation must lie in [0, 1]"))
    0.0 <= s.frac_elite <= 1.0 || throw(ArgumentError("frac_elite must lie in [0, 1]"))
    0.0 <= s.frac_levy <= 1.0 || throw(ArgumentError("frac_levy must lie in [0, 1]"))
    _GnoweeHeuristics(lb, ub, objective, s, rng)
end

# ─── Heuristics (port of GnoweeHeuristics) ──────────────────────────────────

"""
    _gnowee_initialize(h, num_samples, method=nothing)

Initial population within `[lb, ub]`: jittered Latin hypercube (`"lhc"`) or
uniform (`"random"`); returns an `(num_samples, n_dim)` matrix.
"""
function _gnowee_initialize(h::_GnoweeHeuristics, num_samples::Integer,
                           method::Union{Nothing,AbstractString}=nothing)
    s = h.s
    m = method
    if m === nothing || m == ""
        m = (s.init_sampling === nothing || s.init_sampling == "") ? "lhc" : s.init_sampling
    end
    m = lowercase(String(m))
    n_dim = length(h.lb)
    ns = Int(num_samples)
    if m == "random"
        unit = rand(h.rng, ns, n_dim)
    elseif m == "lhc"
        unit = Matrix{Float64}(undef, ns, n_dim)
        for j in 1:n_dim
            perm = randperm(h.rng, ns)
            jitter = rand(h.rng, ns)
            @inbounds for i in 1:ns
                unit[i, j] = (perm[i] - 1 + jitter[i]) / ns
            end
        end
    else
        throw(ArgumentError("Unsupported init_sampling '$m'. Use 'lhc' or 'random'."))
    end
    # rows = individuals, columns = dimensions (Python's (num_samples, n_dim))
    out = Matrix{Float64}(undef, ns, n_dim)
    @inbounds for j in 1:n_dim
        lo, sp = h.lb[j], h.ub[j] - h.lb[j]
        for i in 1:ns
            out[i, j] = lo + sp * unit[i, j]
        end
    end
    return out
end

"""
    _gnowee_cont_levy_flight(h, pop) -> (children, used)

`x_r + step/scaling_factor` with a Lévy step, box-constrained by
`rejection_bounds`.
"""
function _gnowee_cont_levy_flight(h::_GnoweeHeuristics, pop::Vector{_GnoweeParent})
    s = h.s
    isempty(pop) && return Vector{Vector{Float64}}(), Int[]
    n_take = max(1, trunc(Int, s.frac_levy * s.population))
    n_take = min(n_take, length(pop))
    idx = randperm(h.rng, length(pop))[1:n_take]

    dim = length(pop[1].variables)
    step = _gnowee_levy(h.rng, dim, n_take; alpha=s.alpha, gam=s.gamma, n=s.n)
    children = Vector{Vector{Float64}}()
    used = Int[]
    for k in 1:n_take
        parent_idx = idx[k]
        base = copy(pop[parent_idx].variables)
        step_size = Vector{Float64}(view(step, k, :)) / s.scaling_factor
        child = _gnowee_rejection_bounds(base, base .+ step_size, step_size, h.lb, h.ub)
        push!(children, child)
        push!(used, parent_idx)
    end
    return children, used
end

"""
    _gnowee_crossover(h, pop) -> (children, used)

Golden-ratio weighted recombination of a random partner with elite parent `i`:
`child = partner + |parent_i - partner| / phi`.
"""
function _gnowee_crossover(h::_GnoweeHeuristics, pop::Vector{_GnoweeParent})
    s = h.s
    n_pop = length(pop)
    n_take = max(0, trunc(Int, s.frac_elite * n_pop))
    (n_take == 0 || n_pop < 2) && return Vector{Vector{Float64}}(), Int[]
    golden = (1.0 + sqrt(5.0)) / 2.0
    children = Vector{Vector{Float64}}()
    used = Int[]
    for i in 1:n_take
        r = rand(h.rng, 1:n_pop)
        attempts = 0
        while r == i && attempts < 10
            r = rand(h.rng, 1:n_pop)
            attempts += 1
        end
        r == i && continue
        push!(used, i)
        base = copy(pop[r].variables)
        dx = abs.(copy(pop[i].variables) .- base) / golden
        child = _gnowee_simple_bounds(base .+ dx, h.lb, h.ub)
        push!(children, child)
    end
    return children, used
end

"""
    _gnowee_scatter_search(h, pop) -> (children, used)

Egea (2009) path-relinking between elite parents `i` and `j`.
"""
function _gnowee_scatter_search(h::_GnoweeHeuristics, pop::Vector{_GnoweeParent})
    s = h.s
    n_pop = length(pop)
    n_take = max(0, trunc(Int, s.frac_elite * n_pop))
    (n_take == 0 || n_pop < 2) && return Vector{Vector{Float64}}(), Int[]
    children = Vector{Vector{Float64}}()
    used = Int[]
    for i in 1:n_take
        j = rand(h.rng, 1:n_pop)
        attempts = 0
        while (j == i || j in used) && attempts < 10
            j = rand(h.rng, 1:n_pop)
            attempts += 1
        end
        if j == i || j in used
            continue
        end
        push!(used, i)
        xi = copy(pop[i].variables)
        xj = copy(pop[j].variables)
        d = (xj .- xi) / 2.0
        alpha_ = i < j ? 1.0 : -1.0
        beta = (abs(j - i) - 1) / max(n_pop - 2, 1)
        c1 = xi .- d .* (1.0 + alpha_ * beta)
        c2 = xi .+ d .* (1.0 - alpha_ * beta)
        r = rand(h.rng, length(xi))
        child = _gnowee_simple_bounds(c1 .+ (c2 .- c1) .* r, h.lb, h.ub)
        push!(children, child)
    end
    return children, used
end

"""
    _gnowee_mutate(h, pop) -> children

DE-style differential mutation `child = parent + r * (perm1 - perm2) * k` with
a per-component mask `k` driven by `frac_mutation`.
"""
function _gnowee_mutate(h::_GnoweeHeuristics, pop::Vector{_GnoweeParent})
    s = h.s
    isempty(pop) && return Vector{Vector{Float64}}()
    n = length(pop)
    dim = length(pop[1].variables)
    pop_arr = Matrix{Float64}(undef, n, dim)
    @inbounds for p in 1:n
        pop_arr[p, :] = pop[p].variables
    end
    perm1 = randperm(h.rng, n)
    perm2 = randperm(h.rng, n)
    r = rand(h.rng)
    k = rand(h.rng, n, dim) .> (s.frac_mutation * rand(h.rng))
    diff = pop_arr[perm1, :] .- pop_arr[perm2, :]
    children_arr = pop_arr .+ (r .* diff) .* k
    return [_gnowee_simple_bounds(children_arr[p, :], h.lb, h.ub) for p in 1:n]
end

"""
    _gnowee_population_update(h, parents, children, timeline;
                              adopted_parents=Int[], mh_frac=0.0,
                              random_parents=false)

Evaluate children, replace the compared parent when better, keep parents sorted
by fitness (stable, as in Python), apply the Metropolis-Hastings fallback, the
`change_count >= 25` reinitialisation and the long-stall restart.
"""
function _gnowee_population_update(h::_GnoweeHeuristics, parents::Vector{_GnoweeParent},
                                  children::Vector{Vector{Float64}},
                                  timeline::Union{Nothing,Vector{_GnoweeEvent}};
                                  adopted_parents::Vector{Int}=Int[],
                                  mh_frac::Real=0.0,
                                  random_parents::Bool=false)
    s = h.s
    n_parents = length(parents)
    n_children = length(children)
    n_children == 0 && return parents, 0, timeline

    replace = 0
    feval = 0
    for i in 1:n_children
        child_vars = children[i]
        fnew = Float64(h.objective(child_vars))
        fnew > s.penalty && (s.penalty = fnew)
        feval += 1

        if random_parents
            j = rand(h.rng, 1:n_parents)
        elseif length(adopted_parents) == n_children
            j = adopted_parents[i]
        else
            j = i <= n_parents ? i : rand(h.rng, 1:n_parents)
        end

        if fnew < parents[j].fitness
            parents[j].fitness = fnew
            parents[j].variables = copy(child_vars)
            parents[j].change_count += 1
            parents[j].stall_count = 0
            replace += 1
            if parents[j].change_count >= 25 && j > trunc(Int, s.population * s.frac_elite)
                new_vars = Vector{Float64}(view(_gnowee_initialize(h, 1, "random"), 1, :))
                fnew2 = Float64(h.objective(new_vars))
                parents[j].variables = new_vars
                parents[j].fitness = fnew2
                parents[j].change_count = 0
                feval += 1
            end
        else
            parents[j].stall_count += 1
            if parents[j].stall_count > 50_000 && j != 1
                new_vars = Vector{Float64}(view(_gnowee_initialize(h, 1, "random"), 1, :))
                fnew2 = Float64(h.objective(new_vars))
                parents[j].variables = new_vars
                parents[j].fitness = fnew2
                parents[j].change_count = 0
                parents[j].stall_count = 0
                feval += 1
            end
            if mh_frac > 0.0 && rand(h.rng) < mh_frac
                r2 = rand(h.rng, 1:n_parents)
                if fnew < parents[r2].fitness
                    parents[r2].fitness = fnew
                    parents[r2].variables = copy(child_vars)
                    parents[r2].change_count += 1
                    parents[r2].stall_count += 1
                    replace += 1
                end
            end
        end
    end

    sort!(parents; by = p -> p.fitness)

    if timeline !== nothing
        if length(timeline) < 2
            push!(timeline, _GnoweeEvent(1, feval, parents[1].fitness,
                                         copy(parents[1].variables)))
        elseif parents[1].fitness < timeline[end].fitness &&
               abs((timeline[end].fitness - parents[1].fitness) /
                   max(abs(parents[1].fitness), 1.0e-300)) > s.conv_tol
            push!(timeline, _GnoweeEvent(timeline[end].generation,
                                         timeline[end].evaluations + feval,
                                         parents[1].fitness,
                                         copy(parents[1].variables)))
        else
            timeline[end].evaluations += feval
        end
    end
    return parents, replace, timeline
end

# ─── Main optimizer loop (port of run_gnowee) ───────────────────────────────

"""
    _gnowee_run(lb, ub, objective, s, rng; seed_solution=nothing,
                extra_starting=nothing) -> (best_x, best_f, timeline)

Gnowee hybrid loop: Lévy flight → crossover → scatter search → mutation, with
elitist population update, stall-driven termination and fitness convergence.
"""
function _gnowee_run(lb::Vector{Float64}, ub::Vector{Float64}, objective,
                    s::_GnoweeSettings, rng::AbstractRNG;
                    seed_solution::Union{Nothing,AbstractVector{Float64}}=nothing,
                    extra_starting::Union{Nothing,AbstractVector{Float64}}=nothing)
    h = _GnoweeHeuristics(lb, ub, objective, s, rng)

    init_num = max(s.population * 2, length(h.lb) * 10)
    init_vars = _gnowee_initialize(h, init_num, s.init_sampling)
    if seed_solution !== nothing
        init_vars[1, :] = _gnowee_simple_bounds(seed_solution, h.lb, h.ub)
        if extra_starting !== nothing && size(init_vars, 1) > 1
            init_vars[2, :] = _gnowee_simple_bounds(extra_starting, h.lb, h.ub)
        end
    end
    init_num = min(init_num, size(init_vars, 1))

    pop = [_GnoweeParent(Vector{Float64}(view(init_vars, i, :))) for i in 1:init_num]
    for p in pop
        p.fitness = Float64(objective(p.variables))
    end
    sort!(pop; by = p -> p.fitness)
    if length(pop) > s.population
        pop = pop[1:s.population]
    else
        s.population = length(pop)
    end

    timeline = [_GnoweeEvent(0, length(pop), pop[1].fitness, copy(pop[1].variables))]

    converge = false
    while !converge
        children, ind = _gnowee_cont_levy_flight(h, pop)
        if !isempty(children)
            pop, _, timeline = _gnowee_population_update(
                h, pop, children, timeline;
                adopted_parents=ind, mh_frac=0.2, random_parents=true)
        end

        children, ind = _gnowee_crossover(h, pop)
        if !isempty(children)
            pop, _, timeline = _gnowee_population_update(h, pop, children, timeline)
        end

        children, ind = _gnowee_scatter_search(h, pop)
        if !isempty(children)
            pop, _, timeline = _gnowee_population_update(
                h, pop, children, timeline; adopted_parents=ind)
        end

        children = _gnowee_mutate(h, pop)
        if !isempty(children)
            pop, _, timeline = _gnowee_population_update(h, pop, children, timeline)
        end

        gen = timeline[end].generation + 1
        evals = timeline[end].evaluations
        if s.verbose && gen % 10 == 0
            println("Gnowee gen=$gen evals=$evals best=",
                    @sprintf("%.6e", pop[1].fitness))
        end

        if evals > s.stall_limit && length(timeline) >= 2
            if evals > timeline[end - 1].evaluations + s.stall_limit
                converge = true
                s.verbose && println("Gnowee: stall at evaluation #$evals")
            end
        end
        if gen > s.max_gens
            converge = true
            s.verbose && println("Gnowee: max generations reached.")
        end
        if evals > s.max_fevals
            converge = true
            s.verbose && println("Gnowee: max function evaluations reached.")
        end

        if s.optimum == 0.0
            if pop[1].fitness < s.opt_conv_tol
                converge = true
                s.verbose && println("Gnowee: fitness convergence (absolute).")
            end
        elseif abs((pop[1].fitness - s.optimum) / s.optimum) <= s.opt_conv_tol
            converge = true
            s.verbose && println("Gnowee: fitness convergence (relative).")
        elseif pop[1].fitness < s.optimum
            converge = true
            s.verbose && println("Gnowee: fitness below optimum.")
        end

        timeline[end].generation = gen
    end

    return copy(pop[1].variables), Float64(pop[1].fitness), timeline
end

# ─── Warm-start seed, log bounds and objective (shared with unfold_genetic) ─

"""
    _build_seed(A, b, x0) -> Vector{Float64}

Warm-start seed spectrum: `x0` when non-trivial, otherwise a short Landweber
iteration; flat-spectrum fallback if Landweber is unavailable.
"""
function _build_seed(A::AbstractMatrix{<:Real}, b::AbstractVector{<:Real},
                    x0::Union{Nothing,AbstractVector{<:Real}})
    n = size(A, 2)
    if x0 !== nothing && any(>(0), x0)
        return max.(Float64.(collect(x0)), 1e-12)
    end
    try
        lw = solve_landweber(A, b, zeros(Float64, n); max_iterations=500)
        return max.(Float64.(collect(lw.spectrum)), 1e-12)
    catch
        A_fro = Float64(LinearAlgebra.norm(A))
        A_fro == 0.0 && (A_fro = 1.0)
        x_scale = Float64(LinearAlgebra.norm(b)) / A_fro
        return fill(max(x_scale / sqrt(n), 1e-12), n)
    end
end

"""
    _build_log_bounds(seed, half_range) -> (lb, ub)

Log-space bounds centred on the seed: `log(seed) ± half_range` decades.
"""
function _build_log_bounds(seed::AbstractVector{<:Real}, half_range::Real)
    y0 = log.(max.(Float64.(collect(seed)), 1e-300))
    span = Float64(half_range) * log(10.0)
    return y0 .- span, y0 .+ span
end

"""
    _build_fitness(A, b, alpha, norm, L, smoothness_weight, entropy_weight)

Scale-consistent log-space objective `f(y)` with `x = exp(y)`:

    f(y) = ||b - A exp(y)||^2 / ||b||^2
         + alpha * ||exp(y)||_norm / x_scale^p
         + smoothness_weight * ||L exp(y)||^2 / x_scale^2
         - entropy_weight * H(exp(y))
"""
function _build_fitness(A::AbstractMatrix{Float64}, b::Vector{Float64},
                       alpha::Float64, norm::Int,
                       L::Union{Nothing,Matrix{Float64}},
                       smoothness_weight::Float64, entropy_weight::Float64)
    denom = Float64(dot(b, b))
    denom <= 0.0 && (denom = 1.0)
    A_fro = Float64(LinearAlgebra.norm(A))
    A_fro <= 0.0 && (A_fro = 1.0)
    x_scale = sqrt(denom) / A_fro
    x_scale2 = x_scale * x_scale

    function fitness(y::AbstractVector{Float64})
        x = exp.(y)
        residual = A * x .- b
        value = Float64(dot(residual, residual)) / denom
        if alpha > 0
            if norm == 2
                value += alpha * Float64(dot(x, x)) / x_scale2
            elseif norm == 1
                value += alpha * sum(abs.(x)) / x_scale
            end
        end
        if L !== nothing && smoothness_weight > 0
            Lx = L * x
            value += smoothness_weight * Float64(dot(Lx, Lx)) / x_scale2
        end
        if entropy_weight > 0
            total = sum(x)
            if total > 0
                p_ = x ./ total
                logp = log.(max.(p_, 1e-300))
                value -= entropy_weight * Float64(dot(p_, logp))
            end
        end
        return value
    end
    return fitness
end

# ─── Solver ─────────────────────────────────────────────────────────────────

"""
    solve_gnowee(A, b, x0; population=25, max_gens=200, max_fevals=5000,
                 stall_limit=200, conv_tol=1e-6, opt_conv_tol=1e-2,
                 frac_elite=0.2, frac_levy=1.0, frac_mutation=0.2,
                 alpha_levy=1.5, gamma_levy=1.0, n_levy=1, scaling_factor=10.0,
                 init_sampling="lhc", regularization=1e-2, norm=2,
                 smoothness_order=2, smoothness_weight=1.0, entropy_weight=0.0,
                 half_range=2.0, random_state=nothing, verbose=false)

Solve the unfolding problem with the Gnowee hybrid metaheuristic (port of
`solve_gnowee`).  The optimizer searches in log space, seeded with the Landweber
warm start (or `x0` when non-trivial; pass `x0 = fill(0.0, n)` or `nothing` for
the warm start), bounded to `log(seed) ± half_range` decades, minimizing the
scale-consistent objective.

`iterations` reports the number of objective evaluations (Python's `n_evals`);
`converged` is true when neither the `max_fevals` nor the `max_gens` cap was
hit.  `extra` carries the knobs plus `best_fitness`, `generations`,
`evaluations` and `timeline_len`.
"""
function solve_gnowee(A::AbstractMatrix{T}, b::AbstractVector{T},
                      x0::Union{Nothing,AbstractVector{T}};
                      population::Integer=25,
                      max_gens::Integer=200,
                      max_fevals::Integer=5_000,
                      stall_limit::Integer=200,
                      conv_tol::Real=1e-6,
                      opt_conv_tol::Real=1e-2,
                      frac_elite::Real=0.2,
                      frac_levy::Real=1.0,
                      frac_mutation::Real=0.2,
                      alpha_levy::Real=1.5,
                      gamma_levy::Real=1.0,
                      n_levy::Integer=1,
                      scaling_factor::Real=10.0,
                      init_sampling::AbstractString="lhc",
                      regularization::Real=1e-2,
                      norm::Integer=2,
                      smoothness_order::Integer=2,
                      smoothness_weight::Real=1.0,
                      entropy_weight::Real=0.0,
                      half_range::Real=2.0,
                      random_state::Union{Integer,Nothing}=nothing,
                      verbose::Bool=false) where T<:AbstractFloat
    m, n = size(A)
    length(b) == m || throw(ArgumentError("b length ($(length(b))) must match A rows ($m)"))

    norm in (1, 2) || throw(ArgumentError("Unsupported norm type: $norm. Use 1 or 2."))
    smoothness_order in (0, 1, 2) || throw(ArgumentError(
        "Unsupported smoothness order: $smoothness_order. Use 0, 1 or 2."))
    sampler = lowercase(String(init_sampling))
    sampler in ("lhc", "lhs", "random") || throw(ArgumentError(
        "Unsupported init_sampling: $init_sampling. Use 'lhc' or 'random'."))

    L = smoothness_order in (1, 2) ?
        create_derivative_matrix(Float64, n, Int(smoothness_order)) : nothing

    A_f = Float64.(Matrix(A))
    b_f = Float64.(collect(b))
    fitness = _build_fitness(A_f, b_f, Float64(regularization), Int(norm), L,
                             Float64(smoothness_weight), Float64(entropy_weight))
    seed = _build_seed(A_f, b_f, x0)
    lb, ub = _build_log_bounds(seed, half_range)

    rng = random_state === nothing ? MersenneTwister() : MersenneTwister(Int(random_state))

    s = _GnoweeSettings(
        population=Int(population),
        init_sampling=sampler in ("lhc", "lhs") ? "lhc" : "random",
        frac_mutation=Float64(frac_mutation),
        frac_elite=Float64(frac_elite),
        frac_levy=Float64(frac_levy),
        alpha=Float64(alpha_levy),
        gamma=Float64(gamma_levy),
        n=Int(n_levy),
        scaling_factor=Float64(scaling_factor),
        max_gens=Int(max_gens),
        max_fevals=Int(max_fevals),
        conv_tol=Float64(conv_tol),
        stall_limit=Int(stall_limit),
        opt_conv_tol=Float64(opt_conv_tol),
        verbose=Bool(verbose))

    best_y, best_f, timeline = _gnowee_run(
        lb, ub, fitness, s, rng; seed_solution=log.(max.(seed, 1e-300)))
    spectrum = max.(exp.(best_y), 0.0)

    n_evals = isempty(timeline) ? 0 : timeline[end].evaluations
    hit_feval_cap = n_evals >= s.max_fevals
    hit_gen_cap = (isempty(timeline) ? 0 : timeline[end].generation) >= s.max_gens
    converged = !(hit_feval_cap || hit_gen_cap)

    residual = b_f .- A_f * spectrum
    return UnfoldResult(
        Vector{T}(spectrum), Int(n_evals), converged,
        T(sqrt(sum(abs2, residual))),
        Dict{String,Any}(
            "population" => Int(population),
            "max_gens" => Int(max_gens),
            "max_fevals" => Int(max_fevals),
            "stall_limit" => Int(stall_limit),
            "conv_tol" => Float64(conv_tol),
            "opt_conv_tol" => Float64(opt_conv_tol),
            "frac_elite" => Float64(frac_elite),
            "frac_levy" => Float64(frac_levy),
            "frac_mutation" => Float64(frac_mutation),
            "alpha_levy" => Float64(alpha_levy),
            "gamma_levy" => Float64(gamma_levy),
            "n_levy" => Int(n_levy),
            "scaling_factor" => Float64(scaling_factor),
            "init_sampling" => String(init_sampling),
            "regularization" => Float64(regularization),
            "norm" => Int(norm),
            "smoothness_order" => Int(smoothness_order),
            "smoothness_weight" => Float64(smoothness_weight),
            "entropy_weight" => Float64(entropy_weight),
            "half_range" => Float64(half_range),
            "random_state" => random_state === nothing ? nothing : Int(random_state),
            "best_fitness" => best_f,
            "evaluations" => Int(n_evals),
            "generations" => isempty(timeline) ? 0 : Int(timeline[end].generation),
            "timeline_len" => length(timeline),
            "timeline" => [(t.generation, t.evaluations, t.fitness) for t in timeline],
        ))
end
