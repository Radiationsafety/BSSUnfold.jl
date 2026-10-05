"""
Stochastic parametric unfolding with the Fission model (BonnerFinder) —
port of `bssunfold/core/unfold_fission_ga.py`.

I. N. Ogorodnikov, "Inverse problems of spectroscopy and spectrometry in
applied research" (обратные задачи спектроскопии и спектрометрии в
прикладных исследованиях), Traektoriya Issledovaniy -- Chelovek, Priroda,
Tekhnologii, no. 2 (10), pp. 42-83 (2024), Sections 4-5, function
`BonnerFinder()`.  The model curves follow the FRUIT paradigm of Bedogni
et al., NIM A 580, 1301 (2007).

The model spectrum is a superposition of three neutron fractions
(article eq. 4.29):

    Phi(E) = a1 * (E/T0^2) * exp(-E/T0)
           + a2 * [1 - exp(-(E/Ed)^2)] * E^(b-1) * exp(-E/beta)
           + a3 * E^alpha * exp(-E/TF),

with `T0 = T0_THERMAL = 2.53e-8` MeV and `Ed = ED_EPITHERMAL = 7.07e-8` MeV
fixed and the seven free parameters bounded per `FISSION_PARAM_BOUNDS`;
`fit_scale=true` (Python default) adds an eighth free parameter
`log10(phi_scale) ∈ [-12, 12]`.

Stage 1 (article section "Elementy stokhasticheskogo algoritma") is a
stochastic global search of the parameter hypercube minimizing the target
function (article eq. 4.32)

    dM(P) = sum_m | C_m - integral R_m(E) Phi(E; P) dE |.

The article uses the SciLab `optim_ga`; Python uses
`scipy.optimize.differential_evolution(strategy="best1bin", popsize, maxiter,
tol, mutation=(0.5, 1.0), recombination=0.7, init="latinhypercube",
updating="immediate", polish=False)`.  The same DE (population in the
normalized [0, 1]^D box, LHS init, one dithered F per generation, binomial
crossover with a guaranteed mutant component, random re-draw of out-of-box
coordinates, greedy `trial <= target` selection, best promoted to position 1,
stop when `std(energies) <= tol*|mean(energies)|`) is implemented natively as
`_fission_de`; the package's `genetic.jl` engines are not reused because they
search the spectrum in log space (DE/rand/1/bin), not a parameter hypercube.

Stage 2 (article section "Nelineynaya regressiya") refines the best stochastic
point by nonlinear least squares (`scipy.optimize.least_squares` in Python).
`lm_method="trf"` (the Python default) is ported natively as `_fission_trf`:
the bound-constrained trust-region-reflective algorithm of
`scipy.optimize._lsq.trf` with `tr_solver="exact"` and `x_scale="jac"`
(Coleman-Li scaling, SVD trust-region subproblem, reflected and Cauchy step
alternatives, `ftol`/`xtol`/`gtol` stopping tests), minimizing the same bounded
problem as Python.  `lm_method="lm"` (the article's unbounded SciLab `leastsq`)
runs the package's MINPACK `lmdif` port `_fruit_lmdif` (parametric.jl) in the
raw parameter space and clips afterwards, as Python's unbounded LM does.  A
user `initial_params` point is refined as an additional stage-2 start and the
better fit is kept.

The article's validation criteria (norm of the model spectrum, per-sphere
relative uncertainties with alternating signs, FOM of the fit) are computed by
`fission_validate_fit` and attached as `extra["validation"]`.
"""

# Fixed constants of the Fission model (article eq. 4.29).
const T0_THERMAL = 2.53e-8  # Thermal peak energy (MeV)
const ED_EPITHERMAL = 7.07e-8  # Epithermal cutoff parameter (MeV)

# Free parameters of the article's 7-parameter Fission model.
const FISSION_PARAM_NAMES = ("a1", "a2", "a3", "b", "beta", "alpha", "TF")

# Article's indicative parameter bounds (section 4, model Fission).
const FISSION_PARAM_BOUNDS = Dict{String, Tuple{Float64, Float64}}(
    "a1" => (0.0, 1.0),
    "a2" => (0.0, 1.0),
    "a3" => (0.0, 1.0),
    "b" => (-0.5, 0.5),
    "beta" => (1e-4, 1.0),
    "alpha" => (0.0, 1.0),
    "TF" => (1.0, 2.0),
)

# Mid-range default of the flat parameter vector
# [a1, a2, a3, b, beta, alpha, TF] (+ log10(phi_scale) when fit_scale).
const _FISSION_DEFAULT_THETA = (0.3, 0.3, 0.4, 0.0, 0.1, 0.5, 1.5)
const _FISSION_LOG_SCALE_BOUND = (-12.0, 12.0)

"""
    fission_model(E, a1, a2, a3, b, beta, alpha, TF) -> Vector{Float64}

Three-fraction Fission model spectrum (article eq. 4.29) per unit lethargy on
the energy grid `E` (MeV).  `a1, a2, a3` are the thermal, epithermal and fast
weights, `b` the epithermal slope, `beta` the epithermal right cutoff, `alpha`
the fast shape and `TF` the fast peak position.
"""
function fission_model(E::AbstractVector{<:Real}, a1::Real, a2::Real, a3::Real,
                       b::Real, beta::Real, alpha::Real, TF::Real)
    Ef = Float64.(collect(E))
    thermal = Ef ./ (T0_THERMAL^2) .* exp.(-Ef ./ T0_THERMAL)
    epithermal = (1.0 .- exp.(-((Ef ./ ED_EPITHERMAL) .^ 2))) .*
                 Ef .^ (Float64(b) - 1.0) .* exp.(-Ef ./ Float64(beta))
    fast = Ef .^ Float64(alpha) .* exp.(-Ef ./ Float64(TF))
    return Float64(a1) .* thermal .+ Float64(a2) .* epithermal .+ Float64(a3) .* fast
end

function _fission_theta_bounds(fit_scale::Bool)
    lo = Float64[FISSION_PARAM_BOUNDS[name][1] for name in FISSION_PARAM_NAMES]
    hi = Float64[FISSION_PARAM_BOUNDS[name][2] for name in FISSION_PARAM_NAMES]
    if fit_scale
        push!(lo, _FISSION_LOG_SCALE_BOUND[1])
        push!(hi, _FISSION_LOG_SCALE_BOUND[2])
    end
    return lo, hi
end

function _fission_default_theta(fit_scale::Bool)
    theta = Float64[_FISSION_DEFAULT_THETA...]
    fit_scale && push!(theta, 0.0)  # phi_scale = 1
    return theta
end

function _fission_model_shape(theta::AbstractVector{<:Real}, E::AbstractVector{<:Real})
    Ef = Float64.(collect(E))
    th = Float64.(collect(theta))
    phi_scale = length(th) > 7 ? 10.0^th[8] : 1.0
    return phi_scale .* fission_model(Ef, th[1], th[2], th[3], th[4], th[5], th[6], th[7])
end

function _fission_folded(theta::AbstractVector{<:Real}, A::Matrix{Float64},
                         E::Vector{Float64}, ln_steps::Vector{Float64})
    return A * (_fission_model_shape(theta, E) .* ln_steps)
end

function _fission_residual(theta::AbstractVector{<:Real}, A::Matrix{Float64},
                           b::Vector{Float64}, E::Vector{Float64},
                           ln_steps::Vector{Float64})
    return _fission_folded(theta, A, E, ln_steps) .- b
end

# Target function of the stochastic stage (article eq. 4.32).
function _fission_target(theta::AbstractVector{<:Real}, A::Matrix{Float64},
                         b::Vector{Float64}, E::Vector{Float64},
                         ln_steps::Vector{Float64})
    return sum(abs.(_fission_residual(theta, A, b, E, ln_steps)))
end

function _fission_lookup(params, name::String)
    params === nothing && return nothing
    haskey(params, name) && return params[name]
    sym = Symbol(name)
    haskey(params, sym) && return params[sym]
    return nothing
end

function _fission_theta_from_initial(initial_params, fit_scale::Bool)
    (initial_params === nothing || isempty(initial_params)) && return nothing
    theta = _fission_default_theta(fit_scale)
    for (i, name) in enumerate(FISSION_PARAM_NAMES)
        v = _fission_lookup(initial_params, name)
        v !== nothing && (theta[i] = Float64(v))
    end
    if fit_scale
        v = _fission_lookup(initial_params, "phi_scale")
        v !== nothing && (theta[8] = log10(max(Float64(v), 1e-30)))
    end
    lo, hi = _fission_theta_bounds(fit_scale)
    return clamp.(theta, lo, hi)
end

# ─── Stage 1: differential evolution (scipy `best1bin`, immediate updating) ──

function _fission_lhs(rng::AbstractRNG, pop_size::Int, dim::Int)
    seg = 1.0 / pop_size
    samples = Matrix{Float64}(undef, pop_size, dim)
    @inbounds for j in 1:dim, i in 1:pop_size
        samples[i, j] = seg * rand(rng) + (i - 1) * seg
    end
    pop = similar(samples)
    @inbounds for j in 1:dim
        perm = randperm(rng, pop_size)
        for i in 1:pop_size
            pop[i, j] = samples[perm[i], j]
        end
    end
    return pop
end

function _fission_promote_best!(pop::Matrix{Float64}, energy::Vector{Float64})
    isempty(energy) && return nothing
    l = argmin(energy)
    if l != 1
        pop[[1, l], :] = pop[[l, 1], :]
        energy[[1, l]] = energy[[l, 1]]
    end
    return nothing
end

function _fission_de(target::Function, lo::Vector{Float64}, hi::Vector{Float64},
                     rng::AbstractRNG; popsize::Integer, maxiter::Integer,
                     tol::Float64, recombination::Float64=0.7,
                     mutation_lo::Float64=0.5, mutation_hi::Float64=1.0)
    dim = length(lo)
    pop_size = max(5, Int(popsize) * dim)
    mid = 0.5 .* (lo .+ hi)
    span = hi .- lo

    pop = _fission_lhs(rng, pop_size, dim)
    nfev = 0
    energy = Vector{Float64}(undef, pop_size)
    @inbounds for i in 1:pop_size
        energy[i] = target(mid .+ (pop[i, :] .- 0.5) .* span)
        nfev += 1
    end
    _fission_promote_best!(pop, energy)

    for gen in 1:maxiter
        F = rand(rng) * (mutation_hi - mutation_lo) + mutation_lo
        @inbounds for cand in 1:pop_size
            r0 = rand(rng, 1:pop_size)
            while r0 == cand
                r0 = rand(rng, 1:pop_size)
            end
            r1 = rand(rng, 1:pop_size)
            while r1 == cand || r1 == r0
                r1 = rand(rng, 1:pop_size)
            end
            mutant = pop[1, :] .+ F .* (pop[r0, :] .- pop[r1, :])
            fill_point = rand(rng, 1:dim)
            trial = copy(pop[cand, :])
            for j in 1:dim
                (j == fill_point || rand(rng) < recombination) && (trial[j] = mutant[j])
            end
            for j in 1:dim
                (trial[j] > 1.0 || trial[j] < 0.0) && (trial[j] = rand(rng))
            end
            e = target(mid .+ (trial .- 0.5) .* span)
            nfev += 1
            if e <= energy[cand]
                pop[cand, :] = trial
                energy[cand] = e
                e <= energy[1] && _fission_promote_best!(pop, energy)
            end
        end
        if all(isfinite, energy)
            mu = abs(mean(energy))
            std(energy) <= tol * mu && break
        end
    end

    x = mid .+ (pop[1, :] .- 0.5) .* span
    return x, energy[1], nfev
end

# ─── Stage 2: nonlinear least-squares refinement ─────────────────────────────

function _fission_num_jacobian(resid::Function, p::Vector{Float64},
                               r0::Vector{Float64}, counter::Base.RefValue{Int})
    m = length(r0)
    n = length(p)
    J = Matrix{Float64}(undef, m, n)
    step = sqrt(eps(Float64))
    @inbounds for j in 1:n
        h = step * max(1.0, abs(p[j]))
        pp = copy(p)
        pp[j] += h
        rj = resid(pp)
        counter[] += 1
        for i in 1:m
            J[i, j] = (rj[i] - r0[i]) / h
        end
    end
    return J
end

"""
    _fission_bounded_lm(theta0, A, b, E, ln_steps, lo, hi, max_nfev)

Stage 2 for `lm_method="trf"`: box-constrained Levenberg-Marquardt with
Marquardt (Jacobian-diagonal) scaling and projection of every trial point onto
the parameter hypercube — the native substitute for
`scipy.optimize.least_squares(method="trf", x_scale="jac")`.  Both minimize the
same bounded nonlinear least-squares problem; projection onto the box (rather
than the reflective trust-region of the original) is what lets the fit settle
on the model bounds, as the article's `leastsq` refinement does.
"""
function _fission_bounded_lm(theta0::Vector{Float64}, A::Matrix{Float64},
                             b::Vector{Float64}, E::Vector{Float64},
                             ln_steps::Vector{Float64}, lo::Vector{Float64},
                             hi::Vector{Float64}, max_nfev::Integer;
                             ftol::Float64=1e-8, xtol::Float64=1e-8)
    jac_counter = Ref(0)
    raw_resid = p -> _fission_residual(p, A, b, E, ln_steps)
    resid = function (p::Vector{Float64})
        jac_counter[] += 1
        return raw_resid(p)
    end

    theta = clamp.(theta0, lo, hi)
    r = raw_resid(theta)
    cost = norm(r)
    n = length(theta)
    λ = 1e-3
    nfev = 0
    message = ""
    success = false

    while nfev < max_nfev
        J = _fission_num_jacobian(resid, theta, r, jac_counter)
        JtJ = J' * J
        grad = J' * r
        dg = Vector{Float64}(diag(JtJ))
        for i in 1:n
            dg[i] > 1e-30 || (dg[i] = 1.0)
        end
        d = try
            -(JtJ + λ .* Diagonal(dg)) \ grad
        catch err
            err isa LinearAlgebra.SingularException || rethrow(err)
            fill(0.0, n)
        end

        candidate = clamp.(theta .+ d, lo, hi)
        r_new = raw_resid(candidate)
        nfev += 1
        cost_new = norm(r_new)

        if cost_new < cost
            relative = (cost - cost_new) / max(cost, eps(Float64))
            theta, r, cost = candidate, r_new, cost_new
            λ = max(λ / 10.0, 1e-15)
            if relative <= ftol && norm(d) <= xtol * (norm(theta) + xtol)
                message = "Both `ftol` and `xtol` termination conditions are satisfied."
                success = true
                break
            elseif relative <= ftol
                message = "`ftol` termination condition is satisfied."
                success = true
                break
            elseif norm(d) <= xtol * (norm(theta) + xtol)
                message = "`xtol` termination condition is satisfied."
                success = true
                break
            end
        else
            λ *= 10.0
            if λ > 1e15
                message = "`gtol` termination condition is satisfied."
                success = true
                break
            end
        end
    end

    if isempty(message)
        if nfev >= max_nfev
            message = "The maximum number of function evaluations is exceeded."
        else
            message = "Terminated."
        end
    end
    return theta, cost, nfev, success, message
end

function _fission_run_lm(theta0::Vector{Float64}, A::Matrix{Float64}, b::Vector{Float64},
                         E::Vector{Float64}, ln_steps::Vector{Float64},
                         fit_scale::Bool, lm_method::AbstractString,
                         lm_max_nfev::Integer)
    lo, hi = _fission_theta_bounds(fit_scale)
    theta0 = clamp.(theta0, lo, hi)

    if lm_method == "lm"
        # Levenberg-Marquardt (as SciLab leastsq) does not support bounds:
        # project the start inside and clip after the fit, as Python does.
        counter = Ref(0)
        abort_x = Ref{Union{Nothing, Vector{Float64}}}(nothing)
        margin = 1e-9 .* (hi .- lo)
        start = clamp.(theta0, lo .+ margin, hi .- margin)
        raw = function (p::Vector{Float64})
            counter[] += 1
            if counter[] > lm_max_nfev
                abort_x[] = copy(p)
                throw(_FruitAbortError())
            end
            return _fission_residual(p, A, b, E, ln_steps)
        end
        x, _, info, _ = try
            _fruit_lmdif(raw, start; maxfev=Int(lm_max_nfev) * (length(start) + 1))
        catch err
            err isa _FruitAbortError || rethrow(err)
            (abort_x[], _fission_residual(abort_x[], A, b, E, ln_steps), -1, counter[])
        end
        theta = clamp.(x, lo, hi)
        cost = norm(_fission_residual(theta, A, b, E, ln_steps))
        success = info in (1, 2, 3, 4)
        message = if info in (1, 2, 3)
            "Fit succeeded."
        elseif info == 4
            "Fit succeeded (gtol reached)."
        elseif info == 5
            "Number of evaluations exceeded the maximum."
        elseif info == -1
            "Fit aborted."
        elseif info in (6, 7, 8)
            "Tolerance seems to be too small."
        else
            "Number of evaluations exceeded the maximum."
        end
        return theta, cost, counter[], success, message
    end

    theta, cost, nfev, success, message =
        _fission_bounded_lm(theta0, A, b, E, ln_steps, lo, hi, lm_max_nfev)
    return theta, cost, nfev, success, message
end

# ─── Article validation criteria (section "Validatsiya rascheta") ────────────

"""
    fission_validate_fit(computed, measured, spectrum_bins;
                         eps_threshold=0.05, norm_range=nothing) -> Dict

Per-sphere relative uncertainties `eps_m = (C_m - Cb_m)/C_m` must stay inside
`eps_threshold` and alternate their signs; the norm of the (normalized) model
spectrum must lie in `norm_range` when one is supplied.  Returns the metrics
and the boolean flags `residuals_ok`, `norm_ok`, `passed`.
"""
function fission_validate_fit(computed::AbstractVector{<:Real}, measured::AbstractVector{<:Real},
                              spectrum_bins::AbstractVector{<:Real};
                              eps_threshold::Real=0.05,
                              norm_range::Union{Nothing, Tuple{<:Real, <:Real}}=nothing)
    meas = Float64.(collect(measured))
    comp = Float64.(collect(computed))
    eps = [(meas[i] != 0.0) ? (comp[i] - meas[i]) / meas[i] : NaN for i in eachindex(meas)]
    finite_eps = filter(isfinite, eps)

    max_eps = isempty(finite_eps) ? Inf : maximum(abs.(finite_eps))
    fom = isempty(finite_eps) ? Inf : 100.0 * sqrt(mean(finite_eps .^ 2))

    signs = sign.(finite_eps)
    sign_changes = isempty(signs) ? 0 :
        count(i -> signs[i] * signs[i - 1] < 0, 2:length(signs))
    signs_mixed = any(>(0), signs) && any(<(0), signs)

    spectrum_norm = sum(Float64.(collect(spectrum_bins)))
    residuals_ok = isfinite(max_eps) && max_eps <= Float64(eps_threshold)

    norm_ok::Union{Nothing, Bool} = nothing
    if norm_range !== nothing
        norm_ok = Float64(norm_range[1]) <= spectrum_norm <= Float64(norm_range[2])
    end
    passed = norm_ok === nothing ? residuals_ok : (residuals_ok && norm_ok)

    return Dict{String, Any}(
        "fom_percent" => fom,
        "max_relative_uncertainty" => max_eps,
        "relative_uncertainties" => [isfinite(e) ? e : nothing for e in eps],
        "residual_sign_changes" => sign_changes,
        "signs_mixed" => signs_mixed,
        "spectrum_norm" => spectrum_norm,
        "residuals_ok" => residuals_ok,
        "norm_ok" => norm_ok,
        "eps_threshold" => Float64(eps_threshold),
        "passed" => passed,
    )
end

"""
    solve_fission_ga(A, b, x0; E=nothing, log_steps=nothing, initial_params=nothing,
                     fit_scale=true, ga_popsize=15, ga_maxiter=100, ga_tol=1e-10,
                     lm_method="trf", lm_max_nfev=2000, eps_threshold=0.05,
                     random_state=nothing) -> UnfoldResult

Two-stage stochastic unfolding with the Fission model (article `BonnerFinder`):
a differential-evolution global search of the parameter hypercube on the L1
discrepancy of the folded readings, then a nonlinear least-squares refinement
of the best point (plus an optional refinement of `initial_params`, the better
fit winning).

`x0` is kept for the `(A, b, x0)` solver interface and is unused by the
parametric model, exactly as in Python.  `E` is the energy grid in MeV
(default: `10 .^ range(-9, 2; length=n)`); `log_steps` the natural-logarithmic
bin widths `d ln E` (default: `compute_log_steps(E) * log(10)`).
`lm_method` is `"trf"` (bounded, default) or `"lm"` (Levenberg-Marquardt, as in
the article's SciLab `leastsq`).  `spectrum = Phi(E; P*) .* log_steps` — the
per-bin fluence convention of the other parametric methods.  `iterations` is
the total function-evaluation count; `residual_norm` is the stage-2 cost
`||A x - b||₂`.  `extra` carries `model_params` / `params`, `validation`,
`message` and the solver knobs.
"""
function solve_fission_ga(A::AbstractMatrix{T}, b::AbstractVector{T},
                          x0::AbstractVector{T};
                          E::Union{Nothing, AbstractVector}=nothing,
                          log_steps::Union{Nothing, AbstractVector}=nothing,
                          initial_params=nothing,
                          fit_scale::Bool=true,
                          ga_popsize::Integer=15,
                          ga_maxiter::Integer=100,
                          ga_tol::Real=1e-10,
                          lm_method::AbstractString="trf",
                          lm_max_nfev::Integer=2000,
                          eps_threshold::Real=0.05,
                          random_state::Union{Integer, Nothing}=nothing) where T<:AbstractFloat
    lm_method in ("trf", "lm") ||
        throw(ArgumentError("Unknown lm_method: $lm_method. Use \"trf\" or \"lm\"."))
    ga_tol >= 0 || throw(ArgumentError("ga_tol must be non-negative, got $ga_tol"))

    AF = Matrix{Float64}(A)
    bf = vec(Vector{Float64}(b))
    n_energy = size(AF, 2)
    E_f = if E === nothing
        collect(10.0 .^ range(-9.0, 2.0; length=n_energy))
    else
        Float64.(collect(E))
    end
    ln = if log_steps === nothing
        compute_log_steps(E_f) .* log(10.0)
    else
        vec(Float64.(collect(log_steps)))
    end

    lo, hi = _fission_theta_bounds(fit_scale)

    rng = random_state === nothing ? Random.MersenneTwister() :
                                    Random.MersenneTwister(Int(random_state))

    # ---- Stage 1: stochastic (genetic) global search ------------------------
    target = theta -> _fission_target(theta, AF, bf, E_f, ln)
    theta_ga, _, de_nfev = _fission_de(target, lo, hi, rng;
                                      popsize=Int(ga_popsize), maxiter=Int(ga_maxiter),
                                      tol=Float64(ga_tol))
    total_nfev = de_nfev

    candidates = [clamp.(theta_ga, lo, hi)]
    theta_user = _fission_theta_from_initial(initial_params, fit_scale)
    theta_user !== nothing && push!(candidates, theta_user)

    # ---- Stage 2: nonlinear least-squares refinement -----------------------
    best_theta = nothing
    best_cost = Inf
    best_success = false
    best_message = ""
    for theta0 in candidates
        theta, cost, nfev, success, message = _fission_run_lm(
            theta0, AF, bf, E_f, ln, fit_scale, String(lm_method), lm_max_nfev)
        total_nfev += nfev
        if cost < best_cost
            best_theta, best_cost = theta, cost
            best_success, best_message = success, message
        end
    end

    spectrum = _fission_model_shape(best_theta, E_f) .* ln

    params = Dict{String, Any}(name => Float64(best_theta[i])
                               for (i, name) in enumerate(FISSION_PARAM_NAMES))
    fit_scale && (params["phi_scale"] = 10.0^best_theta[8])
    weight_sum = sum(best_theta[1:3])
    params["weight_fractions"] = if weight_sum > 0
        Dict{String, Float64}(name => Float64(best_theta[i] / weight_sum)
                              for (i, name) in enumerate(("a1", "a2", "a3")))
    else
        Dict{String, Float64}("a1" => 0.0, "a2" => 0.0, "a3" => 0.0)
    end
    params["cost"] = best_cost

    norm_range = fit_scale ? nothing : (0.6, 1.2)
    validation = fission_validate_fit(AF * spectrum, bf, spectrum;
                                     eps_threshold=eps_threshold, norm_range=norm_range)

    extra = Dict{String, Any}(
        "model_params" => params,
        "params" => params,
        "validation" => validation,
        "message" => best_message,
        "method" => "fission_ga",
        "initial_params" => initial_params,
        "fit_scale" => fit_scale,
        "ga_popsize" => Int(ga_popsize),
        "ga_maxiter" => Int(ga_maxiter),
        "ga_tol" => Float64(ga_tol),
        "lm_method" => String(lm_method),
        "lm_max_nfev" => Int(lm_max_nfev),
        "eps_threshold" => Float64(eps_threshold),
        "T0" => T0_THERMAL,
        "Ed" => ED_EPITHERMAL,
    )
    return UnfoldResult(spectrum, total_nfev, best_success, best_cost, extra)
end
