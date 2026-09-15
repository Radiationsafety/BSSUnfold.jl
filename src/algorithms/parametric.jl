"""
FRUIT-based parametric unfolding (Bedogni et al., NIM A 580, 1301-1309,
2007; Pyshkina et al., 2021).

The spectrum is a weighted superposition of three components:

    Thermal    (E < 1e-7 MeV):  (E/T0^2) * exp(-E/T0)
    Epithermal (1e-7 < E < 0.1):[1 - exp(-(E/Ed)^2)] * E^(b-1) * exp(-E/beta')
    Fast       (E > 0.1 MeV):   E^alpha * exp(-E/beta)

with the constraint `P_th + P_epi + P_f = 1` (`P_f = 1 - P_th - P_epi`).
FRUIT constants: `_T0 = 2.53e-8`, `_Ed = 7.07e-8` MeV.

Optimizers (lmfit / qpsolvers are replaced by in-house implementations):
`solve_parametric` — multi-start Levenberg-Marquardt with parameter
bounds; `solve_parametric_cvxpy` and `solve_parametric_qpsolvers` —
SQP iterations: linearization by a numerical Jacobian and a projectional
Tikhonov substep solved by regularized normal equations
(regularized Newton) with clamping to the bounds; `solve_parametric_combined`
— first a leastsq fit, then a QP refinement of the spectrum via NNLS
on the augmented matrix (`solve_nnls` of the BSSUnfold package).
"""
const PARAMETRIC_T0 = 2.53e-8
const PARAMETRIC_ED = 7.07e-8
const PARAMETRIC_THERMAL_MAX = 1e-7
const PARAMETRIC_FAST_MIN = 0.1

const PARAM_NAMES = ["b", "beta_prime", "alpha", "beta", "P_th", "P_epi"]
const PARAM_DEFAULTS = [(1.0, 0.5, 2.0), (0.01, 1e-4, 1.0), (0.5, 0.0, 5.0),
                        (2.0, 0.1, 20.0), (1.0, 0.0, 1.0), (1.0, 0.0, 1.0)]

"""
    compute_log_steps(E_MeV) -> Vector{Float64}

Logarithmic steps d log10 E over the energy grid: edge bins use a
one-sided difference, interior bins a central difference.  For d ln E
multiply by `log(10)` (convention of the python package).
"""
function compute_log_steps(E::AbstractVector{<:Real})
    E_f = Float64.(collect(E))
    n = length(E_f)
    log_steps = zeros(n)
    log_e = log10.(E_f .+ 1e-15)
    if n > 1
        log_steps[1] = log_e[2] - log_e[1]
        log_steps[end] = log_e[end] - log_e[end-1]
    else
        log_steps[1] = 1.0
    end
    if n > 2
        for i in 2:(n-1)
            log_steps[i] = (log_e[i+1] - log_e[i-1]) / 2.0
        end
    end
    return log_steps
end

function _param_th(E_f::Vector{Float64})
    out = zeros(length(E_f))
    m = findall(<(PARAMETRIC_THERMAL_MAX), E_f)
    for j in m
        out[j] = (E_f[j] / PARAMETRIC_T0^2) * exp(-E_f[j] / PARAMETRIC_T0)
    end
    return out
end

function _param_epi(E_f::Vector{Float64}, b::Float64, beta_prime::Float64)
    out = zeros(length(E_f))
    m = findall(x -> x >= PARAMETRIC_THERMAL_MAX && x < PARAMETRIC_FAST_MIN, E_f)
    for j in m
        out[j] = (1.0 - exp(-((E_f[j] / PARAMETRIC_ED)^2))) *
                 E_f[j]^(b - 1.0) * exp(-E_f[j] / beta_prime)
    end
    return out
end

function _param_fast(E_f::Vector{Float64}, alpha::Float64, beta::Float64)
    out = zeros(length(E_f))
    m = findall(>=(PARAMETRIC_FAST_MIN), E_f)
    for j in m
        out[j] = E_f[j]^alpha * exp(-E_f[j] / beta)
    end
    return out
end

"""
    parametric_model(E, b, beta_prime, alpha, beta, P_th, P_epi) -> Vector{Float64}

Three-component parametric FRUIT model of the neutron spectrum
(fl per energy bin).  `P_f = max(0, 1 - P_th - P_epi)`.
"""
function parametric_model(E::AbstractVector{<:Real}, b::Real, beta_prime::Real,
                          alpha::Real, beta::Real, P_th::Real, P_epi::Real)
    E_f = Float64.(collect(E))
    P_f = max(0.0, 1.0 - Float64(P_th) - Float64(P_epi))
    return Float64(P_th) .* _param_th(E_f) .+
           Float64(P_epi) .* _param_epi(E_f, Float64(b), Float64(beta_prime)) .+
           P_f .* _param_fast(E_f, Float64(alpha), Float64(beta))
end

function _param_default_vec()
    return Float64[d[1] for d in PARAM_DEFAULTS]
end

function _param_lo_vec()
    return Float64[d[2] for d in PARAM_DEFAULTS]
end

function _param_hi_vec()
    return Float64[d[3] for d in PARAM_DEFAULTS]
end

function _param_clamp!(p::Vector{Float64})
    lo = _param_lo_vec()
    hi = _param_hi_vec()
    for i in eachindex(p)
        p[i] = clamp(p[i], lo[i], hi[i])
    end
    return p
end

function _param_merge_user(initial_params)
    p = _param_default_vec()
    if initial_params !== nothing
        if initial_params isa AbstractDict
            for (i, name) in enumerate(PARAM_NAMES)
                if haskey(initial_params, name)
                    p[i] = Float64(initial_params[name])
                end
            end
        elseif initial_params isa AbstractVector
            for i in 1:min(length(initial_params), length(p))
                p[i] = Float64(initial_params[i])
            end
        end
    end
    return _param_clamp!(p)
end

function _param_dict(p::Vector{Float64})
    return Dict{String,Float64}(n => p[i] for (i, n) in enumerate(PARAM_NAMES))
end

"""
    find_initial_params(A, b, E, log_steps; n_grid=5, n_restarts=1)

Coarse grid search over `P_th x P_epi` (the remaining parameters — FRUIT defaults);
candidates are sorted by the residual norm.  With `n_restarts == 1` a single best
`Dict{String,Float64}` is returned, otherwise — a list of the best by residual.
"""
function find_initial_params(A::AbstractMatrix, b::AbstractVector, E::AbstractVector,
                             log_steps::AbstractVector; n_grid::Integer=5, n_restarts::Integer=1)
    candidates = Tuple{Float64,Vector{Float64}}[]
    p0 = _param_default_vec()
    idx_th = findfirst(==("P_th"), PARAM_NAMES)
    idx_epi = findfirst(==("P_epi"), PARAM_NAMES)
    for kk in 1:n_grid
        p_th = (kk - 1) / max(n_grid - 1, 1)
        for m in 1:n_grid
            p_epi = (m - 1) / max(n_grid - 1, 1)
            p_th + p_epi > 1.0 && continue
            p = copy(p0)
            p[idx_th] = p_th
            p[idx_epi] = p_epi
            spectrum = parametric_model(E, p[1], p[2], p[3], p[4], p[5], p[6]) .* log_steps
            residual = A * spectrum .- b
            push!(candidates, (norm(residual), p))
        end
    end
    isempty(candidates) && return n_restarts > 1 ? Vector{Vector{Float64}}[] : [[p0]]
    sort!(candidates, by = c -> c[1])
    tops = [c[2] for c in candidates[1:min(length(candidates), n_restarts)]]
    return n_restarts > 1 ? tops : tops[1]
end

"""
    compute_parametric_jacobian(E, log_steps, p::Vector{Float64}; delta=1e-8)

Numerical Jacobian of `(parametric_model .* log_steps)` with respect to the 6 parameters
with clamping of perturbations to the bounds; at the boundary — a backward difference.
"""
function compute_parametric_jacobian(E::AbstractVector, log_steps::AbstractVector,
                                     p::Vector{Float64}; delta::Float64=1e-8)
    E_f = Float64.(collect(E))
    ls = Float64.(collect(log_steps))
    lo = _param_lo_vec()
    hi = _param_hi_vec()
    np = length(p)
    J = zeros(length(E_f), np)
    s0 = parametric_model(E_f, p[1], p[2], p[3], p[4], p[5], p[6]) .* ls

    for i in 1:np
        d = delta
        if p[i] + d > hi[i]
            d = max(0.0, hi[i] - p[i]) * 0.5
        end
        p[i] + d < lo[i] && (d = 0.0)
        if d < 1e-15
            d = delta
            if lo[i] >= 0.0 && p[i] - d >= lo[i]
                p_pert = copy(p)
                p_pert[i] = p[i] - d
                s_pert = parametric_model(E_f, p_pert[1], p_pert[2], p_pert[3], p_pert[4],
                                          p_pert[5], p_pert[6]) .* ls
                J[:, i] .= (s0 .- s_pert) ./ d
            else
                J[:, i] .= 0.0
            end
            continue
        end
        p_plus = copy(p)
        p_plus[i] = p[i] + d
        s_plus = parametric_model(E_f, p_plus[1], p_plus[2], p_plus[3], p_plus[4],
                                  p_plus[5], p_plus[6]) .* ls
        J[:, i] .= (s_plus .- s0) ./ d
    end
    return J, s0
end

function _param_residual_and_jac(p::Vector{Float64}, A::Matrix{Float64}, b::Vector{Float64},
                                 E_f::Vector{Float64}, ls::Vector{Float64},
                                 reg_alpha::Float64, p0::Union{Nothing,Vector{Float64}})
    J_s, s0 = compute_parametric_jacobian(E_f, ls, p)
    r = A * s0 .- b
    if reg_alpha > 0 && p0 !== nothing
        J = vcat(A * J_s, sqrt(reg_alpha) .* Matrix{Float64}(I, length(p), length(p)))
        r = vcat(r, sqrt(reg_alpha) .* (p .- p0))
    else
        J = A * J_s
    end
    return J, r, s0
end

function _param_lm_fit(p_start::Vector{Float64}, A::Matrix{Float64}, b::Vector{Float64},
                       E_f::Vector{Float64}, ls::Vector{Float64};
                       reg_alpha::Float64=0.0, max_iter::Integer=120, tol::Float64=1e-6)
    p = copy(p_start)
    _param_clamp!(p)
    p0 = reg_alpha > 0 ? copy(p) : nothing
    mu = 1e-3
    nfev = 0
    converged = false
    message = ""
    for k in 0:max_iter
        J, r, _ = _param_residual_and_jac(p, A, b, E_f, ls, reg_alpha, p0)
        nfev += 1
        last_r_norm = norm(r)
        if last_r_norm < tol
            converged = true
            message = "Converged in $k iterations"
            break
        end
        JTJ = J' * J
        JTr = J' * r
        accepted = false
        for _ in 1:40
            Hm = JTJ + (mu + 1e-12) .* Matrix{Float64}(I, length(p), length(p))
            dp = -(Hm \ JTr)
            p_trial = p .+ dp
            _param_clamp!(p_trial)
            Jr, _rj, _s0j = _param_residual_and_jac(p_trial, A, b, E_f, ls, reg_alpha, p0)
            if norm(Jr) < last_r_norm
                p = p_trial
                mu = max(mu / 3, 1e-12)
                accepted = true
                break
            else
                mu *= 10
                mu > 1e12 && break
            end
        end
        if !accepted
            message = "No further reduction"
            break
        end
    end
    isempty(message) && (message = "Max iterations ($max_iter) reached")
    return p, converged, message, nfev
end

"""
    solve_parametric(A, b, x0=nothing; E_MeV=nothing, initial_params=nothing,
                     method="leastsq", alpha=0.0, alpha_auto=false, n_restarts=5)
      -> UnfoldResult

Nonlinear LS fit of the FRUIT model parameters (Levenberg-Marquardt with
a numerical Jacobian, parameter bounds, and multiple restarts from the
top-N points of a coarse scan by `find_initial_params`).  `alpha > 0`
adds a Tikhonov penalty `sqrt(alpha)*||p - p0||` to the residuals;
`x0` — kept for API compatibility (optional).  `method` is kept as
a label (lmfit methods are unavailable in the port).
"""
function solve_parametric(A::AbstractMatrix, b::AbstractVector, x0::Union{Nothing,AbstractVector}=nothing;
                          E_MeV::Union{Nothing,AbstractVector}=nothing,
                          initial_params=nothing,
                          method::String="leastsq",
                          alpha::Real=0.0,
                          alpha_auto::Bool=false,
                          n_restarts::Integer=5)
    AF = Matrix{Float64}(A)
    bf = Vector{Float64}(b)
    n_energy = size(AF, 2)
    E_f = E_MeV === nothing ? collect(10.0 .^ range(-9, 2, length=n_energy)) : Float64.(collect(E_MeV))
    log_steps = compute_log_steps(E_f)
    ln_steps = log_steps .* log(10)

    starts = find_initial_params(AF, bf, E_f, ln_steps; n_grid=7, n_restarts=Int(n_restarts))
    starts isa AbstractVector || (starts = [starts])

    best_spectrum = nothing
    best_residual = Inf
    best_success = false
    best_message = ""
    total_nfev = 0
    best_params = nothing

    for sp in starts
        p_start = _param_merge_user(sp isa AbstractDict ? sp : _param_dict(sp))
        reg_alpha = Float64(alpha) > 0 ? Float64(alpha) : 0.0
        p_opt, success, message, nfev = _param_lm_fit(p_start, AF, bf, E_f, ln_steps;
                                                      reg_alpha=reg_alpha)
        total_nfev += nfev
        spectrum = parametric_model(E_f, p_opt[1], p_opt[2], p_opt[3], p_opt[4],
                                    p_opt[5], p_opt[6]) .* ln_steps
        res = norm(AF * spectrum .- bf)
        if res < best_residual
            best_residual = res
            best_spectrum = spectrum
            best_success = success
            best_message = message
            best_params = _param_dict(p_opt)
        end
    end

    extra = Dict{String,Any}(
        "params" => best_params,
        "message" => best_message,
        "method" => "parametric",
        "T0" => PARAMETRIC_T0,
        "Ed" => PARAMETRIC_ED,
    )
    return UnfoldResult(max.(best_spectrum, 0.0), total_nfev, best_success,
                        best_residual, extra)
end

function _param_sqp_core(A::Matrix{Float64}, b::Vector{Float64}, E_f::Vector{Float64},
                         ln_steps::Vector{Float64}, p::Vector{Float64};
                         alpha::Float64, max_iter::Integer, tol::Float64)
    message = ""
    nfev = 0
    for k in 0:max_iter
        J_s, s_k = compute_parametric_jacobian(E_f, ln_steps, p)
        nfev += 1
        residual = A * s_k .- b
        if norm(residual) < tol
            return p, true, "Converged in $k iterations", nfev
        end

        A_eff = A * J_s
        n_p = length(p)
        P = A_eff' * A_eff .+ (max(alpha, 1e-300)) .* Matrix{Float64}(I, n_p, n_p)
        q = A_eff' * residual

        delta_val = -(P \ q)
        lo = _param_lo_vec()
        hi = _param_hi_vec()
        for i in 1:n_p
            delta_val[i] = clamp(p[i] + delta_val[i], lo[i], hi[i]) - p[i]
        end

        p = p .+ delta_val

        if norm(delta_val) < tol
            return p, true, "Converged in $(k+1) iterations", nfev
        end
    end
    isempty(message) && (message = "Max iterations ($max_iter) reached")
    return p, false, message, nfev
end

"""
    solve_parametric_cvxpy(A, b, E, log_steps; initial_params=nothing, alpha=1e-4,
                           max_iter=50, tol=1e-6) -> UnfoldResult

SQP unfolding: at each iteration linearization `A_eff = A @ J`
(where J is the Jacobian of the spectrum with respect to the parameters), the subproblem

    min ||A_eff*delta + residual||^2 + alpha*||delta||^2,  bounded delta

is solved by regularized normal equations (Newton) with clamping
to the bounds (in the python port the substep was solved via cvxpy; here — the same
math directly).  `E` — energy grid in MeV; `log_steps` — d(log10 E)
or d ln E steps.
"""
function solve_parametric_cvxpy(A::AbstractMatrix, b::AbstractVector, E::AbstractVector,
                                log_steps::AbstractVector; initial_params=nothing,
                                alpha::Real=1e-4, max_iter::Integer=50, tol::Real=1e-6)
    AF = Matrix{Float64}(A)
    bf = Vector{Float64}(b)
    E_f = Float64.(collect(E))
    ls = Float64.(collect(log_steps))
    p = _param_merge_user(initial_params)
    p, success, message, nfev = _param_sqp_core(AF, bf, E_f, ls, p;
                                                alpha=Float64(alpha), max_iter=Int(max_iter),
                                                tol=Float64(tol))
    spectrum = parametric_model(E_f, p[1], p[2], p[3], p[4], p[5], p[6]) .* ls
    res = norm(AF * spectrum .- bf)
    extra = Dict{String,Any}(
        "params" => _param_dict(p),
        "message" => message,
        "optimizer" => "cvxpy-analytic-QP",
    )
    return UnfoldResult(max.(spectrum, 0.0), nfev, success, res, extra)
end

"""
    solve_parametric_qpsolvers(A, b, E, log_steps; initial_params=nothing,
                               alpha=1e-4, max_iter=50, tol=1e-6) -> UnfoldResult

SQP unfolding equivalent to `solve_parametric_cvxpy`, but the subproblem
is written in the standard QP form of the project (`P = A_effᵀ A_eff +
alpha*I`, `q = A_effᵀ residual`, minimizing `0.5 dᵀ P d + qᵀ d` under
bound constraints) and solved directly (`d = -P\\q`) — qpsolvers is replaced
by an in-house solution via regularized Newton.
"""
function solve_parametric_qpsolvers(A::AbstractMatrix, b::AbstractVector, E::AbstractVector,
                                    log_steps::AbstractVector; initial_params=nothing,
                                    alpha::Real=1e-4, max_iter::Integer=50, tol::Real=1e-6)
    AF = Matrix{Float64}(A)
    bf = Vector{Float64}(b)
    E_f = Float64.(collect(E))
    ls = Float64.(collect(log_steps))
    p = _param_merge_user(initial_params)
    message = ""
    nfev = 0
    success = false
    for k in 0:max_iter
        J_s, s_k = compute_parametric_jacobian(E_f, ls, p)
        nfev += 1
        residual = AF * s_k .- bf
        if norm(residual) < tol
            success = true
            message = "Converged in $k iterations"
            break
        end
        A_eff = AF * J_s
        n_p = length(p)
        P = A_eff' * A_eff .+ (max(Float64(alpha), 1e-300)) .* Matrix{Float64}(I, n_p, n_p)
        q = A_eff' * residual
        delta_val = -(P \ q)
        lo = _param_lo_vec()
        hi = _param_hi_vec()
        for i in 1:n_p
            delta_val[i] = clamp(p[i] + delta_val[i], lo[i], hi[i]) - p[i]
        end
        p = p .+ delta_val
        if norm(delta_val) < tol
            success = true
            message = "Converged in $(k+1) iterations"
            break
        end
    end
    isempty(message) && (message = "Max iterations ($max_iter) reached")
    spectrum = parametric_model(E_f, p[1], p[2], p[3], p[4], p[5], p[6]) .* ls
    res = norm(AF * spectrum .- bf)
    extra = Dict{String,Any}(
        "params" => _param_dict(p),
        "message" => message,
        "optimizer" => "qpsolvers-analytic-QP",
    )
    return UnfoldResult(max.(spectrum, 0.0), nfev, success, res, extra)
end

"""
    solve_parametric_combined(A, b, E, log_steps; initial_params=nothing,
                              method="leastsq", alpha=1e-4, solver_backend="auto",max_iter=50, tol=1e-6) -> UnfoldResult

Combined pipeline: (1) leastsq fit (LM) of the FRUIT model parameters,
(2) QP refinement of the spectrum: `min ||A x - b||^2 + alpha||x - x_init||^2,
x >= 0` via the augmented matrix and `solve_nnls` of the BSSUnfold package
(in the python port the refinement went through cvxpy/qpsolvers).  The resulting spectrum
is returned multiplied by `log_steps` (log convention of the port).
"""
function solve_parametric_combined(A::AbstractMatrix, b::AbstractVector, E::AbstractVector,
                                   log_steps::AbstractVector; initial_params=nothing,
                                   method::String="leastsq", alpha::Real=1e-4,
                                   solver_backend="auto", max_iter::Integer=50, tol::Real=1e-6)
    AF = Matrix{Float64}(A)
    bf = Vector{Float64}(b)
    E_f = Float64.(collect(E))
    ls = Float64.(collect(log_steps))

    lm_res = solve_parametric(AF, bf, nothing; E_MeV=E_f, initial_params=initial_params,
                              method=method, alpha=0.0)
    spectrum_lmfit = lm_res.spectrum

    np_bins = size(AF, 2)
    p0 = max.(spectrum_lmfit ./ max.(ls, 1e-300), 0.0)
    xa = solve_nnls(
        vcat(AF, sqrt(max(Float64(alpha), 0.0)) .* Matrix{Float64}(I, np_bins, np_bins)),
        vcat(bf, sqrt(max(Float64(alpha), 0.0)) .* p0))
    refined = xa .* ls

    msg = "leastsq + QP refinement OK"
    extra = Dict{String,Any}(
        "message" => msg,
        "optimizer" => "combined",
        "solver_backend" => string(solver_backend),
        "params" => getfield(lm_res, :extra)["params"],
    )
    return UnfoldResult(max.(refined, 0.0), lm_res.iterations, lm_res.converged,
                        norm(AF * max.(refined, 0.0) .- bf), extra)
end
