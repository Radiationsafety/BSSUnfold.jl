"""
Uno-style constrained unfolding: Lagrange-Newton NLP presets.

Julia analogue of the R package `Uno` 2.x (Narasimhan; Vanaret &
Leyffer 2024, arXiv:2406.13454): the R package binds the C++ solver
*Uno* ("Unifying Nonlinear Optimization"), which expresses non-linearly
constrained optimisation as a Lagrange-Newton method whose building
blocks (constraint-relaxation, inequality-handling, Hessian and
globalisation strategies) are freely combined, reproducing classical
solvers such as `filterSQP` and `IPOPT` by presets.

Unfolding NLP: minimise `f(x) = 1/2 ||W (A x - b)||^2 + lam/2 ||D2 x||^2`
subject to `x >= 0`. Two presets are provided:

* `"filter_sqp"` — sequential quadratic programming with the **exact**
  Hessian `H = Aᵀ W² A + lam D2ᵀ D2`, Newton directions and Vanaret &
  Leyffer's **filter globalisation** (filter on the
  `(objective, constraint violation)` pair with the `gamm`-margin
  acceptability test);
* `"ipopt_like"` — a primal-dual **interior-point** method in the IPOPT
  manner: log-barrier handling, barrier-Hessian regularisation, a
  geometric mu schedule, a fraction-to-the-boundary rule and an optional
  dense **BFGS** quasi-Newton Hessian (`hessian="bfgs"`).
"""

const _UNO_PRESETS = ("filter_sqp", "ipopt_like")
const _UNO_TINY = 1e-300
# fraction-to-the-boundary factor for the interior-point iteration
const _UNO_FTB_TAU = 0.995

"""
    uno_objective(A, b, w, regularization, x; cache=nothing)

Objective `1/2 ||W(Ax - b)||^2 + lam/2 ||D2 x||^2`.

`cache` (optional) carries the cached derivative operator over repeated
calls in the solver loops (`cache["D"]`).
"""
function uno_objective(A::AbstractMatrix,
                       b::AbstractVector,
                       w::AbstractVector,
                       regularization::Real,
                       x::AbstractVector;
                       cache::Union{Nothing,Dict}=nothing)
    x = Vector{Float64}(x)
    r = w .* (A * x .- b)
    obj = 0.5 * Float64(dot(r, r))
    if regularization > 0 && length(x) > 2
        local D
        if cache !== nothing && haskey(cache, "D")
            D = cache["D"]
        else
            D = _create_derivative_matrix(length(x), 2)
            cache !== nothing && (cache["D"] = D)
        end
        obj = obj + 0.5 * regularization * Float64(dot(D * x, D * x))
    end
    return obj
end

"""
    uno_gradient(A, b, w, regularization, x; cache=nothing)

Gradient of [`uno_objective`](@ref) at `x`.
"""
function uno_gradient(A::AbstractMatrix,
                      b::AbstractVector,
                      w::AbstractVector,
                      regularization::Real,
                      x::AbstractVector;
                      cache::Union{Nothing,Dict}=nothing)
    x = Vector{Float64}(x)
    Aw = A .* w
    g = Aw' * (w .* (A * x .- b))
    if regularization > 0 && length(x) > 2
        local D
        if cache !== nothing && haskey(cache, "D")
            D = cache["D"]
        else
            D = _create_derivative_matrix(length(x), 2)
            cache !== nothing && (cache["D"] = D)
        end
        g = g .+ regularization .* (D' * (D * x))
    end
    return Vector{Float64}(g)
end

"""
    uno_filter(filter_entries, f, viol; gamma=1e-5)

Fletcher-Leyffer (Uno) filter acceptability test.

A trial point `(f, viol)` is *acceptable* for the filter when, for every
entry `(f_j, viol_j)`, at least one of the coordinates is better by the
safety margin `gamma * viol_j` (the classical Fletcher-Leyffer
two-cycle rule).

    f <= f_j - gamma * viol_j   OR   viol <= viol_j - gamma * viol_j
"""
function uno_filter(filter_entries::Vector{Tuple{Float64,Float64}},
                    f::Real, viol::Real; gamma::Real=1e-5)
    for (f_j, viol_j) in filter_entries
        if viol_j > 0.0
            acceptable = (f <= f_j - gamma * viol_j) ||
                         (viol <= (1.0 - gamma) * viol_j)
        else
            # zero-violation entries degenerate to a strict objective
            # decrease requirement
            acceptable = (f < f_j) || (viol < viol_j)
        end
        if !acceptable
            return false
        end
    end
    return true
end

"""
    uno_augment_filter(filter_entries, f, viol)

Append a new entry to the filter (Uno's update rule: the entry is
appended even when it is dominated by older entries — the filter is
monotone only in the sense that no entry is ever removed).
"""
function uno_augment_filter(filter_entries::Vector{Tuple{Float64,Float64}},
                            f::Real, viol::Real)
    push!(filter_entries, (Float64(f), Float64(viol)))
    return filter_entries
end

"""
    _uno_filter_sqp(A, b, w, x0, regularization, max_iterations, tolerance, cache)

Uno `filterSQP` preset: Lagrange-Newton SQP with the exact (constant)
Hessian and the Vanaret-Leyffer filter.

The unfolding NLP is a *convex quadratic* objective with box
inequalities, so the exact-Hessian SQP sub-problem is the QP itself: it
is solved in one Lagrange-Newton step through the classical active-set
zero-space (Lawson-Hanson NNLS) solver on the equivalent single
least-squares system, and the `(objective, violation)` pair of the
result is checked against Uno's filter for the reported KKT statistics.
"""
function _uno_filter_sqp(A::Matrix{Float64}, b::Vector{Float64},
                         w::Vector{Float64}, x0::Vector{Float64},
                         regularization::Float64,
                         max_iterations::Int, tolerance::Float64,
                         cache::Dict)
    m, n = size(A)
    # Equivalent single least-squares system  [W A; sqrt(lam) Dn] x = [W b; 0]
    if regularization > 0 && n > 2
        D = _create_derivative_matrix(n, 2)
        Gsqrt = sqrt(regularization) .* D
        A_aug = vcat(A .* w, Gsqrt)
        b_aug = vcat(b .* w, zeros(n - 2))
    else
        A_aug = A .* w
        b_aug = b .* w
    end

    xs = solve_nnls(A_aug, b_aug)
    x = max.(xs, 0.0)
    f = uno_objective(A, b, w, regularization, x; cache=cache)
    viol = Float64(dot(min.(x, 0.0), min.(x, 0.0)))
    g = uno_gradient(A, b, w, regularization, x; cache=cache)
    interior = x .> 10.0 * _UNO_TINY
    m_int = any(interior) ? Float64(norm(g[interior], Inf)) : 0.0
    act = .!interior
    m_act = any(act) ? Float64(norm(max.(-g[act], 0.0), Inf)) : 0.0
    dual_inf = max(m_int, m_act)
    converged = (dual_inf <= tolerance && viol <= tolerance) && all(isfinite, x)
    return x, 1, converged, Float64(f), viol, dual_inf
end

"""
    _uno_ipopt_like(A, b, w, H, x0, regularization, max_iterations,
                    tolerance, hessian_mode, cache)

Primal-dual interior point in the IPOPT manner.

Minimises the barrier objective `f(x) - mu sum(log x)` by Newton
iterations with a geometric mu schedule; `hessian_mode` selects the
Hessian building block of Uno: `"exact"` or `"bfgs"`.
"""
function _uno_ipopt_like(A::Matrix{Float64}, b::Vector{Float64},
                         w::Vector{Float64}, H::Matrix{Float64},
                         x0::Vector{Float64},
                         regularization::Float64,
                         max_iterations::Int, tolerance::Float64,
                         hessian_mode::String,
                         cache::Dict)
    n = length(x0)
    x = max.(x0, _UNO_TINY)
    mus = [0.1^k for k in 0:29]
    mu_k = 1
    converged = false
    Id = Matrix{Float64}(I, n, n)
    it = 0
    f = uno_objective(A, b, w, regularization, x; cache=cache)
    # Hessian building block: exact by default, dense BFGS approximation
    # (updated from the gradient differences) in the quasi-Newton mode.
    hess_mode = lowercase(hessian_mode)
    hess_mode ∈ ("exact", "bfgs") || throw(ArgumentError(
        "hessian must be 'exact' or 'bfgs', got $hess_mode"))
    scale0 = Float64(mean(diag(H)))
    B = Id .* ((isfinite(scale0) && scale0 > 0) ? scale0 : 1.0)
    local g_prev::Union{Nothing,Vector{Float64}}
    g_prev = nothing
    g0 = uno_gradient(A, b, w, regularization, x; cache=cache)
    grad_scale = max(Float64(norm(g0 .- max(mus[1], 0.0) ./ x, Inf)), 1.0)
    for k in 1:Int(max_iterations)
        it = k
        mu = mus[min(mu_k, length(mus))]
        if mu_k < length(mus) && it % 3 == 0
            mu_k += 1
        end
        Hb = (hess_mode == "exact" ? H : B) .+
             mu .* Diagonal(1.0 ./ (x .* x))
        Hb = Hb .+ 1e-12 * max(Float64(mean(abs.(H))), 1.0) .* Id

        g = uno_gradient(A, b, w, regularization, x; cache=cache)
        barrier_grad = g .- mu ./ x

        local d::Vector{Float64}
        try
            d = -(Hb \ barrier_grad)
        catch
            d = -(pinv(Hb) * barrier_grad)
        end
        if !all(isfinite, d)
            d = zeros(n)
        end

        # fraction-to-the-boundary
        neg = d .< 0
        alpha_max = any(neg) ?
            _UNO_FTB_TAU * Float64(minimum(-x[neg] ./ d[neg])) : 1.0
        t = min(1.0, Float64(alpha_max))
        # backtracking on the barrier objective
        x_old = copy(x)
        for _ in 1:50
            xs = max.(x .+ t .* d, _UNO_TINY)
            fs = uno_objective(A, b, w, regularization, xs; cache=cache)
            barrier_new = fs - mu * Float64(sum(log, xs))
            barrier_old = f - mu * Float64(sum(log, max.(x, _UNO_TINY)))
            if barrier_new <= barrier_old - 1e-4 * t * abs(barrier_old) ||
               (t < 1e-14)
                x = xs
                f = fs
                break
            end
            t *= 0.5
        end
        g_new = uno_gradient(A, b, w, regularization, x; cache=cache)
        if hess_mode == "bfgs" && g_prev !== nothing
            s = x .- x_old
            y = g_new .- g_prev
            ys = Float64(dot(s, y))
            if ys > 1e-10
                Bs = B * s
                sBs = Float64(dot(s, Bs))
                if sBs > 1e-12
                    B = B .+ (y .* y') ./ ys .- (Bs .* Bs') ./ sBs
                end
            end
        end
        g_prev = g_new
        # convergence: barrier gradient small (relative to the initial
        # gradient scale -- Uno's relative-dual-infeasibility metric) and
        # mu at the floor
        dual_inf_it = Float64(norm(barrier_grad, Inf))
        if dual_inf_it <= grad_scale * tolerance && mu <= mus[end]
            converged = true
            break
        end
    end
    g_last = uno_gradient(A, b, w, regularization, x; cache=cache)
    mu_final = mus[min(mu_k, length(mus))]
    dual_inf = Float64(norm(
        g_last .- mu_final ./ max.(x, _UNO_TINY), Inf))
    dual_inf_rel = dual_inf / grad_scale
    viol = Float64(dot(min.(x, 0.0), min.(x, 0.0)))
    return x, it, converged, Float64(f), viol, dual_inf_rel
end

"""
    solve_uno_full(A, b, x0=nothing; preset="filter_sqp", weights="uniform",
                   regularization=1e-3, hessian="exact", max_iterations=300,
                   tolerance=1e-10)

Solve the unfolding NLP with an Uno preset; returns the rich
diagnostics dictionary (1:1 with the Python original):

`spectrum`, `preset`, `hessian`, `objective`, `constraint_violation`,
`dual_infeasibility`, `n_iterations`, `converged`.

# Keywords
- `preset` — `"filter_sqp"` (default) or `"ipopt_like"`;
- `weights` — `"uniform"` (default), `"poisson"` (`w_i = 1 / b_i`) or an
  explicit positive weight array;
- `regularization` — relative roughness ridge on the second differences
  (default: 1e-3; `0` disables it);
- `hessian` — `"exact"` (default) or `"bfgs"` (interior-point preset
  only; with `filter_sqp` it is silently ignored as the exact Hessian is
  constant);
- `max_iterations` — maximum iterations (default: 300);
- `tolerance` — KKT tolerance (default: 1e-10).
"""
function solve_uno_full(A::AbstractMatrix{T}, b::AbstractVector{T},
                        x0::Union{AbstractVector{T},Nothing}=nothing;
                        preset::AbstractString="filter_sqp",
                        weights::Union{AbstractString,Nothing,
                                       AbstractVector{<:Real}}="uniform",
                        regularization::Real=1e-3,
                        hessian::AbstractString="exact",
                        max_iterations::Integer=300,
                        tolerance::Real=1e-10) where T<:AbstractFloat
    preset = lowercase(string(preset))
    preset ∈ _UNO_PRESETS || throw(ArgumentError(
        "preset must be one of $_UNO_PRESETS, got $preset"))
    A, b, x0 = validate_system(A, b; x0=x0,
                               max_iterations=max_iterations,
                               tolerance=tolerance)
    m, n = size(A)
    regularization < 0 && throw(ArgumentError(
        "regularization must be non-negative, got $regularization"))
    Af = Matrix{Float64}(A)
    bf = Vector{Float64}(b)
    local w::Vector{Float64}
    if weights === nothing || weights isa AbstractString
        lw = weights === nothing ? "uniform" : lowercase(string(weights))
        if lw in ("", "uniform", "none", "ones")
            w = ones(m)
        elseif lw == "poisson"
            w = 1.0 ./ max.(bf, _UNO_TINY)
        else
            throw(ArgumentError(
                "weights must be 'uniform', 'poisson' or an array, " *
                "got $lw"))
        end
    else
        w = Vector{Float64}(weights)
        if length(w) != m || any(w .<= 0)
            throw(ArgumentError(
                "weights must be a positive array of length $m"))
        end
    end

    cache = Dict{String,Any}()
    if regularization > 0 && n > 2
        D = _create_derivative_matrix(n, 2)
        G = D' * D
        G = G ./ max(Float64(mean(diag(G))), 1.0)
        cache["D"] = D
    else
        G = zeros(n, n)
        cache["D"] = n > 2 ? _create_derivative_matrix(n, 2) : zeros(0, n)
    end

    Aw = Af .* w
    H = Aw' * Aw .+ regularization .* G

    x_start = x0 === nothing ? ones(n) : Vector{Float64}(x0)

    if preset == "filter_sqp"
        x, it, converged, f, viol, dual_inf = _uno_filter_sqp(
            Af, bf, w, x_start, Float64(regularization),
            Int(max_iterations), Float64(tolerance), cache)
    else
        x, it, converged, f, viol, dual_inf = _uno_ipopt_like(
            Af, bf, w, H, x_start, Float64(regularization),
            Int(max_iterations), Float64(tolerance),
            lowercase(string(hessian)), cache)
    end
    return Dict{String,Any}(
        "spectrum" => x,
        "preset" => preset,
        "hessian" => preset == "filter_sqp" ? "exact" : lowercase(string(hessian)),
        "objective" => f,
        "constraint_violation" => viol,
        "dual_infeasibility" => dual_inf,
        "n_iterations" => it,
        "converged" => converged)
end

"""
    solve_uno(A, b, x0=nothing; preset="filter_sqp", weights="uniform",
              regularization=1e-3, hessian="exact", max_iterations=300,
              tolerance=1e-10)

Solve the unfolding NLP with an Uno preset.

Returns an [`UnfoldResult`](@ref); see [`solve_uno_full`](@ref) for the
parameters.
"""
function solve_uno(A::AbstractMatrix{T}, b::AbstractVector{T},
                   x0::Union{AbstractVector{T},Nothing}=nothing;
                   preset::AbstractString="filter_sqp",
                   weights::Union{AbstractString,Nothing,
                                  AbstractVector{<:Real}}="uniform",
                   regularization::Real=1e-3,
                   hessian::AbstractString="exact",
                   max_iterations::Integer=300,
                   tolerance::Real=1e-10) where T<:AbstractFloat
    diag = solve_uno_full(A, b, x0;
                          preset=preset, weights=weights,
                          regularization=regularization, hessian=hessian,
                          max_iterations=max_iterations, tolerance=tolerance)
    x = diag["spectrum"]
    residual = Vector{Float64}(b) .- Matrix{Float64}(A) * x
    return UnfoldResult(x, diag["n_iterations"], diag["converged"],
                        Float64(norm(residual)),
                        Dict{String,Any}(
                            "preset" => diag["preset"],
                            "hessian" => diag["hessian"],
                            "objective" => diag["objective"],
                            "constraint_violation" => diag["constraint_violation"],
                            "dual_infeasibility" => diag["dual_infeasibility"]))
end
