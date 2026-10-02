"""
L-BFGS-B — bound-constrained limited-memory quasi-Newton unfolding.

Solves the smooth Tikhonov objective

    min_{x_min <= x <= x_max}  1/2||Ax - b||² + (regularization/2)||x||²
                               + (smoothness/2)||D2 x||²

with analytic gradients, using a native implementation of L-BFGS-B (no
Optim.jl / NLopt): a limited-memory inverse-Hessian model built by the
two-loop recursion over the last `lbfgs_history` correction pairs, box
handling through the Cauchy point of the scaled projected gradient, an
Armijo backtracking line search along the *projected* path (every trial point
is clamped into the box, so components stop at their bound while the others
keep moving), and a curvature guard that discards pairs with non-positive
`y's`.  Convergence is tested on the projected gradient (gtol) and on the
relative reduction of the objective (ftol), as in scipy's L-BFGS-B.

Reference: bssunfold core/unfold_lbfgsb.py; Byrd, Lu, Nocedal & Zhu, "A
Limited Memory Algorithm for Bound Constrained Optimization", Math. Comp.
64 (1995) 523-548.
"""

"""
    _lbfgsb_second_difference(n, T)

Second-difference operator `D2` (discrete 1D Laplacian), `max(n-2, 0)` rows.
"""
function _lbfgsb_second_difference(n::Integer, ::Type{T}) where {T<:AbstractFloat}
    rows = max(Int(n) - 2, 0)
    D2 = spzeros(T, rows, n)
    @inbounds for i in 1:rows
        D2[i, i] = one(T)
        D2[i, i + 1] = T(-2)
        D2[i, i + 2] = one(T)
    end
    return D2
end

"""
    _lbfgsb_project(v, lo, hi)

Box projection (clamping); exact, so the iterate is always feasible.
"""
function _lbfgsb_project(v::AbstractVector{T}, lo::T, hi::T) where {T<:AbstractFloat}
    return min.(max.(v, lo), hi)
end

"""
    _lbfgsb_objgrad(A, b, reg, smo, D2, x) -> (f, g)

Objective and analytic gradient, mirroring `objective_and_grad` in Python.
"""
function _lbfgsb_objgrad(A::AbstractMatrix{T}, b::AbstractVector{T},
                         reg::T, smo::T,
                         D2::Union{AbstractMatrix{T},Nothing},
                         x::AbstractVector{T}) where {T<:AbstractFloat}
    r = A * x .- b
    f = T(0.5) * dot(r, r)
    g = A' * r
    if reg > 0
        f += T(0.5) * reg * dot(x, x)
        g = g .+ reg .* x
    end
    if D2 !== nothing
        dx = D2 * x
        f += T(0.5) * smo * dot(dx, dx)
        g = g .+ smo .* (D2' * dx)
    end
    return f, g
end

"""
    _lbfgsb_curvature_scale(A, reg, smo, D2, n, T)

Largest eigenvalue of the (constant) Hessian `AᵀA + reg I + smo D2ᵀD2` by
power iteration; used as the model curvature `1/h0` for the first Cauchy
step.  Matrix-free, so it works for dense and sparse `A`.
"""
function _lbfgsb_curvature_scale(A::AbstractMatrix{T}, reg::T, smo::T,
                                 D2::Union{AbstractMatrix{T},Nothing},
                                 n::Integer, ::Type{T}) where {T<:AbstractFloat}
    v = ones(T, Int(n))
    nv = norm(v)
    nv > 0 && (v ./= nv)
    λ = zero(T)
    @inbounds for _ in 1:50
        w = A' * (A * v)
        reg > 0 && (w = w .+ reg .* v)
        if D2 !== nothing
            w = w .+ smo .* (D2' * (D2 * v))
        end
        nrm = norm(w)
        (isfinite(nrm) && nrm > 0) || break
        v = w ./ nrm
        λ = nrm
    end
    return λ
end

"""
    _lbfgsb_projected_gradient(x, g, lo, hi) -> (pg, pgmax)

The L-BFGS-B projected gradient `pg_i = -g_i`, except that a component sitting
on a bound whose gradient points out of the box is set to zero.  `pgmax` is
`||pg||_inf`, the quantity scipy's `gtol` is applied to.  (Note this is *not*
the clipped displacement `clip(x-g)-x`, whose magnitude would be destroyed by a
narrow box.)
"""
function _lbfgsb_projected_gradient(x::AbstractVector{T}, g::AbstractVector{T},
                                    lo::T, hi::T) where {T<:AbstractFloat}
    n = length(x)
    pg = Vector{T}(undef, n)
    pgmax = zero(T)
    @inbounds for i in 1:n
        xi = x[i]
        gi = g[i]
        p = if (xi <= lo && gi > 0) || (xi >= hi && gi < 0)
            zero(T)                                  # blocked by an active bound
        else
            -gi
        end
        pg[i] = p
        a = abs(p)
        a > pgmax && (pgmax = a)
    end
    return pg, pgmax
end

"""
    _lbfgsb_mask_outward!(p, x, lo, hi)

Zero the components of a quasi-Newton direction that would leave the box at
an exactly active bound (they carry no feasible movement).
"""
function _lbfgsb_mask_outward!(p::AbstractVector{T}, x::AbstractVector{T},
                               lo::T, hi::T) where {T<:AbstractFloat}
    @inbounds for i in eachindex(p)
        xi = x[i]
        if xi <= lo && p[i] < 0
            p[i] = zero(T)
        elseif xi >= hi && p[i] > 0
            p[i] = zero(T)
        end
    end
    return p
end

"""
    _lbfgsb_two_loop(q, S, Y, sy, h0) -> H q

L-BFGS two-loop recursion for the inverse model Hessian, with initial scale
`h0` (= 1/theta).  Every `sy[k]` is guaranteed positive by the curvature
guard in the caller, so no division by a non-positive curvature occurs.
"""
function _lbfgsb_two_loop(q::AbstractVector{T}, S::Vector{Vector{T}},
                          Y::Vector{Vector{T}}, sy::Vector{T},
                          h0::T) where {T<:AbstractFloat}
    kmax = length(S)
    alpha = Vector{T}(undef, kmax)
    @inbounds for k in kmax:-1:1
        a = dot(S[k], q) / sy[k]
        alpha[k] = a
        q .-= a .* Y[k]
    end
    r = h0 .* q
    @inbounds for k in 1:kmax
        b = dot(Y[k], r) / sy[k]
        r .+= (alpha[k] - b) .* S[k]
    end
    return r
end

"""
    _lbfgsb_cauchy_point(x, pg, h0, lo, hi)

Cauchy point of the L-BFGS-B model: with `H0 = theta*I` the piecewise-linear
model gradient decreases up to `alpha = 1/theta = h0`, so the minimizer along
the steepest-descent ray is the projection of `x + h0 * pg`.
"""
function _lbfgsb_cauchy_point(x::AbstractVector{T}, pg::AbstractVector{T},
                              h0::T, lo::T, hi::T) where {T<:AbstractFloat}
    return _lbfgsb_project(x .+ h0 .* pg, lo, hi) .- x
end

"""
    _lbfgsb_backtrack(...) -> (ok, x_new, f_new, g_new, alpha, nfev, ngev)

Armijo backtracking line search along the *projected* path
`alpha -> clip(x + alpha*p, lo, hi)`.  Projection keeps every trial feasible
and lets components stop at their bound while the others keep moving, which is
the piecewise-linear search path of L-BFGS-B (a plain capped line segment would
be throttled by a single nearly-active component).
"""
function _lbfgsb_backtrack(f::T, slope::T, x::AbstractVector{T}, p::AbstractVector{T},
                           lo::T, hi::T,
                           A::AbstractMatrix{T}, b::AbstractVector{T},
                           reg::T, smo::T,
                           D2::Union{AbstractMatrix{T},Nothing}) where {T<:AbstractFloat}
    c1 = T(1e-4)
    alpha = one(T)
    alpha_min = T(0.01) * eps(T)
    # smallest displacement of x that is still resolvable at this magnitude
    step_floor2 = (eps(T) * (norm(x) + one(T)))^2
    nfev = 0
    ngev = 0
    x_new = x
    f_new = f
    g_new = x
    @inbounds for trial in 1:40
        x_trial = _lbfgsb_project(x .+ alpha .* p, lo, hi)
        # the trial point stopped moving: no further decrease is resolvable
        sum(abs2, x_trial .- x) <= step_floor2 && break
        x_new = x_trial
        f_new, g_new = _lbfgsb_objgrad(A, b, reg, smo, D2, x_new)
        nfev += 1
        ngev += 1
        if isfinite(f_new) && f_new <= f + c1 * alpha * slope
            return (true, x_new, f_new, g_new, alpha, nfev, ngev)
        end
        alpha <= alpha_min && break
        alpha *= T(0.5)
    end
    return (false, x_new, f_new, g_new, alpha, nfev, ngev)
end

"""
    solve_lbfgsb(A, b, x0; ...)

Unfold `b` with the response matrix `A` using bound-constrained L-BFGS.

# Keywords
- `max_iterations::Int=500` — iteration cap (scipy `maxiter`)
- `tolerance::T=T(1e-8)` — projected-gradient tolerance (`gtol`)
- `regularization::T=T(0.0)` — Tikhonov (L2) weight
- `smoothness::T=T(0.0)` — second-difference (curvature) penalty weight
- `x_min::T=T(0.0)` — lower bound (nonnegativity by default)
- `x_max::T=T(Inf)` — upper bound
- `lbfgs_history::Int=10` — number of stored correction pairs (scipy `maxcor`)
- `ftol::T=T(2.22e-9)` — relative function-decrease tolerance, applied as
  `ftol*eps(T)` (scipy passes `factr` to the Fortran code, which multiplies it
  by `epsmch`, so this stop only fires on a genuinely stagnant decrease)

# Returns
`UnfoldResult` with `extra` keys `function_evaluations`, `gradient_evaluations`,
`projected_gradient`, `message`, `regularization`, `smoothness`, `x_min`,
`x_max`, `lbfgs_history`.
"""
function solve_lbfgsb(A::AbstractMatrix{T}, b::AbstractVector{T}, x0::AbstractVector{T};
                      max_iterations::Integer=500,
                      tolerance::T=T(1e-8),
                      regularization::T=T(0.0),
                      smoothness::T=T(0.0),
                      x_min::T=T(0.0),
                      x_max::T=T(Inf),
                      lbfgs_history::Integer=10,
                      ftol::T=T(2.220446049250313e-9)) where {T<:AbstractFloat}
    validate_system(A, b; x0=x0, max_iterations=max_iterations, tolerance=tolerance)

    n = size(A, 2)
    lo = T(x_min)
    hi = T(x_max)
    if !(hi > lo)
        throw(ArgumentError("x_max ($hi) must be greater than x_min ($lo)"))
    end
    reg = max(T(regularization), zero(T))
    smo = max(T(smoothness), zero(T))
    mem = max(Int(lbfgs_history), 1)
    maxiter = Int(max_iterations)

    D2 = smo > 0 ? _lbfgsb_second_difference(n, T) : nothing

    # Initial model curvature: L = lambda_max(AᵀA + reg I + smo D2ᵀD2), h0 = 1/L.
    L = _lbfgsb_curvature_scale(A, reg, smo, D2, n, T)
    h0_default = (isfinite(L) && L > 0) ? inv(L) : one(T)
    h0 = h0_default

    # scipy starts from max(x0, x_min); clamp both ways for feasibility.
    x = _lbfgsb_project(vec(collect(T, x0)), lo, hi)
    f, g = _lbfgsb_objgrad(A, b, reg, smo, D2, x)
    nfev = 1
    ngev = 1
    f_init = f

    S = Vector{Vector{T}}()
    Y = Vector{Vector{T}}()
    sy_hist = Vector{T}()

    iters = 0
    converged = false
    message = "maximum number of iterations reached"
    ftol_eff = ftol * eps(T)                 # scipy passes factr, Fortran multiplies by epsmach
    _, pgmax = _lbfgsb_projected_gradient(x, g, lo, hi)

    for k in 1:maxiter
        iters = k

        pg, pgmax = _lbfgsb_projected_gradient(x, g, lo, hi)
        if pgmax <= tolerance
            converged = true
            message = "convergence: norm of projected gradient <= gtol"
            break
        end

        # --- search direction: quasi-Newton, else Cauchy (projected gradient)
        if isempty(S)
            p = _lbfgsb_cauchy_point(x, pg, h0, lo, hi)
        else
            p = _lbfgsb_two_loop(copy(pg), S, Y, sy_hist, h0)
            _lbfgsb_mask_outward!(p, x, lo, hi)
            if dot(g, p) >= 0
                p = _lbfgsb_cauchy_point(x, pg, h0, lo, hi)
            end
        end

        slope = dot(g, p)
        if !(slope < 0)
            converged = true
            message = "convergence: no feasible descent direction"
            break
        end

        # --- line search; on failure drop the memory and retry on the Cauchy path
        ok, x_new, f_new, g_new, _, e_f, e_g =
            _lbfgsb_backtrack(f, slope, x, p, lo, hi, A, b, reg, smo, D2)
        nfev += e_f
        ngev += e_g
        if !ok && !isempty(S)
            empty!(S); empty!(Y); empty!(sy_hist)
            h0 = h0_default
            p = _lbfgsb_cauchy_point(x, pg, h0, lo, hi)
            slope = dot(g, p)
            if slope < 0
                ok, x_new, f_new, g_new, _, e_f, e_g =
                    _lbfgsb_backtrack(f, slope, x, p, lo, hi, A, b, reg, smo, D2)
                nfev += e_f
                ngev += e_g
            end
        end
        if !ok
            converged = true
            message = "convergence: line search failed (roundoff)"
            break
        end

        # --- relative reduction of f (scipy ftol semantics: factr*epsmch)
        drop_total = max(f_init - f_new, zero(T))
        if f - f_new <= ftol_eff * (drop_total + ftol_eff)
            x, f, g = x_new, f_new, g_new
            converged = true
            message = "convergence: relative reduction of f <= ftol"
            break
        end

        # --- step size roundoff guard
        s = x_new .- x
        y = g_new .- g
        ss = dot(s, s)
        step_floor = eps(T) * (norm(x) + one(T))
        if ss <= step_floor * step_floor
            x, f, g = x_new, f_new, g_new
            converged = true
            message = "convergence: step is at roundoff"
            break
        end

        # --- curvature guard: store only positively curved pairs
        sy = dot(s, y)
        if sy > T(1e-10) * ss
            yy = dot(y, y)
            push!(S, s)
            push!(Y, y)
            push!(sy_hist, sy)
            if length(S) > mem
                deleteat!(S, 1)
                deleteat!(Y, 1)
                deleteat!(sy_hist, 1)
            end
            scale = sy / yy
            h0 = (isfinite(scale) && scale > 0) ? scale : h0_default
        end

        x, f, g = x_new, f_new, g_new
    end

    x = _lbfgsb_project(x, lo, hi)
    residual = b .- A * x
    extra = Dict{String,Any}(
        "function_evaluations" => nfev,
        "gradient_evaluations" => ngev,
        "projected_gradient" => pgmax,
        "message" => message,
        "regularization" => reg,
        "smoothness" => smo,
        "x_min" => lo,
        "x_max" => hi,
        "lbfgs_history" => mem,
    )
    return UnfoldResult(x, iters, converged, norm(residual), extra)
end
