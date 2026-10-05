"""
EPIC Tikhonov regularization unfolding — port of
`bssunfold/core/unfold_epic.py`.

Implements the Equal Posterior Information Condition (EPIC) Tikhonov
regularization for least squares inversion, ported from the EPIC_LS package
(https://github.com/frortega/EPIC_LS). See:

    Ortega-Culaciati, F., Simons, M., Ruiz, J., Rivera, L., & Diaz-Salazar, N.
    (2021). An EPIC Tikhonov regularization: Application to quasi-static fault
    slip inversion. Journal of Geophysical Research: Solid Earth, 126,
    e2020JB021141. https://doi.org/10.1029/2020JB021141

The regularization weights are chosen so that the a posteriori variances of the
model parameters match user-supplied target variances (the EPIC condition).
Once the weights are known, the general linear least squares problem is solved,
optionally under a non-negativity constraint on the model parameters.

`scipy.optimize.least_squares(method='trf', bounds=..., x_scale='jac')` is
replaced by the native bounded trust-region-reflective solver `_lsq_trf`
defined below. It is a faithful port of scipy's TRF (Coleman-Li "hat"-space
scaling, the `exact` More-Sorensen SVD subproblem and the `lsmr` 2-D subspace
alternative, the reflective step selection of `select_step`, the same
ftol/xtol/gtol termination logic and `x_scale='jac'` column rescaling), so the
EPIC betas converge to the same solution as in Python.
"""

"""
    _EpicLsqSolution

Minimal stand-in for scipy's `OptimizeResult`, carrying exactly the fields the
EPIC port reads: `x`, `success`, `cost`, `nfev` (plus `status`, `njev` and
`optimality` for diagnostics). `_epic_weights` accesses `sol.x`, `sol.success`,
`sol.cost` and `sol.nfev` line-for-line as in Python.
"""
struct _EpicLsqSolution
    x::Vector{Float64}
    success::Bool
    cost::Float64
    nfev::Int
    status::Int
    njev::Int
    optimality::Float64
end

"""
    _epic_lstsq(A, b)

Minimum-norm least-squares solve equivalent to `numpy.linalg.lstsq`
(LAPACK gelsd): uses the pseudo-inverse, which yields the minimum-norm
solution for under-determined systems and the least-squares solution
otherwise.
"""
function _epic_lstsq(A::AbstractMatrix{Float64}, b::AbstractVector{Float64})::Vector{Float64}
    return Vector{Float64}(pinv(A) * b)
end

"""
    _compute_bounds(k_center=0.0, distance=2.0) -> Tuple{Float64, Float64}

Compute bounds for the betas to avoid floating point rounding errors.

Port of `beta_bounds.compute_bounds` from EPIC_LS. Returns the largest
`k` such that `exp(k_center - k) + exp(k_center + k)` is still well
represented in machine precision, keeping some distance from that limit.
"""
function _compute_bounds(k_center::Real=0.0, distance::Real=2.0)::Tuple{Float64,Float64}
    eps_machine = eps(Float64)
    kc = Float64(k_center)
    k_test = 0.0
    for _ in 1:999999
        k_test += 0.01
        if abs(exp(kc - k_test) + exp(kc + k_test) - exp(kc + k_test)) < eps_machine
            break
        end
    end
    k_test -= Float64(distance)
    return (kc - k_test, kc + k_test)
end

"""
    _default_target_sigmas(A, b, n, sigma_frac)

Derive default target sigmas from the naive least-squares solution.

The scale is taken as `sigma_frac` times the largest magnitude of the
unregularized least-squares solution, falling back to the measurement
scale when that solution is degenerate.
"""
function _default_target_sigmas(A::Matrix{Float64}, b::Vector{Float64},
                                n::Integer, sigma_frac::Real)
    x_ls = _epic_lstsq(A, b)
    scale = isempty(x_ls) ? 1.0 : Float64(maximum(abs, x_ls))
    if !isfinite(scale) || scale <= 0
        scale = isempty(b) ? 1.0 : Float64(maximum(abs, b))
        if !isfinite(scale) || scale <= 0
            scale = 1.0
        end
    end
    return fill(Float64(sigma_frac) * scale, Int(n))
end

"""
    _build_precision(A, noise_var) -> (P, Wx)

Build the precision matrix P = A^T inv(Cx) A and the misfit weight Wx.

With `noise_var=nothing` the misfit covariance is the identity matrix;
otherwise `Cx = noise_var * I` and `Wx' * Wx = inv(Cx)`.
"""
function _build_precision(A::Matrix{Float64}, noise_var::Union{Nothing,Real})
    m, _ = size(A)
    if noise_var === nothing
        P = Matrix{Float64}(A' * A)
        Wx = Matrix{Float64}(I, m, m)
    else
        nv = Float64(noise_var)
        nv > 0 || throw(ArgumentError("noise_var must be positive, got $noise_var"))
        inv_cx = (1.0 / nv) * Matrix{Float64}(I, m, m)
        P = Matrix{Float64}(A' * (inv_cx * A))
        # numpy.linalg.cholesky returns L (lower); scipy port used L' — in Julia
        # cholesky(M).U is the upper factor R = L'.
        Wx = Matrix{Float64}(LinearAlgebra.cholesky(inv_cx).U)
    end
    return P, Wx
end

"""
    _build_regularization_matrix(n, order)

Build the dense regularization operator H.

`order=0` gives the identity (minimum-norm), `order=1` the first
derivative and `order=2` the second derivative operator.
"""
function _build_regularization_matrix(n::Integer, order::Integer)
    order == 0 && return Matrix{Float64}(I, Int(n), Int(n))
    return create_derivative_matrix(Int(n), Int(order))
end

# ---------------------------------------------------------------------------
# Native bounded trust-region-reflective nonlinear least squares
# (port of scipy.optimize._lsq.trf.trf_bounds + common.py helpers)
# ---------------------------------------------------------------------------

const _EPIC_EPS = eps(Float64)
const _EPIC_SQRT_EPS = sqrt(eps(Float64))

_epic_norm_inf(v::AbstractVector) = isempty(v) ? 0.0 : Float64(norm(v, Inf))

function _epic_in_bounds(x::AbstractVector{Float64}, lb::AbstractVector{Float64},
                         ub::AbstractVector{Float64})
    @inbounds for i in eachindex(x)
        (x[i] >= lb[i] && x[i] <= ub[i]) || return false
    end
    return true
end

"""Port of scipy `find_active_constraints` (finite bounds only)."""
function _epic_find_active_constraints(x::AbstractVector{Float64},
                                       lb::AbstractVector{Float64},
                                       ub::AbstractVector{Float64},
                                       rtol::Real)
    n = length(x)
    active = zeros(Int, n)
    if rtol == 0
        @inbounds for i in 1:n
            x[i] <= lb[i] && (active[i] = -1)
            x[i] >= ub[i] && (active[i] = 1)
        end
        return active
    end
    @inbounds for i in 1:n
        lower_dist = x[i] - lb[i]
        upper_dist = ub[i] - x[i]
        lower_threshold = rtol * max(1.0, abs(lb[i]))
        upper_threshold = rtol * max(1.0, abs(ub[i]))
        # scipy sets lower-active first and lets upper-active overwrite it
        if lower_dist <= min(upper_dist, lower_threshold)
            active[i] = -1
        end
        if upper_dist <= min(lower_dist, upper_threshold)
            active[i] = 1
        end
    end
    return active
end

"""Port of scipy `make_strictly_feasible` (finite bounds only)."""
function _epic_make_strictly_feasible(x::AbstractVector{Float64},
                                      lb::AbstractVector{Float64},
                                      ub::AbstractVector{Float64},
                                      rstep::Real=1e-10)
    x_new = copy(x)
    active = _epic_find_active_constraints(x, lb, ub, rstep)
    @inbounds for i in eachindex(x_new)
        if active[i] == -1
            x_new[i] = (rstep == 0 ? nextfloat(lb[i])
                                   : lb[i] + rstep * max(1.0, abs(lb[i])))
        elseif active[i] == 1
            x_new[i] = (rstep == 0 ? prevfloat(ub[i])
                                   : ub[i] - rstep * max(1.0, abs(ub[i])))
        end
    end
    @inbounds for i in eachindex(x_new)
        if x_new[i] < lb[i] || x_new[i] > ub[i]
            x_new[i] = 0.5 * (lb[i] + ub[i])
        end
    end
    return x_new
end

"""Port of scipy `CL_scaling_vector` (Coleman-Li scaling; finite bounds)."""
function _epic_cl_scaling_vector(x::AbstractVector{Float64}, g::AbstractVector{Float64},
                                 lb::AbstractVector{Float64}, ub::AbstractVector{Float64})
    n = length(x)
    v = ones(Float64, n)
    dv = zeros(Float64, n)
    @inbounds for i in 1:n
        if g[i] < 0
            v[i] = ub[i] - x[i]
            dv[i] = -1
        elseif g[i] > 0
            v[i] = x[i] - lb[i]
            dv[i] = 1
        end
    end
    return v, dv
end

"""Port of scipy `compute_jac_scale` (dense J; scale_inv_old: nothing on first call)."""
function _epic_compute_jac_scale(J::Matrix{Float64},
                                 scale_inv_old::Union{Nothing,AbstractVector{Float64}})
    scale_inv = vec(sqrt.(sum(J .* J; dims=1)))
    if scale_inv_old === nothing
        scale_inv[scale_inv .== 0] .= 1
    else
        scale_inv = max.(scale_inv, scale_inv_old)
    end
    return 1 ./ scale_inv, scale_inv
end

"""Port of scipy `step_size_to_bound`. Returns (min_step, hits) with hits in {-1,0,1}."""
function _epic_step_size_to_bound(x::AbstractVector{Float64}, s::AbstractVector{Float64},
                                  lb::AbstractVector{Float64}, ub::AbstractVector{Float64})
    n = length(x)
    steps = fill(Inf, n)
    @inbounds for i in 1:n
        if s[i] != 0
            steps[i] = max((lb[i] - x[i]) / s[i], (ub[i] - x[i]) / s[i])
        end
    end
    min_step = minimum(steps)
    hits = zeros(Int, n)
    @inbounds for i in 1:n
        if steps[i] == min_step
            hits[i] = s[i] > 0 ? 1 : s[i] < 0 ? -1 : 0
        end
    end
    return min_step, hits
end

"""Port of scipy `intersect_trust_region`. Returns (t_neg, t_pos)."""
function _epic_intersect_trust_region(x::AbstractVector{Float64}, s::AbstractVector{Float64},
                                      Delta::Real)
    a = dot(s, s)
    a == 0 && throw(ErrorException("`s` is zero."))
    b = dot(x, s)
    c = dot(x, x) - Delta^2
    c > 0 && throw(ErrorException("`x` is not within the trust region."))
    d = sqrt(b * b - a * c)
    q = -(b + copysign(d, b))
    t1 = q / a
    t2 = c / q
    return t1 < t2 ? (t1, t2) : (t2, t1)
end

"""Port of scipy `build_quadratic_1d`. Returns (a, b) or (a, b, c) when s0 given."""
function _epic_build_quadratic_1d(J::Matrix{Float64}, g::AbstractVector{Float64},
                                  s::AbstractVector{Float64},
                                  diag::Union{Nothing,AbstractVector{Float64}},
                                  s0::Union{Nothing,AbstractVector{Float64}})
    v = J * s
    a = dot(v, v)
    if diag !== nothing
        a += dot(s .* diag, s)
    end
    a *= 0.5
    b = dot(g, s)
    if s0 !== nothing
        u = J * s0
        b += dot(u, v)
        c = 0.5 * dot(u, u) + dot(g, s0)
        if diag !== nothing
            b += dot(s0 .* diag, s)
            c += 0.5 * dot(s0 .* diag, s0)
        end
        return a, b, c
    end
    return a, b, nothing
end

"""Port of scipy `minimize_quadratic_1d`. Returns (t, y)."""
function _epic_minimize_quadratic_1d(a::Real, b::Real, lb::Real, ub::Real, c::Real=0.0)
    t = [Float64(lb), Float64(ub)]
    if a != 0
        extremum = -0.5 * b / a
        if lb < extremum < ub
            push!(t, extremum)
        end
    end
    best_i, best_y = 1, typemax(Float64)
    for (i, ti) in enumerate(t)
        y = ti * (a * ti + b) + c
        if y < best_y
            best_y = y
            best_i = i
        end
    end
    return t[best_i], best_y
end

"""Port of scipy `evaluate_quadratic` (1-D `s` only, as used by TRF)."""
function _epic_evaluate_quadratic(J::Matrix{Float64}, g::AbstractVector{Float64},
                                  s::AbstractVector{Float64},
                                  diag::Union{Nothing,AbstractVector{Float64}})
    Js = J * s
    q = dot(Js, Js)
    if diag !== nothing
        q += dot(s .* diag, s)
    end
    return 0.5 * q + dot(s, g)
end

"""Port of scipy `update_tr_radius`. Returns (Delta_new, ratio)."""
function _epic_update_tr_radius(Delta::Real, actual_reduction::Real,
                                predicted_reduction::Real, step_norm::Real,
                                bound_hit::Bool)
    if predicted_reduction > 0
        ratio = actual_reduction / predicted_reduction
    elseif predicted_reduction == 0 && actual_reduction == 0
        ratio = 1.0
    else
        ratio = 0.0
    end
    Delta_new = Float64(Delta)
    if ratio < 0.25
        Delta_new = 0.25 * step_norm
    elseif ratio > 0.75 && bound_hit
        Delta_new *= 2.0
    end
    return Delta_new, ratio
end

"""Port of scipy `check_termination`. Returns status::Nothing when not terminated."""
function _epic_check_termination(dF::Real, F::Real, dx_norm::Real, x_norm::Real,
                                 ratio::Real, ftol::Real, xtol::Real)
    ftol_satisfied = dF < ftol * F && ratio > 0.25
    xtol_satisfied = dx_norm < xtol * (xtol + x_norm)
    if ftol_satisfied && xtol_satisfied
        return 4
    elseif ftol_satisfied
        return 2
    elseif xtol_satisfied
        return 3
    else
        return nothing
    end
end

"""
    _epic_solve_lsq_trust_region(n, m, uf, s, V, Delta, initial_alpha)

Port of scipy `solve_lsq_trust_region` (More-Sorensen in the eigenspace of the
SVD of the augmented Jacobian). Returns `(p, alpha, n_iter)`.
"""
function _epic_solve_lsq_trust_region(n::Integer, m::Integer,
                                      uf::AbstractVector{Float64},
                                      s::AbstractVector{Float64},
                                      V::Matrix{Float64}, Delta::Real,
                                      initial_alpha::Real;
                                      rtol::Real=0.01, max_iter::Integer=10)
    suf = s .* uf

    # Check if J has full rank and try Gauss-Newton step.
    if m >= n
        threshold = _EPIC_EPS * m * s[1]
        full_rank = s[end] > threshold
    else
        full_rank = false
    end

    if full_rank
        p = -(V * (uf ./ s))
        if norm(p) <= Delta
            return p, 0.0, 0
        end
    end

    alpha_upper = norm(suf) / Delta

    phi_and_derivative(alpha) = begin
        denom = s .^ 2 .+ alpha
        p_norm = norm(suf ./ denom)
        phi = p_norm - Delta
        phi_prime = -sum(suf .^ 2 ./ denom .^ 3) / p_norm
        (phi, phi_prime)
    end

    if full_rank
        phi0, phi_prime0 = phi_and_derivative(0.0)
        alpha_lower = -phi0 / phi_prime0
    else
        alpha_lower = 0.0
    end

    if initial_alpha === nothing || (!full_rank && initial_alpha == 0)
        alpha = max(0.001 * alpha_upper, sqrt(alpha_lower * alpha_upper))
    else
        alpha = Float64(initial_alpha)
    end

    it = 0
    for k in 1:Int(max_iter)
        it = k
        if alpha < alpha_lower || alpha > alpha_upper
            alpha = max(0.001 * alpha_upper, sqrt(alpha_lower * alpha_upper))
        end
        phi, phi_prime = phi_and_derivative(alpha)
        if phi < 0
            alpha_upper = alpha
        end
        ratio = phi / phi_prime
        alpha_lower = max(alpha_lower, alpha - ratio)
        alpha -= (phi + Delta) * ratio / Delta
        if abs(phi) < rtol * Delta
            break
        end
    end

    p = -(V * (suf ./ (s .^ 2 .+ alpha)))
    # Make the norm of p equal to Delta, p is changed only slightly during
    # this (prevents p lying outside the trust region, which causes problems
    # later).
    p *= Delta / norm(p)
    return p, alpha, it
end

"""Port of scipy `solve_trust_region_2d` (returns the step only)."""
function _epic_solve_trust_region_2d(B::Matrix{Float64}, g::AbstractVector{Float64},
                                     Delta::Real)
    try
        p = -(B \ g)
        if all(isfinite, p) && dot(p, p) <= Delta^2
            return p
        end
    catch
        # singular / indefinite: fall through to the 4th-order equation
    end
    a = B[1, 1] * Delta^2
    bb = B[1, 2] * Delta^2
    c = B[2, 2] * Delta^2
    d = g[1] * Delta
    ff = g[2] * Delta
    coeffs = [-bb + d, 2 * (a - c + ff), 6 * bb, 2 * (-a + c + ff), -bb - d]
    ts = _epic_real_roots(coeffs)
    best_p = nothing
    best_val = Inf
    for t in ts
        p = Delta .* [2 * t / (1 + t^2); (1 - t^2) / (1 + t^2)]
        val = 0.5 * dot(p, B * p) + dot(g, p)
        if val < best_val
            best_val = val
            best_p = p
        end
    end
    if best_p === nothing
        # No real roots (should not happen): steepest-descent step clipped to
        # the trust region.
        gn = norm(g)
        best_p = gn > 0 ? -(Delta / gn) .* g : zeros(Float64, 2)
    end
    return best_p
end

"""Real roots of a polynomial with real coefficients (highest order first).
Companion-matrix eigenvalues, port of the `np.roots` usage in
`solve_trust_region_2d`; roots with (numerically) zero imaginary part."""
function _epic_real_roots(coeffs_in::Vector{Float64})
    coeffs = copy(coeffs_in)
    while length(coeffs) > 1 && coeffs[1] == 0
        popfirst!(coeffs)
    end
    deg = length(coeffs) - 1
    deg <= 0 && return Float64[]
    lead = coeffs[1]
    cs = coeffs ./ lead
    if deg == 1
        return [-cs[2]]
    end
    C = zeros(Float64, deg, deg)
    for i in 2:deg
        C[i, i - 1] = 1.0
    end
    C[:, deg] .= .- cs[2:end]
    rts = eigvals(C)
    out = Float64[]
    for r in rts
        if abs(imag(r)) < 1e-8 * (1 + abs(real(r)))
            push!(out, real(r))
        end
    end
    return out
end

"""
    _epic_select_step(x, J_h, diag_h, g_h, p, p_h, d, Delta, lb, ub, theta)

Port of scipy `select_step`: the best step among the constrained
trust-region step, a reflected step and the constrained Cauchy step.
Returns `(step, step_h, predicted_reduction)`.
"""
function _epic_select_step(x::Vector{Float64}, J_h::Matrix{Float64},
                           diag_h::Vector{Float64}, g_h::Vector{Float64},
                           p::Vector{Float64}, p_h::Vector{Float64},
                           d::Vector{Float64}, Delta::Real,
                           lb::Vector{Float64}, ub::Vector{Float64},
                           theta::Real)
    if _epic_in_bounds(x .+ p, lb, ub)
        p_value = _epic_evaluate_quadratic(J_h, g_h, p_h, diag_h)
        return p, p_h, -p_value
    end

    p_stride, hits = _epic_step_size_to_bound(x, p, lb, ub)

    # Compute the reflected direction.
    r_h = copy(p_h)
    @inbounds for i in eachindex(hits)
        hits[i] != 0 && (r_h[i] *= -1)
    end
    r = d .* r_h

    # Restrict the trust-region step so that it hits the bound.
    p = p .* p_stride
    p_h = p_h .* p_stride
    x_on_bound = x .+ p

    # The reflected direction crosses either the feasible region or the
    # trust-region boundary first.
    _, to_tr = _epic_intersect_trust_region(p_h, r_h, Delta)
    to_bound, _ = _epic_step_size_to_bound(x_on_bound, r, lb, ub)

    r_stride = min(to_bound, to_tr)
    if r_stride > 0
        r_stride_l = (1 - theta) * p_stride / r_stride
        r_stride_u = r_stride == to_bound ? theta * to_bound : to_tr
    else
        r_stride_l = 0.0
        r_stride_u = -1.0
    end

    # Check if the reflection step is available.
    if r_stride_l <= r_stride_u
        a, b, c = _epic_build_quadratic_1d(J_h, g_h, r_h, diag_h, p_h)
        s_t, r_value = _epic_minimize_quadratic_1d(a, b, r_stride_l, r_stride_u, c)
        r_h = r_h .* s_t .+ p_h
        r = d .* r_h
    else
        r_value = Inf
    end

    # Make the bound-constrained TR step strictly interior.
    p = p .* theta
    p_h = p_h .* theta
    p_value = _epic_evaluate_quadratic(J_h, g_h, p_h, diag_h)

    ag_h = .- g_h
    ag = d .* ag_h

    to_tr = Delta / norm(ag_h)
    to_bound, _ = _epic_step_size_to_bound(x, ag, lb, ub)
    if to_bound < to_tr
        ag_stride = theta * to_bound
    else
        ag_stride = to_tr
    end

    a, b, _ = _epic_build_quadratic_1d(J_h, g_h, ag_h, diag_h, nothing)
    ag_stride, ag_value = _epic_minimize_quadratic_1d(a, b, 0.0, ag_stride)
    ag_h = ag_h .* ag_stride
    ag = ag .* ag_stride

    if p_value < r_value && p_value < ag_value
        return p, p_h, -p_value
    elseif r_value < p_value && r_value < ag_value
        return r, r_h, -r_value
    else
        return ag, ag_h, -ag_value
    end
end

"""
    _epic_fd_jac(fun, x, f0, lb, ub)

Two-point finite-difference Jacobian (port of scipy's `jac='2-point'` scheme:
central differences `h = sqrt(eps) * max(1, |x|)`, one-sided when the stencil
would leave the box). Used only by the homogeneous preliminary EPIC step.
"""
function _epic_fd_jac(fun::Function, x::Vector{Float64}, f0::Vector{Float64},
                      lb::Vector{Float64}, ub::Vector{Float64})
    n = length(x)
    J = Matrix{Float64}(undef, length(f0), n)
    @inbounds for i in 1:n
        h = _EPIC_SQRT_EPS * max(1.0, abs(x[i]))
        if x[i] + h > ub[i]
            xm = copy(x); xm[i] -= h
            J[:, i] = (f0 .- fun(xm)) / h
        elseif x[i] - h < lb[i]
            xp = copy(x); xp[i] += h
            J[:, i] = (fun(xp) .- f0) / h
        else
            xp = copy(x); xp[i] += h
            xm = copy(x); xm[i] -= h
            J[:, i] = (fun(xp) .- fun(xm)) / (2 * h)
        end
    end
    return J
end

"""
    _lsq_trf(fun, jac, x0, lb, ub; ftol, xtol, gtol, x_scale, max_nfev,
             tr_solver, tr_options, verbose)

Bounded nonlinear least squares, `min 0.5*||fun(x)||^2`,
`lb <= x <= ub` — a port of `scipy.optimize.least_squares(method='trf')`
(trf_bounds with a linear loss; robust losses are not supported).

`jac` is either a callable `x -> Matrix` (analytic) or `:fd` for 2-point
finite differences. `x_scale` is `:unit` (scipy's default `None` for trf),
`:jac` or an explicit vector. `tr_solver` is `"exact"` (More-Sorensen on the
SVD of the augmented Jacobian) or `"lsmr"` (2-D subspace; the inner lsmr
iteration is replaced by the exact dense regularized solve, the fixed point
of lsmr in exact arithmetic). `tr_options` may carry `"regularize"` and
`"damp"`. Returns a `_EpicLsqSolution` with the scipy fields `x`, `success`
(`status > 0`), `cost` (`0.5*||fun(x)||^2`) and `nfev`.
"""
function _lsq_trf(fun::Function, jac, x0::AbstractVector{<:Real},
                  lb::Real, ub::Real;
                  ftol::Real, xtol::Real, gtol::Real,
                  x_scale=:unit,
                  max_nfev::Union{Nothing,Integer}=nothing,
                  tr_solver::AbstractString="exact",
                  tr_options::Union{Nothing,AbstractDict}=nothing,
                  verbose::Integer=0)
    n = length(x0)
    x0v = Float64.(collect(x0))
    lbv = fill(Float64(lb), n)
    ubv = fill(Float64(ub), n)

    # least_squares preprocessing
    if any(xi -> xi < lbv[1] || xi > ubv[1], x0v)
        throw(ArgumentError("Initial guess is outside of provided bounds"))
    end
    x = _epic_make_strictly_feasible(x0v, lbv, ubv, 1e-10)

    f = Vector{Float64}(vec(fun(x)))
    all(isfinite, f) || throw(ArgumentError("Residuals are not finite in the initial point."))
    m = length(f)
    nfev = 1
    J = jac === :fd ? _epic_fd_jac(fun, x, f, lbv, ubv) :
        Matrix{Float64}(jac(x))
    njev = 1
    size(J) == (m, n) || throw(ArgumentError("`jac` returned a matrix of the wrong shape"))
    cost = 0.5 * dot(f, f)
    g = J' * f

    jac_scale = x_scale === :jac
    if jac_scale
        scale, scale_inv = _epic_compute_jac_scale(J, nothing)
    elseif x_scale isa Symbol
        scale = ones(Float64, n)
        scale_inv = ones(Float64, n)
    else
        scale = Float64.(vec(x_scale))
        scale_inv = 1 ./ scale
    end

    v, dv = _epic_cl_scaling_vector(x, g, lbv, ubv)
    @inbounds for i in 1:n
        dv[i] != 0 && (v[i] *= scale_inv[i])
    end
    Delta = norm(x .* scale_inv ./ sqrt.(v))
    Delta == 0 && (Delta = 1.0)
    g_norm = _epic_norm_inf(g .* v)

    isnothing(max_nfev) || (max_nfev = Int(max_nfev))
    maxnfev = something(max_nfev, n * 100)

    regularize_tr = true
    damp_tr = 0.0
    if tr_solver == "lsmr"
        if tr_options !== nothing
            regularize_tr = something(get(tr_options, "regularize", true), true)
            damp_tr = Float64(something(get(tr_options, "damp", 0.0), 0.0))
        end
    end
    reg_term = 0.0

    alpha = 0.0  # "Levenberg-Marquardt" parameter
    status = nothing
    iteration = 0
    step_norm = 0.0
    actual_reduction = 0.0

    if verbose == 2
        @printf("     Iteration       Total nfev         Cost       Cost reduction    Step norm      Optimality   \n")
    end

    while true
        v, dv = _epic_cl_scaling_vector(x, g, lbv, ubv)
        g_norm = _epic_norm_inf(g .* v)
        if g_norm < gtol
            status = 1
        end
        if verbose == 2
            @printf("%15d%15d%15.4e%15.2e%15.2e%15.2e\n",
                    iteration, nfev, cost, actual_reduction, step_norm, g_norm)
        end
        (status !== nothing || nfev >= maxnfev) && break

        # Recompute the "hat"-space quantities: first `x_scale`, then the
        # Coleman-Li scaling in the new variables (mirrors scipy exactly).
        @inbounds for i in 1:n
            dv[i] != 0 && (v[i] *= scale_inv[i])
        end
        d = sqrt.(v) .* scale
        diag_h = g .* dv .* scale   # C = diag(g * scale) Jv
        g_h = d .* g
        theta = max(0.995, 1 - g_norm)

        J_h = J .* permutedims(d)
        if tr_solver == "exact"
            J_aug = vcat(J_h, Diagonal(sqrt.(diag_h)))
            f_aug = vcat(f, zeros(Float64, n))
            U, s_svd, Vt = svd(J_aug)
            uf = U' * f_aug
            Vmat = Matrix(Vt')
            svec = Vector{Float64}(s_svd)
        else
            if regularize_tr
                aq, bq = _epic_build_quadratic_1d(J_h, g_h, .-g_h, diag_h, nothing)[1:2]
                to_tr = Delta / norm(g_h)
                ag_value = _epic_minimize_quadratic_1d(aq, bq, 0.0, to_tr)[2]
                reg_term = -ag_value / Delta^2
            end
            w2 = diag_h .+ reg_term
            # min ||J_h p - f||^2 + ||diag(sqrt(w2)) p||^2 + damp^2 ||p||^2
            G = Symmetric(J_h' * J_h + Diagonal(w2 .+ damp_tr^2), :U)
            gn_h = try
                G \ (J_h' * f)
            catch
                Matrix(G) \ (J_h' * f)
            end
            S = qr([g_h gn_h], Val(false))
            Q = Matrix{Float64}(S.Q)[:, 1:2]   # economic Q, 2 columns
            JS = J_h * Q
            B_S = JS' * JS + Q' * (diag_h .* Q)
            g_S = Q' * g_h
        end

        actual_reduction = -1.0
        x_new = x
        f_new = f
        cost_new = cost
        step_h_norm = 0.0
        step = zeros(Float64, n)
        while actual_reduction <= 0 && nfev < maxnfev
            if tr_solver == "exact"
                p_h, alpha, _ = _epic_solve_lsq_trust_region(n, m, uf, svec,
                                                             Vmat, Delta, alpha)
            else
                p_S = _epic_solve_trust_region_2d(B_S, g_S, Delta)
                p_h = Q * p_S
            end
            p = d .* p_h   # trust-region solution in the original space
            step, step_h, predicted_reduction = _epic_select_step(
                x, J_h, diag_h, g_h, p, copy(p_h), d, Delta, lbv, ubv, theta)

            x_new = _epic_make_strictly_feasible(x .+ step, lbv, ubv, 0)
            f_new = Vector{Float64}(vec(fun(x_new)))
            nfev += 1
            step_h_norm = norm(step_h)

            if !all(isfinite, f_new)
                Delta = 0.25 * step_h_norm
                continue
            end

            # usual trust-region step-quality estimation
            cost_new = 0.5 * dot(f_new, f_new)
            actual_reduction = cost - cost_new
            Delta_new, ratio = _epic_update_tr_radius(
                Delta, actual_reduction, predicted_reduction,
                step_h_norm, step_h_norm > 0.95 * Delta)

            step_norm = norm(step)
            tstat = _epic_check_termination(actual_reduction, cost, step_norm,
                                            norm(x), ratio, ftol, xtol)
            if tstat !== nothing
                status = tstat
                break
            end

            alpha *= Delta / Delta_new
            Delta = Delta_new
        end

        if actual_reduction > 0
            x = x_new
            f = f_new
            cost = cost_new
            J = jac === :fd ? _epic_fd_jac(fun, x, f, lbv, ubv) :
                Matrix{Float64}(jac(x))
            njev += 1
            g = J' * f
            if jac_scale
                scale, scale_inv = _epic_compute_jac_scale(J, scale_inv)
            end
        else
            step_norm = 0.0
            actual_reduction = 0.0
        end

        iteration += 1
    end

    status === nothing && (status = 0)
    success = status > 0
    if verbose >= 1
        @printf("Method 'trf' terminated, status %d, %d function evaluations, final cost %.4e\n",
                status, nfev, cost)
    end
    return _EpicLsqSolution(x, success, cost, nfev, status, njev, g_norm)
end

"""
    _calc_epic_ch(P, H, target_sigmas; X0, V, LSQpar, homogeneous_step,
                  beta_shift_k, beta_distance, EPIC_bool, regularize)

Solve the EPIC condition for the prior variances.

Port of `calc_EPIC_Ch` from EPIC_LS. Returns the `_EpicLsqSolution` (the
scipy-`OptimizeResult` stand-in) whose `x` holds the betas (natural logarithms
of the reciprocal prior variances). `LSQpar` may carry solver tuning:
`TolX1/TolFun1/TolG1` (homogeneous step), `TolX2/TolFun2/TolG2`
(heterogeneous step) and `method/loss/verbose` (defaults `trf`/`linear`/`0`).
`tr_solver` defaults to `"exact"` (the upstream EPIC_LS port uses `"lsmr"`,
which stalls on small unfolding problems and exhausts `max_nfev`).
"""
function _calc_epic_ch(P::Matrix{Float64}, H::Matrix{Float64},
                       target_sigmas::Vector{Float64};
                       X0::Union{Nothing,AbstractVector}=nothing,
                       V::Union{Nothing,AbstractMatrix}=nothing,
                       LSQpar::Union{Nothing,AbstractDict}=nothing,
                       homogeneous_step::Bool=true,
                       beta_shift_k::Real=0,
                       beta_distance::Real=2,
                       EPIC_bool::Union{Nothing,AbstractVector}=nothing,
                       regularize::Union{Nothing,AbstractDict}=nothing)
    Nh, Nm = size(H)
    params = Dict{String,Any}()
    if LSQpar !== nothing
        for (k, v) in LSQpar
            params[String(k)] = v
        end
    end

    _setdefault(key, val) = haskey(params, key) || (params[key] = val)
    if Nh > Nm
        _setdefault("TolX1", 1e-6)
        _setdefault("TolFun1", 1e-6)
        _setdefault("TolG1", 1e-6)
        _setdefault("TolX2", 1e-6)
        _setdefault("TolFun2", 1e-6)
        _setdefault("TolG2", 1e-8)
        _setdefault("damp_trf", 1e-3)
    else
        _setdefault("TolX1", 1e-6)
        _setdefault("TolFun1", 1e-6)
        _setdefault("TolG1", 1e-6)
        _setdefault("TolX2", 1e-8)
        _setdefault("TolFun2", 1e-8)
        _setdefault("TolG2", 1e-10)
        _setdefault("damp_trf", 1e-9)
    end

    method = String(get(params, "method", "trf"))
    method == "trf" || throw(ArgumentError(
        "Unsupported LSQpar method: $method (the native port implements 'trf' only)"))
    loss = String(get(params, "loss", "linear"))
    loss == "linear" || @warn "LSQpar loss '$loss' is not implemented by the native TRF port; falling back to 'linear'." maxlog = 1
    verbose = Int(get(params, "verbose", 0))

    bounds = _compute_bounds(beta_shift_k, beta_distance)
    blo, bhi = bounds

    X0v = if X0 === nothing
        n_unknowns = V !== nothing ? size(V, 2) : Nh
        fill((blo + bhi) / 2, Int(n_unknowns))
    else
        Vector{Float64}(vec(X0))
    end
    tgt = Vector{Float64}(vec(target_sigmas))
    target_var = tgt .^ 2

    # np.finfo(float).precision == 16
    sigma_weight_default = exp(16.0 / 4)

    function _calc_F(X::AbstractVector{Float64})
        beta = V === nothing ? X : Vector{Float64}(V * X)
        inv_ch = exp.(beta)
        invA = inv(P + H' * (inv_ch .* H))
        F = Vector{Float64}(diag(invA))
        if EPIC_bool !== nothing
            F = F[EPIC_bool]
        end
        F = (F .- target_var) ./ target_var
        if regularize !== nothing
            sigma_weight = Float64(get(regularize, "sigma_weight", sigma_weight_default))
            F = vcat(F, exp.(beta ./ 2) ./ sigma_weight)
        end
        return F
    end

    function _calc_JF(X::AbstractVector{Float64})
        beta = V === nothing ? X : Vector{Float64}(V * X)
        E = exp.(beta)
        invA = inv(P + H' * (E .* H))
        B = H * invA
        JF = Matrix{Float64}(-(E .* (B .* B))')
        if V !== nothing
            JF = JF * Matrix{Float64}(V)
        end
        if EPIC_bool !== nothing
            JF = JF[EPIC_bool, :]
        end
        JF = (1.0 ./ target_var) .* JF
        if regularize !== nothing
            sigma_weight = Float64(get(regularize, "sigma_weight", sigma_weight_default))
            JF2 = Matrix{Float64}(Diagonal(0.5 .* exp.(beta ./ 2) ./ sigma_weight))
            JF = vcat(JF, JF2)
        end
        return JF
    end

    if homogeneous_step
        fun_const = x -> _calc_F(x .+ X0v)
        sol0 = _lsq_trf(fun_const, :fd, [0.0], blo, bhi;
                        ftol=Float64(params["TolFun1"]),
                        xtol=Float64(params["TolX1"]),
                        gtol=Float64(params["TolG1"]),
                        verbose=verbose)
        Xnext = sol0.x .+ X0v
    else
        Xnext = X0v
    end

    tr_solver = String(get(params, "tr_solver", "exact"))
    if tr_solver == "lsmr"
        if Nh > Nm
            tr_options = Dict{String,Any}("regularize" => true, "damp" => 1e-3)
        else
            tr_options = Dict{String,Any}("regularize" => false,
                                          "damp" => Float64(get(params, "damp_trf", 1e-9)))
        end
    else
        tr_options = nothing
    end

    sol = _lsq_trf(_calc_F, _calc_JF, Xnext, blo, bhi;
                   ftol=Float64(params["TolFun2"]),
                   xtol=Float64(params["TolX2"]),
                   gtol=Float64(params["TolG2"]),
                   x_scale=:jac,
                   tr_solver=tr_solver,
                   tr_options=tr_options,
                   verbose=verbose)

    return sol
end

"""
    _final_solve(A, b, Wx, H, ho, Wh, non_neg)

Solve the augmented least squares problem

    min ||Wx (A m - b)||^2 + ||Wh (H m - ho)||^2

by stacking the misfit and regularization blocks into an equivalent simple
least squares problem. Applies non-negativity constraints with the Lawson-
Hanson NNLS (`solve_nnls`, the scipy.optimize.nnls semantics of `_nnls.jl`)
when `non_neg` is true. Port of `LeastSquaresRegNonNeg` from EPIC_LS.
"""
function _final_solve(A::Matrix{Float64}, b::Vector{Float64},
                      Wx::Matrix{Float64}, H::Matrix{Float64},
                      ho::Vector{Float64},
                      Wh::Union{Matrix{Float64},Diagonal{Float64}},
                      non_neg::Bool)
    WxG = Wx * A
    Wxd = Wx * b
    WhH = Wh * H
    Whho = Wh * ho

    F = vcat(WxG, WhH)
    D = vcat(Wxd, Whho)

    if non_neg
        x = solve_nnls(F, D)
    else
        x = _epic_lstsq(F, D)
    end

    return max.(Vector{Float64}(x), 0.0)
end

"""
    _epic_weights(A, b; target_sigmas, ...) -> (Wx, H, Wh, meta)

Compute the EPIC regularization weights and optimization metadata.

Returns `(Wx, H, Wh, meta)` where `Wx` and `Wh` are the misfit and prior
weight matrices, `H` the regularization operator and `meta` the EPIC
optimization status together with the resolved target sigmas.
"""
function _epic_weights(A::Matrix{Float64}, b::Vector{Float64};
                       target_sigmas::Union{Nothing,AbstractVector}=nothing,
                       regularization_order::Integer=1,
                       noise_var::Union{Nothing,Real}=nothing,
                       homogeneous_step::Bool=true,
                       regularize::Union{Nothing,AbstractDict}=nothing,
                       beta_shift_k::Real=0,
                       beta_distance::Real=2,
                       EPIC_bool::Union{Nothing,AbstractVector}=nothing,
                       V::Union{Nothing,AbstractMatrix}=nothing,
                       LSQpar::Union{Nothing,AbstractDict}=nothing,
                       sigma_frac::Real=0.1)
    n = size(A, 2)

    P, Wx = _build_precision(A, noise_var)
    H = _build_regularization_matrix(n, regularization_order)

    Vl = V === nothing ? nothing : Matrix{Float64}(V)

    tgt = if target_sigmas === nothing
        _default_target_sigmas(A, b, n, sigma_frac)
    else
        Vector{Float64}(vec(target_sigmas))
    end

    mask = if EPIC_bool !== nothing
        vec(EPIC_bool) .!= 0
    else
        nothing
    end

    if mask !== nothing
        if length(mask) != n
            throw(ArgumentError(
                "EPIC_bool length ($(length(mask))) must match " *
                "number of energy bins ($n)"))
        end
        target_epic = length(tgt) == n ? tgt[mask] : tgt
        expected = Int(sum(mask))
    else
        target_epic = tgt
        expected = n
    end

    if length(target_epic) != expected
        throw(ArgumentError(
            "target_sigmas length ($(length(target_epic))) must match " *
            "the number of parameters subject to the EPIC ($expected)"))
    end
    if !all(isfinite, target_epic) || any(target_epic .<= 0)
        throw(ArgumentError("target_sigmas must be finite and strictly positive"))
    end

    sol = _calc_epic_ch(P, H, Vector{Float64}(target_epic);
                        X0=nothing,
                        V=Vl,
                        LSQpar=LSQpar,
                        homogeneous_step=homogeneous_step,
                        beta_shift_k=beta_shift_k,
                        beta_distance=beta_distance,
                        EPIC_bool=mask,
                        regularize=regularize)

    beta = Vector{Float64}(sol.x)
    if Vl !== nothing
        beta = Vl * beta
    end
    Wh = Diagonal(exp.(beta ./ 2))

    meta = Dict{String,Any}(
        "epic_converged" => Bool(sol.success),
        "epic_cost" => Float64(sol.cost),
        "epic_nfev" => Int(sol.nfev),
        "beta_min" => Float64(minimum(beta)),
        "beta_max" => Float64(maximum(beta)),
        "target_sigmas" => Vector{Float64}(target_epic),
    )

    return Wx, H, Wh, meta
end

"""
    solve_epic(A, b, x0=nothing; target_sigmas=nothing, sigma_frac=0.1,
               regularization_order=1, non_neg=true, noise_var=nothing,
               homogeneous_step=true, regularize=nothing, beta_shift_k=0,
               beta_distance=2, EPIC_bool=nothing, V=nothing, LSQpar=nothing)

Solve the unfolding problem with EPIC Tikhonov regularization.

Selects the prior variances of the regularization operator H such that the
a posteriori variances of the model parameters equal the squared target
sigmas, then solves the weighted least squares problem.

`x0` is not used (kept for API compatibility with the other solvers).
`target_sigmas` defaults to `sigma_frac` times the magnitude of the naive
least-squares solution and must be strictly positive. `regularize`, when
given (it can be empty), damps the EPIC weights towards a minimum-norm
solution and may carry `sigma_weight`. `beta_shift_k`/`beta_distance` shift
and widen the beta search interval, `EPIC_bool` selects which parameters are
subject to the EPIC condition, `V` maps the searched betas to the
regularization rows (`beta = V * y`) and `LSQpar` tunes the TRF solver
(`TolX1/TolFun1/TolG1`, `TolX2/TolFun2/TolG2`, `method`, `loss`, `verbose`,
`tr_solver`, `damp_trf`). Returns an `UnfoldResult` whose `extra` carries
`regularization_order`, `non_neg` and the EPIC meta
(`epic_converged/epic_cost/epic_nfev/beta_min/beta_max/target_sigmas`).
"""
function solve_epic(A::AbstractMatrix, b::AbstractVector,
                    x0::Union{Nothing,AbstractVector}=nothing;
                    target_sigmas::Union{Nothing,AbstractVector}=nothing,
                    sigma_frac::Real=0.1,
                    regularization_order::Integer=1,
                    non_neg::Bool=true,
                    noise_var::Union{Nothing,Real}=nothing,
                    homogeneous_step::Bool=true,
                    regularize::Union{Nothing,AbstractDict}=nothing,
                    beta_shift_k::Real=0,
                    beta_distance::Real=2,
                    EPIC_bool::Union{Nothing,AbstractVector}=nothing,
                    V::Union{Nothing,AbstractMatrix}=nothing,
                    LSQpar::Union{Nothing,AbstractDict}=nothing)
    Af = Matrix{Float64}(A)
    bf = Vector{Float64}(vec(b))
    m, _ = size(Af)
    length(bf) == m || throw(ArgumentError(
        "b length ($(length(bf))) must match the number of A rows ($m)"))
    regularization_order in (0, 1, 2) || throw(ArgumentError(
        "Unsupported regularization_order: $regularization_order. " *
        "Use 0 (identity), 1 or 2."))

    Wx, H, Wh, epic_meta = _epic_weights(Af, bf;
                                         target_sigmas=target_sigmas,
                                         regularization_order=regularization_order,
                                         noise_var=noise_var,
                                         homogeneous_step=homogeneous_step,
                                         regularize=regularize,
                                         beta_shift_k=beta_shift_k,
                                         beta_distance=beta_distance,
                                         EPIC_bool=EPIC_bool,
                                         V=V,
                                         LSQpar=LSQpar,
                                         sigma_frac=sigma_frac)

    if !epic_meta["epic_converged"]
        @warn "EPIC nonlinear optimization did not converge fully " *
              "(cost=" * @sprintf("%.3e", epic_meta["epic_cost"]) *
              "); returning the best-effort regularized solution."
    end

    ho = zeros(Float64, size(H, 1))
    x = _final_solve(Af, bf, Wx, H, ho, Wh, non_neg)

    extra = Dict{String,Any}(
        "regularization_order" => Int(regularization_order),
        "non_neg" => Bool(non_neg),
    )
    for (k, v) in epic_meta
        extra[k] = v
    end

    return UnfoldResult(x, 1, Bool(epic_meta["epic_converged"]),
                        norm(bf .- Af * x), extra)
end
