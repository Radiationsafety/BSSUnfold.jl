"""
Projected subgradient descent for nonsmooth neutron-spectrum unfolding.

Solves: min_{x >= 0} 1/2 ||A x - b||^2 + l1_penalty ||x||_1 + tv_penalty ||D x||_1,
with D the first-order difference operator, (D x)_i = x_{i+1} - x_i.
Subgradient: g = A^T (A x - b) + l1 * sign(x) + tv * D^T sign(D x).
Step policies: 'polyak' (t = (f(x_k) - f*)/||g||^2, f* = best-so-far shrunk
by polyak_margin), 'diminishing' (t = t0/(1 + decay*k)), 'fixed' (t = t0),
with scale-aware base t0 = step_size * (||b||/||A||_2) / ||g(x0)||.
The BEST iterate by objective value is returned (standard subgradient practice).
Reference: MIPT optimization course, lecture 8 / homework 12 (subgradient and
adaptive methods for nonsmooth optimization; cf. Shor, Minimization Methods).
"""
function solve_subgradient(A::AbstractMatrix{T}, b::AbstractVector{T}, x0::AbstractVector{T};
                           max_iterations::Int=3000,
                           tolerance::T=T(1e-8),
                           l1_penalty::T=T(0.0),
                           tv_penalty::T=T(0.0),
                           step_policy::String="diminishing",
                           step_size::T=T(1.0),
                           decay::T=T(1.0),
                           polyak_margin::T=T(0.05)) where T<:AbstractFloat
    _STEPS = ("polyak", "diminishing", "fixed")
    step_policy in _STEPS ||
        throw(ArgumentError("step_policy must be one of $(_STEPS)"))

    validate_system(A, b; x0=x0, max_iterations=max_iterations, tolerance=tolerance)

    n = size(A, 2)
    l1 = max(T(l1_penalty), T(0))
    tv = max(T(tv_penalty), T(0))
    use_tv = tv > zero(T)

    # Scale-aware base step: t0 * ||g(x0)|| ~ ||x||_ref (||b||/||A||_2).
    x_proj0 = max.(x0, T(0))
    x_scale = norm(b) / max(opnorm(A), T(1e-30))
    g0 = _subgradient_subgrad(A, b, x_proj0, l1, tv)
    g0_norm = norm(g0)
    t0 = T(step_size) * max(x_scale, T(1e-300)) / max(g0_norm, T(1e-300))

    x = x_proj0
    best_x = copy(x)
    best_f = _subgradient_objective(A, b, x, l1, tv)
    margin = max(T(1) - polyak_margin, T(0))
    f_star_est = best_f * margin

    converged = false
    iters = 0
    for k in 1:max_iterations
        g = _subgradient_subgrad(A, b, x, l1, tv)
        g_norm_sq = dot(g, g)

        # k here is 1-based; Python's 0-based index is kk = k - 1
        if step_policy == "polyak"
            f_cur = _subgradient_objective(A, b, x, l1, tv)
            if f_cur < best_f
                best_f = f_cur
                best_x = copy(x)
                f_star_est = best_f * margin
            end
            t = g_norm_sq > 0 ? max((f_cur - f_star_est) / g_norm_sq, T(0)) : T(0)
        elseif step_policy == "diminishing"
            t = t0 / (T(1) + decay * T(k - 1))
        else
            t = t0
        end

        x_new = max.(x .- t .* g, T(0))  # projection onto the nonnegative orthant
        rel_change = norm(x_new .- x) / max(norm(x), T(1e-30))
        x = x_new
        iters = k

        f_new = _subgradient_objective(A, b, x, l1, tv)
        if f_new < best_f
            best_f = f_new
            best_x = copy(x)
        end

        if rel_change < tolerance
            converged = true
            break
        end
    end

    residual = b .- A * best_x
    return UnfoldResult(best_x, iters, converged, norm(residual))
end


function _subgradient_objective(A::AbstractMatrix{T}, b::AbstractVector{T}, z::AbstractVector{T},
                                l1::T, tv::T) where T<:AbstractFloat
    r = A * z .- b
    val = T(0.5) * dot(r, r)
    l1 > zero(T) && (val += l1 * sum(abs, z))
    tv > zero(T) && (val += tv * _subgradient_tv_l1(z))
    return T(val)
end


function _subgradient_subgrad(A::AbstractMatrix{T}, b::AbstractVector{T}, z::AbstractVector{T},
                              l1::T, tv::T) where T<:AbstractFloat
    g = A' * (A * z .- b)
    l1 > zero(T) && (g .+= l1 .* sign.(z))
    tv > zero(T) && _subgradient_add_tv!(g, z, tv)
    return g
end


# ||D z||_1 with (D z)_i = z_{i+1} - z_i
function _subgradient_tv_l1(z::AbstractVector{T}) where T<:AbstractFloat
    n = length(z)
    s = zero(T)
    @inbounds for j in 1:(n - 1)
        s += abs(z[j + 1] - z[j])
    end
    return s
end


# g += tv * D' * sign(D z), computed without forming D.
# (D' s)_1 = -s_1; (D' s)_j = s_{j-1} - s_j (2 <= j <= n-1); (D' s)_n = s_{n-1}
function _subgradient_add_tv!(g::AbstractVector{T}, z::AbstractVector{T}, tv::T) where T<:AbstractFloat
    n = length(z)
    n < 2 && return g
    @inbounds begin
        prev_s = sign(z[2] - z[1])          # (D z)_1
        g[1] -= tv * prev_s
        for j in 2:(n - 1)
            cur_s = sign(z[j + 1] - z[j])
            g[j] += tv * (prev_s - cur_s)
            prev_s = cur_s
        end
        g[n] += tv * prev_s                 # (D z)_{n-1}
    end
    return g
end
