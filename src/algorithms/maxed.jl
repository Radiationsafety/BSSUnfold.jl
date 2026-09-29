"""
MAXED — Maximum Entropy Deconvolution (Reginatto & Goldhagen 1999).

Faithful port of `solve_maxed` from bssunfold 0.28.0 (`unfold_maxed.py`).
Minimizes in log-space y = ln(x):

    f(y) = sum_i [e^{y_i}(y_i - ln x0_i) - e^{y_i} + x0_i]
         + 1/2 sum_j (b_j - (A e^y)_j)^2 / sigma_j^2,   sigma = sigma_factor * b

with gradient g_i = x_i [ln(x_i/x0_i) - (A^T (b - A x)/sigma^2)_i],
using L-BFGS (two-loop recursion) and a safeguarded line search on the
convex restriction phi(t) = f(y + t p), stopping when max|g| <= tolerance
(scipy L-BFGS-B gtol).
"""
function solve_maxed(A::AbstractMatrix{T}, b::AbstractVector{T}, x0::AbstractVector{T};
                     sigma_factor::Real=T(0.1),
                     max_iterations::Integer=5000,
                     tolerance::Real=T(1e-6)) where T<:AbstractFloat
    m, n = size(A)
    length(b) == m || throw(DimensionMismatch("length(b) != size(A,1)"))
    length(x0) == n || throw(DimensionMismatch("length(x0) != size(A,2)"))
    gtol = T(tolerance)
    sig = T(sigma_factor)

    tiny = T(1e-300) > zero(T) ? T(1e-300) : floatmin(T)
    b_safe = max.(Vector{T}(b), tiny)
    sigma2_inv = one(T) ./ ((sig .* b_safe) .^ 2)
    phi_0 = max.(Vector{T}(x0), tiny)
    log_phi_0 = log.(phi_0)

    # Objective f(y) and gradient g = df/dy in log space.
    function _f_and_g!(g, y)
        x = exp.(y)
        resid = b .- A * x
        w = resid .* sigma2_inv
        f_ent = sum(x .* (y .- log_phi_0) .- x .+ phi_0)
        f_chi = T(0.5) * dot(resid, w)
        g .= x .* (y .- log_phi_0 .- A' * w)
        return f_ent + f_chi
    end

    # Minimize convex phi(t) = f(y + t p), phi'(0) = gp0 < 0.
    # Returns (t, f(t), g(t)) with phi'(t) >= 0 (at/after the line minimum),
    # or t = 0 on failure.
    function _line_search(y, p, gp0)
        lo_t = zero(T); lo_d = gp0
        lo_f = typemax(T); lo_g = similar(g)
        hi_t = typemax(T) / 8
        hi_f = typemax(T); hi_g = nothing
        t = one(T)
        bracketed = false
        for _ in 1:60
            gt = similar(g)
            ft = _f_and_g!(gt, y .+ t .* p)
            dt = dot(gt, p)
            if !isfinite(ft) || !isfinite(dt) || dt >= zero(T)
                hi_t, hi_f = t, ft
                bracketed = true
                isfinite(ft) && isfinite(dt) && (hi_g = gt)
                break
            end
            lo_t, lo_d, lo_f, lo_g = t, dt, ft, gt
            t *= 2
        end
        bracketed || return zero(T), typemax(T), nothing

        for _ in 1:80
            hi_t - lo_t <= T(1e-12) * max(hi_t, one(T)) && break
            mid = if hi_g !== nothing
                hi_d = dot(hi_g, p)
                wdt = hi_t - lo_t
                cand = lo_t - lo_d * wdt / (hi_d - lo_d)
                clamp(cand, lo_t + T(0.1) * wdt, hi_t - T(0.05) * wdt)
            else
                T(0.5) * (lo_t + hi_t)
            end
            gt = similar(g)
            ft = _f_and_g!(gt, y .+ mid .* p)
            dt = dot(gt, p)
            if !isfinite(ft) || !isfinite(dt)
                hi_t, hi_f, hi_g = mid, typemax(T), nothing
                continue
            end
            if dt >= zero(T)
                hi_t, hi_f, hi_g = mid, ft, gt
            else
                lo_t, lo_d, lo_f, lo_g = mid, dt, ft, gt
            end
        end
        # No usable right point: accept last finite strictly-decreasing point
        # (still an f-decrease and preserves the curvature condition).
        hi_g === nothing && return lo_t, lo_f, copy(lo_g)
        return hi_t, hi_f, hi_g
    end

    y = copy(log_phi_0)
    g = similar(y)
    f = _f_and_g!(g, y)

    iterations = 0
    converged = norm(g, Inf) <= gtol

    mem = 10
    S = Vector{T}[]
    Ys = Vector{T}[]
    Rho = T[]
    alphas_buf = fill(zero(T), mem)

    while !converged && iterations < max_iterations
        # two-loop recursion: H ≈ (sᵀy / yᵀy) scaled identity + low-rank updates
        q = copy(g)
        for i in length(S):-1:1
            a = Rho[i] * dot(S[i], q)
            alphas_buf[i] = a
            q .-= a .* Ys[i]
        end
        gamma = if isempty(S)
            one(T)
        else
            dot(S[end], Ys[end]) / max(dot(Ys[end], Ys[end]), floatmin(T))
        end
        r = q .* gamma
        for i in eachindex(S)
            be = Rho[i] * dot(Ys[i], r)
            r .+= (alphas_buf[i] - be) .* S[i]
        end
        p = .- r
        gp0 = dot(g, p)
        if !(gp0 < zero(T))
            empty!(S); empty!(Ys); empty!(Rho)
            p = .- g
            gp0 = dot(g, p)
            gp0 < zero(T) || break
        end

        t, f_new, g_new = _line_search(y, p, gp0)
        (t > zero(T) && isfinite(f_new) && g_new !== nothing) || break

        s_vec = t .* p
        y_vec = g_new .- g
        sy = dot(s_vec, y_vec)
        if sy > T(1e-16) * norm(s_vec) * norm(y_vec)
            push!(S, s_vec); push!(Ys, y_vec); push!(Rho, one(T) / sy)
            if length(S) > mem
                popfirst!(S); popfirst!(Ys); popfirst!(Rho)
            end
        end

        y .+= s_vec
        g .= g_new
        f = f_new
        iterations += 1
        converged = norm(g, Inf) <= gtol
    end

    x_opt = exp.(y)
    return UnfoldResult(x_opt, iterations, converged, norm(b .- A * x_opt))
end
