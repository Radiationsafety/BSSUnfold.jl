"""
Frank--Wolfe (conditional gradient) unfolding.

Solves: min_x (1/2)||Ax - b||²   s.t.  x >= 0, sum(x) = total_fluence

The feasible set is the simplex of level F = total_fluence; the linear
minimization oracle (LMO) puts all mass on the single bin with the smallest
gradient component, so every iterate is a convex combination of vertex
spectra and the total fluence is preserved exactly at every iteration.
Convergence is certified by the Frank--Wolfe duality gap <grad, x - s>.
Optional Wolfe away-steps reduce zig-zagging near the optimum.

References: Frank & Wolfe (1956); Levitin & Polyak variant; Wolfe away step /
Lacoste-Julien & Jaggi (2015); bssunfold core/unfold_frank_wolfe.py.
"""

"""
    _frank_wolfe_lmo_simplex(grad, total_fluence)

LMO over the simplex: all mass on argmin(grad). Port of `_lmo_simplex`
in unfold_frank_wolfe.py.
"""
function _frank_wolfe_lmo_simplex(grad::AbstractVector{T}, total::T) where T<:AbstractFloat
    s = zeros(T, length(grad))
    s[argmin(grad)] = total
    return s
end


"""
    _frank_wolfe_estimate_total_fluence(A, b)

Data-driven fluence estimate from an unconstrained NNLS fit: total = sum(x_nnls)
with fallback mean(b)/max(mean(A),1e-30)*n. Port of `estimate_total_fluence`
in _matrix_utils.py (ln_steps=None branch).
"""
function _frank_wolfe_estimate_total_fluence(A::AbstractMatrix{T}, b::AbstractVector{T}) where T<:AbstractFloat
    m, n = size(A)
    total = T(0)
    try
        x_nnls, _ = lawson_hanson(A, b; max_iterations=10 * n)
        total = sum(x_nnls)
    catch
        total = T(0)
    end
    if isfinite(total) && total > 0
        return T(total)
    end
    mean_response = max(T(mean(A)), T(1e-30))
    return T(mean(b) / mean_response * n)
end


"""
    _frank_wolfe_golden_section(func, lo, hi; tolerance, max_iterations)

Golden-section minimization of a unimodal scalar function on [lo, hi],
returning t_opt. Port of `golden_section_minimize` in _line_search.py
(the 'backtracking' line_search branch of the FW solver).
"""
function _frank_wolfe_golden_section(func, lo::T, hi::T;
                                     tolerance::T=T(1e-8),
                                     max_iterations::Int=200) where T<:AbstractFloat
    isfinite(lo) && isfinite(hi) || throw(ArgumentError("golden_section requires finite bounds"))
    hi <= lo && return lo, T(func(lo))

    inv_phi = (sqrt(T(5)) - T(1)) / T(2)
    a, b = lo, hi
    c = b - inv_phi * (b - a)
    d = a + inv_phi * (b - a)
    fc, fd = T(func(c)), T(func(d))

    for _ in 1:max_iterations
        (b - a) <= tolerance && break
        if fc < fd
            b, d, fd = d, c, fc
            c = b - inv_phi * (b - a)
            fc = T(func(c))
        else
            a, c, fc = c, d, fd
            d = a + inv_phi * (b - a)
            fd = T(func(d))
        end
    end
    t_opt = T(0.5) * (a + b)
    return t_opt, T(func(t_opt))
end


function solve_frank_wolfe(A::AbstractMatrix{T}, b::AbstractVector{T}, x0::AbstractVector{T};
                           total_fluence::Union{Nothing,T}=nothing,
                           max_iterations::Int=1000,
                           tolerance::T=T(1e-8),
                           away_steps::Bool=true,
                           line_search::AbstractString="exact") where T<:AbstractFloat
    validate_system(A, b; x0=x0, max_iterations=max_iterations, tolerance=tolerance)

    F = total_fluence === nothing ? _frank_wolfe_estimate_total_fluence(A, b) : T(total_fluence)
    if !(F > 0)
        throw(ArgumentError("total_fluence must be positive"))
    end

    n = length(x0)
    x = max.(copy(x0), T(0))
    sx = sum(x)
    x = sx > 0 ? (F / sx) .* x : fill(F / T(n), n)

    ATb = A' * b
    G = A' * A  # n x n Gram matrix; n is small for BSS, precompute once

    converged = false
    iterations = 0

    for k in 1:max_iterations
        grad = G * x .- ATb

        # Frank-Wolfe (duality) gap: <grad, x - s> >= f(x) - f*
        s = _frank_wolfe_lmo_simplex(grad, F)
        fw_gap = T(dot(grad, x .- s))
        iterations = k
        if fw_gap <= tolerance * max(T(1), abs(T(dot(grad, x))))
            converged = true
            break
        end

        # Regular FW step: move from x towards the LMO vertex
        d = s .- x
        gamma_max = T(1)

        # Away-step candidate: drop mass from the supported vertex with the
        # largest gradient component, towards the renormalized remaining support.
        if away_steps
            mask = x .> 0
            if count(mask) > 1
                i_away = argmax([mask[j] ? grad[j] : typemin(T) for j in 1:n])
                x_rest = copy(x)
                x_rest[i_away] = T(0)
                sr = sum(x_rest)
                if sr > 0
                    x_rest = (F / sr) .* x_rest
                    away_gap = T(dot(grad, x .- x_rest))  # == -<grad, d_away>
                    if away_gap > fw_gap
                        d = x_rest .- x
                        gamma_max = T(1)
                    end
                end
            end
        end

        local gamma::T
        if line_search == "exact"
            g_d = T(dot(grad, d))
            gd = T(dot(d, G * d))
            gamma = gd <= 0 ? gamma_max : T(clamp(-g_d / gd, T(0), gamma_max))
        else
            function phi(t::T)
                r = A * (x .+ t .* d) .- b
                return T(0.5) * dot(r, r)
            end
            t_opt, _ = _frank_wolfe_golden_section(phi, T(0), gamma_max)
            gamma = t_opt
        end

        if gamma <= 1e-16
            # no descent possible along the chosen direction: the FW gap is
            # (numerically) the best achievable decrease -> converged
            converged = true
            break
        end
        x = max.(x .+ gamma .* d, T(0))
        sx = sum(x)
        x = sx > 0 ? (F / sx) .* x : fill(F / T(n), n)
    end

    residual = b .- A * x
    return UnfoldResult(x, iterations, converged, norm(residual))
end
