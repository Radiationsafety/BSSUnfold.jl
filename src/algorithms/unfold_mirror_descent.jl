"""
Mirror descent — Bregman-proximal first-order method on the simplex /
positive orthant. Port of `bssunfold/core/unfold_mirror_descent.py`.

Mirror maps (`mirror_map`):
* `:entropy` — multiplicative `x · exp(−η g)`, renormalised to
  `Σx = total_fluence`. Generalises MLEM / GRAVEL. Requires F > 0.
* `:log` — log-barrier prox `1 / (1/x + η g)`, positive orthant.
* `:l2` — plain projected gradient `max(x − η g, 0)`.
* `:pnorm` — `max(x^{p−1} − η g, 0)^{1/(p−1)}`, `p > 1`.

When `step_size === nothing` we run a per-iteration golden-section line
search along the mirror trajectory (`_md_golden_section`); the entropic
geometry makes fixed η very slow, matching the Python behaviour.
"""

const _MIRROR_MAPS = (:entropy, :log, :l2, :pnorm)

function _md_mirror_next(x::AbstractVector{T}, g::AbstractVector{T},
                         eta::T, mm::Symbol, F::Union{Nothing,T}, p::T) where T<:AbstractFloat
    if mm === :entropy
        log_step = .- eta .* g
        shift = maximum(log_step)
        xn = x .* exp.(log_step .- shift)
        s = sum(xn)
        (isfinite(s) && s > 0) || return copy(x)
        return F .* xn ./ s
    elseif mm === :log
        inv = 1.0 ./ max.(x, T(1e-300)) .+ eta .* g
        all(isfinite, inv) && all(>(0), inv) || return copy(x)
        return 1.0 ./ inv
    elseif mm === :l2
        return max.(x .- eta .* g, T(0))
    elseif mm === :pnorm
        pp = p - one(T)
        xp = max.(x, T(1e-300)) .^ pp .- eta .* g
        return max.(xp, T(0)) .^ (1.0 / pp)
    else
        throw(ArgumentError("unknown mirror map: $mm"))
    end
end

function _md_eta_max(g::AbstractVector{T}, L::T, mm::Symbol, x::AbstractVector{T}) where T<:AbstractFloat
    g_max = maximum(abs, g)
    if mm === :entropy || mm === :log
        hi = T(4) / max(g_max, T(1e-300))
        if mm === :log
            neg_idx = findall(<(0), g)
            if !isempty(neg_idx)
                inv_x = 1.0 ./ max.(x[neg_idx], T(1e-300))
                limit = minimum(inv_x ./ abs.(g[neg_idx]))
                hi = min(hi, T(0.25) * limit)
            end
        end
        return hi
    end
    return L > 0 ? T(2) / L : T(1)
end

function _md_golden_section(func, lo::T, hi::T;
                            tolerance::T=T(1e-8), max_iterations::Int=200) where T<:AbstractFloat
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
    t = T(0.5) * (a + b)
    return t, T(func(t))
end

function solve_mirror_descent(A::AbstractMatrix{T}, b::AbstractVector{T},
                              x0::AbstractVector{T};
                              max_iterations::Integer=1000,
                              tolerance::T=T(1e-8),
                              mirror_map::Union{Symbol,String}=:entropy,
                              step_size::Union{Real,Nothing}=nothing,
                              total_fluence::Union{Real,Nothing}=nothing,
                              regularization::Real=T(0.0),
                              p::Real=T(3.0)) where T<:AbstractFloat
    m, n = size(A)
    mm = Symbol(mirror_map)
    mm in _MIRROR_MAPS || throw(ArgumentError("mirror_map must be one of $_MIRROR_MAPS"))
    reg = T(regularization)
    pv = T(p)
    mm === :pnorm && pv ≤ 1 && throw(ArgumentError("p-norm mirror map requires p > 1"))

    F::Union{Nothing,T} = nothing
    if mm === :entropy
        Fv = total_fluence === nothing ? sum(x0) : T(total_fluence)
        Fv > 0 || throw(ArgumentError("entropy mirror map requires positive total fluence"))
        F = T(Fv)
        x = max.(Vector{T}(x0), T(1e-300))
        x = F .* x ./ sum(x)
    else
        max_A = max(maximum(A), T(1e-30))
        floor = (max(maximum(b), T(1e-12)) / max_A) / T(n)
        x = max.(Vector{T}(x0), T(floor))
    end

    L = opnorm(A)^2 + max(reg, T(0))
    obj(z) = begin
        r = A * z .- b
        v = T(0.5) * sum(r .* r)
        reg > 0 && (v += T(0.5) * reg * sum(z .* z))
        v
    end

    converged = false
    iters = 0
    for k in 1:max_iterations
        g = A' * (A * x .- b)
        reg > 0 && (g = g .+ reg .* x)
        eta_k = if step_size === nothing
            hi = _md_eta_max(g, T(L), mm, x)
            hi > 0 || break
            phi(t) = obj(_md_mirror_next(x, g, T(t), mm, F, pv))
            tt, _ = _md_golden_section(phi, T(0), hi; tolerance=hi * T(1e-4))
            tt
        else
            T(step_size)
        end
        x_new = _md_mirror_next(x, g, eta_k, mm, F, pv)
        rel = norm(x_new - x) / max(norm(x), T(1e-300))
        x = x_new
        iters = k
        rel < tolerance && (converged = true; break)
    end
    UnfoldResult(x, iters, converged, norm(b .- A * x))
end
