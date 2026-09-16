"""
SSR unfolding: Sign-Simplicity-Regression solver (sisireg port).

Faithful Julia port of `bssunfold/core/unfold_ssr.py`, itself a port of
the R package `sisireg` 1.2.1 (Lars Metzner, CRAN, GPL>=2).  The SSR
model seeks the most parsimonious (fewest extrema) regression function
that is statistically adequate with respect to two sign criteria:

* the **partial sum criterion** (R `fnR`): for every interval length
  `k` the maximum absolute sum of consecutive residual signs must not
  exceed `F(n, k) = min(sqrt(1 + 2.33 ln(n) k), k)`;
* the **maximum run criterion** (R `maxRunR`): the 95% quantile of the
  maximum run length of equal residual signs is
  `k_run = trunc(3.3 + 1.44 ln(n))`.

The regression function is computed with a quantised Gauss-Seidel
(QSOR) iteration (C `ssrC` / `ssr_neC`): every interior point is
replaced by the simplicitic linear interpolation of its neighbours and
the update is reverted whenever it would violate the partial sum
criterion with threshold `fn`.

Unfolding integration: the method alternates a multiplicative MLEM
update (data fidelity) with a non-equidistant SSR QSOR sweep of the
spectrum over the energy grid.  Following Metzner's minimum statistic,
`fn="auto"` descends a ladder of thresholds starting at
`trunc(0.66 * F(n, k_run))` while the folded residuals stay
sign-adequate and the spectrum does not gain extrema.

References: L. Metzner, "Trendbasierte Prognostik" (2020);
"Adaequates Maschinelles Lernen" (2021); sisireg 1.2.1 (CRAN, 2025).
"""

const _SSR_TINY = 1e-300
const _SSR_MIN_DATA_POINTS = 8

_ssr_sgn(v::Real) = (v > 0.0) - (v < 0.0)  # NaN -> 0, C macro semantics

"""
    max_run_quantile(n) -> Int

95% quantile of the maximum run length of residual signs
(R `maxRunR`: `as.integer(3.3 + 1.44 log(n))`).
"""
function max_run_quantile(n::Integer)
    n < 1 && throw(ArgumentError("n must be positive, got $n"))
    return trunc(Int, 3.3 + 1.44 * log(n))
end

"""
    partial_sum_quantile(n, k)

95% quantile of partial sums of residual signs (R `fnR`):
`F(n, k) = min(sqrt(1 + 2.33 ln(n) k), k)`, element-wise for vector `k`.
"""
function partial_sum_quantile(n::Integer, k::Real)
    n < 1 && throw(ArgumentError("n must be positive, got $n"))
    kf = float(k)
    fn = sqrt(1.0 + 2.33 * log(n) * kf)
    return min(fn, kf)
end

function partial_sum_quantile(n::Integer, k::AbstractVector{<:Real})
    n < 1 && throw(ArgumentError("n must be positive, got $n"))
    kf = float.(k)
    fn = sqrt.(1.0 .+ 2.33 .* log(n) .* kf)
    return min.(fn, kf)
end

"""
    rolling_median(v, k) -> Vector

Centre-aligned rolling median with `n - k + 1` values
(`zoo::rollapply(zoo(v), k, median, align='center')`): element `j` of
the result is `median(v[j : j + k - 1])` (1-based).
"""
function rolling_median(v::AbstractVector{<:Real}, k::Integer)
    vf = collect(float.(v))
    n = length(vf)
    k < 1 && throw(ArgumentError("window k must be >= 1, got $k"))
    k > n && throw(ArgumentError(
        "window k ($k) must not exceed the data length ($n)"))
    return [BSSUnfold.Statistics.median(view(vf, j:j + k - 1)) for j in 1:(n - k + 1)]
end

"""
    _ssr_start_values(dat, k) -> Vector

Initial QSOR values: head/middle/tail rolling medians (layout of
R `ssrR`/`ssr_neR`).  Head/tail are filled with the median of the
first/last `k2 = k>>1` points, the middle with the centre-aligned
rolling median of window `k`; for even `k` the middle segment overlaps
the head by one point and wins (R assignment order).
"""
function _ssr_start_values(dat::Vector{Float64}, k::Int)
    n = length(dat)
    (k < 1 || k > n) && throw(ArgumentError(
        "window k ($k) must be in [1, $n]"))
    k2 = k >> 1
    k1 = k - 2 * k2
    s = zeros(Float64, n)
    if k2 > 0
        s[1:k2] .= BSSUnfold.Statistics.median(view(dat, 1:k2))
    end
    lo = k2 + k1            # 1-based start of the middle segment
    s[lo:(n - k2)] .= rolling_median(dat, k)
    if k2 > 0
        s[(n - k2 + 1):n] .= BSSUnfold.Statistics.median(view(dat, (n - k2 + 1):n))
    end
    return s
end

"""
    partial_sum_max(dat, mu, k) -> Int

Maximum absolute partial sum of residual signs (R `psmaxR`):
`max_t |sum(sign(dat - mu)[t : t + k])|`.
"""
function partial_sum_max(dat::AbstractVector{<:Real},
                         mu::AbstractVector{<:Real}, k::Integer)
    length(dat) == length(mu) || throw(ArgumentError(
        "dat and mu must have the same length, got $(length(dat)) " *
        "and $(length(mu))"))
    n = length(dat)
    k < 1 && throw(ArgumentError("window k must be >= 1, got $k"))
    k > n && throw(ArgumentError(
        "window k ($k) must not exceed the data length ($n)"))
    d = collect(float.(dat)); m = collect(float.(mu))
    s = [_ssr_sgn(d[i] - m[i]) for i in 1:n]
    best = 0
    for j in 1:(n - k + 1)
        psum = 0
        for t in j:(j + k - 1)
            psum += s[t]
        end
        best = max(best, abs(psum))
    end
    return best
end

"""
    number_of_extrema(mu) -> Int

Number of local extrema of a discrete function (R `numberOfExtremaR`):
counts the slope sign changes, classified by the previous slope
direction.
"""
function number_of_extrema(mu::AbstractVector{<:Real})
    n = length(mu)
    n < 2 && return 0
    m = collect(float.(mu))
    n_min = 0
    n_max = 0
    slope = _ssr_sgn(m[2] - m[1])
    for i in 3:n
        s = _ssr_sgn(m[i] - m[i-1])
        if s != slope
            if slope < 0
                n_min += 1
            else
                n_max += 1
            end
            slope = s
        end
    end
    return n_min + n_max
end

"""
    partial_sum_valid(dat, mu) -> Bool

Partial sum adequacy test for all interval lengths (R `psvalid`):
`partial_sum_max(dat, mu, k) <= partial_sum_quantile(n, k)` for every
`k in [5, n>>2]`... precisely `[5, n//5]`; empty for `n < 25`.
"""
function partial_sum_valid(dat::AbstractVector{<:Real},
                           mu::AbstractVector{<:Real})
    n = length(dat)
    maxint = n >> 2 == 0 ? 0 : n ÷ 5
    for k in 5:maxint
        if partial_sum_max(dat, mu, k) > partial_sum_quantile(n, k)
            return false
        end
    end
    return true
end

"""
    run_valid(dat, mu, k=max_run_quantile(n)) -> Bool

Maximum run adequacy test (R `runvalid`): the maximum absolute sum
over windows of `k + 1` consecutive residual signs must not exceed `k`.
"""
function run_valid(dat::AbstractVector{<:Real},
                   mu::AbstractVector{<:Real}, k::Union{Nothing,Integer}=nothing)
    n = length(dat)
    kk = k === nothing ? max_run_quantile(n) : Int(k)
    runmax = partial_sum_max(dat, mu, kk + 1)
    return runmax <= kk
end

"""
    _ssr_equidistant_core(y, mu, funk, k, h, ps, simanz)

QSOR iteration for equidistant data (port of C `ssrC`).  `funk = 1`:
L1 simplicitic neighbour interpolation; `funk = 2`: standardised 4th
order difference system with relaxation 1.9.  `ps` selects the partial
sum threshold mode (`h` threshold) vs the maximum run criterion.
"""
function _ssr_equidistant_core(y::Vector{Float64}, mu::Vector{Float64},
                               funk::Int, k::Int, h::Int, ps::Bool,
                               simanz::Int)
    n = length(y)
    if funk == 1
        for _ in 1:max(simanz, 0)
            chng = false
            for i0 in 1:(n - 2)
                i = i0 + 1
                oldval = mu[i]
                oldsig = _ssr_sgn(y[i] - mu[i])
                mu[i] = 0.5 * (mu[i-1] + mu[i+1])
                newsig = _ssr_sgn(y[i] - mu[i])
                if ps
                    if k < i0 < n - k && oldsig != newsig
                        psum = 0
                        for mm in -k:k
                            psum += _ssr_sgn(y[i0 + mm + 1] - mu[i0 + mm + 1])
                        end
                        if abs(psum) > h
                            mu[i] = oldval
                        end
                    end
                else
                    if oldsig != newsig
                        for j0 in max(0, i0 - k):(min(n - k, i0 + k) - 1)
                            psum = 0
                            for mm in 0:k
                                psum += _ssr_sgn(y[j0 + mm + 1] - mu[j0 + mm + 1])
                            end
                            if abs(psum) > k
                                mu[i] = oldval
                                break
                            end
                        end
                    end
                end
                if mu[i] != oldval
                    chng = true
                end
            end
            chng || break
        end
        return mu
    end

    # funk == 2: standardised QSOR on the 4th-order difference system.
    a2 = sqrt(2.0); a10 = sqrt(10.0); a12 = sqrt(12.0)
    q12 = 1.0 / (a2 * a10); q13 = 1.0 / (a2 * a12); q23 = 1.0 / (a10 * a12)
    a012 = -4.0 * q12; a023 = -8.0 * q23; a013 = 2.0 * q13; a024 = 2.0 * q23
    as2 = 2.0 / 12.0; as8 = -8.0 / 12.0
    omega = 1.9

    ys = copy(y); mus = copy(mu)
    ys[1] *= a2;   mus[1] *= a2
    ys[2] *= a10;  mus[2] *= a10
    ys[3:n-2] .*= a12; mus[3:n-2] .*= a12
    ys[n-1] *= a10; mus[n-1] *= a10
    ys[n] *= a2;   mus[n] *= a2

    for _ in 1:max(simanz, 0)
        chng = false
        for i0 in 1:(n - 2)
            i = i0 + 1
            oldval = mus[i]
            oldsig = _ssr_sgn(ys[i] - mus[i])
            if i0 == 1
                mus[i] -= omega * (mus[i-1] * a012 + mus[i] +
                                   mus[i+1] * a023 + mus[i+2] * a024)
            elseif i0 == 2
                mus[i] -= omega * (mus[i-2] * a013 + mus[i-1] * a023 +
                                   mus[i] + mus[i+1] * as8 + mus[i+2] * as2)
            elseif i0 == 3
                mus[i] -= omega * (mus[i-2] * a024 + mus[i-1] * as8 +
                                   mus[i] + mus[i+1] * as8 + mus[i+2] * as2)
            elseif i0 < n - 4
                mus[i] -= omega * (mus[i-2] * as2 + mus[i-1] * as8 +
                                   mus[i] + mus[i+1] * as8 + mus[i+2] * as2)
            elseif i0 == n - 4
                mus[i] -= omega * (mus[i-2] * as2 + mus[i-1] * as8 +
                                   mus[i] + mus[i+1] * as8 + mus[i+2] * a024)
            elseif i0 == n - 3
                mus[i] -= omega * (mus[i-2] * as2 + mus[i-1] * as8 +
                                   mus[i] + mus[i+1] * a023 + mus[i+2] * a013)
            else # i0 == n - 2
                mus[i] -= omega * (mus[i-2] * a024 + mus[i-1] * a023 +
                                   mus[i] + mus[i+1] * a012)
            end
            newsig = _ssr_sgn(ys[i] - mus[i])
            if ps
                if k < i0 < n - k && oldsig != newsig
                    psum = 0
                    for mm in -k:k
                        psum += _ssr_sgn(ys[i0 + mm + 1] - mus[i0 + mm + 1])
                    end
                    if abs(psum) > h
                        mus[i] = oldval
                    end
                end
            else
                if oldsig != newsig
                    for j0 in max(0, i0 - k):min(n - k, i0 + k)
                        psum = 0
                        for mm in 0:k
                            psum += _ssr_sgn(ys[j0 + mm + 1] - mus[j0 + mm + 1])
                        end
                        if abs(psum) > k
                            mus[i] = oldval
                            break
                        end
                    end
                end
            end
            if mus[i] != oldval
                chng = true
            end
        end
        chng || break
    end

    mus[1] /= a2; mus[2] /= a10
    mus[3:n-2] ./= a12
    mus[n-1] /= a10; mus[n] /= a2
    return mus
end

"""
    _ssr_ne_core(x, y, mu, k, h, simanz)

QSOR iteration, non-equidistant L1 (port of C `ssr_neC`); `x` sorted
ascending.  The partial sum criterion with threshold `h` is always
applied.
"""
function _ssr_ne_core(x::Vector{Float64}, y::Vector{Float64},
                      mu::Vector{Float64}, k::Int, h::Int, simanz::Int)
    n = length(y)
    for _ in 1:max(simanz, 0)
        chng = false
        for i in 2:(n - 1)
            oldval = mu[i]
            oldsig = _ssr_sgn(y[i] - mu[i])
            if x[i-1] != x[i+1]
                mu[i] = mu[i-1] + (x[i] - x[i-1]) *
                    (mu[i+1] - mu[i-1]) / (x[i+1] - x[i-1])
            else
                mu[i] = 0.5 * (mu[i-1] + mu[i+1])
            end
            newsig = _ssr_sgn(y[i] - mu[i])
            if k < i < n - k && oldsig != newsig
                psum = 0
                for mm in -k:k
                    psum += _ssr_sgn(y[i + mm] - mu[i + mm])
                end
                if abs(psum) > h
                    mu[i] = oldval
                end
            end
            if mu[i] != oldval
                chng = true
            end
        end
        chng || break
    end
    return mu
end

"""
    _ssr_ne_sweep(E_grid, y_ref, mu, k, h)

Single non-equidistant QSOR pass continuing from `mu` (no rolling
median re-initialisation): the unfolding solver calls this after every
MLEM step, so `mu` already carries the current spectrum estimate.
Adequacy is measured against the fixed pre-sweep reference `y_ref`;
`mu` is updated in place and returned.
"""
function _ssr_ne_sweep(E_grid::Vector{Float64}, y_ref::Vector{Float64},
                       mu::Vector{Float64}, k::Int, h::Int)
    n = length(y_ref)
    for i in 2:(n - 1)
        oldval = mu[i]
        oldsig = _ssr_sgn(y_ref[i] - mu[i])
        if E_grid[i-1] != E_grid[i+1]
            mu[i] = mu[i-1] + (E_grid[i] - E_grid[i-1]) *
                (mu[i+1] - mu[i-1]) / (E_grid[i+1] - E_grid[i-1])
        else
            mu[i] = 0.5 * (mu[i-1] + mu[i+1])
        end
        newsig = _ssr_sgn(y_ref[i] - mu[i])
        if k < i < n - k && oldsig != newsig
            psum = 0
            for mm in -k:k
                psum += _ssr_sgn(y_ref[i + mm] - mu[i + mm])
            end
            if abs(psum) > h
                mu[i] = oldval
            end
        end
    end
    return mu
end

"""
    ssr(y; fn=0.0, ps=true, funk=1, y1=nothing, yn=nothing, simanz=10000)

Equidistant SSR QSOR regression (R `ssr`/`ssrR` + C `ssrC`).
`fn` is the partial sum threshold; non-positive values select the
automatic default `max(2, log(n) - 2)` in partial sum mode or the run
length quantile otherwise.  `funk = 1` (L1) or `2` (L2).  `y1`/`yn`
fix the boundary values of the regression function.
"""
function ssr(y::AbstractVector{<:Real};
             fn::Real=0.0, ps::Bool=true, funk::Int=1,
             y1::Union{Nothing,Real}=nothing,
             yn::Union{Nothing,Real}=nothing, simanz::Int=10000)
    yv = collect(float.(y))
    n = length(yv)
    n < _SSR_MIN_DATA_POINTS && throw(ArgumentError(
        "SSR requires at least $_SSR_MIN_DATA_POINTS data points, got $n"))
    funk in (1, 2) || throw(ArgumentError("funk must be 1 (L1) or 2 (L2), got $funk"))
    simanz >= 1 || throw(ArgumentError(
        "simanz must be a positive integer, got $simanz"))
    fnv = float(fn)
    if ps
        fnv < 2 && (fnv = max(2.0, log(n) - 2.0))
        k = trunc(Int, 3.3 + 1.44 * log(n))
        h = trunc(Int, fnv)
    else
        fnv < 2 && (fnv = trunc(Int, 3.3 + 1.44 * log(n)))
        k = trunc(Int, fnv)
        h = 0
    end
    mu = _ssr_start_values(yv, k)
    y1 !== nothing && (mu[1] = float(y1))
    yn !== nothing && (mu[end] = float(yn))
    return _ssr_equidistant_core(yv, mu, funk, k, h, ps, simanz)
end

"""
    ssr_ne(x, y; fn=0.0, simanz=10000) -> (x_sorted, mu)

Non-equidistant SSR QSOR regression, L1 (R `ssr_neR`).  `x` is sorted
internally (stable sort); a non-positive `fn` selects the default
`trunc(max(2, log(n) - 2))`.
"""
function ssr_ne(x::AbstractVector{<:Real}, y::AbstractVector{<:Real};
                fn::Real=0.0, simanz::Int=10000)
    xf = collect(float.(x)); yf = collect(float.(y))
    length(xf) == length(yf) || throw(ArgumentError(
        "x and y must have the same length, got $(length(xf)) " *
        "and $(length(yf))"))
    n = length(yf)
    n < _SSR_MIN_DATA_POINTS && throw(ArgumentError(
        "SSR requires at least $_SSR_MIN_DATA_POINTS data points, got $n"))
    simanz >= 1 || throw(ArgumentError(
        "simanz must be a positive integer, got $simanz"))
    order = sortperm(xf)                     # stable
    xs = xf[order]; ys = yf[order]
    fnv = float(fn)
    fnv < 2 && (fnv = trunc(Int, max(2.0, log(n) - 2.0)))
    h = trunc(Int, fnv)
    k = trunc(Int, 3.3 + 1.44 * log(n))      # == max_run_quantile(n)
    mu = _ssr_start_values(ys, k)
    mu = _ssr_ne_core(xs, ys, mu, k, h, simanz)
    return (xs, mu)
end

"""
    ssr_min_statistic(y; funk=1, y1=nothing, yn=nothing, ps=true, simanz=10000)
        -> (mu, fn)

Minimum statistic SSR (R `ssr_minR`): starts from
`fn = 0.66 * partial_sum_quantile(n, k_run)` and decreases `fn` while
the model stays statistically adequate and does not gain extrema; the
final model is recomputed at the last adequate `fn + 1`.
"""
function ssr_min_statistic(y::AbstractVector{<:Real};
                           funk::Int=1, y1::Union{Nothing,Real}=nothing,
                           yn::Union{Nothing,Real}=nothing,
                           ps::Bool=true, simanz::Int=10000)
    yv = collect(float.(y))
    n = length(yv)
    n < _SSR_MIN_DATA_POINTS && throw(ArgumentError(
        "SSR requires at least $_SSR_MIN_DATA_POINTS data points, got $n"))
    if ps
        k = max_run_quantile(n)
        fnv = 0.66 * partial_sum_quantile(n, k)
    else
        fnv = float(max_run_quantile(n))
    end
    mu = ssr(yv; funk=funk, y1=y1, yn=yn, fn=fnv, ps=ps, simanz=simanz)
    valid = ps ? partial_sum_valid(yv, mu) : run_valid(yv, mu)
    extrema_opt = number_of_extrema(mu)
    extrema = extrema_opt
    while valid && extrema <= extrema_opt && fnv > 0
        fnv -= 1.0
        mu = ssr(yv; funk=funk, y1=y1, yn=yn, fn=fnv, ps=ps, simanz=simanz)
        valid = ps ? partial_sum_valid(yv, mu) : run_valid(yv, mu)
        extrema = number_of_extrema(mu)
    end
    fnv += 1.0
    mu = ssr(yv; funk=funk, y1=y1, yn=yn, fn=fnv, ps=ps, simanz=simanz)
    return (mu, trunc(Int, fnv))
end

"""
    ssr_min_statistic_ne(x, y; simanz=10000) -> (x_sorted, mu, fn)

Minimum statistic SSR for non-equidistant data (R `ssr_ne_minR`).
"""
function ssr_min_statistic_ne(x::AbstractVector{<:Real},
                              y::AbstractVector{<:Real}; simanz::Int=10000)
    xf = collect(float.(x)); yf = collect(float.(y))
    length(xf) == length(yf) || throw(ArgumentError(
        "x and y must have the same length, got $(length(xf)) " *
        "and $(length(yf))"))
    n = length(yf)
    n < _SSR_MIN_DATA_POINTS && throw(ArgumentError(
        "SSR requires at least $_SSR_MIN_DATA_POINTS data points, got $n"))
    order = sortperm(xf)                     # stable
    xs = xf[order]; ys = yf[order]
    k = max_run_quantile(n)
    fnv = 0.66 * partial_sum_quantile(n, k)
    _, mu = ssr_ne(xs, ys; fn=fnv, simanz=simanz)
    valid = partial_sum_valid(ys, mu)
    extrema_opt = number_of_extrema(mu)
    extrema = extrema_opt
    while valid && extrema <= extrema_opt && fnv > 0
        fnv -= 1.0
        _, mu = ssr_ne(xs, ys; fn=fnv, simanz=simanz)
        valid = partial_sum_valid(ys, mu)
        extrema = number_of_extrema(mu)
    end
    fnv += 1.0
    xs_final, mu = ssr_ne(xs, ys; fn=fnv, simanz=simanz)
    return (xs_final, mu, trunc(Int, fnv))
end

"""
    ssr_predict(x, mu, xx) -> Vector

Piecewise-linear prediction of an SSR model (R `ssr_predict`):
linear interpolation between model points with linear extrapolation
from the end segments outside the support.  Duplicate argument values
are reduced to their first occurrence.
"""
function ssr_predict(x::AbstractVector{<:Real}, mu::AbstractVector{<:Real},
                     xx::AbstractVector{<:Real})
    xf = collect(float.(x)); muf = collect(float.(mu))
    length(xf) == length(muf) || throw(ArgumentError(
        "x and mu must have the same length, got $(length(xf)) " *
        "and $(length(muf))"))
    length(xf) >= 2 || throw(ArgumentError(
        "ssr_predict requires at least 2 model points, got $(length(xf))"))
    order = sortperm(xf)                     # stable
    xs = xf[order]; mus = muf[order]
    if any(diff(xs) .== 0)
        keep = trues(length(xs))
        keep[1] = true
        for i in 2:length(xs)
            keep[i] = xs[i] != xs[i-1]
        end
        xs = xs[keep]; mus = mus[keep]
    end
    out = zeros(Float64, length(xx))
    for (idx, xq) in enumerate(float.(xx))
        if xq <= xs[1]
            out[idx] = xs[2] != xs[1] ?
                mus[1] - (xs[1] - xq) * (mus[2] - mus[1]) / (xs[2] - xs[1]) :
                mus[1]
        elseif xq >= xs[end]
            out[idx] = xs[end] != xs[end-1] ?
                mus[end] + (xq - xs[end]) * (mus[end] - mus[end-1]) /
                    (xs[end] - xs[end-1]) :
                mus[end]
        else
            j = searchsortedlast(xs, xq)
            out[idx] = xs[j] == xq ? mus[j] :
                mus[j] + (xq - xs[j]) / (xs[j+1] - xs[j]) * (mus[j+1] - mus[j])
        end
    end
    return out
end

# ---------------------------------------------------------------------------
# SSR unfolding (MLEM data step + SSR parsimony sweep)
# ---------------------------------------------------------------------------

"""
    _ssr_folded_adequacy(b, fit, k_run) -> (ps_ok, run_ok, max_run)

Data-space sign adequacy of a candidate solution: the SSR criteria are
applied to the residuals `b - A x` of the folded model.
"""
function _ssr_folded_adequacy(b::Vector{Float64}, fit::Vector{Float64},
                              k_run::Int)
    max_run = partial_sum_max(b, fit, k_run + 1)
    ps_ok = partial_sum_valid(b, fit)
    return (ps_ok, max_run <= k_run, max_run)
end

"""
    _ssr_unfold_fixed_fn(A, b, x_init, E_sorted, k_run, fn_try, ...
        max_iterations, tolerance, smooth_every, inner_sweeps)
        -> (spectrum, iterations, converged, n_sweeps)

Alternating MLEM / SSR-sweep scheme for one fixed `fn`: one
multiplicative MLEM update per outer iteration followed -- every
`smooth_every` iterations -- by `inner_sweeps` non-equidistant SSR QSOR
sweeps on the current spectrum over the energy grid.
"""
function _ssr_unfold_fixed_fn(A::Matrix{Float64}, b::Vector{Float64},
                              x_init::Vector{Float64},
                              E_sorted::Vector{Float64}, k_run::Int,
                              fn_try::Int, max_iterations::Int,
                              tolerance::Float64, smooth_every::Int,
                              inner_sweeps::Int)
    floor_ = max(sum(b), 1.0) * 1e-12
    x = max.(copy(x_init), floor_)
    colsum = vec(sum(A, dims=1))
    colsum_safe = [colsum[j] > _SSR_TINY ? colsum[j] : 1.0 for j in eachindex(colsum)]
    AT = Matrix(A')
    n_sweeps = 0
    converged = false
    iterations = max_iterations
    for it in 1:max_iterations
        x_prev = x
        # (1) data fidelity: one MLEM update
        Ax = A * x
        Ax = max.(Ax, _SSR_TINY)
        x_new = x .* (AT * (b ./ Ax)) ./ colsum_safe
        x_new = max.(x_new, 0.0)
        # (2) SSR parsimony step on the energy grid
        if smooth_every > 0 && it % smooth_every == 0
            y_ref = copy(x_new)
            for _ in 1:max(inner_sweeps, 1)
                _ssr_ne_sweep(E_sorted, y_ref, x_new, k_run, fn_try)
            end
            n_sweeps += 1
            x_new = max.(x_new, 0.0)
        end
        # convergence: relative L2 change of the spectrum
        diff = norm(x_new .- x_prev)
        base = norm(x_prev)
        x = x_new
        if diff <= tolerance * max(base, _SSR_TINY)
            converged = true
            iterations = it
            break
        end
    end
    return (x, iterations, converged, n_sweeps)
end

"""
    solve_ssr_full(A, b; x0=nothing, E_MeV=nothing, fn="auto",
                   max_iterations=500, tolerance=1e-6, smooth_every=1,
                   inner_sweeps=1, fn_ladder_cap=8) -> Dict{String,Any}

SSR unfolding returning rich diagnostics: `spectrum`, `n_iterations`,
`converged`, `fn` (threshold used), `fn_start`, `k_run`, `n_extrema`,
`ps_valid_data`, `run_valid_data`, `max_run_data`, `ssr_sweeps` and
`fn_ladder` (per candidate adequacy results).
"""
function solve_ssr_full(A::AbstractMatrix{<:Real}, b::AbstractVector{<:Real};
                        x0::Union{Nothing,AbstractVector{<:Real}}=nothing,
                        E_MeV::Union{Nothing,AbstractVector{<:Real}}=nothing,
                        fn::Union{String,Integer}="auto",
                        max_iterations::Integer=500,
                        tolerance::Real=1e-6,
                        smooth_every::Integer=1,
                        inner_sweeps::Integer=1,
                        fn_ladder_cap::Integer=8)
    Af = Matrix{Float64}(A)
    bf = collect(float.(b))
    m, n = size(Af)
    length(bf) == m || throw(ArgumentError(
        "Length of b ($(length(bf))) must match number of rows in A ($m)"))
    m < 3 && throw(ArgumentError(
        "SSR unfolding requires at least 3 detector readings, got $m"))
    n < _SSR_MIN_DATA_POINTS && throw(ArgumentError(
        "SSR unfolding requires at least $_SSR_MIN_DATA_POINTS energy " *
        "bins, got $n"))
    max_iterations > 0 || throw(ArgumentError(
        "max_iterations must be positive, got $max_iterations"))
    tolerance > 0 || throw(ArgumentError(
        "tolerance must be positive, got $tolerance"))

    # --- energy grid (the SSR sweep needs ascending arguments) ---------------
    inverse = nothing
    order = nothing
    if E_MeV === nothing
        E_sorted = collect(0.0:(n - 1.0))
    else
        E = collect(float.(E_MeV))
        length(E) == n || throw(ArgumentError(
            "Length of E_MeV ($(length(E))) must match number of energy " *
            "bins ($n)"))
        if all(diff(E) .> 0)
            E_sorted = E
        else
            order = sortperm(E)          # stable
            E_sorted = E[order]
            inverse = sortperm(order)
        end
    end

    A_use = order === nothing ? Af : Af[:, order]
    x_init = nothing
    if x0 !== nothing && any(collect(float.(x0)) .> 0)
        x_init = collect(float.(x0))
    else
        total = sum(bf)
        x_init = fill(total > 0 ? total / n : 1.0, n)
    end
    order !== nothing && (x_init = x_init[order])

    # --- threshold ladder -----------------------------------------------------
    k_run = max_run_quantile(n)
    fn_start = max(2, trunc(Int, 0.66 * partial_sum_quantile(n, k_run)))
    ladder = Int[]
    if fn isa String
        fn == "auto" || throw(ArgumentError(
            "fn must be 'auto' or a positive integer"))
        bottom = max(2, fn_start - Int(fn_ladder_cap) + 1)
        ladder = collect(fn_start:-1:bottom)
    else
        Int(fn) >= 1 || throw(ArgumentError(
            "fn must be 'auto' or a positive integer, got $fn"))
        fn_start = Int(fn)
        ladder = [fn_start]
    end

    ladder_results = Dict{String,Any}[]
    first_extrema = nothing
    best = nothing
    for fn_try in ladder
        x_run, iters, conv, sweeps = _ssr_unfold_fixed_fn(
            A_use, bf, x_init, E_sorted, k_run, Int(fn_try),
            Int(max_iterations), Float64(tolerance), Int(smooth_every),
            Int(inner_sweeps))
        ps_ok, run_ok, max_run = _ssr_folded_adequacy(bf, A_use * x_run, k_run)
        extrema = number_of_extrema(x_run)
        push!(ladder_results, Dict{String,Any}(
            "fn" => Int(fn_try),
            "ps_valid_data" => ps_ok,
            "run_valid_data" => run_ok,
            "n_extrema" => extrema,
            "n_iterations" => iters,
            "converged" => conv))
        if first_extrema === nothing
            first_extrema = extrema
        end
        if !(ps_ok && run_ok && extrema <= first_extrema)
            break
        end
        best = (x_run, iters, conv, Int(fn_try), sweeps)
    end

    if best !== nothing
        x_sorted, iters, conv, fn_used, sweeps = best
    else
        # Even the start model is inadequate: fall back to fn_start + 1,
        # as the R ssr_minR/ssr_ne_minR loop does in that situation.
        fn_used = fn_start + 1
        x_sorted, iters, conv, sweeps = _ssr_unfold_fixed_fn(
            A_use, bf, x_init, E_sorted, k_run, fn_used,
            Int(max_iterations), Float64(tolerance), Int(smooth_every),
            Int(inner_sweeps))
        ps_ok, run_ok, max_run = _ssr_folded_adequacy(bf, A_use * x_sorted, k_run)
        push!(ladder_results, Dict{String,Any}(
            "fn" => fn_used,
            "ps_valid_data" => ps_ok,
            "run_valid_data" => run_ok,
            "n_extrema" => number_of_extrema(x_sorted),
            "n_iterations" => iters,
            "converged" => conv))
    end

    spectrum = inverse === nothing ? x_sorted : x_sorted[inverse]
    ps_ok, run_ok, max_run = _ssr_folded_adequacy(bf, Af * spectrum, k_run)
    converged = conv && all(isfinite.(spectrum))
    return Dict{String,Any}(
        "spectrum" => spectrum,
        "n_iterations" => iters,
        "converged" => converged,
        "fn" => fn_used,
        "fn_start" => fn_start,
        "k_run" => k_run,
        "n_extrema" => number_of_extrema(spectrum),
        "ps_valid_data" => ps_ok,
        "run_valid_data" => run_ok,
        "max_run_data" => max_run,
        "ssr_sweeps" => sweeps,
        "fn_ladder" => ladder_results)
end

"""
    solve_ssr(A, b, x0; E_MeV=nothing, fn="auto", max_iterations=500,
              tolerance=1e-6, smooth_every=1, inner_sweeps=1,
              fn_ladder_cap=8) -> UnfoldResult

Solve the unfolding problem with SSR sign-parsimony regularisation:
the spectrum is alternately fitted to the data with one MLEM update and
smoothed with a non-equidistant SSR QSOR sweep whose partial sum
threshold follows Metzner's minimum statistic (see `solve_ssr_full`).
"""
function solve_ssr(A::AbstractMatrix{<:Real}, b::AbstractVector{<:Real},
                   x0::AbstractVector{<:Real};
                   E_MeV::Union{Nothing,AbstractVector{<:Real}}=nothing,
                   fn::Union{String,Integer}="auto",
                   max_iterations::Integer=500,
                   tolerance::Real=1e-6,
                   smooth_every::Integer=1,
                   inner_sweeps::Integer=1,
                   fn_ladder_cap::Integer=8)
    diag = solve_ssr_full(A, b;
                          x0=x0, E_MeV=E_MeV, fn=fn,
                          max_iterations=max_iterations,
                          tolerance=tolerance,
                          smooth_every=smooth_every,
                          inner_sweeps=inner_sweeps,
                          fn_ladder_cap=fn_ladder_cap)
    spectrum = diag["spectrum"]
    residual = b .- A * spectrum
    extra = Dict{String,Any}(
        "fn" => diag["fn"], "fn_start" => diag["fn_start"],
        "k_run" => diag["k_run"], "n_extrema" => diag["n_extrema"],
        "ps_valid_data" => diag["ps_valid_data"],
        "run_valid_data" => diag["run_valid_data"],
        "max_run_data" => diag["max_run_data"],
        "ssr_sweeps" => diag["ssr_sweeps"],
        "fn_ladder" => diag["fn_ladder"])
    return UnfoldResult(max.(spectrum, 0.0), diag["n_iterations"],
                        diag["converged"], norm(residual), extra)
end
