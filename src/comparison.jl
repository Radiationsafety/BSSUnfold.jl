"""
Spectrum comparison (port of utils/comparison.py, 1769 lines).

Each metric operates on 1-D Float64 vectors and returns a Float64.
SciPy dependencies are replaced by native implementations of the real-valued
formulas: wasserstein/energy/KS — via the pooled ECDF (port of scipy._cdf_distance),
Anderson–Darling (k-sample, "right" variant in permetry METRIC), Wilcoxon/MWU — W/U statistics,
power divergences (chi2/G/FT/Cressie–Read) — power_divergence formulas.
"""

const CMP_EPS = 1e-15

# ─── Helpers ───────────────────────────────────────────────────────────────────

function _check_same_length_cmp(s1::AbstractVector, s2::AbstractVector)
    length(s1) == length(s2) || throw(ArgumentError(
        "Spectra must have same length, got $(length(s1)) and $(length(s2))"))
    return nothing
end

function _normalize_cmp(p::AbstractVector{<:Real})
    v = max.(Float64.(p), CMP_EPS)
    return v ./ sum(v)
end

"""
    _compute_log_steps(energy)

Logarithmic integration steps: "central differences" of log10(E),
multiplied by ln(10) (port of `_compute_log_steps`).
"""
function _compute_log_steps(energy::AbstractVector{<:Real})
    e = Float64.(energy)
    log_e = log10.(e .+ CMP_EPS)
    n = length(e)
    log_steps = zeros(n)
    if n > 1
        log_steps[1] = log_e[2] - log_e[1]
        log_steps[end] = log_e[end] - log_e[end-1]
        if n > 2
            log_steps[2:n-1] .= (log_e[3:n] .- log_e[1:n-2]) ./ 2.0
        end
    end
    return log_steps .* log(10)
end

"""
    _extract_cc_array(cc, energy; preferred_geom="AP")

Extract a 1-D array of conversion coefficients from a Dict/Vector
(`nothing` → ones, Dict → preferred geometry or the first key other than `E_MeV`).
"""
function _extract_cc_array(cc::Union{Nothing,Dict{String,<:Vector{<:Real}},
                                     AbstractVector{<:Real}},
                           energy::AbstractVector{<:Real};
                           preferred_geom::AbstractString="AP")
    if cc === nothing
        return ones(Float64, length(energy))
    elseif cc isa Dict
        haskey(cc, preferred_geom) && return Float64.(cc[preferred_geom])
        for (key, val) in cc
            key != "E_MeV" && return Float64.(val)
        end
        return ones(Float64, length(energy))
    end
    return Float64.(collect(cc))
end

"""
    _get_ade_cc(energy, cc_ade)

ICRP-74 ADE conversion coefficients interpolated onto the `energy` grid
(`nothing` → the `ICRP74_operational` set, Dict → extract with key "ADE").
"""
function _get_ade_cc(energy::AbstractVector{<:Real},
                     cc_ade::Union{Nothing,Dict{String,<:Vector{<:Real}},
                                   AbstractVector{<:Real}})
    e = Float64.(energy)
    if cc_ade === nothing
        cc = get_coefficients("ICRP74_operational")
        e_src = Float64.(cc["E_MeV"])
        v_src = Float64.(cc["ADE"])
        interp = similar(e)
        for (i, x) in enumerate(e)
            interp[i] = BSSUnfold._linear_interp(e_src, v_src, x)
        end
        interp[e .< e_src[1]] .= 0.0
        interp[e .> e_src[end]] .= 0.0
        return interp
    elseif cc_ade isa Dict
        return _extract_cc_array(cc_ade, e; preferred_geom="ADE")
    end
    return Float64.(collect(cc_ade))
end

# ─── Flux ─────────────────────────────────────────────────────────────────────

"Sum of the spectrum bins (fluence)."
function total_flux(s::AbstractVector{<:Real})::Float64
    return sum(Float64.(s))
end

# ─── Entropy ─────────────────────────────────────────────────────────────────

function _prob(p::AbstractVector{<:Real})
    v = max.(Float64.(p), CMP_EPS)
    return v ./ sum(v)
end

"""
    kl_divergence(p, q)

KL divergence D_KL(p‖q); both inputs are normalized as probabilities.
"""
function kl_divergence(p::AbstractVector{<:Real}, q::AbstractVector{<:Real})::Float64
    _check_same_length_cmp(p, q)
    pn = _prob(p)
    qn = _prob(q)
    return sum(pn .* log.(pn ./ qn))
end

"Cross-entropy H(p, q)."
function cross_entropy(p::AbstractVector{<:Real}, q::AbstractVector{<:Real})::Float64
    _check_same_length_cmp(p, q)
    pn = _prob(p)
    qn = _prob(q)
    return -sum(pn .* log.(qn))
end

"Shannon entropy H(p)."
function entropy(p::AbstractVector{<:Real})::Float64
    pn = _prob(p)
    return -sum(pn .* log.(pn))
end

"100·(H(p,q) − H(p)) / H(p)."
function entropy_difference_percent(p::AbstractVector{<:Real}, q::AbstractVector{<:Real})::Float64
    _check_same_length_cmp(p, q)
    pn = _prob(p)
    qn = _prob(q)
    h_p = -sum(pn .* log.(pn))
    h_pq = -sum(pn .* log.(qn))
    h_p == 0.0 && return 0.0
    return 100.0 * (h_pq - h_p) / h_p
end

# ─── Distribution distances (port of scipy._cdf_distance) ────────────────────

function _cdf_distance(p_int::Integer, u_values::AbstractVector{<:Real},
                       v_values::AbstractVector{<:Real})
    u = sort(Float64.(collect(u_values)))
    v = sort(Float64.(collect(v_values)))
    all_values = sort(vcat(u, v))
    deltas = diff(all_values)                    # length N_all − 1

    # ECDF at points all_values[:-1]: number of values <= the points
    u_cdf = Float64[searchsortedlast(u, x) / length(u) for x in all_values[1:end-1]]
    v_cdf = Float64[searchsortedlast(v, x) / length(v) for x in all_values[1:end-1]]

    if p_int == 1
        return sum(abs(u_cdf[i] - v_cdf[i]) * deltas[i] for i in eachindex(deltas))
    elseif p_int == 2
        return sqrt(sum((u_cdf[i] - v_cdf[i])^2 * deltas[i] for i in eachindex(deltas)))
    end
    return (sum(abs(u_cdf[i] - v_cdf[i])^p_int * deltas[i]
                for i in eachindex(deltas)))^(1 / p_int)
end

"1D Wasserstein-1 distance (port of scipy.stats.wasserstein_distance)."
function wasserstein_dist(p::AbstractVector{<:Real}, q::AbstractVector{<:Real})::Float64
    return _cdf_distance(1, p, q)
end

"Energy distance (port of scipy.stats.energy_distance: √2·W₂)."
function energy_dist(p::AbstractVector{<:Real}, q::AbstractVector{<:Real})::Float64
    return sqrt(2) * _cdf_distance(2, p, q)
end

"KS statistic of two samples: max |ECDF_p − ECDF_q| over all observed points."
function kolmogorov_smirnov_stat(p::AbstractVector{<:Real}, q::AbstractVector{<:Real})::Float64
    u = sort(Float64.(collect(p)))
    v = sort(Float64.(collect(q)))
    all_values = sort(vcat(u, v))
    dmax = 0.0
    for x in all_values
        cu = searchsortedlast(u, x) / length(u)
        cv = searchsortedlast(v, x) / length(v)
        dmax = max(dmax, abs(cu - cv))
    end
    return dmax
end

# ─── Correlations ────────────────────────────────────────────────────────────

"Pearson correlation (0.0 if either input has zero variance)."
function pearson_r(p::AbstractVector{<:Real}, q::AbstractVector{<:Real})::Float64
    _check_same_length_cmp(p, q)
    pn = Float64.(p)
    qn = Float64.(q)
    sp = std(pn)
    sq = std(qn)
    (sp == 0.0 || sq == 0.0) && return 0.0
    return sum((pn .- mean(pn)) .* (qn .- mean(qn))) /
           ((length(pn) - 1) * sp * sq)
end

"""
    _rankdata_average(v)

Ranks with the average assigned at ties (analog of scipy.stats.rankdata(mode='average')).
"""
function _rankdata_average(v::AbstractVector{<:Real})
    n = length(v)
    order = sortperm(collect(Float64.(v)))
    ranks = zeros(Float64, n)
    i = 1
    while i <= n
        j = i
        while j < n && v[order[j+1]] == v[order[j]]
            j += 1
        end
        rk = (i + j) / 2.0
        for k in i:j
            ranks[order[k]] = rk
        end
        i = j + 1
    end
    return ranks
end

"Spearman correlation (rank-based; 0 at zero variance)."
function spearman_r(p::AbstractVector{<:Real}, q::AbstractVector{<:Real})::Float64
    _check_same_length_cmp(p, q)
    pn = Float64.(p)
    qn = Float64.(q)
    (std(pn) == 0.0 || std(qn) == 0.0) && return 0.0
    rp = _rankdata_average(pn)
    rq = _rankdata_average(qn)
    return sum((rp .- mean(rp)) .* (rq .- mean(rq))) /
           (norm(rp .- mean(rp)) * norm(rq .- mean(rq)))
end

# ─── Error metrics ───────────────────────────────────────────────────────────

function mean_squared_error(p::AbstractVector{<:Real}, q::AbstractVector{<:Real})::Float64
    _check_same_length_cmp(p, q)
    pn = Float64.(p); qn = Float64.(q)
    return mean((pn .- qn) .^ 2)
end

function root_mean_squared_error(p::AbstractVector{<:Real}, q::AbstractVector{<:Real})::Float64
    return sqrt(mean_squared_error(p, q))
end

function mean_absolute_error(p::AbstractVector{<:Real}, q::AbstractVector{<:Real})::Float64
    _check_same_length_cmp(p, q)
    return mean(abs.(Float64.(p) .- Float64.(q)))
end

"""
    mape(p, q)

Mean absolute percentage error in percent (0–100); points with |p| ≤ 1e-15
are skipped.
"""
function mape(p::AbstractVector{<:Real}, q::AbstractVector{<:Real})::Float64
    _check_same_length_cmp(p, q)
    pn = Float64.(p); qn = Float64.(q)
    mask = abs.(pn) .> CMP_EPS
    any(mask) || return 0.0
    return mean(abs.((pn[mask] .- qn[mask]) ./ pn[mask])) * 100.0
end

function r2_score(p::AbstractVector{<:Real}, q::AbstractVector{<:Real})::Float64
    _check_same_length_cmp(p, q)
    pn = Float64.(p); qn = Float64.(q)
    ss_res = sum((pn .- qn) .^ 2)
    ss_tot = sum((pn .- mean(pn)) .^ 2)
    ss_tot == 0.0 && return 0.0
    return 1.0 - ss_res / ss_tot
end

function max_error(p::AbstractVector{<:Real}, q::AbstractVector{<:Real})::Float64
    _check_same_length_cmp(p, q)
    return maximum(abs.(Float64.(p) .- Float64.(q)))
end

function median_absolute_error(p::AbstractVector{<:Real}, q::AbstractVector{<:Real})::Float64
    _check_same_length_cmp(p, q)
    return median(abs.(Float64.(p) .- Float64.(q)))
end

"""
    total_flux_ratio(p, q)

Ratio of total fluences (1.0 = perfect conservation); 0 if q is empty.
"""
function total_flux_ratio(p::AbstractVector{<:Real}, q::AbstractVector{<:Real})::Float64
    _check_same_length_cmp(p, q)
    q_sum = sum(Float64.(q))
    q_sum == 0.0 && return 0.0
    return sum(Float64.(p)) / q_sum
end

"Cosine similarity (0 if either vector has zero norm)."
function cosine_similarity(p::AbstractVector{<:Real}, q::AbstractVector{<:Real})::Float64
    _check_same_length_cmp(p, q)
    pn = Float64.(p)
    qn = Float64.(q)
    np_ = norm(pn)
    nq = norm(qn)
    (np_ == 0.0 || nq == 0.0) && return 0.0
    return dot(pn, qn) / (np_ * nq)
end

# ─── Kernel ──────────────────────────────────────────────────────────────────

"""
    mmd_rbf(p, q; gamma=nothing)

Maximum Mean Discrepancy with an RBF kernel. When `gamma=nothing`,
1/(2·med²), med = median of pairwise distances of the pooled points.
"""
function mmd_rbf(p::AbstractVector{<:Real}, q::AbstractVector{<:Real};
                 gamma::Union{Nothing,Real}=nothing)::Float64
    _check_same_length_cmp(p, q)
    x = Float64.(collect(p))
    y = Float64.(collect(q))
    if gamma === nothing
        # med — median of pairwise Euclidean distances of the pooled points
        all_pts = vcat(x, y)
        dists = [sqrt(abs2(xi - xj)) for xi in all_pts, xj in all_pts]
        med = median(collect(Iterators.flatten(dists)))
        γ = 1.0 / (2.0 * max(med, CMP_EPS)^2)
    else
        γ = Float64(gamma)
    end
    XX = mean(exp(-γ * abs2(xi - xj)) for xi in x, xj in x)
    YY = mean(exp(-γ * abs2(yi - yj)) for yi in y, yj in y)
    XY = mean(exp(-γ * abs2(xi - yj)) for xi in x, yj in y)
    return XX + YY - 2.0 * XY
end

# ─── Power divergences (port of scipy.stats.power_divergence) ───────────────

function _power_divergence(p::Vector{Float64}, q::Vector{Float64}, λ::Float64)::Float64
    o = _prob(p)
    e = _prob(q)
    if λ == -1.0
        # log-likelihood (G-test): 2·Σ o·ln(o/e)
        return 2.0 * sum(o[i] > 0 ? o[i] * log(o[i] / e[i]) : 0.0
                         for i in eachindex(o))
    elseif λ == 0.0
        # limit λ→0 (multinomial): 2·Σ o·ln(o/e)
        return 2.0 * sum(o[i] > 0 ? o[i] * log(o[i] / e[i]) : 0.0
                         for i in eachindex(o))
    elseif λ == 0.5
        # Freeman–Tukey: 4·Σ (√o − √e)²
        return 4.0 * sum(abs2(sqrt(o[i]) - sqrt(e[i])) for i in eachindex(o))
    elseif λ == 2.0 / 3.0
        # Cressie–Read: (9/5)·Σ o·[(o/e)^(2/3) − 1]  (2/(λ(λ+1)) = 9/5)
        return 9.0 / 5.0 * sum(o[i] * ((o[i] / e[i])^(2 / 3) - 1)
                               for i in eachindex(o))
    end
    s = 0.0
    for i in eachindex(o)
        s += o[i] * ((o[i] / e[i])^λ - 1)
    end
    return (2.0 / (λ * (λ + 1))) * s
end

"""
    chi_squared(p, q)

χ² statistic (Pearson, λ=1) for normalized probabilities.
"""
function chi_squared(p::AbstractVector{<:Real}, q::AbstractVector{<:Real})::Float64
    _check_same_length_cmp(p, q)
    return _power_divergence(Float64.(collect(p)), Float64.(collect(q)), Float64(1.0))
end

"G-test (λ=−1, log-likelihood)."
function g_test(p::AbstractVector{<:Real}, q::AbstractVector{<:Real})::Float64
    _check_same_length_cmp(p, q)
    return _power_divergence(Float64.(collect(p)), Float64.(collect(q)), Float64(-1.0))
end

" Freeman–Tukey (λ=½)."
function freeman_tukey(p::AbstractVector{<:Real}, q::AbstractVector{<:Real})::Float64
    _check_same_length_cmp(p, q)
    return _power_divergence(Float64.(collect(p)), Float64.(collect(q)), Float64(0.5))
end

"Cressie–Read (λ=⅔)."
function cressie_read(p::AbstractVector{<:Real}, q::AbstractVector{<:Real})::Float64
    _check_same_length_cmp(p, q)
    return _power_divergence(Float64.(collect(p)), Float64.(collect(q)), Float64(2 / 3))
end

# ─── Anderson–Darling (k-sample; port of scipy._morestats) ─────────────────

"""
    anderson_darling(p, q)

k-sample Anderson–Darling statistic (Scholz & Stephens 1987, the
"right" variant, as in bssunfold comparison; port of `_anderson_ksamp_right`).
The value depends only on the data — permutations affect only the p-value.
"""
function anderson_darling(p::AbstractVector{<:Real}, q::AbstractVector{<:Real})::Float64
    samples = [Float64.(collect(p)), Float64.(collect(q))]
    if length(unique(p)) < 2 || length(unique(q)) < 2
        return 0.0
    end
    k = 2
    Z = sort(vcat(samples...))
    N = length(Z)
    N >= 2 || return 0.0
    A2kN = 0.0
    Zstar = unique(Z)
    M = length(Zstar)
    # lj — multiplicities of the pooled sample for the first M−1 unique values
    lj = Float64[searchsortedlast(Z, z) - searchsortedfirst(Z, z) + 1
                 for z in Zstar[1:M-1]]
    Bj = accumulate(+, lj)
    for i in 1:k
        n_i = length(samples[i])
        s = sort(samples[i])
        inner = 0.0
        for j in 1:M-1
            Mij = searchsortedlast(s, Zstar[j])  # count of s ≤ Zstar[j] (side='right')
            den = Bj[j] * Float64(N - Bj[j])
            inner += lj[j] / Float64(N) * (N * Mij - Bj[j] * n_i)^2 / den
        end
        A2kN += inner / n_i
    end

    # Normalization (as in scipy.stats.anderson_ksamp, variant='right')
    m = Float64(k - 1)
    H = sum(1.0 / n_i for n_i in [length(samples[i]) for i in 1:k])
    t_terms = [1.0 / r for r in (N-1):(-1):2]             # arange(N-1, 1, -1)
    hs_cum = accumulate(+, t_terms)                       # cumsum
    h = last(hs_cum) + 1.0
    g = sum(hs_cum[j] / Float64(2 + j - 1) for j in eachindex(hs_cum))
    a_c = (4g - 6)*(k - 1) + (10 - 6*g)*H
    b_c = (2g - 4)*k^2 + 8h*k + (2g - 14h - 4)*H - 8h + 4g - 6
    c_c = (6h + 2g - 2)*k^2 + (4h - 4g + 6)*k + (2h - 6)*H + 4h
    d_c = (2h + 6)*k^2 - 4h*k
    sigmasq = (a_c*N^3 + b_c*N^2 + c_c*N + d_c) /
              ((N - 1.0)*(N - 2.0)*(N - 3.0))
    return (A2kN - m) / sqrt(sigmasq)
end

"Standardized difference of means (Cohen's d)."
function standardized_mean_difference(p::AbstractVector{<:Real}, q::AbstractVector{<:Real})::Float64
    pn = Float64.(collect(p))
    qn = Float64.(collect(q))
    denom = sqrt((var(pn) + var(qn)) / 2.0)
    denom == 0.0 && return 0.0
    return (mean(pn) - mean(qn)) / denom
end

"""
    wilcoxon_test(p, q)

Wilcoxon signed-rank statistic (two-sided: min(W+, W−), zeros dropped;
0.0 for identical inputs).
"""
function wilcoxon_test(p::AbstractVector{<:Real}, q::AbstractVector{<:Real})::Float64
    _check_same_length_cmp(p, q)
    pn = Float64.(collect(p))
    qn = Float64.(collect(q))
    if all(isapprox(pn[i], qn[i]) for i in eachindex(pn))
        return 0.0
    end
    diffs = [(pn[i] - qn[i], i) for i in eachindex(pn) if pn[i] != qn[i]]
    isempty(diffs) && return 0.0
    ranks = _rankdata_average([abs(d) for (d, _) in diffs])
    w_plus = sum(ranks[i] for i in eachindex(diffs) if diffs[i][1] > 0; init=0.0)
    w_minus = sum(ranks[i] for i in eachindex(diffs) if diffs[i][1] < 0; init=0.0)
    return min(w_plus, w_minus)
end

"""
    mannwhitneyu_test(p, q)

Mann–Whitney U statistic (two-sided: min(U1, U2)).
"""
function mannwhitneyu_test(p::AbstractVector{<:Real}, q::AbstractVector{<:Real})::Float64
    pn = Float64.(collect(p))
    qn = Float64.(collect(q))
    ranks = _rankdata_average(vcat(pn, qn))
    n1 = length(pn)
    n2 = length(qn)
    R1 = sum(ranks[1:n1])
    U1 = R1 - n1 * (n1 + 1) / 2
    return U1
end

# ─── Spectrum shape ─────────────────────────────────────────────────────────

"Cosine similarity of spectra normalized to total fluence."
function spectral_shape_similarity(s1::AbstractVector{<:Real},
                                  s2::AbstractVector{<:Real})::Float64
    _check_same_length_cmp(s1, s2)
    s1f = Float64.(collect(s1))
    s2f = Float64.(collect(s2))
    sum1 = sum(s1f)
    sum2 = sum(s2f)
    (sum1 < CMP_EPS || sum2 < CMP_EPS) && return 0.0
    n1 = s1f ./ sum1
    n2 = s2f ./ sum2
    nrm1 = norm(n1)
    nrm2 = norm(n2)
    (nrm1 < CMP_EPS || nrm2 < CMP_EPS) && return 0.0
    return dot(n1, n2) / (nrm1 * nrm2)
end

# ─── Xu et al. (NIMA 2026) ──────────────────────────────────────────────────

"""
    relative_flux_error(phi_true, phi_hat)

Relative fluence error `‖φ_true − φ̂‖₂ / ‖φ_true‖₂` (0 = exact,
≥1 = reconstruction of the same order as the reference).
"""
function relative_flux_error(phi_true::AbstractVector{<:Real},
                            phi_hat::AbstractVector{<:Real})::Float64
    _check_same_length_cmp(phi_true, phi_hat)
    p_true = Float64.(collect(phi_true))
    p_hat = Float64.(collect(phi_hat))
    denom = norm(p_true)
    denom == 0.0 && return norm(p_hat) == 0.0 ? 0.0 : 1.0
    return norm(p_true .- p_hat) / denom
end

"""
    comprehensive_score(phi_true, phi_hat)

Composite index of Xu et al. (NIMA 2026): `flux_err − 0.5·pearson_r`;
lower is better (minimum −0.5).
"""
function comprehensive_score(phi_true::AbstractVector{<:Real},
                            phi_hat::AbstractVector{<:Real})::Float64
    _check_same_length_cmp(phi_true, phi_hat)
    return relative_flux_error(phi_true, phi_hat) - 0.5 * pearson_r(phi_true, phi_hat)
end

# ─── EURADOS metrics with parameters ───────────────────────────────────────

"""
    fluence_difference_percent(s1, s2; energy_bins=nothing)

Percent difference of total fluence `100·(Σs2 − Σs1)/Σs1`
(`energy_bins` — bin widths for weighted summation).
"""
function fluence_difference_percent(s1::AbstractVector{<:Real},
                                   s2::AbstractVector{<:Real};
                                   energy_bins=nothing)::Float64
    _check_same_length_cmp(s1, s2)
    s1f = Float64.(collect(s1))
    s2f = Float64.(collect(s2))
    if energy_bins !== nothing
        bins = Float64.(collect(energy_bins))
        total1 = sum(s1f .* bins)
        total2 = sum(s2f .* bins)
    else
        total1 = sum(s1f)
        total2 = sum(s2f)
    end
    abs(total1) < CMP_EPS && return 0.0
    return 100.0 * (total2 - total1) / total1
end

"""
    energy_group_fluence_diff(s1, s2; thermal_max=0.4e-6, epithermal_max=0.1)

Percent differences of fluence by thermal/epithermal/fast groups. Returns
`Dict("thermal"=>..., "epithermal"=>..., "fast"=>...)`.
"""
function energy_group_fluence_diff(s1::AbstractVector{<:Real},
                                  s2::AbstractVector{<:Real},
                                  energy::AbstractVector{<:Real};
                                  thermal_max::Real=0.4e-6,
                                  epithermal_max::Real=0.1)
    _check_same_length_cmp(s1, s2)
    length(energy) == length(s1) || throw(ArgumentError(
        "Energy array must match spectrum length"))
    g1 = energy_group_fluence(s1, energy; thermal_max=thermal_max,
                              epithermal_max=epithermal_max)
    g2 = energy_group_fluence(s2, energy; thermal_max=thermal_max,
                              epithermal_max=epithermal_max)
    s1f = Float64.(collect(s1))
    s2f = Float64.(collect(s2))
    e = Float64.(collect(energy))
    result = Dict{String,Float64}()
    for (grp, mask) in (("thermal", e .< thermal_max),
                        ("epithermal", (e .>= thermal_max) .& (e .< epithermal_max)),
                        ("fast", e .>= epithermal_max))
        any(mask) || (result[grp] = 0.0; continue)
        t1 = sum(s1f[mask])
        t2 = sum(s2f[mask])
        result[grp] = abs(t1) < CMP_EPS ? 0.0 : 100.0 * (t2 - t1) / t1
    end
    return result
end

"""
    dose_difference_percent(s1, s2, energy; cc)

Percent difference of doses `Σ φ·cc·dlnE` (conversion coefficients for "AP").
"""
function dose_difference_percent(s1::AbstractVector{<:Real},
                                s2::AbstractVector{<:Real},
                                energy::AbstractVector{<:Real},
                                cc_icrp116=nothing)::Float64
    _check_same_length_cmp(s1, s2)
    length(energy) == length(s1) || throw(ArgumentError(
        "Energy array must match spectrum length"))
    e = Float64.(energy)
    cc = _extract_cc_array(cc_icrp116, e; preferred_geom="AP")
    ln_steps = _compute_log_steps(e)
    dose1 = sum(Float64.(s1) .* cc .* ln_steps)
    dose2 = sum(Float64.(s2) .* cc .* ln_steps)
    abs(dose1) < CMP_EPS && return 0.0
    return 100.0 * (dose2 - dose1) / dose1
end

"""
    fluence_averaged_energy_diff(s1, s2, energy)

Percent difference of fluence-averaged energies `⟨E⟩ = Σ E·φ/Σφ`.
"""
function fluence_averaged_energy_diff(s1::AbstractVector{<:Real},
                                     s2::AbstractVector{<:Real},
                                     energy::AbstractVector{<:Real})::Float64
    _check_same_length_cmp(s1, s2)
    length(energy) == length(s1) || throw(ArgumentError(
        "Energy array must match spectrum length"))
    e = Float64.(energy)
    s1f = Float64.(s1)
    s2f = Float64.(s2)
    e1 = sum(s1f) > 0 ? sum(e .* s1f) ./ sum(s1f) : 0.0
    e2 = sum(s2f) > 0 ? sum(e .* s2f) ./ sum(s2f) : 0.0
    abs(e1) < CMP_EPS && return 0.0
    return 100.0 * (e2 - e1) / e1
end

"""
    dose_averaged_energy_diff(s1, s2, energy; cc_icrp116)

Percent difference of dose-averaged energies `<E>_H`.
"""
function dose_averaged_energy_diff(s1::AbstractVector{<:Real},
                                  s2::AbstractVector{<:Real},
                                  energy::AbstractVector{<:Real},
                                  cc_icrp116=nothing)::Float64
    _check_same_length_cmp(s1, s2)
    length(energy) == length(s1) || throw(ArgumentError(
        "Energy array must match spectrum length"))
    e = Float64.(energy)
    cc = _extract_cc_array(cc_icrp116, e; preferred_geom="AP")
    ln_steps = _compute_log_steps(e)
    w1 = abs(sum(Float64.(s1) .* cc .* ln_steps))
    w2 = abs(sum(Float64.(s2) .* cc .* ln_steps))
    h1 = sum(Float64.(s1) .* e .* cc .* ln_steps) / w1
    h2 = sum(Float64.(s2) .* e .* cc .* ln_steps) / w2
    abs(h1) < CMP_EPS && return 0.0
    return 100.0 * (h2 - h1) / h1
end

"""
    log_lethargy_correlation(s1, s2, energy)

Pearson correlation of spectra in the lethargy representation log10(E)·φ(E).
"""
function log_lethargy_correlation(s1::AbstractVector{<:Real},
                                 s2::AbstractVector{<:Real},
                                 energy::AbstractVector{<:Real})::Float64
    _check_same_length_cmp(s1, s2)
    length(energy) == length(s1) || throw(ArgumentError(
        "Energy array must match spectrum length"))
    e = Float64.(energy)
    leth1 = log10.(e .+ 1e-15) .* Float64.(s1)
    leth2 = log10.(e .+ 1e-15) .* Float64.(s2)
    (std(leth1) == 0.0 || std(leth2) == 0.0) && return 0.0
    return pearson_r(leth1, leth2)
end

"""
    peak_location_error(s1, s2, energy)

Relative difference of fluence peak energies.
"""
function peak_location_error(s1::AbstractVector{<:Real},
                            s2::AbstractVector{<:Real},
                            energy::AbstractVector{<:Real})::Float64
    _check_same_length_cmp(s1, s2)
    length(energy) == length(s1) || throw(ArgumentError(
        "Energy array must match spectrum length"))
    e = Float64.(energy)
    e_peak1 = e[argmax(Float64.(s1))]
    e_peak2 = e[argmax(Float64.(s2))]
    abs(e_peak1) < CMP_EPS && return 0.0
    return 100.0 * (e_peak2 - e_peak1) / e_peak1
end

"""
    peak_width_error(s1, s2, energy)

Relative difference of FWHM (width at half maximum over energy bins).
"""
function peak_width_error(s1::AbstractVector{<:Real},
                         s2::AbstractVector{<:Real},
                         energy::AbstractVector{<:Real})::Float64
    _check_same_length_cmp(s1, s2)
    length(energy) == length(s1) || throw(ArgumentError(
        "Energy array must match spectrum length"))
    e = Float64.(energy)

    function fwhm_endpoints(spec)
        max_val = maximum(spec)
        max_val < CMP_EPS && return 0.0
        half = max_val / 2.0
        idx = findall(x -> x >= half, spec)
        isempty(idx) && return 0.0
        return Float64(e[last(idx)] - e[first(idx)])
    end

    f1 = fwhm_endpoints(Float64.(s1))
    f2 = fwhm_endpoints(Float64.(s2))
    abs(f1) < CMP_EPS && return 0.0
    return 100.0 * (f2 - f1) / f1
end

"""
    dose_weighted_error(s1, s2, energy; cc_icrp116)

Dose-weighted root mean square deviation:
`sqrt(Σ w_i·(s1−s2)² / Σ w_i)`, w = cc·dlnE.
"""
function dose_weighted_error(s1::AbstractVector{<:Real},
                            s2::AbstractVector{<:Real},
                            energy::AbstractVector{<:Real},
                            cc_icrp116=nothing)::Float64
    _check_same_length_cmp(s1, s2)
    length(energy) == length(s1) || throw(ArgumentError(
        "Energy array must match spectrum length"))
    e = Float64.(energy)
    cc = _extract_cc_array(cc_icrp116, e; preferred_geom="AP")
    ln_steps = _compute_log_steps(e)
    weights = cc .* ln_steps
    total_weight = sum(weights)
    total_weight < CMP_EPS && return 0.0
    weighted_mse = sum(weights .* (Float64.(s1) .- Float64.(s2)) .^ 2) / total_weight
    return sqrt(weighted_mse)
end

"""
    response_matrix_consistency(spectrum, readings, response_matrix)

χ² consistency: `Σ (R_meas − A·φ)² / R_meas` (only for R_meas > 0).
"""
function response_matrix_consistency(spectrum::AbstractVector{<:Real},
                                    readings::AbstractVector{<:Real},
                                    response_matrix::AbstractMatrix{<:Real})::Float64
    size(response_matrix, 1) == length(readings) || throw(ArgumentError(
        "readings length must match response matrix rows"))
    s = Float64.(collect(spectrum))
    r = Float64.(collect(readings))
    A = Float64.(response_matrix)
    r_computed = A * s
    mask = r .> CMP_EPS
    any(mask) || return 0.0
    return sum((r[mask] .- r_computed[mask]) .^ 2 ./ r[mask])
end

# ─── Single-spectrum integral quantities (EURADOS) ─────────────────────────

"""
    fluence_averaged_energy(spectrum, energy)

`⟨E⟩ = Σ E·φ / Σφ` (0 for a zero spectrum).
"""
function fluence_averaged_energy(spectrum::AbstractVector{<:Real},
                                energy::AbstractVector{<:Real})::Float64
    length(energy) == length(spectrum) || throw(ArgumentError(
        "Energy array must match spectrum length"))
    s = Float64.(collect(spectrum))
    total = sum(s)
    total <= 0.0 && return 0.0
    return sum(Float64.(energy) .* s) / total
end

"""
    energy_group_fluence(spectrum, energy; thermal_max=0.4e-6, epithermal_max=0.1)

Fluence sums by group: thermal (< `thermal_max` = 0.4 eV),
epithermal, fast (>= `epithermal_max` = 0.1 MeV).
"""
function energy_group_fluence(spectrum::AbstractVector{<:Real},
                             energy::AbstractVector{<:Real};
                             thermal_max::Real=0.4e-6,
                             epithermal_max::Real=0.1)::Dict{String,Float64}
    length(energy) == length(spectrum) || throw(ArgumentError(
        "Energy array must match spectrum length"))
    s = Float64.(collect(spectrum))
    e = Float64.(collect(energy))
    result = Dict{String,Float64}()
    mask_thermal = e .< thermal_max
    mask_epi = (e .>= thermal_max) .& (e .< epithermal_max)
    mask_fast = e .>= epithermal_max
    result["thermal"] = any(mask_thermal) ? sum(s[mask_thermal]) : 0.0
    result["epithermal"] = any(mask_epi) ? sum(s[mask_epi]) : 0.0
    result["fast"] = any(mask_fast) ? sum(s[mask_fast]) : 0.0
    return result
end

"""
    dose_averaged_energy(spectrum, energy; cc_ade)

`<E>_H = Σ E·H·φ / Σ H·φ` with ICRP-74 ADE coefficients.
"""
function dose_averaged_energy(spectrum::AbstractVector{<:Real},
                             energy::AbstractVector{<:Real},
                             cc_ade=nothing)::Float64
    length(energy) == length(spectrum) || throw(ArgumentError(
        "Energy array must match spectrum length"))
    s = Float64.(collect(spectrum))
    e = Float64.(energy)
    cc = _get_ade_cc(e, cc_ade)
    ln_steps = _compute_log_steps(e)
    weight = sum(s .* cc .* ln_steps)
    weight <= 0 && return 0.0
    return sum(e .* s .* cc .* ln_steps) / weight
end

"""
    ambient_dose_equivalent_rate(spectrum, energy; cc_ade)

Ambient dose equivalent rate H*(10) (pSv/s):
`Σ h*(10)·φ·dlnE`, ICRP-74 ADE.
"""
function ambient_dose_equivalent_rate(spectrum::AbstractVector{<:Real},
                                     energy::AbstractVector{<:Real},
                                     cc_ade=nothing)::Float64
    length(energy) == length(spectrum) || throw(ArgumentError(
        "Energy array must match spectrum length"))
    s = Float64.(collect(spectrum))
    cc = _get_ade_cc(Float64.(energy), cc_ade)
    return sum(s .* cc .* _compute_log_steps(Float64.(energy)))
end


# ─── Metric registries (port of _METRIC_FUNCTIONS / _WITH_PARAMS / _SINGLE) ──

const METRIC_FUNCTIONS = Dict{String,Function}(
    "kl_divergence"                  => kl_divergence,
    "cross_entropy"                  => cross_entropy,
    "entropy_difference_percent"     => entropy_difference_percent,
    "wasserstein_dist"               => wasserstein_dist,
    "energy_dist"                    => energy_dist,
    "kolmogorov_smirnov_stat"        => kolmogorov_smirnov_stat,
    "pearson_r"                      => pearson_r,
    "spearman_r"                     => spearman_r,
    "mean_squared_error"             => mean_squared_error,
    "root_mean_squared_error"        => root_mean_squared_error,
    "mean_absolute_error"            => mean_absolute_error,
    "mape"                           => mape,
    "r2_score"                       => r2_score,
    "max_error"                      => max_error,
    "median_absolute_error"          => median_absolute_error,
    "cosine_similarity"              => cosine_similarity,
    "total_flux_ratio"               => total_flux_ratio,
    "chi_squared"                    => chi_squared,
    "g_test"                         => g_test,
    "freeman_tukey"                  => freeman_tukey,
    "cressie_read"                   => cressie_read,
    "anderson_darling"               => anderson_darling,
    "standardized_mean_difference"   => standardized_mean_difference,
    "wilcoxon_test"                  => wilcoxon_test,
    "mannwhitneyu_test"              => mannwhitneyu_test,
    "spectral_shape_similarity"      => spectral_shape_similarity,
    "mmd_rbf"                        => (a, b) -> mmd_rbf(a, b),
    # Xu et al. (NIMA 2026)
    "relative_flux_error"            => relative_flux_error,
    "comprehensive_score"            => comprehensive_score,
)

const METRIC_FUNCTIONS_WITH_PARAMS = Dict{String,Function}(
    "fluence_difference_percent"    => (s1, s2) -> fluence_difference_percent(s1, s2),
    "fluence_averaged_energy_diff"  => (s1, s2, e) -> fluence_averaged_energy_diff(s1, s2, e),
    "log_lethargy_correlation"      => log_lethargy_correlation,
    "peak_location_error"           => peak_location_error,
    "peak_width_error"              => peak_width_error,
    "dose_difference_percent"       => (s1, s2, e, cc) -> dose_difference_percent(s1, s2, e, cc),
    "dose_averaged_energy_diff"     => (s1, s2, e, cc) -> dose_averaged_energy_diff(s1, s2, e, cc),
    "dose_weighted_error"           => (s1, s2, e, cc) -> dose_weighted_error(s1, s2, e, cc),
)

const SINGLE_SPECTRUM_METRICS = Dict{String,Function}(
    "fluence_averaged_energy"      => (s, e) -> fluence_averaged_energy(s, e),
    "dose_averaged_energy"         => (s, e) -> dose_averaged_energy(s, e),
    "ambient_dose_equivalent_rate" => (s, e) -> ambient_dose_equivalent_rate(s, e),
    "energy_group_fluence"         => (s, e) -> energy_group_fluence(s, e),
)

"""
    available_metrics() -> (simple, with_params, single)

Descriptions of the available metric groups.
"""
function available_metrics()
    return (collect(keys(METRIC_FUNCTIONS)),
            collect(keys(METRIC_FUNCTIONS_WITH_PARAMS)),
            collect(keys(SINGLE_SPECTRUM_METRICS)))
end

# ─── compare_spectra / compare_multiple ─────────────────────────────────────

"""
    compare_spectra(spectrum1, spectrum2; metrics=nothing, energy=nothing,
                    cc_icrp116=nothing, readings1=nothing, readings2=nothing,
                    response_matrix=nothing)

Compare two spectra with the selected metrics.

# Arguments
- `spectrum1, spectrum2::Vector`: 1-D spectra of equal length
- `metrics`: name, list of names, or `nothing` (all simple; when `energy`
  is given, EURADOS metrics and single-spectrum metrics are added automatically)
- `energy`: energy grid, MeV (for metrics with parameters)
- `cc_icrp116`: coefficients (Dict/Vector; default AP from ICRP-116)
- `readings1/readings2`, `response_matrix`: for `response_matrix_consistency`

# Returns
`Dict{String,Float64}` — metric name → value.
"""
function compare_spectra(spectrum1::AbstractVector{<:Real},
                         spectrum2::AbstractVector{<:Real};
                         metrics=nothing,
                         energy=nothing,
                         cc_icrp116=nothing,
                         readings1=nothing,
                         readings2=nothing,
                         response_matrix=nothing)::Dict{String,Float64}
    _check_same_length_cmp(spectrum1, spectrum2)
    all_simple = collect(keys(METRIC_FUNCTIONS))
    all_eurados = vcat(collect(keys(METRIC_FUNCTIONS_WITH_PARAMS)), "energy_group_fluence_diff")
    all_single = collect(keys(SINGLE_SPECTRUM_METRICS))

    simple_keys, eurados_keys, single_keys = if metrics === nothing
        (all_simple,
         energy === nothing ? String[] : all_eurados,
         energy === nothing ? String[] : all_single)
    elseif metrics isa AbstractString
        if haskey(METRIC_FUNCTIONS, metrics)
            ([metrics], String[], String[])
        elseif haskey(METRIC_FUNCTIONS_WITH_PARAMS, metrics) ||
               metrics == "energy_group_fluence_diff"
            (String[], [metrics], String[])
        elseif haskey(SINGLE_SPECTRUM_METRICS, metrics)
            (String[], String[], [metrics])
        else
            throw(ArgumentError(
                "Unknown metric '$metrics'. Available: $(vcat(all_simple, all_eurados, all_single))"))
        end
    elseif metrics isa AbstractVector
        sk = String[m for m in metrics if haskey(METRIC_FUNCTIONS, m)]
        ek = String[m for m in metrics if haskey(METRIC_FUNCTIONS_WITH_PARAMS, m) ||
                                        m == "energy_group_fluence_diff"]
        skk = String[m for m in metrics if haskey(SINGLE_SPECTRUM_METRICS, m)]
        unknown = String[m for m in metrics if !(m in vcat(sk, ek, skk))]
        if !isempty(unknown)
            throw(ArgumentError(
                "Unknown metric(s) $unknown. Available: $(vcat(all_simple, all_eurados, all_single))"))
        end
        (sk, ek, skk)
    else
        throw(ArgumentError("metrics must be nothing, String or Vector{String}"))
    end

    results = Dict{String,Float64}()

    # Simple metrics
    for key in simple_keys
        try
            results[key] = METRIC_FUNCTIONS[key](spectrum1, spectrum2)
        catch err
            @debug "Metric $key failed" err
            results[key] = NaN
        end
    end

    # Metrics with parameters (EURADOS)
    for key in eurados_keys
        if energy === nothing
            results[key] = NaN
            continue
        end
        try
            if key == "fluence_difference_percent"
                results[key] = fluence_difference_percent(spectrum1, spectrum2)
            elseif key == "energy_group_fluence_diff"
                groups = energy_group_fluence_diff(spectrum1, spectrum2, energy)
                for (grp, val) in groups
                    results["energy_group_fluence_diff_$grp"] = val
                end
            elseif key == "log_lethargy_correlation"
                results[key] = log_lethargy_correlation(spectrum1, spectrum2, energy)
            elseif key == "peak_location_error"
                results[key] = peak_location_error(spectrum1, spectrum2, energy)
            elseif key == "peak_width_error"
                results[key] = peak_width_error(spectrum1, spectrum2, energy)
            elseif key == "fluence_averaged_energy_diff"
                results[key] = fluence_averaged_energy_diff(spectrum1, spectrum2, energy)
            elseif key == "response_matrix_consistency"
                continue
            else
                fn = METRIC_FUNCTIONS_WITH_PARAMS[key]
                results[key] = fn(spectrum1, spectrum2, energy, cc_icrp116)
            end
        catch err
            @debug "EURADOS metric $key failed" err
            results[key] = NaN
        end
    end
    if (metrics === nothing || (metrics isa AbstractVector &&
        "response_matrix_consistency" in metrics)) &&
       energy !== nothing && readings1 !== nothing && response_matrix !== nothing
        results["response_matrix_consistency_ref"] =
            response_matrix_consistency(spectrum1, readings1, response_matrix)
        results["response_matrix_consistency_test"] =
            response_matrix_consistency(spectrum2,
                readings2 === nothing ? readings1 : readings2, response_matrix)
    end

    # Single-spectrum quantities (EURADOS)
    for key in single_keys
        if energy === nothing
            if key == "energy_group_fluence"
                for grp in ("thermal", "epithermal", "fast")
                    results["energy_group_fluence_$(grp)_ref"] = NaN
                    results["energy_group_fluence_$(grp)_test"] = NaN
                end
            else
                results["$(key)_ref"] = NaN
                results["$(key)_test"] = NaN
            end
            continue
        end
        try
            fn = SINGLE_SPECTRUM_METRICS[key]
            if key == "energy_group_fluence"
                g_ref = energy_group_fluence(spectrum1, energy)
                g_test = energy_group_fluence(spectrum2, energy)
                for (grp, val) in g_ref
                    results["energy_group_fluence_$(grp)_ref"] = val
                end
                for (grp, val) in g_test
                    results["energy_group_fluence_$(grp)_test"] = val
                end
            elseif key == "dose_averaged_energy"
                results["dose_averaged_energy_ref"] =
                    dose_averaged_energy(spectrum1, energy)
                results["dose_averaged_energy_test"] =
                    dose_averaged_energy(spectrum2, energy)
            else
                results["$(key)_ref"] = fn(spectrum1, energy)
                results["$(key)_test"] = fn(spectrum2, energy)
            end
        catch err
            @debug "Integral metric $key failed" err
            results["$(key)_ref"] = NaN
            results["$(key)_test"] = NaN
        end
    end

    return results
end

"""
    compare_multiple(spectra; metrics=nothing, labels=nothing)

Compare a list of spectra (the first is the reference) against the rest.

# Returns
`Dict{String,Dict{String,Float64}}` — "key A vs B" → metrics.
"""
function compare_multiple(spectra::Vector{<:AbstractVector{<:Real}};
                          metrics=nothing, labels=nothing)::Dict{String,Dict{String,Float64}}
    n = length(spectra)
    n >= 2 || throw(ArgumentError("At least two spectra required for comparison"))
    labels = labels === nothing ? ["Spectrum $(i-1)" for i in 1:n] : labels
    length(labels) == n || throw(ArgumentError(
        "Number of labels must match number of spectra"))
    ref = spectra[1]
    results = Dict{String,Dict{String,Float64}}()
    for i in 2:n
        results["$(labels[1]) vs $(labels[i])"] =
            compare_spectra(ref, spectra[i]; metrics=metrics)
    end
    return results
end


# ─── Benchmark harness (port of benchmark_unfold_methods) ──────────────────

"Metric tokens ranked in ascending order (lower is better)."
const _ASCENDING_TOKENS = ("error", "rmse", "mape", "kl_", "wasserstein", "max_error")

"""
    default_unfold_benchmark_methods()

Default registry `{name => (Detector-method name, [NamedTuple of parameters])}`
(port of DEFAULT_UNFOLD_BENCHMARK_METHODS; tsvd/cvxpy — kwargs from the Julia API).
"""
function default_unfold_benchmark_methods()
    km = Dict{String,Tuple{String,Vector{NamedTuple}}}()
    iter = [(max_iterations=10,), (max_iterations=50,), (max_iterations=100,)]
    half = [(max_iterations=10,), (max_iterations=50,)]
    km["mlem"]        = ("unfold_mlem", iter)
    km["maxed"]       = ("unfold_maxed", [(sigma_factor=0.01,), (sigma_factor=0.05,), (sigma_factor=0.1,)])
    km["tsvd"]        = ("unfold_tsvd", [(truncation_rank=3,), (truncation_rank=5,), (truncation_rank=7,)])
    km["bayes"]       = ("unfold_bayes", [(max_iterations=100,), (max_iterations=500,)])
    km["cvxpy"]       = ("unfold_cvxpy", [(regularization=0.001,), (regularization=0.01,)])
    km["landweber"]   = ("unfold_landweber", half)
    km["sart"]        = ("unfold_sart", half)
    km["kaczmarz"]    = ("unfold_kaczmarz", half)
    km["cgls"]        = ("unfold_cgls", half)
    km["lanczos"]     = ("unfold_lanczos", half)
    km["gravel"]      = ("unfold_gravel", half)
    km["doroshenko"]  = ("unfold_doroshenko", half)
    km["bunki"]       = ("unfold_bunki", half)
    km["sandii"]      = ("unfold_sandii", half)
    km["osem"]        = ("unfold_osem", half)
    km["fista"]       = ("unfold_fista", half)
    return km
end

"Default ranking metrics (port of DEFAULT_UNFOLD_BENCHMARK_METRICS)."
const DEFAULT_UNFOLD_BENCHMARK_METRICS = [
    "r2_score",
    "pearson_r",
    "root_mean_squared_error",
    "mape",
    "wasserstein_dist",
    "kl_divergence",
    "cosine_similarity",
    "relative_flux_error",
    "comprehensive_score",
]

"""
    BenchmarkResult

Container of the `benchmark_unfold_methods` result:
- `results::Vector{Dict{String,Any}}` — one row per (spectrum, method, parameters)
- `summary::Vector{Dict{String,Any}}` — mean/std of metrics per method
- `ranking::Vector{Dict{String,Any}}` — methods sorted by `rank_by`
- `report::String` — text summary
"""
struct BenchmarkResult
    results::Vector{Dict{String,Any}}
    summary::Vector{Dict{String,Any}}
    ranking::Vector{Dict{String,Any}}
    report::String
end

"""
    benchmark_unfold_methods(detector, E_ref, reference_spectra;
                             methods=nothing, metrics=nothing,
                             spectrum_names=nothing, rank_by="r2_score",
                             progress=false)

Run a registry of unfolding methods over a set of reference spectra.

# Arguments
- `detector::Detector`
- `E_ref::Vector{Float64}`: reference grid, MeV
- `reference_spectra::Dict{String,Vector{Float64}}`: `{name: φ(E)}`
- `methods::Dict`: `{name => (unfold_name, [param_NamedTuple...])}`
- `metrics::Union{Nothing,Vector{String}}`: `compare_spectra` keys
- `rank_by`: metric to sort by; error-class metrics use ascending order

# Returns
`BenchmarkResult`.
"""
function benchmark_unfold_methods(detector, E_ref::Vector{Float64},
                                  reference_spectra::Dict{String,<:Vector{<:Real}};
                                  methods=nothing,
                                  metrics::Union{Nothing,Vector{String}}=nothing,
                                  spectrum_names::Union{Nothing,Vector{String}}=nothing,
                                  rank_by::AbstractString="r2_score",
                                  progress::Bool=false)::BenchmarkResult
    methods === nothing && (methods = default_unfold_benchmark_methods())
    metrics === nothing && (metrics = copy(DEFAULT_UNFOLD_BENCHMARK_METRICS))
    rank_by in metrics || throw(ArgumentError(
        "rank_by='$rank_by' is not in the requested metrics $metrics"))
    isempty(methods) && throw(ArgumentError("No methods configured"))
    names = spectrum_names === nothing ? collect(keys(reference_spectra)) :
            [n for n in spectrum_names if haskey(reference_spectra, n)]
    isempty(names) && throw(ArgumentError("No reference spectra selected"))

    E_det = detector.config.E_MeV
    cc_det = detector.config.cc_icrp116
    ascending = any(t -> occursin(t, rank_by), _ASCENDING_TOKENS)

    rows = Vector{Dict{String,Any}}()
    for spec_name in names
        ref_spec = Float64.(collect(reference_spectra[spec_name]))
        φ_on_det = if length(E_det) == length(E_ref) &&
                      isapprox(E_det, E_ref; rtol=1e-12)
            Float64.(ref_spec)
        else
            interpolate_spectrum(ref_spec, E_ref, E_det)
        end
        readings = get_effective_readings_for_spectra(detector, E_ref, ref_spec)
        for (mname, (uname, params_list)) in pairs(methods)
            isdefined(BSSUnfold, Symbol(uname)) || continue
            fn = getfield(BSSUnfold, Symbol(uname))
            for params in params_list
                row = Dict{String,Any}(
                    "method" => mname,
                    "params" => _params_to_str(params),
                    "spectrum" => spec_name,
                    "success" => false,
                    "time_sec" => NaN,
                    "error" => "",
                )
                t0 = time()
                try
                    res_fn = fn(detector, readings; params...)
                    row["time_sec"] = time() - t0
                    vals = compare_spectra(φ_on_det, res_fn["spectrum"];
                                           metrics=metrics,
                                           energy=E_det, cc_icrp116=cc_det)
                    merge!(row, vals)
                    row["success"] = true
                catch err
                    row["time_sec"] = time() - t0
                    row["error"] = sprint(showerror, err)
                end
                push!(rows, row)
                progress && println("  $spec_name/$mname: ",
                                    row["success"] ? "ok" : "FAIL",
                                    " ($(round(row["time_sec"]; digits=2))s)")
            end
        end
    end

    summary_rows = _benchmark_summarize(rows, metrics)
    ranking = _rank_benchmark(summary_rows, metrics, rank_by, ascending)
    report = _benchmark_report(rows, summary_rows, metrics)
    return BenchmarkResult(rows, summary_rows, ranking, report)
end

"Deterministic parameter string `k=v, ...`."
function _params_to_str(params)
    return join(["$k=$v" for (k, v) in pairs(params)], ", ")
end

"Mean/std of metrics per method (port of _summarize)."
function _benchmark_summarize(rows::Vector{Dict{String,Any}},
                              metric_cols::Vector{String})::Vector{Dict{String,Any}}
    methods_seen = String[]
    for r in rows
        !haskey(r, "method") && continue
        r["method"] in methods_seen || push!(methods_seen, r["method"])
    end
    summary = Vector{Dict{String,Any}}()
    for meth in methods_seen
        row = Dict{String,Any}("method" => meth)
        for col in metric_cols
            vals = Float64[r[col] for r in rows
                           if r["method"] == meth && r["success"] && haskey(r, col) &&
                              isfinite(r[col])]
            row["$(col)_mean"] = isempty(vals) ? NaN : mean(vals)
            row["$(col)_std"] = isempty(vals) ? NaN :
                               (length(vals) > 1 ? std(vals) : 0.0)
        end
        push!(summary, row)
    end
    return summary
end

"Sort the summary: error metrics ascending (port of _rank)."
function _rank_benchmark(summary::Vector{Dict{String,Any}}, metric_cols,
                         rank_by::AbstractString, ascending::Bool)::Vector{Dict{String,Any}}
    rank_col = "$(rank_by)_mean"
    any(r -> haskey(r, rank_col), summary) || return summary
    key_of(r) = get(r, rank_col, NaN)
    valid = filter(r -> isfinite(key_of(r)), summary)
    invalid = filter(r -> !isfinite(key_of(r)), summary)
    return vcat(sort(valid; by=r -> key_of(r), rev=!ascending), invalid)
end

"Text summary (port of _build_report, without pandas)."
function _benchmark_report(rows::Vector{Dict{String,Any}},
                           summary::Vector{Dict{String,Any}},
                           metrics::Vector{String})::String
    n_ok = count(r -> get(r, "success", false), rows)
    ok_rate = isempty(rows) ? 0.0 : 100.0 * n_ok / length(rows)
    lines = [
        "=" ^ 60,
        "Benchmark: $(length(rows)) runs ($(n_ok) ok, $(length(rows) - n_ok) fail), ok rate $(round(ok_rate; digits=1))%",
        "",
        "Summary by method:",
    ]
    for r in summary
        parts = ["$(r["method"]):"]
        for col in metrics
            hm = get(r, "$(col)_mean", NaN)
            isfinite(hm) && push!(parts, "$col=$(round(hm; sigdigits=4))")
        end
        push!(lines, "  " * join(parts, "  "))
    end
    return join(lines, "\n")
end
