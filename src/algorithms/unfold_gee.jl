"""
GEE (Generalized Estimation Equation) unfolding with robust inference.

Port of `bssunfold/core/unfold_gee.py` — a Python analogue of the R package
`gee` 4.13-30 (Carey, Lumley & Ripley; Liang & Zeger 1986 quasi-score): the
unfolded spectrum comes from the *generalized estimating equations*

    U(x) = A' R(alpha)^-1 (b - A x) - lam * G x = 0,

where the `m` detector spheres are a correlated cluster of repeated
measurements with an `m x m` working correlation `R(alpha)`:

* `"independence"` — `R = I`;
* `"exchangeable"` — `R_ij = alpha` for `i != j` (the R default corstr), with
  the Liang-Zeger moment estimator of `alpha` from the Pearson residuals;
* `"ar1"` — `R_ij = alpha^|i-j|` estimated from the lag-1 products of the
  Pearson residuals.

The families are quasi-likelihood variance functions with the identity link on
`A xp` (`gaussian` `v(mu) = 1`, `poisson` `v(mu) = mu`, `gamma`
`v(mu) = mu^2`).  Because `n > m`, the score equations are not unique and are
ridged on the `diff_order`-th difference roughness penalty `G = D'D`, scaled
relative to the mean diagonal of `A' R^-1 A`.

Inference follows the two Liang-Zeger sandwich estimators: the robust (empirical
meat) covariance and the naive (model-based) covariance, both reported per bin
as square roots of the diagonal.  The iteration is the standard GEE/IRLS loop —
update `x` from the current working correlation, re-estimate `alpha` and the
dispersion `phi` from the Pearson residuals, repeat until the relative change of
the spectrum falls below `tolerance` or the projected (non-negativity clipped)
fixed point stalls.

Module API (mirrors the Python `__all__`):

* `working_correlation(alpha, m, kind)` — build the `R` matrix;
* `estimate_alpha(r_pearson, kind)` — moment estimators of `(alpha, phi)`;
* `gee_fit(A, b, x0; ...)` — core GEE/IRLS loop returning the rich diagnostics
  `Dict` (spectrum, robust/naive SE, `alpha`, `phi`, Pearson chi-square,
  iterations, convergence);
* `solve_gee(A, b, x0; ...)` — solver returning `UnfoldResult`;
* `solve_gee_full` — alias of `gee_fit`, the same solver with diagnostics.

The GEE solution is unconstrained (like R `gee`); the returned spectrum is
clipped at zero, matching the convention of the other solvers.
"""

const FAMILIES = ("gaussian", "poisson", "gamma")
const CORSTRINGS = ("independence", "exchangeable", "ar1")

const _GEE_TINY = 1e-300

function _gee_validate_family_corstr(family::Union{Symbol,String},
                                     corstr::Union{Symbol,String})
    fam = lowercase(String(family))
    cor = lowercase(String(corstr))
    fam in FAMILIES ||
        throw(ArgumentError("family must be one of $FAMILIES, got \"$fam\""))
    cor in CORSTRINGS ||
        throw(ArgumentError("corstr must be one of $CORSTRINGS, got \"$cor\""))
    return fam, cor
end

"""
    working_correlation(alpha, m, kind="exchangeable") -> Matrix{Float64}

Build the `m x m` working correlation matrix `R(alpha)`: `"exchangeable"`
(`R_ii = 1`, `R_ij = alpha`), `"ar1"` (`R_ij = alpha^|i-j|`) or
`"independence"` (identity, which says nothing about `alpha`).
"""
function working_correlation(alpha::Real, m::Integer,
                             kind::Union{Symbol,String}="exchangeable")
    av = Float64(alpha)
    k = lowercase(String(kind))
    m >= 1 || throw(ArgumentError("m must be a positive integer, got $m"))
    if k == "independence"
        return Matrix{Float64}(I, m, m)
    elseif k == "exchangeable"
        lo = -1.0 / (m - 1)
        (lo < av < 1.0) || throw(ArgumentError(
            "exchangeable alpha must be in ($lo, 1) for m=$m, got $av"))
        R = fill(av, m, m)
        for i in 1:m
            R[i, i] = 1.0
        end
        return R
    elseif k == "ar1"
        (-1.0 < av < 1.0) || throw(ArgumentError(
            "ar1 alpha must be in (-1, 1), got $av"))
        R = Matrix{Float64}(undef, m, m)
        for j in 1:m, i in 1:m
            R[i, j] = av ^ abs(i - j)
        end
        return R
    end
    throw(ArgumentError("unknown working correlation kind \"$k\""))
end

"""
    estimate_alpha(r_pearson, kind="exchangeable") -> Tuple{Float64, Float64}

Moment estimates of the working correlation parameter `alpha` and the
dispersion `phi` from the Pearson residuals (Liang & Zeger 1986 Eqs. 6-7): the
off-diagonal products of the standardised residuals estimate the intra-cluster
correlation, the mean squared standardised residual the dispersion.  For
`"independence"` `alpha` is 0; the returned `alpha` is kept inside the SPD
region of the working matrix.
"""
function estimate_alpha(r_pearson::AbstractVector{<:Real},
                        kind::Union{Symbol,String}="exchangeable")
    r = Vector{Float64}(r_pearson)
    m = length(r)
    m < 2 && return (0.0, 1.0)
    phi = dot(r, r) / m
    k = lowercase(String(kind))
    if k == "independence"
        return (0.0, phi)
    elseif k == "exchangeable"
        num = 0.0
        for i in 1:m, j in 1:m
            i == j && continue
            num += r[i] * r[j]
        end
        den = Float64(m * (m - 1))
        den <= _GEE_TINY && return (0.0, phi)
        a = num / den
    elseif k == "ar1"
        num = dot(view(r, 2:m), view(r, 1:m-1))
        den = dot(view(r, 1:m-1), view(r, 1:m-1))
        den <= _GEE_TINY && return (0.0, phi)
        a = num / den
    else
        throw(ArgumentError("unknown working correlation kind \"$k\""))
    end
    lo = -0.95 / max(m - 1, 1)
    return (clamp(a, lo, 0.95), phi)
end

function _gee_variance_mu(mu::AbstractVector{T}, family::AbstractString) where T<:AbstractFloat
    if family == "gaussian"
        return ones(T, length(mu))
    elseif family == "poisson"
        return max.(mu, T(_GEE_TINY))
    elseif family == "gamma"
        return max.(mu .* mu, T(_GEE_TINY))
    end
    throw(ArgumentError("unknown family \"$family\""))
end

function _gee_inv(M::AbstractMatrix{T}) where T<:AbstractFloat
    try
        return inv(M)
    catch err
        if err isa Union{LinearAlgebra.SingularException,
                         LinearAlgebra.NoPivotException,
                         LinearAlgebra.LAPACKException}
            return pinv(M)
        end
        rethrow(err)
    end
end

function _gee_solve(M::AbstractMatrix{T}, rhs::AbstractVector{T}) where T<:AbstractFloat
    try
        return Vector{T}(M \ rhs)
    catch err
        if err isa Union{LinearAlgebra.SingularException,
                         LinearAlgebra.NoPivotException,
                         LinearAlgebra.LAPACKException}
            return Vector{T}(pinv(M) * rhs)
        end
        rethrow(err)
    end
end

"""
    gee_fit(A, b, x0=nothing; family="gaussian", corstr="exchangeable",
            regularization=1e-4, max_iterations=100, tolerance=1e-6,
            diff_order=2) -> Dict{String,Any}

Fit the unfolding model by generalized estimating equations and return the rich
diagnostics: `spectrum`, `cov_robust`, `cov_naive`, `robust_se`, `naive_se`,
`alpha`, `phi`, `residuals`, `pearson_residuals`, `pearson_chi2`, `df`,
`iterations`, `converged`, `family`, `corstr`.

`regularization` is the relative ridge on the `diff_order` difference penalty
(`0` disables it); `x0 === nothing` selects `zeros(n)` for the gaussian family
and `ones(n)` for the multiplicative families.
"""
function gee_fit(A::AbstractMatrix{T}, b::AbstractVector{T},
                 x0::Union{Nothing,AbstractVector{T}}=nothing;
                 family::Union{Symbol,String}="gaussian",
                 corstr::Union{Symbol,String}="exchangeable",
                 regularization::Real=T(1e-4),
                 max_iterations::Integer=100,
                 tolerance::Real=T(1e-6),
                 diff_order::Integer=2) where T<:AbstractFloat
    fam, cor = _gee_validate_family_corstr(family, corstr)
    A, b, x0v = validate_system(A, b; x0=x0, max_iterations=max_iterations,
                                tolerance=tolerance)
    m, n = size(A)
    if !any(!iszero, A)
        throw(ArgumentError("A must contain at least one non-zero entry"))
    end
    reg = T(regularization)
    reg >= zero(T) || throw(ArgumentError(
        "regularization must be non-negative, got $reg"))
    tol = T(tolerance)
    doff = Int(diff_order)
    if reg > zero(T) && !(doff == 1 || doff == 2)
        throw(ArgumentError(
            "diff_order must be 1 or 2 when regularization is active, got $doff"))
    end

    x = if x0v === nothing
        (fam == "poisson" || fam == "gamma") ? ones(T, n) : zeros(T, n)
    else
        max.(Vector{T}(x0v), zero(T))
    end

    G = if reg > zero(T) && doff > 0 && n > doff
        D = create_derivative_matrix(T, n, doff)
        Graw = D' * D
        Graw ./ max(mean(diag(Graw)), one(T))
    else
        zeros(T, n, n)
    end

    alpha = zero(T)
    phi = one(T)
    converged = false
    it = 0
    delta_prev = T(Inf)
    for k in 1:Int(max_iterations)
        it = k
        mu = max.(A * x, T(_GEE_TINY))
        v = _gee_variance_mu(mu, fam)
        r_pear = (b .- mu) ./ sqrt.(v)
        a_np, p_np = estimate_alpha(r_pear, cor)
        alpha_new, phi_new = T(a_np), T(p_np)

        R = Matrix{T}(working_correlation(alpha_new, m, cor))
        AW = A' * _gee_inv(R)
        H = AW * A
        damp = reg * max(mean(diag(H)), one(T))
        x_new = _gee_solve(H .+ damp .* G, AW * b)
        if !all(isfinite, x_new)
            converged = false
            break
        end

        delta = norm(x_new .- x) / max(norm(x_new), norm(x), one(T))
        x = max.(x_new, zero(T))
        alpha, phi = alpha_new, phi_new
        stalled = delta >= delta_prev * (one(T) - T(1e-12)) && delta < T(1e10)
        if delta <= tol || (stalled && it > 1)
            converged = true
            break
        end
        delta_prev = delta
    end

    mu = max.(A * x, T(_GEE_TINY))
    v = _gee_variance_mu(mu, fam)
    r_raw = b .- mu
    r_pear = r_raw ./ sqrt.(v)
    R = Matrix{T}(working_correlation(alpha, m, cor))
    AW = A' * _gee_inv(R)
    H = AW * A
    damp = reg * max(mean(diag(H)), one(T))
    Ninv = _gee_inv(H .+ damp .* G)

    grad_sq = max.(r_pear .^ 2, zero(T))
    meat = A' * (A .* reshape(grad_sq, :, 1))
    cov_robust = (Ninv * meat) * Ninv
    cov_naive = (Ninv * H) * Ninv
    cov_robust = T(0.5) .* (cov_robust .+ cov_robust')
    cov_naive = T(0.5) .* (cov_naive .+ cov_naive')

    robust_se = sqrt.(max.(diag(cov_robust), zero(T)))
    naive_se = sqrt.(max.(diag(cov_naive), zero(T)))
    pearson_chi2 = sum(r_pear .^ 2)
    df = max(m - n, 1)

    return Dict{String,Any}(
        "spectrum" => copy(x),
        "cov_robust" => cov_robust,
        "cov_naive" => cov_naive,
        "robust_se" => robust_se,
        "naive_se" => naive_se,
        "alpha" => alpha,
        "phi" => phi,
        "residuals" => copy(r_raw),
        "pearson_residuals" => copy(r_pear),
        "pearson_chi2" => pearson_chi2,
        "df" => df,
        "iterations" => it,
        "converged" => converged,
        "family" => fam,
        "corstr" => cor,
        "regularization" => reg,
        "max_iterations" => Int(max_iterations),
        "tolerance" => tol,
        "diff_order" => doff)
end

"""
    solve_gee(A, b, x0; family="gaussian", corstr="exchangeable",
              regularization=1e-4, max_iterations=100, tolerance=1e-6)
         -> UnfoldResult

Solve the unfolding problem by generalized estimating equations (see `gee_fit`)
and return `UnfoldResult`; the GEE diagnostics (`alpha`, `phi`, `family`,
`corstr`, `robust_se`, `naive_se`, `pearson_chi2`, ...) are reported on
`result.extra`.
"""
function solve_gee(A::AbstractMatrix{T}, b::AbstractVector{T},
                   x0::Union{Nothing,AbstractVector{T}};
                   family::Union{Symbol,String}="gaussian",
                   corstr::Union{Symbol,String}="exchangeable",
                   regularization::Real=T(1e-4),
                   max_iterations::Integer=100,
                   tolerance::Real=T(1e-6),
                   diff_order::Integer=2) where T<:AbstractFloat
    diag = gee_fit(A, b, x0; family=family, corstr=corstr,
                   regularization=regularization,
                   max_iterations=max_iterations, tolerance=tolerance,
                   diff_order=diff_order)
    spectrum = diag["spectrum"]
    extra = Dict{String,Any}((k, v) for (k, v) in diag if k != "spectrum")
    extra["spectrum_uncert_robust"] = copy(diag["robust_se"])
    extra["gee_converged"] = diag["converged"]
    return UnfoldResult(spectrum, diag["iterations"], diag["converged"],
                        norm(b .- A * spectrum), extra)
end

const solve_gee_full = gee_fit
