"""
P-spline mixed-model unfolding with REML smoothing selection.

Port of `bssunfold/core/unfold_pspline_reml.py` — a Python analogue of the
R package `LMMsolver` (Boer 2023): the unfolded spectrum is represented as
a P-spline (penalised B-spline) `x(E) = B(E) * c` and the smoothness is
selected automatically by restricted maximum likelihood (REML) in a
linear mixed-model (LMM) formulation.

Method summary
--------------
The Fredholm system `b = A x` becomes the mixed model

    b = A B c + eps,
    c ~ N(0, sigma2_e * lam * G^-1),   G = D^(d)' D^(d),

with `D^(d)` the `d`-th order difference matrix (classic P-spline penalty,
Eilers & Marx 1996).  Following the mixed-model reparameterisation
(Wand & Ormerod 2008), the coefficients are split into a fixed
(unpenalised) part spanning the null space of `G` — a polynomial trend of
degree `d - 1` — and a random (penalised) part:

    c = U_fixed * beta + U_random * b_random,  b_random ~ N(0, sigma2_e * lam * L^-1),

with `G = U diag(g) U'` and `L = diag(g_random)` the positive eigenvalues.
For a trial smoothing parameter `lam` the coefficients come from the
Henderson mixed-model equations, and `lam` is estimated by maximising the
REML profile log-likelihood

    l_R(lam) = -1/2 [(m - p_f) log sigma2_hat + log|V| + log|X' V^-1 X|],
    V = I + lam Z L^-1 Z',   sigma2_hat = SS / (m - p_f).

The 1-D profile is optimised with golden-section search on the
log-relative scale, `lam = lam_relative * lam_ref`, where `lam_ref`
equalises the average trace of the data and penalty terms — the method is
therefore insensitive to the absolute problem scale.

API (mirrors the Python module):

* `solve_pspline_reml(A, b, x0; ...)` — core solver returning
  `UnfoldResult` (Python `(spectrum, iterations, converged)`);
* `solve_pspline_reml_full(A, b, x0; ...)` — the same solver returning a
  rich diagnostics `extra` dictionary (`lam`, `lam_relative`, `sigma2`,
  `reml_loglik`, `ed`, `ed_norm`, ...).
"""

const _PSREML_TINY = 1e-300

# Default search interval for the relative smoothing parameter
const _psreml_LAMBDA_REL_BOUNDS = (1e-6, 1e6)

const _psreml_VALID_KNOT_SPACING = ("auto", "uniform", "log")

# Eigenvalues of G below this relative threshold are treated as the
# (exactly zero) null space of the difference penalty (~2.2e-7 relative).
const _PSREML_NULLSPACE_RTOL = 1e9 * eps(Float64)

# ─── Small shared helpers ────────────────────────────────────────────────────

function _psreml_floatvec(v::AbstractVector{<:Real})
    x = Vector{Float64}(undef, length(v))
    @inbounds for i in eachindex(v)
        x[i] = Float64(v[i])
    end
    return x

# B-spline basis: reuse the Cox–de Boor implementation of unfold_mlem_bs
# when it is present (same include scope), otherwise fall back to a local
# copy so that this file is self-contained.
function _psreml_build_bspline_basis(E::Vector{Float64}, n_basis::Int,
                                     spline_order::Int, knot_spacing::AbstractString)
    if isdefined(@__MODULE__, :_mlembs_build_bspline_basis)
        return _mlembs_build_bspline_basis(E, n_basis, spline_order, knot_spacing)
    elseif isdefined(@__MODULE__, :build_bspline_basis) &&
           !(parentmodule(build_bspline_basis) === @__MODULE__)
        # public helper of the package module (mlem_bs.jl include order)
        return build_bspline_basis(E, n_basis, spline_order, knot_spacing)
    else
        return _psreml_build_bspline_basis_local(E, n_basis, spline_order, knot_spacing)
    end

function _psreml_build_bspline_basis_local(E::AbstractVector{Float64},
                                           n_basis::Int, spline_order::Int,
                                           knot_spacing::AbstractString)
    length(E) >= 2 || throw(ArgumentError(
        "E_MeV must be a 1D array with >= 2 points, got length $(length(E))"))
    all(isfinite, E) || throw(ArgumentError("E_MeV contains non-finite values"))
    all(i -> E[i+1] > E[i], 1:(length(E)-1)) || throw(ArgumentError(
        "E_MeV must be strictly increasing"))
    all(>(0), E) || throw(ArgumentError("E_MeV must contain positive energies"))
    2 <= spline_order <= 8 || throw(ArgumentError(
        "spline_order must be in [2, 8], got $spline_order"))
    n_basis >= spline_order || throw(ArgumentError(
        "n_basis (N_s = $n_basis) must be >= spline_order (p = $spline_order)"))

    degree = spline_order - 1
    emin, emax = Float64(E[1]), Float64(E[end])
    if knot_spacing == "auto"
        ratio = maximum(E) / minimum(E)
        knot_spacing = ratio > 100.0 ? "log" : "uniform"
    end
    n_interior = n_basis - spline_order
    interior = if n_interior > 0
        if knot_spacing == "log"
            emin .* (emax / emin) .^ range(0.0, 1.0; length=n_interior + 2)[2:end-1]
        else
            collect(range(emin, emax; length=n_interior + 2))[2:end-1]
        end
    else
        Float64[]
    end
    t = vcat(fill(emin, spline_order), collect(Float64, interior),
             fill(emax, spline_order))

    B = Matrix{Float64}(undef, length(E), n_basis)
    row = zeros(Float64, n_basis)
    for (k, x0) in pairs(E)
        x = clamp(Float64(x0), emin, emax)
        d = zeros(Float64, n_basis)
        @inbounds for j in 1:n_basis
            d[j] = ((t[j] <= x < t[j+1]) ||
                    (x == emax && t[j] < t[j+1] == emax)) ? 1.0 : 0.0
        end
        for p in 1:degree
            nd = zeros(Float64, n_basis)
            @inbounds for j in 1:n_basis
                d1 = t[j+p] - t[j]
                c1 = d1 > 0 ? (x - t[j]) / d1 * d[j] : 0.0
                d2 = t[j+p+1] - t[j+1]
                c2 = d2 > 0 ? (t[j+p+1] - x) / d2 * d[j+1] : 0.0
                nd[j] = c1 + c2
            end
            d = nd
        end
        @inbounds B[k, :] .= row .= d
    end
    return B

function _psreml_validate(Ain::AbstractMatrix, bin::AbstractVector,
                          max_iterations::Integer, tolerance::Real)
    A = Matrix{Float64}(Ain)
    b = _psreml_floatvec(bin)
    m, n = size(A)
    length(b) == m || throw(ArgumentError(
        "Length of b ($(length(b))) must match number of rows in A ($m)"))
    max_iterations > 0 || throw(ArgumentError(
        "max_iterations must be positive, got $max_iterations"))
    tolerance > 0 || throw(ArgumentError("tolerance must be positive, got $tolerance"))
    return A, b

# ─── Difference penalty and mixed-model split ───────────────────────────────

"""
    _psreml_difference_matrix(n, order)

`order`-th order difference matrix `D` of shape `(n - order, n)`: rows are
successive finite differences, so the penalty `||D c||^2` shrinks
polynomial trends of degree `order - 1` but leaves them unpenalised.
"""
function _psreml_difference_matrix(n::Integer, order::Integer=2)
    n = Int(n)
    order = Int(order)
    n >= 1 || throw(ArgumentError("n must be a positive integer, got $n"))
    1 <= order <= 4 || throw(ArgumentError("order must be in [1, 4], got $order"))
    n > order || throw(ArgumentError(
        "n ($n) must be greater than the difference order ($order)"))
    E = Matrix{Float64}(I, n, n)
    for _ in 1:order
        E = E[2:end, :] .- E[1:end-1, :]
    end
    return E

"""
    _psreml_pspline_penalty(n, order=2)

Symmetric P-spline penalty matrix `G = D' D` (positive semi-definite).
"""
function _psreml_pspline_penalty(n::Integer, order::Integer=2)
    D = _psreml_difference_matrix(n, order)
    return Matrix(D' * D)

"""
    _psreml_mixed_model_split(n, order=2)

Split a P-spline space into fixed and random subspaces by diagonalising
the penalty `G = D' D`:

* the null space of `G` (eigenvalues ~ 0, dimension `order`) becomes the
  **fixed** part — an unpenalised polynomial trend;
* the range space (positive eigenvalues) becomes the **random** part with
  prior precision `L = diag(g_random)`.

Returns `(U_fixed, U_random, g_random)` where `U_fixed` is `(n, p_f)`,
`U_random` is `(n, n - p_f)` and `g_random` are the positive penalty
eigenvalues (ascending).
"""
function _psreml_mixed_model_split(n::Integer, order::Integer=2)
    G = _psreml_pspline_penalty(n, order)
    F = eigen(Symmetric(G))
    g = F.values
    U = F.vectors
    g_max = max(Float64(g[end]), _PSREML_TINY)
    is_fixed = g .<= _PSREML_NULLSPACE_RTOL * g_max
    n_fixed = count(is_fixed)
    if n_fixed == 0
        # Degenerate configuration: treat the smallest eigen-direction as
        # fixed to keep a proper mixed model (should not happen for
        # difference penalties of order >= 1).
        is_fixed[argmin(g)] = true
        n_fixed = 1
    end
    U_fixed = U[:, is_fixed]
    U_random = U[:, .!is_fixed]
    g_random = max.(g[.!is_fixed], _PSREML_TINY)
    return U_fixed, U_random, g_random

# ─── REML profile ────────────────────────────────────────────────────────────

"""
    _psreml_reml_profile(y_w, X_w, Z_w, g_random, lam)

REML profile log-likelihood for one smoothing value.  For the weighted
mixed model `y_w = X_w beta + Z_w b_r + eta` with
`b_r ~ N(0, lam^-1 L^-1)` the marginal covariance is
`V = I + lam Z_w L^-1 Z_w'`, and (up to an additive constant)

    l_R = -1/2 [(m - p_f) log sigma2_hat + log|V| + log|X_w' V^-1 X_w|],

with `sigma2_hat = SS / (m - p_f)` and `SS` the residual sum of squares of
the GLS projection.  Returns `(loglik, sigma2)`, or `(-Inf, NaN)` when the
profile is not finite at this `lam`.
"""
function _psreml_reml_profile(y_w::Vector{Float64}, X_w::Matrix{Float64},
                              Z_w::Matrix{Float64}, g_random::Vector{Float64},
                              lam::Float64)
    m = length(y_w)
    p_f = size(X_w, 2)
    df = m - p_f
    df < 1 && return (-Inf, NaN)

    L_inv = 1.0 ./ max.(g_random, _PSREML_TINY)
    V = Matrix{Float64}(I, m, m) .+ lam .* ((Z_w .* permutedims(L_inv)) * Z_w')
    cV = try
        cholesky(Symmetric(V))
    catch
        return (-Inf, NaN)
    end

    _solve_V(rhs) = cV \ rhs   # cholesky solve handles both triangular steps

    Vinv_X = _solve_V(X_w)
    XtVinvX = X_w' * Vinv_X
    sign_det, logdet_XtVX = slogdet(XtVinvX)
    sign_det <= 0 && return (-Inf, NaN)

    XtVinv_y = X_w' * _solve_V(y_w)
    beta = try
        XtVinvX \ XtVinv_y
    catch
        return (-Inf, NaN)
    end

    resid = y_w .- Vinv_X * beta
    ss = Float64(dot(resid, _solve_V(resid)))
    if !isfinite(ss) || ss <= 0
        # Exact fit (ss == 0) is legitimate for clean synthetic data but
        # makes log(sigma2) undefined; clamp to a tiny positive value.
        ss = max(ss, _PSREML_TINY)
    end

    sigma2 = ss / df
    _, logdet_V = slogdet(V)
    loglik = -0.5 * (df * log(sigma2) + logdet_V + logdet_XtVX)
    return (Float64(loglik), Float64(sigma2))

"""
    _psreml_golden_min(f, lo, hi; xatol=1e-5, max_iterations=100)

Golden-section minimisation of `f` over `[lo, hi]` (the analogue of
`scipy.optimize.minimize_scalar(..., method="bounded")`).  Returns
`(x_min, n_fev)`.
"""
function _psreml_golden_min(f, lo::Float64, hi::Float64;
                            xatol::Float64=1e-5, max_iterations::Int=100)
    invphi = (sqrt(5.0) - 1.0) / 2.0
    a, bnd = lo, hi
    c = bnd - invphi * (bnd - lo)
    d = lo + invphi * (bnd - lo)
    fc = f(c)
    fd = f(d)
    nfev = 2
    while abs(bnd - lo) > xatol && nfev < max_iterations
        if fc < fd
            bnd, d, fd = d, c, fc
            c = bnd - invphi * (bnd - lo)
            fc = f(c)
        else
            lo, c, fc = c, d, fd
            d = lo + invphi * (bnd - lo)
            fd = f(d)
        end
        nfev += 1
    end
    return (fc < fd ? ((lo + bnd) / 2.0) : ((c + d) / 2.0)), nfev

"""
    _psreml_select_lambda(y_w, X_w, Z_w, g_random, lam_ref, lam_bounds)

Maximise the REML profile over the relative smoothing parameter: the
profile is scanned with golden-section search in `t = log10(lam_rel)` over
`log10(lam_bounds)`; the absolute smoothing parameter is
`lam = lam_ref * lam_rel`.  Returns a `Dict` with keys `lam`,
`lam_relative`, `sigma2`, `reml_loglik`, `converged`, `n_iterations`
(number of profile evaluations).
"""
function _psreml_select_lambda(y_w::Vector{Float64}, X_w::Matrix{Float64},
                               Z_w::Matrix{Float64}, g_random::Vector{Float64},
                               lam_ref::Float64,
                               lam_bounds::Tuple{Float64,Float64}=_psreml_LAMBDA_REL_BOUNDS)
    lo = log10(max(lam_bounds[1], _PSREML_TINY))
    hi = log10(lam_bounds[2])
    cache = Dict{Float64,Float64}()
    n_fev = Ref(0)

    neg_loglik = function (t::Float64)
        haskey(cache, t) && return cache[t]
        n_fev[] += 1
        lam = lam_ref * 10.0^t
        ll, _ = _psreml_reml_profile(y_w, X_w, Z_w, g_random, lam)
        val = -ll
        cache[t] = val
        return val
    end

    t_opt, _ = _psreml_golden_min(neg_loglik, lo, hi)
    converged = isfinite(get(cache, t_opt, -neg_loglik(t_opt))) &&
                (cache[t_opt] < Inf)
    lam_rel = 10.0^t_opt
    lam = lam_ref * lam_rel
    ll, sigma2 = _psreml_reml_profile(y_w, X_w, Z_w, g_random, lam)
    return Dict{String,Any}(
        "lam"          => Float64(lam),
        "lam_relative" => Float64(lam_rel),
        "sigma2"       => Float64(sigma2),
        "reml_loglik"  => Float64(ll),
        "converged"    => converged,
        "n_iterations" => Int(n_fev[]),
    )

function _psreml_build_weights(weights::Union{AbstractString,AbstractVector{<:Real},Nothing},
                               b::Vector{Float64})
    if weights === nothing || (weights isa AbstractString && weights == "uniform")
        return ones(Float64, length(b))
    end
    if weights isa AbstractString
        if weights == "poisson"
            floor_w = 1e-3 * max(maximum(b), _PSREML_TINY)
            return 1.0 ./ max.(b, floor_w)
        end
        throw(ArgumentError(
            "weights must be \"uniform\", \"poisson\" or an array, got \"$weights\""))
    end
    w = _psreml_floatvec(weights)
    length(w) == length(b) || throw(ArgumentError(
        "weights length ($(length(w))) must match number of readings ($(length(b)))"))
    (all(isfinite, w) && all(>(0), w)) || throw(ArgumentError(
        "weights must be finite and positive"))
    return w

# ─── Public solvers ──────────────────────────────────────────────────────────

"""
    solve_pspline_reml_full(A, b, x0=nothing; E_MeV=nothing, n_basis=nothing,
                            spline_order=4, diff_order=2, knot_spacing="auto",
                            weights="uniform", lam_relative=nothing,
                            lam_bounds=(1e-6, 1e6))

P-spline REML unfolding returning an `UnfoldResult` whose `extra`
dictionary carries the rich Python diagnostics with keys `coefficients`,
`n_basis`, `spline_order`, `diff_order`, `knot_spacing`, `weights`, `lam`,
`lam_relative`, `lam_ref`, `sigma2`, `reml_loglik`, `ed` (effective
dimension), `ed_norm`, `reml_converged` and `n_iterations`.

- `x0` — unused (kept for API compatibility with other solvers).
- `E_MeV` — energy grid (MeV), `n` points (default: uniform grid `1:n`).
- `n_basis` — B-spline space dimension; default `max(diff_order + 2, min(n ÷ 2, 30))`.
- `weights` — `"uniform"`, `"poisson"` (`w_i = 1 / b_i`) or an explicit
  positive weight array.
- `lam_relative` — fixed relative smoothing parameter (skips REML
  optimisation when provided).
- `lam_bounds` — search interval for the relative smoothing parameter.
"""
function solve_pspline_reml_full(Ain::AbstractMatrix{T}, bin::AbstractVector{T};
                                 x0::Union{Nothing,AbstractVector{T}}=nothing,
                                 E_MeV::Union{Nothing,AbstractVector{<:Real}}=nothing,
                                 n_basis::Union{Nothing,Integer}=nothing,
                                 spline_order::Integer=4,
                                 diff_order::Integer=2,
                                 knot_spacing::AbstractString="auto",
                                 weights::Union{AbstractString,AbstractVector{<:Real},Nothing}="uniform",
                                 lam_relative::Union{Nothing,Real}=nothing,
                                 lam_bounds::Tuple{Real,Real}=_psreml_LAMBDA_REL_BOUNDS) where T<:AbstractFloat
    A, b = _psreml_validate(Ain, bin, 1000, 1e-6)
    m, n = size(A)
    m >= diff_order + 2 || throw(ArgumentError(
        "P-spline REML requires at least diff_order + 2 = $(diff_order + 2) " *
        "detector readings, got $m"))

    # ── spline space ──
    ns = n_basis === nothing ? max(diff_order + 2, min(n ÷ 2, 30)) : Int(n_basis)
    ns >= diff_order + 2 || throw(ArgumentError(
        "n_basis ($ns) must be >= diff_order + 2 ($(diff_order + 2))"))
    ns <= n || throw(ArgumentError(
        "n_basis ($ns) cannot exceed the number of energy bins ($n)"))

    E = E_MeV === nothing ? collect(range(1.0, Float64(n); length=n)) :
        _psreml_floatvec(E_MeV)
    length(E) == n || throw(ArgumentError(
        "Length of E_MeV ($(length(E))) must match number of energy bins ($n)"))
    knot_spacing in _psreml_VALID_KNOT_SPACING || throw(ArgumentError(
        "knot_spacing must be one of $_psreml_VALID_KNOT_SPACING, got \"$knot_spacing\""))

    B = _psreml_build_bspline_basis(E, ns, Int(spline_order), knot_spacing)

    # ── mixed-model split of the penalty ──
    U_fixed, U_random, g_random = _psreml_mixed_model_split(ns, Int(diff_order))
    X = A * (B * U_fixed)   # (m, p_f) unpenalised polynomial trend
    Z = A * (B * U_random)  # (m, p_r) penalised wiggly part

    w = _psreml_build_weights(weights, b)
    sw = sqrt.(w)
    y_w = sw .* b
    X_w = sw .* X
    Z_w = sw .* Z

    # ── scale-equalising reference smoothing parameter ──
    data_scale = sum(abs2, Z_w) / max(size(Z_w, 2), 1)
    pen_scale = sum(g_random) / length(g_random)
    lam_ref = max(data_scale / max(pen_scale, _PSREML_TINY), _PSREML_TINY)

    # ── smoothing parameter selection ──
    if lam_relative !== nothing
        lam_rel = Float64(lam_relative)
        (lam_rel > 0 && isfinite(lam_rel)) || throw(ArgumentError(
            "lam_relative must be positive and finite, got $lam_relative"))
        lam = lam_ref * lam_rel
        ll, sigma2 = _psreml_reml_profile(y_w, X_w, Z_w, g_random, lam)
        selection = Dict{String,Any}(
            "lam"          => Float64(lam),
            "lam_relative" => lam_rel,
            "sigma2"       => Float64(sigma2),
            "reml_loglik"  => Float64(ll),
            "converged"    => isfinite(ll),
            "n_iterations" => 0,
        )
    else
        selection = _psreml_select_lambda(y_w, X_w, Z_w, g_random, lam_ref,
                                          (Float64(lam_bounds[1]),
                                           Float64(lam_bounds[2])))
    end
    lam = selection["lam"]::Float64

    # ── Henderson mixed model equations ──
    # [X^T W X   X^T W Z    ] [beta ]   [X^T W y]
    # [Z^T W X   Z^T W Z+lam L] [b_r] = [Z^T W y]
    XtWX = X' * (w .* X)
    XtWZ = X' * (w .* Z)
    ZtWZ = Z' * (w .* Z)
    rhs = vcat(X' * (w .* b), Z' * (w .* b))
    saddle = [XtWX                 XtWZ;
              XtWZ'                ZtWZ .+ lam .* Diagonal(g_random)]
    coef = try
        saddle \ rhs
    catch
        pinv(saddle) * rhs
    end

    p_f = size(U_fixed, 2)
    beta_hat = coef[1:p_f]
    b_random_hat = coef[p_f+1:end]
    spectrum = B * (U_fixed * beta_hat + U_random * b_random_hat)

    # ── effective dimension ──
    ed = try
        ZtZ_lam = ZtWZ .+ lam .* Diagonal(g_random)
        Float64(p_f + tr(ZtZ_lam \ ZtWZ))
    catch
        NaN
    end

    diag = Dict{String,Any}(
        "coefficients"  => copy(coef),
        "n_basis"       => Int(ns),
        "spline_order"  => Int(spline_order),
        "diff_order"    => Int(diff_order),
        "knot_spacing"  => String(knot_spacing),
        "weights"       => weights isa AbstractString ? String(weights) : "array",
        "lam"           => Float64(lam),
        "lam_relative"  => Float64(selection["lam_relative"]),
        "lam_ref"       => Float64(lam_ref),
        "sigma2"        => Float64(selection["sigma2"]),
        "reml_loglik"   => Float64(selection["reml_loglik"]),
        "ed"            => Float64(ed),
        "ed_norm"       => Float64(ed / ns),
        "reml_converged" => selection["converged"]::Bool,
        "n_iterations"  => Int(selection["n_iterations"]),
    )

    spectrum_vec = Vector{T}(spectrum)
    resid = b .- A * spectrum_vec
    converged = diag["reml_converged"] && all(isfinite, spectrum_vec)
    return UnfoldResult(spectrum_vec, diag["n_iterations"], converged,
                        T(norm(resid)), diag)

# Positional-x0 convenience (Python keyword `x0` is also accepted as the
# third positional argument, like in other BSSUnfold.jl solvers).
solve_pspline_reml_full(A::AbstractMatrix{T}, b::AbstractVector{T},
                        x0::AbstractVector{T}; kwargs...) where T<:AbstractFloat =
    solve_pspline_reml_full(A, b; x0=x0, kwargs...)

"""
    solve_pspline_reml(A, b, x0=nothing; E_MeV=nothing, n_basis=nothing,
                       spline_order=4, diff_order=2, knot_spacing="auto",
                       weights="uniform", lam_relative=nothing,
                       lam_bounds=(1e-6, 1e6))

Solve the unfolding problem with P-spline REML smoothing.  The spectrum
is represented by a P-spline; the smoothing parameter is selected by
maximising the REML profile likelihood of the equivalent linear mixed
model, then the Henderson mixed-model equations are solved for the spline
coefficients.  The returned `UnfoldResult` contains the non-negative
clamped spectrum, the number of REML profile evaluations as `iterations`
and the REML solve success as `converged`.
"""
function solve_pspline_reml(Ain::AbstractMatrix{T}, bin::AbstractVector{T};
                            x0::Union{Nothing,AbstractVector{T}}=nothing,
                            E_MeV::Union{Nothing,AbstractVector{<:Real}}=nothing,
                            n_basis::Union{Nothing,Integer}=nothing,
                            spline_order::Integer=4,
                            diff_order::Integer=2,
                            knot_spacing::AbstractString="auto",
                            weights::Union{AbstractString,AbstractVector{<:Real},Nothing}="uniform",
                            lam_relative::Union{Nothing,Real}=nothing,
                            lam_bounds::Tuple{Real,Real}=_psreml_LAMBDA_REL_BOUNDS) where T<:AbstractFloat
    diag = solve_pspline_reml_full(Ain, bin; x0=x0, E_MeV=E_MeV,
                                   n_basis=n_basis, spline_order=spline_order,
                                   diff_order=diff_order,
                                   knot_spacing=knot_spacing,
                                   weights=weights, lam_relative=lam_relative,
                                   lam_bounds=lam_bounds)
    A, b = _psreml_validate(Ain, bin, 1000, 1e-6)
    spectrum = max.(diag.spectrum, T(0))
    converged = diag.extra["reml_converged"]::Bool && all(isfinite, spectrum)
    resid = b .- A * spectrum
    return UnfoldResult(spectrum, diag.iterations, converged,
                        T(norm(resid)))

# Positional-x0 convenience.
solve_pspline_reml(A::AbstractMatrix{T}, b::AbstractVector{T},
                   x0::AbstractVector{T}; kwargs...) where T<:AbstractFloat =
    solve_pspline_reml(A, b; x0=x0, kwargs...)
