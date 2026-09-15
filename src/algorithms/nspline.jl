"""
N-spline unfolding method (port from unfold_nspline.py).

An implementation of the approach of R. F. Islamgulov and V. D. Lartsev,
"Reconstruction of neutron spectra from activation measurements in the form
of N-splines", Atomic Energy 104(5), 295-302 (May 2008) — RFNC-VNIITF named
after E. I. Zababakhin.

The method solves the system of activation integrals

    Q_i = ∫ sigma_i(E) phi(E) dE,   i = 1..N                (eq. 1)

by parameterizing the sought spectrum phi(E) with a specialized "neutron"
spline (N-spline) with basis functions

    N_k(E) = exp(a_k + q_k ln E + r_k E),  E_k <= E <= E_{k+1},
    k = 1..M                                                 (eq. 2)

i.e. piecewise functions whose logarithm is linear both in ln E and in E.
This family contains the classic model spectra (1/E, the Maxwellian
evaporator exp(-E/T), the fission sqrt(E) exp(-bE), ...) as special
cases, so the basis is nearly complete for reactor and accelerator
spectra, and the entire spectrum is described by only 3M parameters.

Three components of the article are implemented:

1. `build_continuity_matrix` / `fit_nspline` — the N-spline itself.  C0/C1 continuity
   at the internal knots (eq. 3-4) is imposed by the block matrix D
   (eq. 5), and the pointwise approximation of a tabulated spectrum (eq. 6-7)
   reduces to a weighted linear LSQ in the log-domain with linear
   equality constraints D X = 0, X = (a, q, r)^T, via a KKT system.

2. `solve_nspline_full` — a directed-divergence minimization loop
   (generalized MIRD algorithm of Lartsev; Tarasko, FEI Preprint No. 1446, 1983).
   With normalized measured activations p_i = Q_i / sum(Q) the functional

        H = Σ_i [pN_i ln(pN_i / p_i) - pN_i + p_i] >= 0   (eq. 8-9)

   (pN_i — normalized calculated activations) is decreased by a
   fluence-preserving gradient iteration

        phi_{n+1}(E) = phi_n(E) [1 - dmu_n (R_n(E) - Rbar_n)],

   where Rbar_n is the fluence-weighted mean of R_n (preserving fluence), and the step
   dmu_n starts from the conservative value 0.1 / sup|R_n - Rbar_n| and
   is halved (backtracking) until H decreases.  After *each*
   iteration the current spectrum is smoothed by refitting the N-spline —
   the key regularization trick of the article: the iteration actually acts
   on the 3M spline parameters instead of the n bin values.

3. The stopping criteria and quality control of the article: iterations stop
   when H reaches a level corresponding to the measurement errors,
        H <= H_target = 0.5 * mean_i (dQ_i / Q_i)^2,
   or when the relative decrease of H per iteration falls below `tol`.
   The suitability of the recovered spectrum is measured by the root-mean-square
   residual nev = sqrt(1/(N-1) Σ_i ((Qr_i - Q_i)/dQ_i)^2), acceptable if
   nev <= 1 + 2/sqrt(N).

Knot sets for the BARS-5, IGRIK (channel and surface) and YAGUAR reactors
are given in `NSPLINE_KNOT_PRESETS`; `auto_knots` builds a log-uniform grid
by default.
"""

const _PHI_FLOOR = 1e-300  # absolute floor for positive values of the spectrum
const _LOG_CLIP = 50.0     # clipping ln(pN/p) to suppress outliers

"""
    NSPLINE_KNOT_PRESETS::Dict{String,Vector{Float64}}

Knot sets (eq. 2, MeV) from the article ("Reconstruction of the spectra of the
BARS-5, IGRIK, YAGUAR reactors").
"""
const NSPLINE_KNOT_PRESETS = Dict{String,Vector{Float64}}(
    # Channel of the BARS-5 reactor
    "BARS5_channel" => [
        1e-10, 1.3e-7, 3.83e-7, 8e-6, 2e-5, 3e-5, 7.3e-5,
        3.2e-3, 0.38, 0.95, 7.0, 17.0, 20.0,
    ],
    # Channel of the IGRIK reactor
    "IGRIK_channel" => [
        1e-10, 2e-8, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 1.5e-4,
        3e-4, 6e-4, 6e-3, 0.27, 1.0, 2.7, 7.0, 13.0, 20.0,
    ],
    # Surface of the IGRIK reactor
    "IGRIK_surface" => [
        1e-10, 2e-8, 1e-7, 2e-7, 3e-6, 5e-6, 2.5e-4, 0.6,
        0.8, 1.5, 2.7, 7.0, 11.5, 14.0, 20.0,
    ],
    # Channel of the YAGUAR reactor
    "YAGUAR_channel" => [
        1e-10, 2e-8, 1e-7, 6e-7, 1e-6, 3e-6, 1e-5, 4.3e-5,
        1.8e-4, 6.3e-4, 5e-3, 0.6, 0.8, 1.0, 2.5, 7.0, 11.0,
        13.0, 20.0,
    ],
)

# ─── Knot utilities ──────────────────────────────────────────────────────────

"""
    auto_knots(E_MeV; n_segments=12) -> Vector{Float64}

Build a log-uniform grid of knots covering the range `E_MeV`
(`n_segments + 1` knots from min(E) to max(E)).
"""
function auto_knots(E_MeV::AbstractVector{<:Real}; n_segments::Integer=12)
    Epos = filter(>(0), collect(Float64, E_MeV))
    length(Epos) >= 2 || throw(ArgumentError(
        "auto_knots requires at least two positive energy points, got $(length(Epos))"))
    emin, emax = extrema(Epos)
    (isfinite(emin) && isfinite(emax) && emin < emax) || throw(ArgumentError(
        "auto_knots requires finite min(E) < max(E), got [$emin, $emax]"))
    n_segments >= 1 || throw(ArgumentError("n_segments must be >= 1, got $n_segments"))
    # geomspace: log-uniform grid
    return collect(emin .* (emax / emin) .^ range(0.0, 1.0, length=n_segments + 1))
end

function _resolve_knots(knots::Union{Nothing,String,AbstractVector{<:Real}},
                       E_MeV::Vector{Float64},
                       n_segments::Union{Nothing,Integer})
    Epos = filter(>(0), E_MeV)
    emin = isempty(Epos) ? 1e-10 : minimum(Epos)
    emax = maximum(E_MeV)

    local kn::Vector{Float64}
    local src::String
    if knots === nothing
        ns = n_segments === nothing ?
             min(12, max(4, length(E_MeV) ÷ 4)) : Int(n_segments)
        kn = auto_knots(E_MeV; n_segments=ns)
        src = "auto"
    elseif knots isa String
        haskey(NSPLINE_KNOT_PRESETS, knots) || throw(ArgumentError(
            "Unknown N-spline knot preset '$(knots)'. Available presets: " *
            join(sorted_keys(NSPLINE_KNOT_PRESETS), ", ")))
        kn = copy(NSPLINE_KNOT_PRESETS[knots])
        src = "preset:$(knots)"
    else
        kn = collect(Float64, knots)
        src = "user"
    end

    length(kn) >= 2 || throw(ArgumentError("N-spline needs at least 2 knots, got $(length(kn))"))
    all(>(0), diff(kn)) || throw(ArgumentError("N-spline knots must be strictly increasing"))

    # Clipping the knots to the grid range and expanding the outer knots so
    # that the spline domain covers the entire grid.
    kn = clamp.(kn, emin, emax)
    kn = sort(unique(kn))
    kn[1] = min(kn[1], emin)
    kn[end] = max(kn[end], emax)
    if length(kn) < 2
        kn = [emin, emax]
    end
    return kn, src
end

sorted_keys(d::Dict{String,Vector{Float64}}) = sort!(collect(keys(d)))

"""
    _segment_indices(E, knots) -> Vector{Int}

Map the energy points onto the spline segment indices 0..M-1
(in Julia — 1..M).
"""
function _segment_indices(E::AbstractVector{<:Real}, knots::Vector{Float64})
    # searchsortedright: index of the last knot <= E, clipping to [1, M]
    k = searchsortedlast.(Ref(knots), E)  # knots[i] <= E
    return clamp.(k, 1, length(knots) - 1)
end

# ─── Definition of the N-spline (eq. 2-5) ────────────────────────────────────

"""
    build_continuity_matrix(knots; continuity="C0C1") -> Matrix{Float64}

Build the spline continuity matrix D (eq. 5).

The vector of N-spline parameters: `X = (a, q, r)^T` with `a = (a_1..a_M)`,
`q = (q_1..q_M)`, `r = (r_1..r_M)`.  The continuity conditions at the internal
knots (eq. 3-4):

    C0: a_k - a_{k+1} + u (q_k - q_{k+1}) + E (r_k - r_{k+1}) = 0
    C1: (q_k - q_{k+1}) + E (r_k - r_{k+1}) = 0,   u = ln E

are assembled into `D = [[A, B, C], [0, A, C]]`, `D X = 0`.

`continuity`: `"C0C1"` (default) — continuity of value and derivative;
`"C0"` — value only; `"none"` — no continuity.
"""
function build_continuity_matrix(knots::AbstractVector{<:Real};
                                continuity::AbstractString="C0C1")
    kn = Float64.(collect(knots))
    M = length(kn) - 1
    M >= 1 || throw(ArgumentError("knots must contain at least 2 values"))
    cont = uppercase(replace(continuity, " " => ""))
    cont in ("C0C1", "C0", "NONE") || throw(ArgumentError(
        "continuity must be one of 'C0C1', 'C0', 'none', got '$(continuity)'"))

    n_int = M - 1  # internal knots
    (cont == "NONE" || n_int == 0) && return zeros(0, 3 * M)

    rows_c0 = cont == "C0C1"
    n_rows = rows_c0 ? 2 * n_int : n_int
    D = zeros(n_rows, 3 * M)
    for k in 1:n_int
        Ek = kn[k + 1]
        uk = log(Ek)
        # C0 row: a_k - a_{k+1} + u(q_k - q_{k+1}) + E(r_k - r_{k+1}) = 0
        D[k, k] = -1.0
        D[k, k + 1] = 1.0
        D[k, M + k] = -uk
        D[k, M + k + 1] = uk
        D[k, 2M + k] = -Ek
        D[k, 2M + k + 1] = Ek
        if rows_c0
            # C1 row: (q_k - q_{k+1}) + E(r_k - r_{k+1}) = 0
            row = n_int + k
            D[row, M + k] = -1.0
            D[row, M + k + 1] = 1.0
            D[row, 2M + k] = -Ek
            D[row, 2M + k + 1] = Ek
        end
    end
    return D
end

"""
    nspline_eval(E, a, q, r, knots) -> Vector{Float64}

Evaluate the N-spline `N(E) = exp(a_k + q_k ln E + r_k E)`.

# Arguments
- `E`: energies (MeV), strictly positive
- `a, q, r`: vectors of length M (number of segments)
- `knots`: M+1 knot values
"""
function nspline_eval(E::AbstractVector{<:Real},
                     a::AbstractVector{<:Real},
                     q::AbstractVector{<:Real},
                     r::AbstractVector{<:Real},
                     knots::AbstractVector{<:Real})
    any(<=(0), E) && throw(ArgumentError("nspline_eval requires strictly positive energies"))
    M = length(knots) - 1
    (length(a) == length(q) == length(r) == M) || throw(ArgumentError(
        "a, q, r must all have length M=$(M) segments, got " *
        "$(length(a)), $(length(q)), $(length(r))"))
    kseg = _segment_indices(E, Float64.(collect(knots)))
    Ef = Float64.(collect(E))
    return [exp(a[k] + q[k] * log(Ef[i]) + r[k] * Ef[i]) for (i, k) in enumerate(kseg)]
end

"""
    directed_divergence(p_calc, p_meas) -> Float64

Directed (Kulback-Leibler-type) divergence of eq. (8-9):

    H = Σ_i [pN_i ln(pN_i / p_i) - pN_i + p_i] >= 0,

H = 0 if and only if the calculated activations equal the measured ones.
"""
function directed_divergence(p_calc::AbstractVector{<:Real},
                            p_meas::AbstractVector{<:Real})
    pN = max.(Float64.(p_calc), 1e-300)
    p = max.(Float64.(p_meas), 1e-300)
    return sum(@. pN * log(pN / p) - pN + p)
end

# ─── Pointwise N-spline approximation (eq. 6-7) ─────────────────────────────

"""
    fit_nspline(E, phi; knots=nothing, rel_err=nothing, continuity="C0C1",
                n_segments=nothing) -> (N_E, info)

Approximation of a pointwise spectrum by an N-spline (eq. 2, 5-7).

Solves a weighted LSQ problem in the log-domain with continuity constraints:

    min_X Σ_j w_j^2 (a_kj + u_j q_kj + E_j r_kj - ln phi_j)^2
    s.t.  D X = 0,   w_j = 1 / eps_j,

via the KKT system (Lagrange multipliers)

    [[G^T W G, D^T], [D, 0]] [X; lam] = [G^T W Y; 0].

# Returns
`(N_E, info)`, where `N_E` is the fitted spline on `E`, and `info` contains
`knots`, `knots_source`, `a`/`q`/`r`, `log_rms_residual`, `continuity`.
"""
function fit_nspline(E::AbstractVector{<:Real},
                    phi::AbstractVector{<:Real};
                    knots::Union{Nothing,String,AbstractVector{<:Real}}=nothing,
                    rel_err::Union{Nothing,AbstractVector{<:Real}}=nothing,
                    continuity::AbstractString="C0C1",
                    n_segments::Union{Nothing,Integer}=nothing)
    E_arr = Float64.(collect(E))
    phi_arr = Float64.(collect(phi))
    length(E_arr) == length(phi_arr) || throw(ArgumentError(
        "E and phi length mismatch: $(length(E_arr)) vs $(length(phi_arr))"))
    all(>(0), E_arr) || throw(ArgumentError("fit_nspline requires strictly positive energies"))
    length(phi_arr) >= 3 || throw(ArgumentError("fit_nspline requires at least 3 spectrum points"))

    kn, src = _resolve_knots(knots, E_arr, n_segments)
    M = length(kn) - 1
    n = length(E_arr)

    # Point → segment and log-domain projection matrix G (n x 3M).
    kseg = _segment_indices(E_arr, kn)
    u = log.(E_arr)
    G = zeros(n, 3M)
    for i in 1:n
        k = kseg[i]
        G[i, k] = 1.0
        G[i, M + k] = u[i]
        G[i, 2M + k] = E_arr[i]
    end

    # Floor for microscopic/zero bins with weight attenuation.
    phi_max = maximum(phi_arr)
    tiny = max(_PHI_FLOOR, 1e-12 * phi_max)
    floored = phi_arr .< tiny
    y = log.([f ? tiny : v for (f, v) in zip(floored, phi_arr)])

    w = rel_err === nothing ? ones(n) :
        1.0 ./ max.(Float64.(collect(rel_err)), 1e-12)
    w = [f ? 1e-3 * wi : wi for (f, wi) in zip(floored, w)]  # strong relative weight penalty

    D = build_continuity_matrix(kn; continuity=continuity)
    nc = size(D, 1)

    # Weighted normal equations + KKT constraint block.
    # No ridge is added: the pivoted-QR solver returns the minimum-norm
    # solution for rank-deficient systems (empty segments).
    Gw = G .* w
    yw = y .* w
    H_norm = Gw' * Gw
    KKT = zeros(3M + nc, 3M + nc)
    KKT[1:3M, 1:3M] .= H_norm
    if nc > 0
        KKT[1:3M, 3M+1:end] .= D'
        KKT[3M+1:end, 1:3M] .= D
    end
    rhs = vcat(Gw' * yw, zeros(nc))

    sol = qr(KKT, ColumnNorm()) \ rhs
    X = sol[1:3M]

    N_E = exp.(G * X)
    resid = w .* (G * X .- y)
    rms = sqrt(mean(resid .^ 2)) / max(mean(w), 1e-300)

    info = Dict{String,Any}(
        "knots" => kn,
        "knots_source" => src,
        "continuity" => continuity,
        "a" => X[1:M],
        "q" => X[M+1:2M],
        "r" => X[2M+1:3M],
        "log_rms_residual" => rms,
    )
    return N_E, info
end

# ─── Trapezoid ──────────────────────────────────────────────────────────────

function _trapz(y::AbstractVector{<:Real}, x::AbstractVector{<:Real})
    n = length(y)
    n == length(x) || throw(ArgumentError("y and x must have the same length"))
    n < 2 && return 0.0
    s = 0.0
    @inbounds for i in 1:(n - 1)
        s += (y[i] + y[i + 1]) * (x[i + 1] - x[i])
    end
    return s / 2.0
end

# ─── Directed divergence unfolding with per-iteration N-spline
#     smoothing ──────────────────────────────────────────────────────────

"""
    solve_nspline_full(A, b, x0, E_MeV; knots=nothing, sigma_rel=nothing,
                       continuity="C0C1", max_iterations=200, tol=1e-3,
                       step_theta=0.1, smoothing=true, n_segments=nothing) -> Dict

Full N-spline unfolding with diagnostics (Islamgulov & Lartsev, 2008).

Iteratively minimizes the directed divergence H between the measured and
calculated normalized activations, smoothing the spectrum by an N-spline fit
at each iteration (the regularization of the article).  Uses the stopping criteria
of the article (H at the level of the measurement errors or a stalled relative
decrease) and returns the residual statistic `nev` with the acceptability bound
`nev <= 1 + 2/sqrt(N)`.

# Arguments
- `A::AbstractMatrix`: response matrix of activation detectors (m, n)
- `b::AbstractVector`: measured readings / activation integrals (m,)
- `x0`: initial guess of the spectrum (n,); `nothing` — flat spectrum
- `E_MeV`: energy grid (MeV), strictly positive (required)
- `knots`: preset name (`NSPLINE_KNOT_PRESETS`), explicit vector
  or `nothing` (auto log-grid)
- `sigma_rel`: relative measurement errors dQ_i/Q_i (m,);
  `nothing` — 0.1 for each detector
- `continuity`: `"C0C1"` (default), `"C0"` or `"none"`
- `max_iterations`: iteration budget (default 200)
- `tol`: threshold of relative decrease of H (default 1e-3)
- `step_theta`: conservative initial step factor: dmu = step_theta /
  sup|R-Rbar| (the article value is 0.1); backtracking halves it while H grows
- `smoothing`: refit the N-spline after each iteration
  (default true, the article procedure; `false` reduces to a plain MIRD loop)
- `n_segments`: number of segments when `knots=nothing`

# Returns
A `Dict{String,Any}` with keys `spectrum`, `iterations`, `converged`,
`stop_reason`, `H`, `H_history`, `H_target`, `nev`, `nev_limit`,
`acceptable`, `Qr`, `relative_residuals`, `fluence`, `mean_energy`,
`knots`, `knots_source`, `continuity`, `params`.
"""
function solve_nspline_full(A::AbstractMatrix{<:Real},
                           b::AbstractVector{<:Real},
                           x0::Union{Nothing,AbstractVector{<:Real}},
                           E_MeV::AbstractVector{<:Real};
                           knots::Union{Nothing,String,AbstractVector{<:Real}}=nothing,
                           sigma_rel::Union{Nothing,AbstractVector{<:Real}}=nothing,
                           continuity::AbstractString="C0C1",
                           max_iterations::Integer=200,
                           tol::Real=1e-3,
                           step_theta::Real=0.1,
                           smoothing::Bool=true,
                           n_segments::Union{Nothing,Integer}=nothing)
    A_arr = Float64.(Matrix(A))
    b_arr = Float64.(collect(b))
    m, n = size(A_arr)
    length(b_arr) == m || throw(ArgumentError(
        "b length ($(length(b_arr))) does not match A rows ($m)"))
    m >= 1 || throw(ArgumentError("At least one measurement is required"))
    E = Float64.(collect(E_MeV))
    length(E) == n || throw(ArgumentError(
        "E_MeV length ($(length(E))) does not match A columns ($n)"))
    all(>(0), E) || throw(ArgumentError("E_MeV must contain strictly positive energies"))
    max_iterations >= 1 || throw(ArgumentError("max_iterations must be >= 1, got $max_iterations"))
    0 < step_theta <= 1 || throw(ArgumentError("step_theta must be in (0, 1], got $step_theta"))
    tol > 0 || throw(ArgumentError("tol must be positive, got $tol"))

    # Detectors with positive readings (zero measurements carry no
    # information for divergence minimization).
    valid = b_arr .> 0
    any(valid) || throw(ArgumentError(
        "solve_nspline requires at least one positive measurement"))
    A_v = A_arr[valid, :]
    b_v = b_arr[valid]

    sigma_v = sigma_rel === nothing ? fill(0.1, length(b_v)) :
              max.(Float64.(collect(sigma_rel))[valid], 1e-12)

    kn, knot_src = _resolve_knots(knots, E, n_segments)

    # Normalized measured activations and the article H-target: the expected
    # directed divergence, when all calculated activentions deviate
    # by 1 sigma from the measurements, E[H] ~ 0.5 Σ_i p_i delta_i^2.
    p = b_v ./ sum(b_v)
    H_target = 0.5 * sum(p .* sigma_v .^ 2)

    # Initial spectrum: rescale x0 to the measured total response, then
    # N-spline smoothing (in the spirit of the article — the spline of the MC spectrum as
    # the initial approximation).
    x = x0 === nothing ? ones(n) :
        [isfinite(v) ? max(v, 0.0) : 0.0 for v in Float64.(collect(x0))]
    length(x) == n || throw(ArgumentError("x0 length ($(length(x))) does not match A columns ($n)"))
    sum(x) <= 0 && (x = ones(n))

    Qc0 = A_v * x
    scale = sum(b_v) / max(sum(Qc0), 1e-300)
    x = max.(x .* scale, _PHI_FLOOR)

    # Pointwise relative errors for the per-iteration smoothings
    # (the article weight w = 1/eps, eq. 7): bins with a low total
    # sensitivity of the detectors carry less information and get
    # proportionally larger assumed errors (Poisson
    # sqrt-scaling), so as not to pull the spline.
    sens = vec(sum(A_v, dims=1))
    sens_max = isempty(sens) ? 0.0 : maximum(sens)
    smooth_rel_err = (smoothing && sens_max > 0) ?
        sqrt.(clamp.(sens_max ./ max.(sens, 1e-300), 1.0, 1e12)) : nothing

    if smoothing
        x, fit_info = fit_nspline(E, x; knots=kn, rel_err=smooth_rel_err,
                                  continuity=continuity)
        x = max.(x, _PHI_FLOOR)
        # Preserve the activation scale after the shape-only spline fit.
        x .*= sum(b_v) / max(sum(A_v * x), 1e-300)
    else
        fit_info = Dict{String,Any}()
    end

    b_total = sum(b_v)
    eps_scale = 1e-12 * max(b_total, 1e-300)

    _gauge(xx) = xx .* (b_total / max(sum(A_v * xx), 1e-300))

    function _state(xx)
        xx = max.(xx, _PHI_FLOOR)
        Qc_ = max.(A_v * xx, eps_scale)
        pN_ = Qc_ ./ max(sum(Qc_), 1e-300)
        H_ = directed_divergence(pN_, p)
        return xx, Qc_, pN_, H_
    end

    x, _Qc, pN, H = _state(_gauge(x))
    H_history = [H]
    converged = false
    stop_reason = "max_iterations"
    iterations = 0

    if H <= H_target
        converged = true
        stop_reason = "H_target (initial)"
    end

    for iteration in 1:max_iterations
        iterations = iteration

        # Gradient of H with respect to the spectrum (up to the constant 1/sum(Q)):
        # R(E) = Σ_i (p_i / Q_i) sigma_i(E) ln(pN_i / p_i).
        ln_ratio = clamp.(log.(pN ./ p), -_LOG_CLIP, _LOG_CLIP)
        R = (A_v' * ln_ratio) ./ sum(b_v)
        x_sum = sum(x)
        Rbar = dot(x, R) / max(x_sum, 1e-300)
        g = R .- Rbar
        g_max = maximum(abs.(g))
        if !isfinite(g_max) || g_max <= 0.0
            stop_reason = "stalled_gradient"
            iterations -= 1
            break
        end

        # Conservative step of the article (dmu0 = 0.1 / sup|R - Rbar|) with
        # backtracking halving until H stops growing.
        mu = step_theta / g_max
        accepted = false
        local x_new, Qc_new, pN_new, H_new
        for _bt in 1:60
            x_trial = x .* (1.0 .- mu .* g)
            if smoothing
                x_trial, _ = fit_nspline(E, x_trial; knots=kn,
                                         rel_err=smooth_rel_err,
                                         continuity=continuity)
            end
            # Fixing the activation scale after the shape-only update.
            x_new, Qc_new, pN_new, H_new = _state(_gauge(x_trial))
            if isfinite(H_new) && H_new <= H + 1e-4 * max(H, 1e-300)
                accepted = true
                break
            end
            mu *= 0.5
        end
        if !accepted
            stop_reason = "no_further_reduction"
            iterations -= 1
            break
        end

        H_prev = H
        x, _Qc, pN, H = x_new, Qc_new, pN_new, H_new
        push!(H_history, H)

        # Stopping criteria of the article.
        if H <= H_target
            converged = true
            stop_reason = "H_target"
            break
        end
        if abs(H_prev - H) <= tol * max(H_prev, 1e-300)
            converged = true
            stop_reason = "relative_change"
            break
        end
    end

    # Acceptability statistics of the article: nev = RMS((Qr - Q)/dQ),
    # acceptable if nev <= 1 + 2/sqrt(N).
    Qr_full = A_arr * x
    rel_res = zeros(m)
    denom = max.(sigma_v .* b_v, 1e-300)
    rel_res[valid] .= (Qr_full[valid] .- b_v) ./ denom
    cnt = count(valid)
    div = cnt > 1 ? cnt - 1 : cnt
    nev = sqrt(sum(rel_res[valid] .^ 2) / max(div, 1))
    nev_limit = 1.0 + 2.0 / sqrt(cnt)
    acceptable = nev <= nev_limit

    fluence = _trapz(x, E)
    mean_energy = fluence > 0 ? _trapz(E .* x, E) / fluence : NaN

    # Final spline parameterization of the recovered spectrum.
    M = length(kn) - 1
    if haskey(fit_info, "a")
        params = Dict{String,Any}("a" => fit_info["a"], "q" => fit_info["q"],
                                  "r" => fit_info["r"])
    else
        kseg = _segment_indices(E, kn)
        G = zeros(n, 3M)
        for i in 1:n
            k = kseg[i]
            G[i, k] = 1.0
            G[i, M + k] = log(E[i])
            G[i, 2M + k] = E[i]
        end
        Xl = qr(G, ColumnNorm()) \ log.(max.(x, _PHI_FLOOR))
        params = Dict{String,Any}("a" => Xl[1:M], "q" => Xl[M+1:2M],
                                  "r" => Xl[2M+1:3M])
    end

    return Dict{String,Any}(
        "spectrum" => x,
        "iterations" => iterations,
        "converged" => converged,
        "stop_reason" => stop_reason,
        "H" => H,
        "H_history" => H_history,
        "H_target" => H_target,
        "nev" => nev,
        "nev_limit" => nev_limit,
        "acceptable" => acceptable,
        "Qr" => Qr_full,
        "relative_residuals" => rel_res,
        "fluence" => fluence,
        "mean_energy" => mean_energy,
        "knots" => kn,
        "knots_source" => knot_src,
        "continuity" => continuity,
        "params" => params,
    )
end

"""
    solve_nspline(A, b, x0; E_MeV, knots=nothing, sigma_rel=nothing,
                  continuity="C0C1", max_iterations=200, tol=1e-3,
                  step_theta=0.1, smoothing=true, n_segments=nothing) -> UnfoldResult

Standard-API solver of the N-spline method: a thin wrapper over
[`solve_nspline_full`](@ref), returning an `UnfoldResult` with the spectrum,
number of iterations and the convergence flag.

# Arguments
- `A::AbstractMatrix{T}`: response matrix (m × n)
- `b::AbstractVector{T}`: measurements (m,)
- `x0::AbstractVector{T}`: initial guess of the spectrum (n,)
- `E_MeV`: energy grid (MeV), strictly positive — a **required**
  keyword (the `unfold_nspline` wrapper substitutes the detector grid automatically)
- the remaining keyword arguments are identical to `solve_nspline_full`
"""
function solve_nspline(A::AbstractMatrix{T}, b::AbstractVector{T}, x0::AbstractVector{T};
                      E_MeV::Union{Nothing,AbstractVector{<:Real}}=nothing,
                      knots::Union{Nothing,String,AbstractVector{<:Real}}=nothing,
                      sigma_rel::Union{Nothing,AbstractVector{<:Real}}=nothing,
                      continuity::AbstractString="C0C1",
                      max_iterations::Integer=200,
                      tol::Real=1e-3,
                      step_theta::Real=0.1,
                      smoothing::Bool=true,
                      n_segments::Union{Nothing,Integer}=nothing) where T<:AbstractFloat
    E_MeV === nothing && throw(ArgumentError(
        "E_MeV (energy grid in MeV) is required for solve_nspline"))
    result = solve_nspline_full(A, b, x0, E_MeV;
                                knots=knots, sigma_rel=sigma_rel,
                                continuity=continuity,
                                max_iterations=max_iterations, tol=tol,
                                step_theta=step_theta, smoothing=smoothing,
                                n_segments=n_segments)
    spectrum = Vector{T}(result["spectrum"])
    res = b .- A * spectrum
    return UnfoldResult(spectrum, result["iterations"], result["converged"],
                        norm(res),
                        Dict{String,Any}(
                            "stop_reason" => result["stop_reason"],
                            "H" => result["H"],
                            "H_target" => result["H_target"],
                            "nev" => result["nev"],
                            "nev_limit" => result["nev_limit"],
                            "acceptable" => result["acceptable"],
                            "fluence" => result["fluence"],
                            "mean_energy" => result["mean_energy"],
                            "knots_source" => result["knots_source"],
                        ))
end
