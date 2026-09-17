"""
MLEM-BS: B-spline MLEM unfolding with sieve regularization.

Port of `bssunfold/core/unfold_mlem_bs.py` (V. Mazankova, L. Torokova,
D. Trunec, Z. Kopecky, Z. Matej, "Experimental Measurement of Neutron Flux
and Its Mathematical Data Processing", Proc. CNDGS'2026, Brno, 2026,
https://doi.org/10.47459/cndcgs.2026.61).

The spectrum is represented as a linear combination of B-spline basis
functions,

    x(E) = sum_s b_s B_s(E),   s = 1..N_s,

so the effective system matrix becomes `RB = A * B` and the coefficients
are found with the regularized MLEM iteration (MLEM-BS, Eq. 4):

    b_s^(k+1) = b_s^(k) / (sum_i (RB)_is + beta * dP/db_s)
                * sum_i (RB)_is * n_i / (sum_s' (RB)_is' b_s'^(k)),

with the second-derivative penalty `P(b) = ||D2 b||^2` (Eq. 5,
`dP/db = 2 (D2)^T D2 b`), restricted to the non-negative B-spline sieve
(Szkutik 2005).  Iteration count, `N_s` and `beta` are selected by
minimizing the goodness-of-fit statistic (Eq. 6):

    K_S = | sum_i (n_i - model_i)^2 / sum_i model_i - 1 |.

Confidence intervals (paper Eqs. 7-9) are a Detector-layer feature and are
not part of the core solvers here.

Scaling note
------------
The absolute `beta` of Eq. 4 is problem-scale dependent (the paper uses
`beta = 1e-17`).  Alternatively `beta_relative` defines the effective
penalty as `beta = beta_relative * mean column sum of RB`.  Exactly one
of the two may be given; with neither, `beta = 0` (pure sieve MLEM).

API (mirrors the Python module):

* `build_bspline_basis(E_MeV, n_basis, spline_order, knot_spacing)` —
  Cox–de Boor designed B-spline basis (clamped knot vector);
* `second_difference_matrix(n)` — the `D2` penalty matrix;
* `ks_statistic(b, model)` — the `K_S` statistic of Eq. 6;
* `solve_mlem_bs(A, b, x0; ...)` — core solver returning `UnfoldResult`;
* `solve_mlem_bs_full(A, b, x0; ...)` — the solver with a rich
  diagnostics `extra` dictionary (`ks_history`, `ks_final`, `chi2_pearson`,
  `n_basis`, `interior_knots`, `beta_effective`, `auto_selection`, ...).
"""
const _MLEMBS_TINY = 1e-300         # absolute floor for denominators
const _MLEMBS_COUNT_FLOOR = 1e-10   # floor for forward-model counts

# Relative penalty strengths scanned by the KS-based auto selection.
const _mlembs_AUTO_BETA_RELATIVE_GRID = (0.0, 1e-4, 1e-3, 1e-2, 1e-1)

const _mlembs_VALID_KNOT_SPACING = ("auto", "uniform", "log")

# ─── Small shared helpers ────────────────────────────────────────────────────

_mlembs_geomspace(a::Float64, b::Float64, n::Integer) =
    a .* (b / a) .^ range(0.0, 1.0; length=Int(n))

function _mlembs_floatvec(v::AbstractVector{<:Real})
    x = Vector{Float64}(undef, length(v))
    @inbounds for i in eachindex(v)
        x[i] = Float64(v[i])
    end
    return x
end

# NNLS access: works both when included inside the BSSUnfold module
# (`solve_nnls` is defined there) and when included into a test harness
# module where `BSSUnfold` itself is imported.
function _mlembs_solve_nnls(B::AbstractMatrix{Float64}, x0::AbstractVector{Float64})
    if isdefined(@__MODULE__, :solve_nnls)
        return solve_nnls(B, x0)
    elseif isdefined(@__MODULE__, :BSSUnfold)
        return BSSUnfold.solve_nnls(B, x0)
    else
        error("solve_nnls is not available")
    end
end

function _mlembs_validate(Ain::AbstractMatrix, bin::AbstractVector,
                          x0::Union{Nothing,AbstractVector},
                          max_iterations::Integer, tolerance::Real)
    A = Matrix{Float64}(Ain)
    b = _mlembs_floatvec(bin)
    m, n = size(A)
    length(b) == m || throw(ArgumentError(
        "Length of b ($(length(b))) must match number of rows in A ($m)"))
    max_iterations > 0 || throw(ArgumentError(
        "max_iterations must be positive, got $max_iterations"))
    tolerance > 0 || throw(ArgumentError("tolerance must be positive, got $tolerance"))
    if x0 !== nothing
        length(x0) == n || throw(ArgumentError(
            "Length of x0 ($(length(x0))) must match number of columns in A ($n)"))
    end
    return A, b, x0
end

# ─── B-spline basis (local Cox–de Boor, no external spline dependency) ──────

function _mlembs_resolve_knot_spacing(knot_spacing::AbstractString,
                                      E::AbstractVector{Float64})
    knot_spacing in _mlembs_VALID_KNOT_SPACING || throw(ArgumentError(
        "knot_spacing must be one of $_mlembs_VALID_KNOT_SPACING, got \"$knot_spacing\""))
    knot_spacing != "auto" && return knot_spacing
    ratio = !isempty(E) && minimum(E) > 0 ? maximum(E) / minimum(E) : Inf
    return ratio > 100.0 ? "log" : "uniform"
end

"""
    _mlembs_basis_row!(row, t, x, n_basis, degree)

Evaluate all `n_basis` B-splines of the given `degree` on the (clamped)
knot vector `t` at the point `x`, iteratively (Cox–de Boor).
"""
function _mlembs_basis_row!(row::AbstractVector{Float64},
                            t::AbstractVector{Float64}, x::Float64,
                            n_basis::Int, degree::Int)
    fill!(row, 0.0)
    emax = t[end]
    d = zeros(Float64, n_basis)
    @inbounds for j in 1:n_basis
        d[j] = ((t[j] <= x < t[j+1]) || (x == emax && t[j] < t[j+1] == emax)) ? 1.0 : 0.0
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
    copyto!(row, d)
    return row
end

"""
    _mlembs_build_bspline_basis(E, n_basis, spline_order, knot_spacing)

Build the clamped B-spline design matrix `(length(E), n_basis)` with a
Cox–de Boor recursion.  `knot_spacing` must already be resolved (one of
`"uniform"`, `"log"`); interior knots are `n_basis - spline_order` points
uniform (or geometric) in energy between `E[1]` and `E[end]`.  The
clamped basis reproduces the boundary energies exactly and forms a
partition of unity on `[E[1], E[end]]`.
"""
function _mlembs_build_bspline_basis(E::AbstractVector{Float64}, n_basis::Int,
                                     spline_order::Int, knot_spacing::AbstractString)
    length(E) >= 2 || throw(ArgumentError(
        "E_MeV must be a 1D array with >= 2 points, got length $(length(E))"))
    all(isfinite, E) || throw(ArgumentError("E_MeV contains non-finite values"))
    all(i -> E[i+1] > E[i], 1:(length(E)-1)) || throw(ArgumentError(
        "E_MeV must be strictly increasing"))
    all(>(0), E) || throw(ArgumentError("E_MeV must contain positive energies"))
    2 <= spline_order <= 8 || throw(ArgumentError(
        "spline_order must be in [2, 8], got $spline_order"))
    n_basis >= spline_order || throw(ArgumentError(
        "n_basis (N_s = $n_basis) must be >= spline_order (p = $spline_order) " *
        "for a clamped B-spline basis"))

    degree = spline_order - 1
    emin, emax = Float64(E[1]), Float64(E[end])
    n_interior = n_basis - spline_order   # from length(t) = n_basis + degree + 1
    interior = if n_interior > 0
        knot_spacing == "log" ?
            _mlembs_geomspace(emin, emax, n_interior + 2)[2:end-1] :
            collect(range(emin, emax; length=n_interior + 2))[2:end-1]
    else
        Float64[]
    end
    t = vcat(fill(emin, spline_order), collect(Float64, interior),
             fill(emax, spline_order))

    B = Matrix{Float64}(undef, length(E), n_basis)
    row = zeros(Float64, n_basis)
    for (k, x0) in pairs(E)
        x = clamp(Float64(x0), emin, emax)  # clip for float fuzz at bounds
        _mlembs_basis_row!(row, t, x, n_basis, degree)
        @inbounds B[k, :] .= row
    end
    return B
end

"""
    build_bspline_basis(E_MeV, n_basis, spline_order=4, knot_spacing="auto")

B-spline design matrix `B` on an energy grid: the spectrum is represented
as `x(E) = B(E) * b`.  `knot_spacing` is `"auto"` (log-uniform interior
knots when the grid spans more than two decades, uniform otherwise),
`"uniform"` or `"log"`.  Implemented with a local Cox–de Boor recursion
(clamped knot vector, partition of unity).
"""
function build_bspline_basis(E_MeV::AbstractVector{<:Real}, n_basis::Integer,
                             spline_order::Integer=4,
                             knot_spacing::AbstractString="auto")
    E = _mlembs_floatvec(E_MeV)
    resolved = _mlembs_resolve_knot_spacing(knot_spacing, E)
    return _mlembs_build_bspline_basis(E, Int(n_basis), Int(spline_order), resolved)
end

"""
    second_difference_matrix(n)

Second-derivative (second finite difference) matrix `D2` of shape
`(n - 2, n)` with rows `[..., 1, -2, 1, ...]`; used in the penalty
`P(b) = ||D2 b||^2` (Eq. 5).  Requires `n >= 3`.
"""
function second_difference_matrix(n::Integer)
    n = Int(n)
    n >= 3 || throw(ArgumentError(
        "second_difference_matrix requires n >= 3, got $n"))
    D = zeros(Float64, n - 2, n)
    for i in 1:(n-2)
        D[i, i] = 1.0
        D[i, i+1] = -2.0
        D[i, i+2] = 1.0
    end
    return D
end

"""
    ks_statistic(b, model)

The `K_S` goodness-of-fit statistic of Eq. 6:
`K_S = | sum_i (n_i - model_i)^2 / sum_i model_i - 1 |`.
The iteration count, `N_s` and `beta` are selected by minimizing it.
"""
function ks_statistic(b::AbstractVector{<:Real}, model::AbstractVector{<:Real})
    bv = _mlembs_floatvec(b)
    mv = _mlembs_floatvec(model)
    denom = sum(mv)
    denom <= _MLEMBS_TINY && return Inf
    resid = bv .- mv
    return abs(sum(abs2, resid) / denom - 1.0)
end

# ─── Core MLEM-BS iteration ──────────────────────────────────────────────────

function _mlembs_argmin_index(values::AbstractVector{Float64})
    best_i = 1
    best_v = Inf
    for i in eachindex(values)
        v = values[i]
        if isfinite(v) && v < best_v
            best_v = v
            best_i = i
        end
    end
    return best_i
end

function _mlembs_sieve_projection(B::Matrix{Float64}, x0::AbstractVector{Float64})
    ns = size(B, 2)
    mean_x0 = isempty(x0) ? 0.0 : sum(x0) / length(x0)
    fallback = fill(max(mean_x0, _MLEMBS_COUNT_FLOOR), ns)
    b0 = try
        _mlembs_solve_nnls(B, x0)
    catch
        b0 = max.(B \ x0, 0.0)   # unconstrained LSQ clipped at zero
    end
    if !all(isfinite, b0)
        return fallback
    end
    if !any(>(0), b0)
        # pathological all-zero projection: keep the iteration alive
        return fill(max(mean_x0, _MLEMBS_COUNT_FLOOR), ns)
    end
    return b0
end

function _mlembs_resolve_beta(beta::Union{Nothing,Real},
                              beta_relative::Union{Nothing,Real},
                              RB::Matrix{Float64})
    beta === nothing && beta_relative === nothing ||
        (beta !== nothing && beta_relative !== nothing && throw(ArgumentError(
            "Provide either 'beta' (absolute, paper Eq. 4) or " *
            "'beta_relative' (scaled by mean sensitivity), not both")))
    if beta_relative !== nothing
        br = Float64(beta_relative)
        br >= 0 || throw(ArgumentError("beta_relative must be >= 0, got $br"))
        s_bar = isempty(RB) ? 0.0 : sum(RB; dims=1) |> v -> sum(v) / length(v)
        return br * s_bar, br
    end
    beta_eff = beta === nothing ? 0.0 : Float64(beta)
    beta_eff >= 0 || throw(ArgumentError("beta must be >= 0, got $beta_eff"))
    return beta_eff, nothing
end

function _mlembs_iterate(RB::Matrix{Float64}, b::Vector{Float64},
                         b_coef::Vector{Float64},
                         pen_mat::Union{Nothing,Matrix{Float64}},
                         beta::Float64, max_iterations::Int, tolerance::Float64,
                         ks_min_mode::Bool, ks_patience::Int)
    s_s = vec(sum(RB; dims=1))          # MLEM sensitivity: sum_i (RB)_is
    s_s_safe = max.(s_s, _MLEMBS_TINY)

    best_coef = copy(b_coef)
    best_ks = ks_statistic(b, RB * b_coef)
    ks_history = Float64[best_ks]
    converged = false
    iterations = 0

    for k in 1:max_iterations
        # forward projection: sum_s' (RB)_is' b_s'^(k)
        fwd = RB * b_coef
        fwd = max.(fwd, _MLEMBS_COUNT_FLOOR)
        ratio = b ./ fwd

        # multiplicative ML correction
        correction = RB' * ratio

        # penalized denominator: sum_i (RB)_is + beta * dP/db_s   (Eq. 4)
        local denom
        if beta != 0.0 && pen_mat !== nothing
            grad_P = pen_mat * b_coef
            d0 = s_s .+ beta .* grad_P
            # one-step-late guard: fall back to the unpenalized sensitivity
            # where the penalty gradient would flip the denominator
            # non-positive (standard OSL safeguard)
            denom = max.(d0, s_s_safe)
        else
            denom = s_s_safe
        end
        denom = max.(denom, _MLEMBS_TINY)

        b_new = b_coef .* (correction ./ denom)
        b_new = max.(b_new, 0.0)   # B-spline sieve: b_s >= 0

        iterations = k

        # K_S statistic of the new iterate (Eq. 6)
        fwd_new = RB * b_new
        ks = ks_statistic(b, fwd_new)
        push!(ks_history, ks)

        if ks_min_mode
            if ks < best_ks
                best_ks = ks
                best_coef = copy(b_new)
            elseif (k + 1) - _mlembs_argmin_index(ks_history) >= ks_patience
                b_coef = b_new
                break
            end
        else
            best_coef = b_new
        end

        # convergence on relative coefficient change
        diff = norm(b_new .- b_coef) / (norm(b_coef) + _MLEMBS_TINY)
        b_coef = b_new
        if diff < tolerance
            converged = true
            break
        end
    end

    return best_coef, iterations, converged, ks_history
end

# ─── Auto parameter selection (paper: minimize K_S over N_s, beta, iters) ───

function _mlembs_auto_ns_grid(n_energy_bins::Int, spline_order::Int;
                              n_points::Integer=8)
    lo = max(spline_order + 1, 8)
    hi = max(min(Int(n_energy_bins), 150), lo)
    hi <= lo && return [lo]
    grid = _mlembs_geomspace(Float64(lo), Float64(hi), Int(n_points))
    sizes = sort!(collect(Set{Int}(Int(round(v)) for v in grid) ∪ Set{Int}([lo, hi])))
    return [v for v in sizes if v >= spline_order + 1]
end

function _mlembs_interior_knots(E::AbstractVector{Float64}, n_basis::Int,
                                spline_order::Int, knot_spacing::AbstractString)
    n_interior = n_basis - spline_order
    n_interior <= 0 && return Float64[]
    emin, emax = Float64(E[1]), Float64(E[end])
    if knot_spacing == "log"
        return _mlembs_geomspace(emin, emax, n_interior + 2)[2:end-1]
    end
    return collect(range(emin, emax; length=n_interior + 2))[2:end-1]
end

function _mlembs_chi2_pearson(b::Vector{Float64}, model::Vector{Float64})
    acc = 0.0
    any_valid = false
    @inbounds for i in eachindex(b)
        if model[i] > _MLEMBS_COUNT_FLOOR
            acc += (b[i] - model[i])^2 / model[i]
            any_valid = true
        end
    end
    return any_valid ? acc : Inf
end

# ─── Public solvers ──────────────────────────────────────────────────────────

function _mlembs_run_single(A::Matrix{Float64}, b::Vector{Float64},
                            x0_in::Vector{Float64}, E::Vector{Float64},
                            ns::Int, spline_order::Int, knot_spacing::AbstractString,
                            beta_eff::Float64, beta_rel::Union{Nothing,Float64},
                            max_iter::Int, tolerance::Float64,
                            ks_mode::Bool, ks_patience::Int)
    B = _mlembs_build_bspline_basis(E, ns, spline_order, knot_spacing)
    RB = A * B
    pen_mat = nothing
    if beta_eff != 0.0
        D2 = second_difference_matrix(ns)
        pen_mat = 2.0 .* (D2' * D2)
    end
    b0 = _mlembs_sieve_projection(B, x0_in)
    best_coef, iterations, converged, ks_history = _mlembs_iterate(
        RB, b, b0, pen_mat, beta_eff, max_iter, tolerance, ks_mode, ks_patience)
    spectrum = B * best_coef
    model = RB * best_coef
    return Dict{String,Any}(
        "spectrum"       => spectrum,
        "coefficients"   => best_coef,
        "iterations"     => Int(iterations),
        "converged"      => converged,
        "ks_history"     => ks_history,
        "ks_final"       => isempty(ks_history) ? Inf : ks_history[end],
        "chi2_pearson"   => _mlembs_chi2_pearson(b, model),
        "n_basis"        => Int(ns),
        "spline_order"   => Int(spline_order),
        "knot_spacing"   => knot_spacing,
        "interior_knots" => _mlembs_interior_knots(E, ns, spline_order, knot_spacing),
        "beta_effective" => Float64(beta_eff),
        "beta_relative"  => beta_rel,
        "method"         => "MLEM-BS",
    )
end

"""
    solve_mlem_bs_full(A, b, x0=nothing; E_MeV=nothing, n_basis=nothing,
                       spline_order=4, beta=nothing, beta_relative=nothing,
                       knot_spacing="auto", max_iterations=1000,
                       tolerance=1e-6, auto_params=false, ks_patience=50)

Solve the unfolding problem with MLEM-BS and return an `UnfoldResult`
whose `extra` dictionary carries the rich Python diagnostics: keys
`coefficients`, `iterations`, `converged`, `ks_history`, `ks_final`,
`chi2_pearson`, `n_basis`, `spline_order`, `knot_spacing`,
`interior_knots`, `beta_effective`, `beta_relative`, `method` and, when
`auto_params=true`, `auto_selection` (candidate table with the chosen
`N_s`, penalty strength and iteration count selected by minimizing
`K_S`, Eq. 6).

- `E_MeV` — energy grid (MeV), `size(A, 2)` points; when `nothing` a
  uniform grid `1:J` is used (the Python API requires the grid
  explicitly — the default here keeps the solver usable without it).
- `n_basis` — B-spline space dimension `N_s`; default selects
  `max(spline_order + 1, min(J ÷ 2, 40))`.
- `beta` / `beta_relative` — absolute or scale-aware penalty of Eq. 4
  (mutually exclusive; neither → no penalty beyond the sieve).
- `knot_spacing` — `"auto"` (log knots for grids spanning > 2 decades),
  `"uniform"` or `"log"`.
- `auto_params=true` selects `N_s`, the penalty strength and the
  iteration count by minimization of `K_S` (paper's optimization
  procedure).
"""
function solve_mlem_bs_full(Ain::AbstractMatrix{T}, bin::AbstractVector{T};
                            x0::Union{Nothing,AbstractVector{T}}=nothing,
                            E_MeV::Union{Nothing,AbstractVector{<:Real}}=nothing,
                            n_basis::Union{Nothing,Integer}=nothing,
                            spline_order::Integer=4,
                            beta::Union{Nothing,Real}=nothing,
                            beta_relative::Union{Nothing,Real}=nothing,
                            knot_spacing::AbstractString="auto",
                            max_iterations::Integer=1000,
                            tolerance::T=T(1e-6),
                            auto_params::Bool=false,
                            ks_patience::Integer=50) where T<:AbstractFloat
    A, b, x0_in = _mlembs_validate(Ain, bin, x0, max_iterations, tolerance)
    n_bins = size(A, 2)
    E = E_MeV === nothing ? collect(range(1.0, Float64(n_bins); length=n_bins)) :
        _mlembs_floatvec(E_MeV)
    length(E) == n_bins || throw(ArgumentError(
        "Length of E_MeV ($(length(E))) must match the number of columns of A ($n_bins)"))
    if x0_in === nothing
        x0_in = fill(0.5, n_bins)
    else
        x0_in = _mlembs_floatvec(x0_in)
    end
    knot_spacing = _mlembs_resolve_knot_spacing(knot_spacing, E)

    # ── auto selection of (N_s, beta, iterations) by minimizing K_S ──
    if auto_params
        ns_grid = _mlembs_auto_ns_grid(n_bins, Int(spline_order))
        # cap scan iterations for tractability; final run uses the budget
        scan_iter = Int(min(Int(max_iterations), 400))
        candidates = Dict{String,Any}[]
        best = nothing
        for ns in ns_grid
            for rho in _mlembs_AUTO_BETA_RELATIVE_GRID
                beta_eff = 0.0
                if rho > 0.0
                    B = _mlembs_build_bspline_basis(E, ns, Int(spline_order), knot_spacing)
                    colsums = vec(sum(A * B; dims=1))
                    s_bar = sum(colsums) / length(colsums)
                    beta_eff = rho * s_bar
                end
                res = _mlembs_run_single(A, b, x0_in, E, ns, Int(spline_order),
                                         knot_spacing, beta_eff,
                                         rho > 0 ? rho : 0.0,
                                         scan_iter, Float64(tolerance),
                                         true, Int(ks_patience))
                hist = res["ks_history"]::Vector{Float64}
                entry = Dict{String,Any}(
                    "n_basis"         => Int(ns),
                    "beta_relative"   => Float64(rho),
                    "ks"              => minimum(hist),
                    "ks_iteration"    => Int(_mlembs_argmin_index(hist) - 1),
                    "iterations"      => res["iterations"],
                )
                push!(candidates, entry)
                if best === nothing || entry["ks"] < best["ks"]
                    best = entry
                end
            end
        end
        # final run with the full iteration budget in K_S-minimizing mode
        chosen_ns::Int = best["n_basis"]
        chosen_rho::Float64 = best["beta_relative"]
        beta_eff = 0.0
        if chosen_rho > 0.0
            Bc = _mlembs_build_bspline_basis(E, chosen_ns, Int(spline_order), knot_spacing)
            colsums = vec(sum(A * Bc; dims=1))
            beta_eff = chosen_rho * (sum(colsums) / length(colsums))
        end
        result = _mlembs_run_single(A, b, x0_in, E, chosen_ns, Int(spline_order),
                                    knot_spacing, beta_eff,
                                    chosen_rho > 0 ? chosen_rho : 0.0,
                                    Int(max_iterations), Float64(tolerance),
                                    true, Int(ks_patience))
        result["auto_selection"] = Dict{String,Any}(
            "candidates"         => candidates,
            "chosen"             => best,
            "ns_grid"            => ns_grid,
            "beta_relative_grid" => collect(_mlembs_AUTO_BETA_RELATIVE_GRID),
        )
        result["beta_relative"] = chosen_rho > 0 ? chosen_rho : nothing
        result["beta_effective"] = beta_eff

        spectrum = result["spectrum"]::Vector{Float64}
        extra = Dict{String,Any}(k => v for (k, v) in result if k != "spectrum")
        resid = b .- A * spectrum
        return UnfoldResult(Vector{T}(spectrum), result["iterations"]::Int,
                            result["converged"]::Bool, T(norm(resid)), extra)
    end

    # ── fixed parameters ──
    ns = n_basis === nothing ?
        max(Int(spline_order) + 1, min(n_bins ÷ 2, 40)) : Int(n_basis)
    beta_eff, beta_rel = _mlembs_resolve_beta(
        beta, beta_relative, zeros(Float64, length(b), 1))
    # resolve beta against the actual RB (s_bar depends on the basis)
    if beta_rel !== nothing
        B = _mlembs_build_bspline_basis(E, ns, Int(spline_order), knot_spacing)
        colsums = vec(sum(A * B; dims=1))
        beta_eff = beta_rel * (sum(colsums) / length(colsums))
    end
    result = _mlembs_run_single(A, b, x0_in, E, ns, Int(spline_order),
                                knot_spacing, beta_eff, beta_rel,
                                Int(max_iterations), Float64(tolerance),
                                false, Int(ks_patience))

    spectrum = result["spectrum"]::Vector{Float64}
    extra = Dict{String,Any}(k => v for (k, v) in result if k != "spectrum")
    resid = b .- A * spectrum
    return UnfoldResult(Vector{T}(spectrum), result["iterations"]::Int,
                        result["converged"]::Bool, T(norm(resid)), extra)
end


"""
    solve_mlem_bs(A, b, x0=nothing; E_MeV=nothing, n_basis=nothing,
                  spline_order=4, beta=nothing, beta_relative=nothing,
                  knot_spacing="auto", max_iterations=1000, tolerance=1e-6,
                  auto_params=false, ks_patience=50)

Solve unfolding with the B-spline MLEM (MLEM-BS) algorithm: convenience
wrapper around [`solve_mlem_bs_full`](@ref); the `spectrum`, `iterations`
and `converged` components of the returned `UnfoldResult` correspond to
the Python `(spectrum, iterations, converged)` tuple.
"""
function solve_mlem_bs(Ain::AbstractMatrix{T}, bin::AbstractVector{T};
                       x0::Union{Nothing,AbstractVector{T}}=nothing,
                       E_MeV::Union{Nothing,AbstractVector{<:Real}}=nothing,
                       n_basis::Union{Nothing,Integer}=nothing,
                       spline_order::Integer=4,
                       beta::Union{Nothing,Real}=nothing,
                       beta_relative::Union{Nothing,Real}=nothing,
                       knot_spacing::AbstractString="auto",
                       max_iterations::Integer=1000,
                       tolerance::T=T(1e-6),
                       auto_params::Bool=false,
                       ks_patience::Integer=50) where T<:AbstractFloat
    res = solve_mlem_bs_full(Ain, bin; x0=x0,
                             E_MeV=E_MeV, n_basis=n_basis,
                             spline_order=spline_order,
                             beta=beta, beta_relative=beta_relative,
                             knot_spacing=knot_spacing,
                             max_iterations=max_iterations,
                             tolerance=tolerance,
                             auto_params=auto_params,
                             ks_patience=ks_patience)
    return UnfoldResult(res.spectrum, res.iterations, res.converged,
                        res.residual_norm)
end

# Positional-x0 convenience (Python keyword `x0` is also accepted as the
# third positional argument, like in other BSSUnfold.jl solvers).
solve_mlem_bs_full(A::AbstractMatrix{T}, b::AbstractVector{T},
                   x0::AbstractVector{T}; kwargs...) where T<:AbstractFloat =
    solve_mlem_bs_full(A, b; x0=x0, kwargs...)
solve_mlem_bs(A::AbstractMatrix{T}, b::AbstractVector{T},
              x0::AbstractVector{T}; kwargs...) where T<:AbstractFloat =
    solve_mlem_bs(A, b; x0=x0, kwargs...)
