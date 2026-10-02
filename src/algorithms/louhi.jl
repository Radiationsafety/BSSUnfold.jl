"""
LOUHI — general-purpose unfolding with generalized smoothing (linear/nonlinear mode).

Faithful port of `solve_louhi` from bssunfold (`unfold_louhi.py`), after the
LOUHI78 program of J. T. Routti and V. Sandberg, "General purpose unfolding
program LOUHI78 with linear and nonlinear regressions", Computer Physics
Communications 21 (1980) 119-135, doi:10.1016/0010-4655(80)90021-4.

Minimizes the constrained weighted least-squares functional

    chi2(phi) = sum_i [(b_i - (A phi)_i) / sigma_i]^2
                + lambda^2 * sum_k [L (phi - phi0)]_k^2

with the a-priori (default) spectrum `x0` as smoothing reference, by Hildreth's
iterative coordinate quadratic-programming sweeps with non-negativity projection
(LOUHI's LSI step). In nonlinear mode (`auto_smooth=true`) the smoothing weight
lambda is adjusted by a golden-section search on log10(lambda) so the data
chi-square reaches `chi2_target` (default: number of detectors).
"""

const _LOUHI_EPS = 1e-12
const _LOUHI_INV_PHI = (sqrt(5.0) - 1.0) / 2.0  # 1/phi ~ 0.618

function _louhi_smoothing_matrix(n::Int, smooth_order::Int)
    if !(smooth_order in (0, 1, 2))
        throw(ArgumentError("smooth_order must be one of (0, 1, 2), got $smooth_order"))
    end
    if n < 1
        throw(ArgumentError("n must be positive, got $n"))
    end
    if smooth_order == 0 || n == 1
        return Matrix{Float64}(I, n, n)
    end
    first = zeros(n - 1, n)
    for i in 1:n-1
        first[i, i] = -1.0
        first[i, i + 1] = 1.0
    end
    if smooth_order == 1 || n == 2
        return first
    end
    second = zeros(n - 2, n)
    for i in 1:n-2
        second[i, i] = 1.0
        second[i, i + 1] = -2.0
        second[i, i + 2] = 1.0
    end
    return second
end

"""
    _louhi_golden_section(func, lo, hi, tolerance=1e-8, max_iterations=200)

Golden-section minimization of a unimodal scalar function on `[lo, hi]`
(inlined from bssunfold `core/_line_search.py::golden_section_minimize`).
Returns `(t_opt, f_opt)`.
"""
function _louhi_golden_section(func, lo::Real, hi::Real,
                               tolerance::Real=1e-8, max_iterations::Int=200)
    isfinite(lo) && isfinite(hi) || throw(ArgumentError("golden section requires finite bounds"))
    if hi <= lo
        t = Float64(lo)
        return t, Float64(func(t))
    end
    invphi = _LOUHI_INV_PHI
    a, b = Float64(lo), Float64(hi)
    c = b - invphi * (b - a)
    d = a + invphi * (b - a)
    fc, fd = Float64(func(c)), Float64(func(d))
    for _ in 1:max_iterations
        if b - a <= tolerance
            break
        end
        if fc < fd
            b, d, fd = d, c, fc
            c = b - invphi * (b - a)
            fc = Float64(func(c))
        else
            a, c, fc = c, d, fd
            d = a + invphi * (b - a)
            fd = Float64(func(d))
        end
    end
    t_opt = 0.5 * (a + b)
    return t_opt, Float64(func(t_opt))
end

"""
    _louhi_hildreth_qp(H, g, x0, max_iterations, tolerance)

Hildreth's coordinate QP: minimize `1/2 x'Hx - g'x` s.t. `x >= 0` by cyclic
coordinate minimization with projection onto the non-negativity box.
Convergence: relative change of the quadratic objective between consecutive
sweeps below `tolerance`. Returns `(x, sweeps, converged)`.
"""
function _louhi_hildreth_qp(H::AbstractMatrix{T}, g::AbstractVector{T},
                            x0::AbstractVector{T},
                            max_iterations::Int, tolerance::T) where T<:AbstractFloat
    x = max.(Vector{T}(x0), zero(T))
    n = length(x)
    diag_H = max.(diag(H), T(_LOUHI_EPS))
    converged = false
    sweeps = 0
    prev_obj = typemax(T)
    for sweep in 1:max_iterations
        sweeps = sweep
        for j in 1:n
            grad_j = dot(H[j, :], x) - g[j]
            x[j] = max(zero(T), x[j] - grad_j / diag_H[j])
        end
        obj = T(0.5) * dot(x, H * x) - dot(g, x)
        if abs(prev_obj - obj) <= tolerance * max(one(T), abs(obj))
            converged = true
            break
        end
        prev_obj = obj
    end
    return x, sweeps, converged
end

function _louhi_weighted_normal_equations(A::AbstractMatrix{T}, b::AbstractVector{T},
                                          sigma::AbstractVector{T}, x0::AbstractVector{T},
                                          L::AbstractMatrix, smoothness::T) where T<:AbstractFloat
    w = T(1) ./ (sigma .^ 2)
    ata = A' * (A .* reshape(w, :, 1))   # row scaling, == A * diag(w)
    atb = A' * (b .* w)
    ltl = Matrix{T}(L' * L)
    H = T(2) * (ata + smoothness^2 * ltl)
    g = T(2) * (atb + smoothness^2 * (ltl * x0))
    return H, g
end

function _louhi_data_chi2(A::AbstractMatrix{T}, b::AbstractVector{T},
                          sigma::AbstractVector{T}, x::AbstractVector{T}) where T<:AbstractFloat
    resid = (b .- A * x) ./ sigma
    return dot(resid, resid)
end

"""
    _louhi_smoothness_bracket(objective, lo=-6, hi=6, n_scan=25)

Scan a uniform grid on log10(lambda) in [lo, hi]; return the first adjacent
pair where the (monotone) objective changes sign, else the full interval.
"""
function _louhi_smoothness_bracket(objective, lo::Real=-6.0, hi::Real=6.0, n_scan::Int=25)
    grid = range(Float64(lo), Float64(hi), length=n_scan)
    values = [Float64(objective(t)) for t in grid]
    for k in 1:length(grid)-1
        if values[k] == 0.0 || values[k] * values[k + 1] <= 0.0
            return Float64(grid[k]), Float64(grid[k + 1])
        end
    end
    return Float64(lo), Float64(hi)
end

function solve_louhi(A::AbstractMatrix{T}, b::AbstractVector{T}, x0::AbstractVector{T};
                     smoothness::Real=T(1.0),
                     smooth_order::Integer=1,
                     max_iterations::Integer=500,
                     tolerance::Real=T(1e-6),
                     relative_uncertainty::Real=0.1,
                     sigma::Union{AbstractVector,Nothing}=nothing,
                     auto_smooth::Bool=false,
                     chi2_target::Union{Real,Nothing}=nothing) where T<:AbstractFloat
    m, n = size(A)
    if m == 0
        throw(ArgumentError("Response matrix must be a non-empty 2-D array, got shape ($m, $n)"))
    end
    length(b) == m || throw(DimensionMismatch("Measurement vector length $(length(b)) does not match response matrix with $m rows"))
    if !any(b .> zero(T))
        throw(ArgumentError("At least one positive measurement is required; all readings are zero or negative"))
    end
    sig = if sigma !== nothing
        length(sigma) == m || throw(DimensionMismatch("sigma must have length $m, got $(length(sigma))"))
        max.(Vector{T}(sigma), T(_LOUHI_EPS))
    else
        T(relative_uncertainty) .* max.(b, T(_LOUHI_EPS))
    end

    apriori = max.(Vector{T}(x0), zero(T))
    length(apriori) == n || throw(DimensionMismatch("Default spectrum length $(length(apriori)) does not match the number of energy bins $n"))
    lam0 = T(smoothness)
    lam0 >= zero(T) || throw(ArgumentError("smoothness must be non-negative, got $lam0"))
    L = _louhi_smoothing_matrix(n, Int(smooth_order))
    tol = T(tolerance)

    target = T(chi2_target === nothing ? m : chi2_target)

    function solve_with(lam::T)
        H, g = _louhi_weighted_normal_equations(A, b, sig, apriori, L, lam)
        x, _, _ = _louhi_hildreth_qp(H, g, apriori, Int(max_iterations), tol)
        return x
    end

    lambda_used = lam0
    if auto_smooth && max_iterations > 0
        lo, hi = _louhi_smoothness_bracket(t -> _louhi_data_chi2(A, b, sig, solve_with(T(10.0^t))) - target)
        best_log_lam, _ = _louhi_golden_section(log_lam -> abs(_louhi_data_chi2(A, b, sig, solve_with(T(10.0^log_lam))) - target), lo, hi)
        lambda_used = T(10.0^best_log_lam)
    end

    H, g = _louhi_weighted_normal_equations(A, b, sig, apriori, L, lambda_used)
    x, sweeps, converged = _louhi_hildreth_qp(H, g, apriori, Int(max_iterations), tol)

    residual = b .- A * x
    return UnfoldResult(x, sweeps, converged, norm(residual))
end
