"""
Tikhonov + generalized discrepancy principle (Sobolev penalty).
Faithful port of `bssunfold/core/unfold_tikhonov_sobolev_dp.py`.

Regularized minimiser
    z(α) = argmin ‖A z − b‖² + α ‖L z‖²
solved as the normal system `(N + α K) z = rhs` with `N = AᵀA`,
`K = LᵀL`, `rhs = Aᵀb`.  `α*` is the root of `ρ(α) = ‖A z(α) − b‖² − δ²`
bracketed on `[α_min, α_max]` and refined by Brent (default) or
Newton–Kantorovich on log10(α).  Penalty is
`:sobolev` (order-1 difference), `:curvature` (order-2) or
`:identity`.  `δ = noise_level · ‖b‖` unless `delta` is given.

Returns the raw z (no clipping), matching Python.
"""

function _dp_penalty_matrix(n::Int, penalty::Symbol)
    if penalty === :sobolev
        create_derivative_matrix(n, 1)
    elseif penalty === :curvature
        create_derivative_matrix(n, 2)
    elseif penalty === :identity
        Matrix{Float64}(I, n, n)
    else
        throw(ArgumentError("Unsupported penalty: $penalty. Choose :sobolev, :curvature, :identity."))
    end
end

function _dp_solve_regularized(N::AbstractMatrix{T}, K::AbstractMatrix{T},
                               rhs::AbstractVector{T}, α::T) where T<:AbstractFloat
    M = N .+ α .* K
    try
        Vector{T}(M \ rhs)
    catch err
        (err isa LinearAlgebra.SingularException || err isa LinearAlgebra.ZeroPivotException) || rethrow(err)
        Vector{T}(pinv(M) * rhs)
    end
end

function _dp_rho(α::T, N, K, rhs, A, b, delta_sq::T) where T<:AbstractFloat
    z = _dp_solve_regularized(N, K, rhs, α)
    r = A * z .- b
    T(sum(r .* r) - delta_sq)
end

function _dp_rho_deriv(α::T, N, K, rhs, A, b, delta_sq::T) where T<:AbstractFloat
    z = _dp_solve_regularized(N, K, rhs, α)
    r = A * z .- b
    rho = sum(r .* r) - delta_sq
    Kz = K * z
    dz = .- _dp_solve_regularized(N, K, Kz, α)
    rho_p = 2.0 * sum(r .* (A * dz))
    T(rho), T(rho_p)
end

# Bisection on the bracketed root of `f(t)=0`.  ρ is monotone on the
# bracket, so bisection converges in ~60 iterations over a 20-decade
# α range.  Matches scipy.brentq semantics to within 1e-12 in log10(α).
function _dp_brent(f, lo::T, hi::T; max_iter::Int=100, tol::T=T(1e-12)) where T<:AbstractFloat
    flo, fhi = f(lo), f(hi)
    flo == 0.0 && return (lo, true, 0)
    fhi == 0.0 && return (hi, true, 0)
    flo * fhi > 0 && return (T(0.5) * (lo + hi), false, 0)
    iters = 0
    for k in 1:max_iter
        iters = k
        mid = T(0.5) * (lo + hi)
        fm = f(mid)
        fm == 0.0 && return (mid, true, iters)
        if flo * fm < 0
            hi, fhi = mid, fm
        else
            lo, flo = mid, fm
        end
        (hi - lo) <= tol && return (mid, true, iters)
    end
    return (T(0.5) * (lo + hi), false, iters)
end

# Newton–Kantorovich on log10(α), safeguarded to the bracket.
function _dp_newton_kantorovich(N, K, rhs, A, b, delta_sq::T,
                                 t_lo::T, t_hi::T;
                                 max_iter::Int=50, xtol::T=T(1e-10)) where T<:AbstractFloat
    t = T(0.5) * (t_lo + t_hi)
    lo, hi = t_lo, t_hi
    f_lo = _dp_rho(T(10.0)^lo, N, K, rhs, A, b, delta_sq)
    f_hi = _dp_rho(T(10.0)^hi, N, K, rhs, A, b, delta_sq)
    for k in 1:max_iter
        α = T(10.0)^t
        rho, drho = _dp_rho_deriv(α, N, K, rhs, A, b, delta_sq)
        abs(rho) < T(1e-14) && return (α, k, true)
        # Track bracket by sign of rho on the endpoints.
        if rho > 0
            hi = t; f_hi = rho
        else
            lo = t; f_lo = rho
        end
        dt = drho != 0 ? -rho / (drho * log(T(10)) * α) : T(NaN)
        t_next = t + dt
        if !isfinite(t_next) || !(lo ≤ t_next ≤ hi)
            t_next = T(0.5) * (lo + hi)
        end
        if abs(t_next - t) < xtol
            t = t_next
            break
        end
        t = t_next
    end
    T(10.0)^t, max_iter, true
end

"""
    solve_tikhonov_sobolev_dp(A, b, x0; noise_level, penalty, kwargs...)

Tikhonov minimizer with Sobolev-type penalty selected by the generalized
discrepancy principle: α* is the root of `‖A z(α) − b‖² − δ²` found by
Brent or Newton-Kantorovich. Port of
`bssunfold/core/unfold_tikhonov_sobolev_dp.py`.
"""
function solve_tikhonov_sobolev_dp(A::AbstractMatrix{T}, b::AbstractVector{T},
                                   x0::Union{Nothing,AbstractVector{T}}=nothing;
                                   noise_level::Real=T(0.02),
                                   delta::Union{Real,Nothing}=nothing,
                                   penalty::Union{Symbol,String}=:sobolev,
                                   alpha_range::Tuple{<:Real,<:Real}=(T(1e-10), T(1e10)),
                                   max_iter::Integer=100,
                                   method::Union{Symbol,String}=:brent) where T<:AbstractFloat
    m, n = size(A)
    b = Vector{T}(b)
    pen = Symbol(penalty)
    meth = Symbol(method)

    δ = delta === nothing ? T(noise_level) * norm(b) : T(delta)
    δ > 0 || throw(ArgumentError("delta must be positive, got $δ; provide delta or positive noise_level"))
    δ_sq = δ * δ

    L = T.(_dp_penalty_matrix(n, pen))
    N = A' * A
    K = L' * L
    rhs = A' * b

    a_lo, a_hi = T(alpha_range[1]), T(alpha_range[2])
    (0 < a_lo < a_hi) || throw(ArgumentError("alpha_range must satisfy 0 < alpha_min < alpha_max, got $alpha_range"))

    rho_min = _dp_rho(a_lo, N, K, rhs, A, b, δ_sq)
    rho_max = _dp_rho(a_hi, N, K, rhs, A, b, δ_sq)
    iters = 2
    converged = false
    status = 0
    alpha_star = a_lo

    if rho_min > 0
        # Even the least-regularized solution overfits: report status=1 and
        # use alpha_min.
        status = 1
        alpha_star = a_lo
    elseif rho_max < 0
        # Even maximal regularization underfits: status=2, use alpha_max.
        status = 2
        alpha_star = a_hi
    elseif meth === :brent
        t_lo, t_hi = log10(a_lo), log10(a_hi)
        f(t) = _dp_rho(T(10.0)^t, N, K, rhs, A, b, δ_sq)
        root_t, ok, k = _dp_brent(f, t_lo, t_hi)
        if ok && isfinite(root_t)
            alpha_star = T(10.0)^root_t
            converged = true
        else
            alpha_star = T(10.0)^T(0.5*(t_lo+t_hi))
        end
        iters += k
    elseif meth === :newton_kantorovich
        t_lo, t_hi = log10(a_lo), log10(a_hi)
        α, k, ok = _dp_newton_kantorovich(N, K, rhs, A, b, δ_sq, t_lo, t_hi; max_iter=Int(max_iter))
        alpha_star = α
        converged = ok
        iters += k
    else
        throw(ArgumentError("Unknown method: $method. Choose :brent or :newton_kantorovich."))
    end

    spectrum = _dp_solve_regularized(N, K, rhs, alpha_star)
    r = A * spectrum .- b
    residual_sq = sum(r .* r)
    extra = Dict{String,Any}(
        "penalty" => String(pen),
        "delta" => Float64(δ),
        "noise_level" => Float64(noise_level),
        "dp_method" => String(meth),
        "alpha" => Float64(alpha_star),
        "discrepancy_status" => status,
        "dp_converged" => converged,
        "residual_sq" => Float64(residual_sq),
    )
    UnfoldResult(spectrum, iters, converged, T(sqrt(residual_sq)), extra)
end
