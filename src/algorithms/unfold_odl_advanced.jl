"""
Advanced unfolding via the Chambolle–Pock primal-dual hybrid gradient
(PDHG) scheme and Douglas–Rachford operator splitting, both with an
optional 1-D total-variation (TV) penalty. Port of
`bssunfold/core/unfold_odl_advanced.py`.

Despite the module name no external ODL library is used: ODL 1.0's own
PDHG / Douglas-Rachford solvers break on translated data terms, so the
Python reference reimplements them in pure NumPy and we mirror that
arithmetic statement-for-statement.

* `solve_odl_pdhg` builds the augmented operator `K = [A; tv_weight*D]`
  and `d = [b; 0]` so that `0.5‖Ax − b‖² + tv_weight‖Dx‖₁` becomes
  `0.5‖Kx − d‖²`, runs the closed-form primal-dual iteration on that
  quadratic and returns `(x_opt, max_iterations, converged)`.
* `solve_odl_douglas_rachford` splits `ψ₁(x) = 0.5‖Ax − b‖²` (prox via a
  precomputed Cholesky of `I + γAᵀA`, `γ = 1`) from `ψ₂(x) = tv_weight‖Dx‖₁`
  (prox = `_tv_prox`).

`_tv_prox` is Chambolle's dual gradient-ascent TV denoiser with the fixed
`L = 4.0` bound on the largest eigenvalue of `D Dᵀ`.
"""

function _forward_diff_matrix(n::Integer)
    D = zeros(Float64, n - 1, n)
    for i in 1:n - 1
        D[i, i] = -1.0
        D[i, i + 1] = 1.0
    end
    return D
end

function _tv_prox(f::AbstractVector{T}, lam::Real, n_iter::Integer=200) where T<:AbstractFloat
    n = length(f)
    lamv = T(lam)
    (n <= 1 || lamv <= zero(T)) && return copy(f)
    Df = f[2:n] .- f[1:n-1]
    L = T(4.0)
    rho = one(T) / (L * lamv * lamv)
    p = zeros(T, n - 1)
    for _ in 1:Int(n_iter)
        grad = zeros(T, n)
        grad[2:n] = p
        grad[1:n-1] .-= p
        p = p + rho * (lamv .* Df - (lamv * lamv) .* (grad[2:n] - grad[1:n-1]))
        p = clamp.(p, -one(T), one(T))
    end
    grad = zeros(T, n)
    grad[2:n] = p
    grad[1:n-1] .-= p
    return f - lamv .* grad
end

function solve_odl_pdhg(A::AbstractMatrix{T}, b::AbstractVector{T}, x0::AbstractVector{T};
                        max_iterations::Integer=100,
                        tau::Union{Real,Nothing}=nothing,
                        sigma::Union{Real,Nothing}=nothing,
                        use_tv::Bool=true,
                        tv_weight::Real=T(0.1),
                        nonnegativity::Bool=true,
                        tolerance::Real=T(1e-6)) where T<:AbstractFloat
    A = Matrix{T}(A)
    b = Vector{T}(b)
    m, n = size(A)

    x = Vector{T}(x0)

    if use_tv
        D = _forward_diff_matrix(n)
        K = [A; T(tv_weight) .* D]
        d = [b; zeros(T, n - 1)]
    else
        K = A
        d = b
    end

    eig_max = maximum(eigvals(Symmetric(K' * K)))
    op_norm = sqrt(max(T(eig_max), T(1e-12)))
    tau_val = tau === nothing ? T(0.99) / op_norm : T(tau)
    sigma_val = sigma === nothing ? T(0.99) / op_norm : T(sigma)

    b_norm = max(norm(b), T(1e-300))
    res_before = norm(A * x - b) / b_norm

    y = zeros(T, size(K, 1))
    x_bar = copy(x)
    for _ in 1:Int(max_iterations)
        z = y + sigma_val .* (K * x_bar)
        y = (z - sigma_val .* d) / (one(T) + sigma_val)
        x_new = x - tau_val .* (K' * y)
        nonnegativity && (x_new = max.(x_new, zero(T)))
        x_bar = x_new + (x_new - x)
        x = x_new
    end

    x_opt = nonnegativity ? max.(x, zero(T)) : x
    finite = all(isfinite, x_opt)
    res_after = norm(A * x_opt - b) / b_norm
    converged = finite && res_after <= res_before * (one(T) + T(tolerance) * T(100))

    return UnfoldResult(Vector{T}(x_opt), Int(max_iterations), converged,
                        norm(b - A * x_opt),
                        Dict{String,Any}("max_iterations" => Int(max_iterations),
                                         "use_tv" => use_tv,
                                         "tv_weight" => T(tv_weight),
                                         "tau" => tau_val,
                                         "sigma" => sigma_val,
                                         "op_norm" => op_norm))
end

function solve_odl_douglas_rachford(A::AbstractMatrix{T}, b::AbstractVector{T}, x0::AbstractVector{T};
                                    max_iterations::Integer=100,
                                    use_tv::Bool=true,
                                    tv_weight::Real=T(0.1),
                                    nonnegativity::Bool=true,
                                    tolerance::Real=T(1e-6)) where T<:AbstractFloat
    A = Matrix{T}(A)
    b = Vector{T}(b)
    m, n = size(A)

    x = Vector{T}(x0)

    gamma = one(T)
    M = Matrix{T}(I, n, n) + gamma .* (A' * A)
    M_chol = cholesky(Symmetric(M, :L))
    Atb = A' * b

    prox_psi1(v) = M_chol \ (v + gamma .* Atb)

    b_norm = max(norm(b), T(1e-300))
    res_before = norm(A * x - b) / b_norm

    y = copy(x)
    z = copy(x)
    for _ in 1:Int(max_iterations)
        u = prox_psi1(2.0 .* z - y)
        y = use_tv ? _tv_prox(u, gamma * T(tv_weight)) : u
        z = z + y - u
    end

    x_opt = nonnegativity ? max.(y, zero(T)) : y
    finite = all(isfinite, x_opt)
    res_after = norm(A * x_opt - b) / b_norm
    converged = finite && res_after <= res_before * (one(T) + T(tolerance) * T(100))

    return UnfoldResult(Vector{T}(x_opt), Int(max_iterations), converged,
                        norm(b - A * x_opt),
                        Dict{String,Any}("max_iterations" => Int(max_iterations),
                                         "use_tv" => use_tv,
                                         "tv_weight" => T(tv_weight),
                                         "gamma" => gamma))
end
