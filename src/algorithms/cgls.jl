"""CGLS — Conjugate Gradient for Least Squares — faithful port of
`bssunfold.core.unfold_cgls.solve_cgls`.

CG applied implicitly to the normal equations `AᵀA x = Aᵀb`
(Hansen, "Discrete Inverse Problems", Algorithm 6.1).  The solution is
regularized by *early stopping*: the iteration terminates when the
normal-equation residual `‖s‖ = ‖Aᵀ(b − Ax)‖` drops below
`tolerance * ‖Aᵀb‖` (or `tolerance`), or — when `noise_level` is given —
by the discrepancy principle `‖r‖ ≤ 1.01 · noise_level · ‖b‖`.

Nonnegativity is enforced by a single clamping pass **after** the
iteration (clamping inside the loop would destroy CG conjugacy).
Optionally a Tikhonov term `regularization²‖Lx‖²` with a derivative
operator `L` (`smoothness_order` 1 or 2) is added.
"""

"""
    create_derivative_matrix(n, order) -> Matrix

Finite-difference derivative matrix of shape `(n-1, n)` for order 1 or
`(n-2, n)` for order 2 (port of
`bssunfold.core._matrix_utils.create_derivative_matrix`).
"""
function create_derivative_matrix(n::Integer, order::Integer)
    order == 1 && begin
        L = zeros(n - 1, n)
        for i in 1:(n-1)
            L[i, i] = -1.0
            L[i, i+1] = 1.0
        end
        return L
    end
    order == 2 && begin
        L = zeros(n - 2, n)
        for i in 1:(n-2)
            L[i, i] = 1.0
            L[i, i+1] = -2.0
            L[i, i+2] = 1.0
        end
        return L
    end
    throw(ArgumentError("Unsupported derivative order: $order. Use 1 or 2."))
end

"""
    make_regularization_operator(n, smoothness_order; identity_for_zero=true)

Dense regularization operator `L` for derivative order 0/1/2 (port of
`bssunfold.core._matrix_utils.make_regularization_operator`).  With
`identity_for_zero=false` order 0 yields `nothing` so implicit solvers
can skip the regularization term entirely.
"""
function make_regularization_operator(n::Integer, smoothness_order::Integer;
                                      identity_for_zero::Bool=true)
    if smoothness_order == 0
        return identity_for_zero ? Matrix{Float64}(I, n, n) : nothing
    end
    smoothness_order in (1, 2) || throw(ArgumentError(
        "Unsupported smoothness_order: $smoothness_order. Use 0, 1 or 2."))
    return create_derivative_matrix(n, smoothness_order)
end

function solve_cgls(A::AbstractMatrix{T}, b::AbstractVector{T}, x0::AbstractVector{T};
                    max_iterations::Integer=100,
                    tolerance::Real=T(1e-12),
                    noise_level::Union{Nothing,Real}=nothing,
                    regularization::Real=T(0.0),
                    smoothness_order::Integer=0) where T<:AbstractFloat
    A = Matrix{T}(A)
    b = Vector{T}(b)
    m, n = size(A)

    x = Vector{T}(x0)
    if length(x) != n
        throw(DimensionMismatch("x0 length must match the number of energy bins"))
    end

    nrmb = norm(b)
    if nrmb == 0
        return UnfoldResult(zeros(T, n), 0, true, T(0))
    end

    L = regularization > 0 ?
        make_regularization_operator(n, smoothness_order; identity_for_zero=false) :
        nothing

    r = b .- A * x
    s = A' * r
    if L !== nothing
        s = s .- T(regularization) .* (L' * (L * x))
    end
    d = copy(s)

    nrmAtb = norm(A' * b)
    rho = dot(s, s)

    rtol = (noise_level !== nothing && noise_level ≥ 0) ?
           T(1.01 * noise_level * nrmb) : nothing

    iterations = 0
    converged = false

    for k in 1:max_iterations
        Ad = A * d
        normAd2 = dot(Ad, Ad)
        if L !== nothing
            Ld = L * d
            normAd2 += T(regularization)^2 * dot(Ld, Ld)
        end

        normAd2 <= 0 && break

        alpha_k = rho / normAd2
        @. x += alpha_k * d
        @. r -= alpha_k * Ad

        s = A' * r
        if L !== nothing
            s = s .- T(regularization) .* (L' * (L * x))
        end

        rho_new = dot(s, s)
        beta = rho > 0 ? rho_new / rho : T(0)
        rho = rho_new
        d = s .+ beta .* d

        iterations = k

        ne_res = norm(s)
        if rtol !== nothing && norm(r) <= rtol
            converged = true
            break
        end
        if nrmAtb > 0 && ne_res <= T(tolerance) * nrmAtb
            converged = true
            break
        end
        if ne_res <= T(tolerance)
            converged = true
            break
        end
    end

    spectrum = max.(x, T(0))
    residual = b .- A * spectrum
    return UnfoldResult(spectrum, iterations, converged, norm(residual))
end
