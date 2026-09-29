"""
CGLS — Conjugate Gradient Least Squares.

Faithful port of `solve_cgls` from bssunfold 0.28.0 (`unfold_cgls.py`),
itself following IR Tools / TRIPs-Py. CG applied implicitly to the normal
equations (optionally Tikhonov-regularized: (A'A + reg^2 L'L) x = A'b),
with semi-convergence stopping via tolerance or discrepancy principle.
Final solution is clipped to be nonnegative.
"""

# Regularization operator L (mirrors make_regularization_operator with
# identity_for_zero=False): nothing for order 0, derivative matrix (grid-aware
# with lethargy quadrature weights when E_MeV is given) for orders 1 and 2.
function _cgls_reg_operator(n::Integer, smoothness_order::Integer, E_MeV::Union{AbstractVector,Nothing})
    smoothness_order == 0 && return nothing
    smoothness_order in (1, 2) ||
        throw(ArgumentError("Unsupported smoothness_order: $smoothness_order. Use 0, 1 or 2."))
    rows = n - smoothness_order
    L = zeros(Float64, rows, n)
    if E_MeV === nothing
        if smoothness_order == 1
            for i in 1:rows
                L[i, i] = -1.0
                L[i, i + 1] = 1.0
            end
        else
            for i in 1:rows
                L[i, i] = 1.0
                L[i, i + 1] = -2.0
                L[i, i + 2] = 1.0
            end
        end
        return L
    end
    E = Vector{Float64}(E_MeV)
    length(E) == n || throw(ArgumentError("E_MeV must have length $n, got $(length(E))"))
    all(>(0), E) && all(diff(E) .> 0) ||
        throw(ArgumentError("E_MeV must be strictly positive and strictly increasing"))
    h = diff(log.(E))
    if smoothness_order == 1
        for i in 1:rows
            L[i, i] = -1.0 / h[i]
            L[i, i + 1] = 1.0 / h[i]
        end
        w = h
    else
        for i in 1:rows
            hp, hn = h[i], h[i + 1]
            L[i, i] = 2.0 / (hp * (hp + hn))
            L[i, i + 1] = -2.0 / (hp * hn)
            L[i, i + 2] = 2.0 / (hn * (hp + hn))
        end
        w = 0.5 .* (h[1:end-1] .+ h[2:end])
    end
    @. L *= sqrt(w)
    return L
end

function solve_cgls(A::AbstractMatrix{T}, b::AbstractVector{T}, x0::AbstractVector{T};
                    max_iterations::Integer=100,
                    tolerance::Real=T(1e-12),
                    noise_level::Union{Real,Nothing}=nothing,
                    regularization::Real=T(0.0),
                    smoothness_order::Integer=0,
                    E_MeV::Union{AbstractVector,Nothing}=nothing) where T<:AbstractFloat
    m, n = size(A)
    length(b) == m || throw(DimensionMismatch("length(b) != size(A,1)"))
    length(x0) == n || throw(DimensionMismatch("length(x0) != size(A,2)"))
    tol = T(tolerance)
    reg = T(regularization)

    x = Vector{T}(x0)
    nrmb = norm(b)
    if nrmb == 0.0
        return UnfoldResult(zeros(T, n), 0, true, T(0))
    end

    L = reg > 0 ? T.(_cgls_reg_operator(n, Int(smoothness_order), E_MeV)) : nothing

    r = b .- A * x
    s = A' * r
    if L !== nothing
        s -= reg * (L' * (L * x))
    end
    d = copy(s)

    nrmAtb = norm(A' * b)
    rho = dot(s, s)

    rtol = (noise_level !== nothing && noise_level >= 0) ? T(1.01) * T(noise_level) * nrmb : nothing

    iterations = 0
    converged = false

    for k in 1:max_iterations
        Ad = A * d
        normAd2 = if L !== nothing
            Ld = L * d
            dot(Ad, Ad) + reg * reg * dot(Ld, Ld)
        else
            dot(Ad, Ad)
        end
        normAd2 <= 0 && break

        alpha_k = rho / normAd2
        x += alpha_k * d
        r -= alpha_k * Ad

        s = A' * r
        if L !== nothing
            s -= reg * (L' * (L * x))
        end

        rho_new = dot(s, s)
        beta = rho > 0 ? rho_new / rho : T(0)
        rho = rho_new
        d = s + beta * d

        iterations = k

        ne_res = norm(s)
        if rtol !== nothing
            if norm(r) <= rtol
                converged = true
                break
            end
        end
        if nrmAtb > 0 && ne_res <= tol * nrmAtb
            converged = true
            break
        end
        if ne_res <= tol
            converged = true
            break
        end
    end

    x = max.(x, T(0))
    return UnfoldResult(x, iterations, converged, norm(b .- A * x))
end
