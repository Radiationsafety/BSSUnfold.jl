"""
Tikhonov with a Legendre polynomial basis (port of `solve_tikhonov_legendre`).

The spectrum is expanded in the Legendre basis P_k(x), x ∈ [-1,1]; the basis
coefficients are obtained by a regularized LS solution of the combined system
`[A·Φ; δ·L2] c = [b; 0]` (L2 — second difference of the order terms).
Legendre polynomials are computed by the three-term recurrence
`(k+1)P_{k+1} = (2k+1)·x·P_k − k·P_{k−1}` on the grid `linspace(−1,1,n)`.
"""

function _leg_poly_basis(n_energy::Int, n_polynomials::Int)
    X = collect(range(-1.0, 1.0, length=n_energy))
    basis = zeros(Float64, n_energy, n_polynomials)
    n_polynomials >= 1 && (basis[:, 1] .= 1.0)
    np = n_polynomials
    if np >= 2
        basis[:, 2] .= X
    end
    Pprev = ones(Float64, n_energy)
    Pcurr = copy(X)
    for k in 3:np
        Pnext = ((2 * k - 3) .* X .* Pcurr .- (k - 2) .* Pprev) ./ (k - 1)
        basis[:, k] .= Pnext
        Pprev = Pcurr
        Pcurr = Pnext
    end
    return basis
end

function _leg_d2(n::Integer)
    n <= 2 && return zeros(0, n)
    L = zeros(Float64, n - 2, n)
    for j in 1:n - 2
        L[j, j] = 1.0
        L[j, j + 1] = -2.0
        L[j, j + 2] = 1.0
    end
    return L
end

"""
    solve_tikhonov_legendre(A, b, x0=nothing; delta=0.05, n_polynomials=15)

Unfolding with Tikhonov regularization in the Legendre basis.

- `delta` — regularization parameter (weight of the second difference).
- `n_polynomials` — number of Legendre polynomials in the basis.
"""
function solve_tikhonov_legendre(A::AbstractMatrix{Float64}, b::AbstractVector{Float64},
                                 x0::Union{AbstractVector{Float64},Nothing}=nothing;
                                 delta::Real=0.05,
                                 n_polynomials::Integer=15)
    n_energy = size(A, 2)
    np = Int(n_polynomials)
    basis = _leg_poly_basis(n_energy, np)
    L = _leg_d2(np)

    A_proj = A * basis
    M = [A_proj; delta .* L]
    rhs = [b; zeros(size(L, 1))]

    c = try
        M \ rhs
    catch
        pinv(M) * rhs
    end

    spectrum = max.(basis * c[1:np], 0.0)
    resid = b .- A * spectrum
    return UnfoldResult(spectrum, 1, true, norm(resid),
        Dict{String,Any}("delta" => Float64(delta),
                         "n_polynomials" => np))
end
