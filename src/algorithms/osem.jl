"""
OSEM — Ordered Subsets Expectation Maximization.

Faithful port of bssunfold 0.28.0 `solve_osem` (unfold_osem.py).

    x <- max( x * A_S^T ( b_S / (A_S x + eps)) / (A_S^T 1 + eps), 0 )

for each ordered subset S of detectors; n_subsets == 1 reduces to MLEM.
"""
function solve_osem(A::AbstractMatrix{T}, b::AbstractVector{T}, x0::AbstractVector{T};
                   max_iterations::Integer=50,
                   n_subsets::Integer=1,
                   tolerance::Real=1e-6) where T<:AbstractFloat
    m, n = size(A)
    if n_subsets < 1
        throw(ArgumentError("n_subsets must be >= 1"))
    end
    if n_subsets > m
        throw(ArgumentError("n_subsets ($n_subsets) must not exceed the number of detectors ($m)"))
    end

    tol = T(tolerance)
    eps = T(1e-11)

    # np.array_split: first (m % n_subsets) subsets have one extra element
    base, extra = divrem(m, Int(n_subsets))
    subsets = Vector{Int}[]
    start = 1
    for s in 1:n_subsets
        sz = base + (s <= extra ? 1 : 0)
        push!(subsets, collect(start:start+sz-1))
        start += sz
    end

    x = Vector{T}(max.(x0, zero(T)))
    converged = false
    iterations = 0

    for k in 1:max_iterations
        iterations = k
        x_old = copy(x)

        for idx in subsets
            A_sub = view(A, idx, :)
            b_sub = view(b, idx)
            norm_col = vec(sum(A_sub, dims=1))
            ratio = b_sub ./ (A_sub * x .+ eps)
            correction = A_sub' * ratio
            x = max.(x .* correction ./ (norm_col .+ eps), zero(T))
        end

        rel = norm(x .- x_old) / (norm(x_old) + eps)
        if rel < tol
            converged = true
            break
        end
    end

    residual = b .- A * x
    return UnfoldResult(x, iterations, converged, norm(residual))
end
