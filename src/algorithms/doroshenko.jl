"""
Doroshenko — coordinate-update unfolding (Doroshenko et al., 1986).

Faithful port of `solve_doroshenko` from bssunfold 0.28.0
(`unfold_doroshenko.py`). Cyclic coordinate minimization of
||A x - b||^2 + regularization * ||x||^2 with incremental residual updates:

    x_j <- max(0, (A_j . r + c_j x_j) / c_j),  c_j = ||A_j||^2 + regularization

Convergence: ||x - x_old||_2 < tolerance after a full sweep.
"""
function solve_doroshenko(A::AbstractMatrix{T}, b::AbstractVector{T}, x0::AbstractVector{T};
                          max_iterations::Integer=1000,
                          tolerance::Real=T(1e-6),
                          regularization::Real=T(0.0)) where T<:AbstractFloat
    m, n = size(A)
    length(b) == m || throw(DimensionMismatch("length(b) != size(A,1)"))
    length(x0) == n || throw(DimensionMismatch("length(x0) != size(A,2)"))
    tol = T(tolerance)
    reg = T(regularization)

    x = Vector{T}(x0)
    denominator_cache = vec(mapreduce(abs2, +, A; dims=1)) .+ reg

    residual = b .- A * x
    converged = false
    iterations = 0

    for i in 1:max_iterations
        x_old = copy(x)
        for j in 1:n
            den = denominator_cache[j]
            den <= 0 && continue
            Aj = view(A, :, j)
            numerator = dot(Aj, residual) + den * x[j]
            new_xj = max(numerator / den, T(0))
            delta = new_xj - x[j]
            if delta != 0
                @. residual -= delta * Aj
                x[j] = new_xj
            end
        end
        if norm(x .- x_old) < tol
            converged = true
            iterations = i
            break
        end
    end
    converged || (iterations = Int(max_iterations))

    return UnfoldResult(x, iterations, converged, norm(b .- A * x))
end
