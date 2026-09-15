"""
OSEM — Ordered Subset Expectation Maximization (Hudson & Larkin, 1994).

MLEM with subset-based updates — speeds up convergence
by a factor of n_subsets compared to MLEM. A classical algorithm in PET.

For each subset S:
    x_{k+1}[j] = x_k[j] * (Σ_{i∈S} A[i,j] * b_i / (A x_k)_i) / (Σ_{i∈S} A[i,j])
"""
function solve_osem(A::AbstractMatrix{T}, b::AbstractVector{T}, x0::AbstractVector{T};
                   max_iterations::Integer=100,
                   tolerance::T=T(1e-6),
                   n_subsets::Integer=4,
                   eps::T=T(1e-10)) where T<:AbstractFloat
    m, n = size(A)
    x = max.(copy(x0), eps)
    AT = Matrix(A')

    subset_size = max(cld(m, n_subsets), 1)
    subsets = [collect((i-1)*subset_size+1:min(i*subset_size, m)) for i in 1:n_subsets]

    # Pre-compute sensitivity for each subset
    sensitivities = [vec(sum(A[subset, :], dims=1)) for subset in subsets]

    converged = false
    iters = 0

    @inbounds for k in 1:max_iterations
        iters = k
        x_prev = copy(x)
        for (i, subset) in enumerate(subsets)
            if isempty(subset)
                continue
            end
            A_sub = A[subset, :]
            b_sub = b[subset]
            AT_sub = A_sub'

            Ax = A_sub * x
            Ax = max.(Ax, eps)
            ratio = b_sub ./ Ax
            correction = AT_sub * ratio
            s = sensitivities[i]
            x_new = x .* correction ./ max.(s, eps)
            x = max.(x_new, T(0))
        end

        if norm(x .- x_prev) / (norm(x_prev) + eps) < tolerance
            converged = true
            break
        end
    end

    residual = b .- A * x
    return UnfoldResult(x, iters, converged, norm(residual))
end
