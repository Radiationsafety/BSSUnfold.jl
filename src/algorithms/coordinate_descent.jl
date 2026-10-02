"""
Coordinate descent unfolding method for neutron spectra.

Block-free coordinate descent for the non-negative least-squares problem

    min_{x >= 0}  1/2 ||A x - b||^2 + l1_penalty * ||x||_1
                  + (l2_penalty/2) * ||x||^2

(lecture 15 of the MIPT optimization course).  Each step updates a single
coordinate in closed form against a running residual r = b - A x, so every
sweep costs O(m * n) with no matrix products.  Coordinate order is either
cyclic (classical Gauss-Seidel-type CD) or random (seeded RNG).
"""
function solve_coordinate_descent(A::AbstractMatrix{T}, b::AbstractVector{T}, x0::AbstractVector{T};
                                 max_iterations::Integer=2000,
                                 tolerance::T=T(1e-8),
                                 l1_penalty::T=T(0.0),
                                 l2_penalty::T=T(0.0),
                                 selection::Union{AbstractString,Symbol}="cyclic",
                                 random_state::Union{Integer,Nothing}=nothing) where T<:AbstractFloat
    sel = lowercase(String(selection))
    if sel != "cyclic" && sel != "random"
        error("selection must be 'cyclic' or 'random'")
    end

    m, n = size(A)
    x = max.(copy(x0), T(0))
    l1 = max(T(l1_penalty), T(0))
    l2 = max(T(l2_penalty), T(0))

    col_sq = vec(sum(A .^ 2, dims=1))  # ||a_j||^2
    residual = b .- A * x

    rng = random_state === nothing ? MersenneTwister() : MersenneTwister(random_state)

    converged = false
    iterations = 0

    @inbounds for k in 1:max_iterations
        x_prev = copy(x)
        order = sel == "cyclic" ? (1:n) : Random.randperm(rng, n)

        for j in order
            c = col_sq[j]
            if c <= 0
                continue
            end
            # partial correlation: a_j^T (r + a_j x_j) = a_j^T r + c x_j
            rho_j = dot(view(A, :, j), residual) + c * x[j]
            x_new = max((rho_j - l1) / (c + l2), T(0))
            delta = x_new - x[j]
            if delta != 0
                residual -= view(A, :, j) * delta
                x[j] = x_new
            end
        end

        iterations = k
        rel_change = norm(x .- x_prev) / max(norm(x), T(1e-30))
        if rel_change < tolerance
            converged = true
            break
        end
    end

    residual = b .- A * x
    return UnfoldResult(x, iterations, converged, norm(residual))
end
