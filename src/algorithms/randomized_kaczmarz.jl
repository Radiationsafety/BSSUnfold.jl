"""
Randomized Kaczmarz unfolding method.

Port of `bssunfold/src/bssunfold/core/unfold_randomized_kaczmarz.py`.

Randomized Kaczmarz algorithm: rows are selected probabilistically
with probability proportional to the squared row norm.
This yields faster convergence on ill-conditioned systems.

Reference: Strohmer & Vershynin (2009), "A Randomized Kaczmarz Algorithm
with Exponential Convergence".
"""

"""
    solve_randomized_kaczmarz(A, b, x0; max_iterations, omega, tolerance, random_state)

Randomized Kaczmarz algorithm.

# Arguments
- `A::AbstractMatrix{T}`: response matrix (m × n)
- `b::AbstractVector{T}`: measurements (m,)
- `x0::AbstractVector{T}`: initial spectrum (n,)
- `max_iterations`: max number of iterations (default 1000)
- `omega::T`: relaxation parameter, 0 < ω ≤ 2 (default 1.0)
- `tolerance::T`: convergence criterion `||x_k - x_{k-1}||` after a full pass
- `random_state`: seed for reproducibility (default `nothing` = random)

# Returns
- `UnfoldResult{T}` with the spectrum
"""
function solve_randomized_kaczmarz(A::AbstractMatrix{T}, b::AbstractVector{T}, x0::AbstractVector{T};
                                  max_iterations::Integer=1000,
                                  omega::T=T(1.0),
                                  tolerance::T=T(1e-6),
                                  random_state::Union{Integer,Nothing}=nothing,
                                  eps::T=T(1e-30)) where T<:AbstractFloat
    m, n = size(A)
    x = max.(copy(x0), T(0))

    rng = random_state === nothing ? MersenneTwister() : MersenneTwister(random_state)

    # Squared row norms for probabilistic selection
    row_norms_sq = vec(sum(A .^ 2, dims=2))
    total_norm_sq = sum(row_norms_sq)
    if total_norm_sq == 0
        return UnfoldResult(x, 0, true, T(0))
    end
    probabilities = row_norms_sq ./ total_norm_sq

    # Cumulative probabilities for sampling
    cum_probs = cumsum(probabilities)

    converged = false
    iterations = 0
    x_old = copy(x)

    @inbounds for k in 1:max_iterations
        # Sample a row index
        u = rand(rng)
        i = searchsortedfirst(cum_probs, u)
        i = clamp(i, 1, m)

        if row_norms_sq[i] > eps
            ai = view(A, i, :)
            update = (b[i] - dot(ai, x)) / row_norms_sq[i]
            x .+= omega .* update .* ai
            x .= max.(x, T(0))
        end

        # Convergence check after each full cycle (m iterations)
        if k % m == 0
            diff_norm = norm(x .- x_old)
            if diff_norm < tolerance
                converged = true
                iterations = k
                break
            end
            x_old .= x
        end
    end

    if !converged
        iterations = max_iterations
    end

    residual = b .- A * x
    return UnfoldResult(x, iterations, converged, norm(residual))
end
