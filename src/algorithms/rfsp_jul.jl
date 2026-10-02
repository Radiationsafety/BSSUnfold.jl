"""
RFSP-JUL unfolding method for neutron spectrum reconstruction.

Independent open-source reimplementation of the RFSP-JUL algorithm
following its published mathematical description (Fischer; the 1981
review of unfolding codes). The original RFSP-JUL code is a proprietary
package; this implementation is built solely from the published
algorithmic description.

RFSP-JUL is an iterative, damped least-squares method. At each iteration
`k` it minimises the functional

    S^{(k)} = sum_i W_i [ (b_i - (A phi)_i) / b_i ]^2
              + sum_j [ (phi_j - phi_prev_j) / phi_prev_j ]^2

where `phi_prev = phi^{(k-1)}` (the previous iterate) provides a
Marquardt-style damping that keeps the solution from diverging. Both terms
are quadratic in `phi`, so the minimiser is found in closed form from the
normal equations (a symmetric positive-definite system solved directly):

    [ sum_i W_i R_i R_i^T / b_i^2  +  diag(1 / phi_prev^2) ] phi
        = sum_i W_i R_i / b_i  +  phi_prev / phi_prev^2
"""

"""
    solve_rfsp_jul(A, b, x0; max_iterations=200, tolerance=1e-4, weights=nothing)

Solve unfolding problem using the RFSP-JUL algorithm.

# Arguments
- `A` — response matrix (`m x n`);
- `b` — measurement vector (`m`);
- `x0` — initial guess (`n`); also used as the reference iterate `phi_prev`
  at the first iteration.

# Keywords
- `max_iterations` — maximum number of iterations (default: 200);
- `tolerance` — convergence tolerance on the maximum relative spectrum
  change (default: 1e-4);
- `weights` — per-detector weights `W_i` for the residual term; `nothing`
  means all detectors are weighted equally (`W_i = 1`).

Returns an [`UnfoldResult`](@ref).
"""
function solve_rfsp_jul(A::AbstractMatrix{T}, b::AbstractVector{T},
                        x0::AbstractVector{T};
                        max_iterations::Integer=200,
                        tolerance::Real=1e-4,
                        weights::Union{Nothing,AbstractVector{<:Real}}=nothing
                        ) where T<:AbstractFloat
    Af = Matrix{Float64}(A)
    bf = Vector{Float64}(b)
    x0f = Vector{Float64}(x0)

    (isempty(Af) || isempty(bf)) && throw(ArgumentError(
        "Response matrix and measurements must be non-empty"))
    all(bf .<= 0) && throw(ArgumentError(
        "All measurements are zero or negative"))

    m, n = size(Af)
    length(bf) == m || throw(ArgumentError(
        "Length of b ($(length(bf))) must match number of rows of A ($m)"))
    length(x0f) == n || throw(ArgumentError(
        "Length of x0 ($(length(x0f))) must match number of columns of A ($n)"))
    W = weights === nothing ? ones(m) : max.(Vector{Float64}(weights), 0.0)

    # Guard the 1/b_i^2 weighting: only strictly positive measurements enter.
    pos = bf .> 0
    !any(pos) && throw(ArgumentError("All measurements are zero or negative"))

    A_pos = Af[pos, :]
    b_pos = bf[pos]
    W_pos = W[pos]
    b_safe_pos = b_pos  # already > 0 here

    # Precompute the (positive) right-hand data term: sum_i W_i R_i / b_i.
    wb_inv = W_pos ./ (b_safe_pos .^ 2)
    rhs_data = A_pos' * (wb_inv .* b_pos)          # sum_i W_i R_ik / b_i
    # Precompute the (positive) curvature contribution:
    # sum_i W_i R_i R_i^T / b_i^2, stored as M[k, j].
    M = A_pos' * (wb_inv .* A_pos)

    x = max.(x0f, 1e-12)
    converged = false
    iterations = 0

    for iteration in 1:Int(max_iterations)
        iterations = iteration
        phi_prev = max.(x, 1e-12)
        inv_prev2 = 1.0 ./ (phi_prev .^ 2)

        # Symmetric positive-definite system (regularised on the diagonal).
        lhs = M + Diagonal(inv_prev2)
        rhs = rhs_data .+ inv_prev2 .* phi_prev
        x_new = max.(lhs \ rhs, 0.0)

        rel_change = maximum(abs.(x_new .- x) ./ max.(x, 1e-12))
        x = x_new
        if rel_change < tolerance
            converged = true
            break
        end
    end

    residual = bf .- Af * x
    return UnfoldResult(x, iterations, converged, Float64(norm(residual)),
                        Dict{String,Any}("tolerance" => Float64(tolerance)))
end
