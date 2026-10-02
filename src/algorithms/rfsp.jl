"""
RFSP-JUL — iterative damped least-squares unfolding (classical neutron
spectrometry code, Fischer; the 1981 review of unfolding codes).  This is an
independent open-source reimplementation following the published mathematical
description; the original RFSP-JUL package is proprietary.

At each iteration the functional
    S = sum_i W_i [(b_i - (A phi)_i)/b_i]^2 + sum_j [(phi_j - phi_prev_j)/phi_prev_j]^2
is minimised in closed form from the normal equations (Marquardt-style damping
by the previous iterate), solved as an SPD system
    (M + diag(1/phi_prev^2)) phi = rhs_data + 1/phi_prev,
then clamped to be nonnegative; convergence on max relative spectrum change.
"""
function solve_rfsp(A::AbstractMatrix{T}, b::AbstractVector{T}, x0::AbstractVector{T};
                    max_iterations::Int=200,
                    tolerance::T=T(1e-4),
                    weights::Union{Nothing,AbstractVector}=nothing) where T<:AbstractFloat
    isempty(A) && throw(ArgumentError("Response matrix and measurements must be non-empty"))
    isempty(b) && throw(ArgumentError("Response matrix and measurements must be non-empty"))
    !any(b .> zero(T)) && throw(ArgumentError("All measurements are zero or negative"))

    m, n = size(A)
    W = weights === nothing ? fill(one(T), m) : Vector{T}(weights)
    weights === nothing || length(W) == m ||
        throw(ArgumentError("weights must have length $m, got $(length(W))"))
    W = max.(W, zero(T))

    # Guard the 1/b_i^2 weighting: only strictly positive measurements enter.
    pos = b .> zero(T)
    any(pos) || throw(ArgumentError("All measurements are zero or negative"))

    A_pos = A[pos, :]
    b_pos = b[pos]
    W_pos = W[pos]

    # wb_inv = W_i / b_i^2; rhs_data = sum_i W_i A_ik / b_i;
    # M[k,j] = sum_i W_i A_ik A_ij / b_i^2.
    wb_inv = W_pos ./ (b_pos .^ 2)
    rhs_data = A_pos' * (wb_inv .* b_pos)
    M = A_pos' * (A_pos .* reshape(wb_inv, :, 1))

    x = Vector{T}(max.(x0, T(1e-12)))
    converged = false
    iterations = 0

    for iteration in 1:max_iterations
        iterations = iteration
        phi_prev = max.(x, T(1e-12))
        inv_prev2 = T(1.0) ./ (phi_prev .^ 2)

        # Symmetric positive-definite system (damped on the diagonal).
        lhs = M + Diagonal(inv_prev2)
        rhs = rhs_data .+ inv_prev2 .* phi_prev
        x_new = lhs \ rhs
        x_new = max.(x_new, zero(T))

        rel_change = maximum(abs.(x_new .- x) ./ max.(x, T(1e-12)))
        x = x_new
        if rel_change < T(tolerance)
            converged = true
            break
        end
    end

    residual = b .- A * x
    return UnfoldResult(x, iterations, converged, norm(residual))
end
