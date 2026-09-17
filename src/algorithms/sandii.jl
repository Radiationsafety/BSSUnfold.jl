"""SAND-II iterative unfolding — faithful port of
`bssunfold.core.unfold_sandii.solve_sandii`.

Logarithmic SAND-II update with per-detector weights:

    E        = A x                       (calculated readings)
    R_i      = b_i / E_i
    W_ij     = A_ij x_j / E_i
    x_new_j  = x_j * exp( Σ_i W_ij log R_i / Σ_i W_ij )

Convergence (`chi_fac=1`, default): stop when the chi-square of the fit,

    χ² = Σ_i ((b_i - (A x_new)_i) / σ_i)²

is not greater than the number of detectors, with
`σ_i = relative_uncertainty * b_i`.  With `chi_fac=0` the iteration stops
on the maximum relative spectrum change instead.
"""
function solve_sandii(A::AbstractMatrix{T}, b::AbstractVector{T}, x0::AbstractVector{T};
                      max_iterations::Integer=50,
                      tolerance::Real=T(1e-3),
                      chi_fac::Integer=1,
                      relative_uncertainty::Real=T(0.1),
                      sigma::Union{Nothing,AbstractVector{<:Real}}=nothing) where T<:AbstractFloat
    A = Matrix{T}(A)
    b = Vector{T}(b)
    x = max.(Vector{T}(x0), T(0))

    if sigma !== nothing
        sig = max.(Vector{T}(sigma), T(1e-12))
    else
        sig = T(relative_uncertainty) .* max.(b, T(1e-12))
    end

    valid = b .> 0
    any(valid) ||
        throw(ArgumentError("All measurements are zero or negative"))

    A_valid = A[valid, :]
    b_valid = b[valid]
    sig_valid = sig[valid]
    m_valid = length(b_valid)
    eps = T(1e-12)

    converged = false
    iters = 0

    for iteration in 1:max_iterations
        iters = iteration

        E = A_valid * x
        E_safe = max.(E, T(1e-300))
        R = b_valid ./ E_safe
        W = A_valid .* (reshape(x, 1, :) ./ E_safe)

        denom = vec(sum(W, dims=1))
        denom_safe = ifelse.(denom .<= eps, eps, denom)
        numer = vec(sum(W .* log.(R), dims=1))
        x_new = x .* exp.(numer ./ denom_safe)

        if chi_fac == 1
            chi2 = sum(((b_valid .- A_valid * x_new) ./ sig_valid) .^ 2)
            if chi2 <= m_valid
                x = x_new
                converged = true
                break
            end
        else
            rel = abs.(x_new .- x) ./ max.(x, eps)
            if maximum(rel) < T(tolerance)
                x = x_new
                converged = true
                break
            end
        end

        x = x_new
    end

    residual = b .- A * x
    return UnfoldResult(x, iters, converged, norm(residual))
end
