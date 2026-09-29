"""
SAND-II — multiplicative geometric-mean iterative ratio method.

Faithful port of bssunfold 0.28.0 `solve_sandii` (unfold_sandii.py).

    E_i  = sum_j A_ij x_j
    W_ij = A_ij x_j / E_i
    x_j <- x_j * exp( sum_i W_ij ln(b_i / E_i) / sum_i W_ij )

Convergence: chi_fac == 1 -> chi-square <= number of valid detectors;
chi_fac == 0 -> maximum relative spectrum change < tolerance.
"""
function solve_sandii(A::AbstractMatrix{T}, b::AbstractVector{T}, x0::AbstractVector{T};
                     max_iterations::Integer=50,
                     tolerance::Real=1e-3,
                     chi_fac::Integer=1,
                     relative_uncertainty::Real=0.1,
                     sigma::Union{AbstractVector,Nothing}=nothing) where T<:AbstractFloat
    tol = T(tolerance)
    A_valid = A[b .> 0, :]
    b_valid = b[b .> 0]
    if isempty(b_valid)
        throw(ArgumentError("All measurements are zero or negative"))
    end
    m_valid = length(b_valid)

    sig = if sigma !== nothing
        max.(T.(sigma), T(1e-12))[b .> 0]
    else
        T(relative_uncertainty) .* max.(b_valid, T(1e-12))
    end

    x = Vector{T}(max.(x0, zero(T)))
    eps = T(1e-12)

    converged = false
    iterations = 0

    for k in 1:max_iterations
        iterations = k

        E = A_valid * x
        E_safe = max.(E, T(1e-300))
        R = b_valid ./ E_safe
        W = A_valid .* (transpose(x) ./ reshape(E_safe, :, 1))

        denom = vec(sum(W, dims=1))
        denom_safe = ifelse.(denom .<= eps, eps, denom)
        numer = W' * log.(R)
        x_new = x .* exp.(numer ./ denom_safe)

        if chi_fac == 1
            chi2 = sum(((b_valid .- A_valid * x_new) ./ sig) .^ 2)
            if chi2 <= m_valid
                x = x_new
                converged = true
                break
            end
        else
            rel = abs.(x_new .- x) ./ max.(x, eps)
            if maximum(rel) < tol
                x = x_new
                converged = true
                break
            end
        end

        x = x_new
    end

    residual = b .- A * x
    return UnfoldResult(x, iterations, converged, norm(residual))
end
