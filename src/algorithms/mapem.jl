"""
MAP-EM (penalised expectation maximization) — one-step (OSMAPOSL,
one-step-late) weighted EM update with an energy-wise prior model:

    x^{n+1} = x^n * Aᵀ ( b / (A x^n + eps) ) / ( Aᵀ 1 + beta * grad V(x^n) )

Prior models (port of PyTomography): `quadratic`, `logcosh`,
`relative_difference`; `none` corresponds to plain MLEM.
"""
function solve_mapem(A::AbstractMatrix{T}, b::AbstractVector{T}, x0::AbstractVector{T};
                     prior::AbstractString="quadratic",
                     beta::Real=T(1e-3),
                     prior_delta::Real=one(T),
                     gamma::Real=one(T),
                     max_iterations::Integer=50,
                     tolerance::Real=T(1e-6)) where T<:AbstractFloat
    prior_names = ("none", "quadratic", "logcosh", "relative_difference")
    prior_l = lowercase(prior)
    prior_l in prior_names ||
        throw(ArgumentError("Unknown prior '$prior'. Choose from $(collect(prior_names))"))

    eps = T(1e-11)
    x = max.(copy(x0), T(0))
    sensitivity = vec(sum(A, dims=1))
    AT = Matrix(A')
    converged = false
    iters = 0

    for k in 1:max_iterations
        iters = k
        Ax = A * x
        @. Ax = b / (Ax + eps)
        correction = AT * Ax

        if prior_l == "none"
            x_new = max.(x .* correction ./ (sensitivity .+ eps), T(0))
        else
            grad = _mapem_prior_gradient(x, prior_l, T(beta), T(prior_delta), T(gamma))
            x_new = max.(x .* correction ./ (sensitivity .+ grad .+ eps), T(0))
        end

        rel = norm(x_new .- x) / (norm(x) + eps)
        x = x_new
        if rel < T(tolerance)
            converged = true
            break
        end
    end

    residual = b .- A * x
    return UnfoldResult(x, iters, converged, norm(residual))
end

function _mapem_prior_gradient(x::AbstractVector{T}, prior::AbstractString,
                               beta::T, delta::T, gamma::T) where T<:AbstractFloat
    n = length(x)
    left = zeros(T, n)
    right = zeros(T, n)
    if n > 1
        left[2:n] .= x[1:n-1]
        right[1:n-1] .= x[2:n]
    end
    g_r = _mapem_phi1(x, left, prior, delta, gamma)
    g_s = _mapem_phi1(x, right, prior, delta, gamma)
    return beta .* (g_r .+ g_s)
end

function _mapem_phi1(fr::AbstractVector{T}, fs::AbstractVector{T},
                     prior::AbstractString, delta::T, gamma::T) where T<:AbstractFloat
    if prior == "quadratic"
        return (fr .- fs) ./ delta
    elseif prior == "logcosh"
        return tanh.((fr .- fs) ./ delta)
    elseif prior == "relative_difference"
        absd = abs.(fr .- fs)
        denom = gamma .* absd .+ fr .+ fs .+ delta
        return (fr .- fs) .* (gamma .* absd .+ T(3.0) .* fs .+ fr .+ T(2.0) .* delta) ./ denom .^ 2
    end
    throw(ArgumentError("Unknown prior '$prior'"))
end
