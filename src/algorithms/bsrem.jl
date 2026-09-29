"""
BSREM — Block-Sequential Regularized Expectation Maximization.

Faithful port of bssunfold `solve_bsrem` (penalized EM with ordered
detector subsets, relaxation sequence and post-iteration floor clamp):

    x_{n+1} = x_n + alpha(n)/(omega_m * A^T 1 + eps)
                  * ( A_m^T ( b_m/(A_m x_n + eps) ) - A_m^T 1
                      - omega_m * beta * grad V(x_n) )

with zero-padded nearest-neighbour priors `none|quadratic|logcosh|
relative_difference` along the energy axis; after every sub-iteration the
spectrum is clamped to be at least `addition_after_iteration`.
"""
function solve_bsrem(A::AbstractMatrix{T}, b::AbstractVector{T}, x0::AbstractVector{T};
                     prior::AbstractString="none",
                     beta::Real=T(1e-3),
                     prior_delta::Real=one(T),
                     gamma::Real=one(T),
                     max_iterations::Integer=50,
                     n_subsets::Integer=1,
                     tolerance::Real=T(1e-6),
                     relaxation=nothing,
                     addition_after_iteration::Real=T(1e-4)) where T<:AbstractFloat
    m, n = size(A)
    n_subsets >= 1 || throw(ArgumentError("n_subsets must be >= 1"))
    n_subsets <= m || throw(ArgumentError(
        "n_subsets ($n_subsets) must not exceed the number of detectors ($m)"))
    prior_l = lowercase(prior)
    prior_l in ("none", "quadratic", "logcosh", "relative_difference") ||
        throw(ArgumentError("Unknown prior '$prior'. Choose from " *
                            "['none', 'quadratic', 'logcosh', 'relative_difference']"))

    beta = T(beta); prior_delta = T(prior_delta); gamma = T(gamma)
    tol = T(tolerance); floor_ = T(addition_after_iteration)
    relax_val = relaxation isa Function ? nothing : relaxation === nothing ? one(T) : T(relaxation)

    eps = T(1e-11)
    subsets = _bsrem_array_split(m, Int(n_subsets))
    norm_all = vec(sum(A, dims=1))
    x = max.(x0, zero(T))
    converged = false
    iterations = 0

    for it in 1:max_iterations
        iterations = it
        x_old = copy(x)
        alpha = relaxation isa Function ? T(relaxation(it)) : relax_val

        for idx in subsets
            omega = T(length(idx)) / T(m)
            A_sub = view(A, idx, :)
            b_sub = view(b, idx)
            norm_sub = vec(sum(A_sub, dims=1))

            ratio = b_sub ./ (A_sub * x .+ eps)
            correction = A_sub' * ratio
            if prior_l == "none"
                grad = zeros(T, n)
            else
                grad = omega .* _bsrem_prior_gradient(x, prior_l, beta, prior_delta, gamma)
            end

            update = correction .- norm_sub .- grad
            denom = omega .* norm_all
            step = @. ifelse(denom > eps, alpha / (denom + eps), zero(T))
            x = x .+ x .* step .* update
            x = max.(x, floor_)
        end

        rel = norm(x - x_old) / (norm(x_old) + eps)
        if rel < tol
            converged = true
            break
        end
    end

    residual = b .- A * x
    return UnfoldResult(x, iterations, converged, norm(residual))
end

# np.array_split(1:m, k): first rem(m,k) groups are one element larger.
function _bsrem_array_split(m::Int, k::Int)
    q, r = divrem(m, k)
    idxs = Vector{UnitRange{Int}}(undef, k)
    start = 1
    @inbounds for i in 1:k
        len = q + (i <= r ? 1 : 0)
        idxs[i] = start:(start + len - 1)
        start += len
    end
    return idxs
end

# Port of _em_priors.prior_gradient (beta-scaled NN gradient, zero-padded).
function _bsrem_prior_gradient(x::AbstractVector{T}, prior::AbstractString,
                               beta::T, delta::T, gamma::T) where T<:AbstractFloat
    n = length(x)
    left = zeros(T, n)
    right = zeros(T, n)
    if n > 1
        left[2:end] .= @view(x[1:end-1])
        right[1:end-1] .= @view(x[2:end])
    end
    g = _bsrem_phi1(x, left, prior, delta, gamma) .+
        _bsrem_phi1(x, right, prior, delta, gamma)
    return beta .* g
end

function _bsrem_phi1(fr::AbstractVector{T}, fs::AbstractVector{T}, prior::AbstractString,
                     delta::T, gamma::T) where T<:AbstractFloat
    if prior == "quadratic"
        return (fr .- fs) ./ delta
    elseif prior == "logcosh"
        return tanh.((fr .- fs) ./ delta)
    else  # relative_difference
        d = fr .- fs
        absd = abs.(d)
        denom = gamma .* absd .+ fr .+ fs .+ delta
        return d .* (gamma .* absd .+ 3fs .+ fr .+ 2delta) ./ (denom .* denom)
    end
end
