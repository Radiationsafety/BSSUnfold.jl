"""BSREM (Block-Sequential Regularized Expectation Maximization) —
faithful port of `bssunfold.core.unfold_bsrem.solve_bsrem`.

Additive relaxed update over ordered subsets of the detector readings:

    ratio     = b_S ./ (A_S x + eps)
    update    = A_Sᵀ ratio − Σ_j A_Sij − ω ∇V(x)
    step      = α / (ω Σ_j A_ij)
    x        += x .* step .* update
    x         = max.(x, addition_after_iteration)

with the relaxation sequence `α(n)` (constant 1 by default) and
ω = |S|/m.  The prior gradient ∇V supports the `quadratic`, `logcosh`
and `relative_difference` nearest-neighbour priors of
`bssunfold.core._em_priors` (beta-scaled, unit neighbour weights along
the energy axis, zero-padded boundaries).
"""

const _BSREM_PRIORS = ("none", "quadratic", "logcosh", "relative_difference")

"Nearest-neighbour (left, right) arrays, zero-padded at the boundaries."
function _em_neighbours(x::Vector{T}) where T<:AbstractFloat
    n = length(x)
    left = zeros(T, n)
    right = zeros(T, n)
    if n > 1
        left[2:end] .= x[1:end-1]
        right[1:end-1] .= x[2:end]
    end
    return left, right
end

"phi1(f_r, f_s) = d/d f_r phi0(f_r, f_s) for the three EM priors."
function _em_phi1(fr::T, fs::T, prior::String, delta::T, gamma::T) where T<:AbstractFloat
    if prior == "quadratic"
        return (fr - fs) / delta
    elseif prior == "logcosh"
        return tanh((fr - fs) / delta)
    elseif prior == "relative_difference"
        absd = abs(fr - fs)
        denom = gamma * absd + fr + fs + delta
        return (fr - fs) * (gamma * absd + 3 * fs + fr + 2 * delta) / denom^2
    end
    throw(ArgumentError("Unknown prior $prior. Choose from 'quadratic', " *
                        "'logcosh', 'relative_difference'."))
end

"""
    prior_gradient(x, prior; beta=1e-3, delta=1.0, gamma=1.0) -> Vector

Beta-scaled nearest-neighbour prior gradient along the energy axis
(port of `bssunfold.core._em_priors.prior_gradient`).
"""
function prior_gradient(x::AbstractVector{T}, prior::String;
                        beta::Real=T(1e-3), delta::Real=T(1.0),
                        gamma::Real=T(1.0)) where T<:AbstractFloat
    prior in ("quadratic", "logcosh", "relative_difference") ||
        throw(ArgumentError("Unknown prior $prior. Choose from 'quadratic', " *
                            "'logcosh', 'relative_difference'."))
    xf = Vector{T}(x)
    d = T(delta)
    g = T(gamma)
    left, right = _em_neighbours(xf)
    grad = [_em_phi1(xf[j], left[j], prior, d, g) +
            _em_phi1(xf[j], right[j], prior, d, g) for j in eachindex(xf)]
    return T(beta) .* grad
end

"numpy.array_split equivalent: k nearly-equal parts of 1:m."
function _array_split(m::Integer, k::Integer)
    base, rem = divrem(m, k)
    parts = Vector{UnitRange{Int}}(undef, k)
    idx = 1
    for i in 1:k
        len = base + (i <= rem ? 1 : 0)
        parts[i] = idx:(idx + len - 1)
        idx += len
    end
    return parts
end

function solve_bsrem(A::AbstractMatrix{T}, b::AbstractVector{T}, x0::AbstractVector{T};
                     prior::String="none",
                     beta::Real=T(1e-3),
                     prior_delta::Real=T(1.0),
                     gamma::Real=T(1.0),
                     max_iterations::Integer=50,
                     n_subsets::Integer=1,
                     tolerance::Real=T(1e-6),
                     relaxation::Union{Nothing,Real,Function}=nothing,
                     addition_after_iteration::Real=T(1e-4)) where T<:AbstractFloat
    A = Matrix{T}(A)
    b = Vector{T}(b)
    m, n = size(A)

    n_subsets ≥ 1 || throw(ArgumentError("n_subsets must be >= 1"))
    n_subsets ≤ m || throw(ArgumentError(
        "n_subsets ($n_subsets) must not exceed the number of detectors ($m)"))
    prior in _BSREM_PRIORS || throw(ArgumentError(
        "Unknown prior $prior. Choose from $(_BSREM_PRIORS)"))

    local relaxation_seq::Function
    if relaxation === nothing
        relaxation_seq = _n -> T(1)
    elseif relaxation isa Function
        relaxation_seq = relaxation
    else
        relax_val = T(relaxation)
        relaxation_seq = _n -> relax_val
    end

    eps = T(1e-11)
    subset_indices = _array_split(m, n_subsets)
    norm_all = vec(sum(A, dims=1))
    x = max.(Vector{T}(x0), T(0))
    floor_val = T(addition_after_iteration)

    converged = false
    iters = 0

    for it in 1:max_iterations
        iters = it
        x_old = copy(x)
        alpha = T(relaxation_seq(it))

        for idx in subset_indices
            omega = length(idx) / m
            A_sub = A[idx, :]
            b_sub = b[idx]
            norm_sub = vec(sum(A_sub, dims=1))

            ratio = b_sub ./ (A_sub * x .+ eps)
            correction = A_sub' * ratio
            if prior == "none"
                grad = zeros(T, n)
            else
                grad = T(omega) .* prior_gradient(x, prior; beta=beta,
                                                  delta=prior_delta, gamma=gamma)
            end

            update = correction .- norm_sub .- grad
            step = ifelse.(omega .* norm_all .> eps,
                           alpha ./ (omega .* norm_all .+ eps), T(0))
            x = x .+ x .* step .* update
            x[x .<= floor_val] .= floor_val
        end

        rel = norm(x .- x_old) / (norm(x_old) + eps)
        if rel < T(tolerance)
            converged = true
            break
        end
    end

    residual = b .- A * x
    return UnfoldResult(x, iters, converged, norm(residual))
end
