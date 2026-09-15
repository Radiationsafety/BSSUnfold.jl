"""
TSVD — Truncated Singular Value Decomposition.

Solves Ax=b by discarding singular values below threshold.

    x_k = Σ_{i=1..k} (u_iᵀb / σ_i) v_i

where k is chosen by the Discrepancy Principle or given by the user.
"""
function solve_tsvd(A::AbstractMatrix{T}, b::AbstractVector{T}, x0::AbstractVector{T};
                   max_iterations::Integer=1,
                   tolerance::T=T(1e-3),
                   regularization::T=T(0.0),
                   truncation_rank::Union{Integer,Nothing}=nothing,
                   eps::T=T(1e-10)) where T<:AbstractFloat
    m, n = size(A)

    # SVD decomposition
    F = svd(A)
    U, σ, V = F.U, F.S, F.Vt

    # Determine truncation rank
    if truncation_rank !== nothing
        k = clamp(truncation_rank, 1, min(m, n))
    elseif regularization > 0
        # Discard σ_i < regularization
        k = count(s -> s > regularization, σ)
        k = max(k, 1)
    else
        # Discrepancy principle: σ_k > ||b_noise|| / ||b||
        noise_level = tolerance
        threshold = noise_level * norm(b)
        k = count(s -> s > threshold, σ)
        k = max(k, 1)
    end
    k = min(k, min(m, n))

    # Reconstruct
    x = V[1:k, :]' * ((U[:, 1:k]' * b) ./ σ[1:k])
    x = max.(x, T(0))

    residual = b .- A * x
    return UnfoldResult(x, k, true, norm(residual),
                       Dict{String,Any}("truncation_rank" => k,
                                       "singular_values" => σ))
end
