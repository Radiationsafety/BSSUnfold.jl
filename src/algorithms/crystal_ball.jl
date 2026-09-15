"""
    solve_crystal_ball(A, b, x0=nothing; regularization=0.0)

The CRYSTAL BALL algorithm — one-step (iteration-free) unfolding.

The spectrum is represented as a linear combination of detector response
functions (rows of the matrix A):  phi = Σ_i alpha_i * A_i.  Substituting
into the measurement equation b = A * phi gives the normal equations

    (A Aᵀ + λ I) alpha = b,

after which the spectrum is recovered as `phi = Aᵀ alpha`.  This is
equivalent to approximating the delta operator with a linear combination
of integral response operators (Kam & Stallmann).

`x0` is not used (accepted for signature uniformity).
`regularization` is the Tikhonov parameter λ used to stabilize the
ill-conditioned Gram matrix.

# Returns
`UnfoldResult` (iterations = 1, converged = true — a one-step method).
"""
function solve_crystal_ball(A::AbstractMatrix{T}, b::AbstractVector{T},
                            x0::Union{Nothing,AbstractVector{T}}=nothing;
                            regularization::T=T(0.0)) where T<:AbstractFloat
    m, n = size(A)

    if isempty(A) || isempty(b)
        throw(ArgumentError("Response matrix and measurements must be non-empty"))
    end
    if all(b .<= 0)
        throw(ArgumentError("All measurements are zero or negative"))
    end

    G = A * A'
    if regularization > 0
        G .+= regularization .* Matrix{T}(I, m, m)
    end

    alpha = G \ b
    spectrum = A' * alpha

    x = max.(spectrum, T(0))
    residual = b .- A * x
    return UnfoldResult(x, 1, true, norm(residual),
                        Dict{String,Any}("regularization" => regularization))
end
