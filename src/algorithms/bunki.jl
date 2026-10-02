"""BUNKI (SPUNIT) unfolding — faithful port of
`bssunfold.core.unfold_bunki.solve_bunki`.

The iteration works on a lethargy-weighted response matrix scaled by the
initial spectrum (`aleth = A .* x0`) with a relative working spectrum
`spl` that starts at ones:

    bcc_i   = Σ_j aleth_ij * spl_j                       (calculated readings)
    spll_j  = spl_j * (Σ_i aleth_ij / bcc_i) / ss_j      (SPUNIT update)
    ss_j    = Σ_i aleth_ij / b_i                         (normalization)
    spl     ← 3-point smoothing of spll (interior bins)

The final spectrum is the back-conversion `x = spl .* x0_safe`.
"""
function solve_bunki(A::AbstractMatrix{T}, b::AbstractVector{T}, x0::AbstractVector{T};
                     smoothing::Real=T(0.1),
                     max_iterations::Integer=1000,
                     tolerance::Real=T(1e-6),
                     lethargy_weights::Union{Nothing,AbstractVector{<:Real}}=nothing) where T<:AbstractFloat
    A = Matrix{T}(A)
    b = Vector{T}(b)
    m, n = size(A)

    if lethargy_weights !== nothing
        length(lethargy_weights) == n ||
            throw(ArgumentError("lethargy_weights must have length ($n,)"))
        A = A .* reshape(Vector{T}(lethargy_weights), 1, n)
    end

    any(b .< 0) &&
        throw(ArgumentError("BUNKI requires strictly positive measurements"))

    # Zero readings carry no usable information for BUNKI and would cause
    # division by zero. Drop those detectors instead of failing the whole
    # unfolding (mirrors the Python implementation).
    keep = b .> 0
    if !all(keep)
        A = A[keep, :]
        b = b[keep]
        isempty(b) &&
            throw(ArgumentError("BUNKI requires strictly positive measurements"))
    end

    x0_safe = max.(Vector{T}(x0), T(0))
    # trans_mat: response scaled by the initial spectrum, spl starts at ones.
    aleth = A .* reshape(x0_safe, 1, n)
    spl = ones(T, n)
    bcc = aleth * spl

    # ss[j] = Σ_i aleth[i, j] / b[i]
    inv_b = [b[i] > 0 ? T(1) / b[i] : T(0) for i in eachindex(b)]
    ss = vec(sum(aleth .* reshape(inv_b, :, 1), dims=1))
    inv_ss = [s > 0 ? T(1) / max(s, T(1e-37)) : T(0) for s in ss]

    denom_s = T(1) + T(2) * T(smoothing)

    converged = false
    iters = 0

    for k in 1:max_iterations
        iters = k

        inv_bcc = [v > 0 ? T(1) / max(v, T(1e-37)) : T(0) for v in bcc]
        spll = spl .* (aleth' * inv_bcc) .* inv_ss
        @. spll = ifelse(spll < T(1e-37), T(0), spll)
        @. spll = ifelse(spl <= T(0), T(0), spll)

        # Vectorized 3-point smoothing (interior bins only)
        new_spl = copy(spll)
        if n > 2
            @inbounds for j in 2:(n-1)
                new_spl[j] = (T(smoothing) * spll[j-1] + spll[j] +
                              T(smoothing) * spll[j+1]) / denom_s
            end
        end

        bcc = aleth * new_spl

        rel = norm(new_spl .- spl) / (norm(spl) + T(1e-12))
        spl = new_spl

        if rel < T(tolerance)
            converged = true
            break
        end
    end

    spectrum = spl .* x0_safe
    residual = b .- A * spectrum
    return UnfoldResult(spectrum, iters, converged, norm(residual))
end
