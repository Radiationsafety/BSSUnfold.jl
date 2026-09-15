"""
BUNKI-UT (BON31G) — Bonner-sphere iterative unfolding (the modern version
of BUNKI from the University of Texas; a port of the reference BUMS2
implementation, MIT).

Operates on a lethargy-weighted response matrix scaled by the initial
spectrum (`alethnew = aleth * x0`), starting spectrum `spl = 1`:

    bk_jm   = Σ_i alethnew_ij * alethnew_im
    vect_j  = Σ_i alethnew_ij * b_i
    ax_j    = Σ_m spl_m * bk_jm
    spll_j  = spl_j * vect_j / ax_j
    spl     ← 3-point smoothing of spll (bins 0, 1 unchanged)

The final spectrum is the back-conversion `x = spl * x0`.
"""
function solve_bunkiut(A::AbstractMatrix{T}, b::AbstractVector{T}, x0::AbstractVector{T};
                       smoothing::Real=T(0.05),
                       max_iterations::Integer=1000,
                       tolerance::Real=T(1e-6),
                       lethargy_weights::Union{Nothing,AbstractVector{T}}=nothing) where T<:AbstractFloat
    m, n = size(A)

    if lethargy_weights !== nothing
        length(lethargy_weights) == n ||
            throw(ArgumentError("lethargy_weights must have length ($n,)"))
        A = A .* reshape(lethargy_weights, 1, n)
    end

    any(b .< 0) && throw(ArgumentError("BUNKI-UT requires strictly positive measurements"))

    keep = b .> 0
    if any(keep .== 0)
        A = A[keep, :]
        b = b[keep]
        isempty(b) && throw(ArgumentError("BUNKI-UT requires strictly positive measurements"))
    end

    x0_safe = max.(x0, T(0))
    aleth = A .* reshape(x0_safe, 1, n)
    spl = ones(T, n)

    bk = Matrix(aleth') * aleth
    vect = vec(aleth' * b)

    denom_s = 1.0 + 2.0 * smoothing

    converged = false
    iters = 0

    @inbounds for k in 1:max_iterations
        iters = k
        spl_old = copy(spl)

        ax = max.(spl' * bk', T(1e-37))
        spll = spl .* vect ./ vec(ax)
        @. spll = ifelse(spll < T(1e-37), T(0), spll)

        new_spl = copy(spll)
        if n > 2
            for j in 3:n
                hi = j + 1 <= n ? spll[j+1] : T(0)
                new_spl[j] = (spll[j-1] * T(smoothing) + spll[j] + hi * T(smoothing)) /
                             T(denom_s)
            end
        end
        new_spl[1] = spll[1]
        if n > 1
            new_spl[2] = spll[2]
        end
        spl = new_spl

        rel = norm(spl .- spl_old) / (norm(spl_old) + 1e-12)
        if rel < T(tolerance)
            converged = true
            break
        end
    end

    spectrum = spl .* x0_safe
    residual = b .- A * spectrum
    return UnfoldResult(spectrum, iters, converged, norm(residual))
end
