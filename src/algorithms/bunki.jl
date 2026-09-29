"""
    solve_bunki(A, b, x0; smoothing=0.1, max_iterations=1000, tolerance=1e-6, lethargy_weights=nothing)

BUNKI (SPUNIT) unfolding: SPUNIT iteration with three-point smoothing on the
lethargy-weighted matrix transformed by the initial spectrum (`aleth = A * x0`),
final spectrum `x = spl * x0`.
"""
function solve_bunki(A::AbstractMatrix{T}, b::AbstractVector{T}, x0::AbstractVector{T};
                     smoothing::Real=T(0.1),
                     max_iterations::Integer=1000,
                     tolerance::Real=T(1e-6),
                     lethargy_weights::Union{Nothing,AbstractVector}=nothing) where T<:AbstractFloat
    m, n = size(A)
    s = T(smoothing)
    tol = T(tolerance)

    W = if lethargy_weights === nothing
        A
    else
        lw = Vector{T}(lethargy_weights)
        length(lw) == n || throw(ArgumentError("lethargy_weights must have length $n"))
        A .* lw'
    end

    any(<(zero(T)), b) && throw(ArgumentError("BUNKI requires strictly positive measurements"))

    # Zero readings carry no usable information and would divide by zero:
    # drop those detectors instead of failing the whole unfolding.
    if any(==(zero(T)), b)
        keep = findall(>(zero(T)), b)
        isempty(keep) && throw(ArgumentError("BUNKI requires strictly positive measurements"))
        W = W[keep, :]
        b = b[keep]
    end

    x0_safe = max.(x0, zero(T))
    # trans_mat: response scaled by the initial spectrum, spl starts at ones.
    aleth = W .* x0_safe'
    spl = ones(T, n)
    bcc = aleth * spl

    # ss[j] = sum_i aleth[i, j] / b[i]
    inv_b = T[bi > 0 ? 1 / bi : zero(T) for bi in b]
    ss = aleth' * inv_b
    tiny = T(1e-37)
    inv_ss = T[si > 0 ? 1 / max(si, tiny) : zero(T) for si in ss]
    denom_s = 1 + 2 * s

    converged = false
    iterations = 0

    for k in 1:Int(max_iterations)
        iterations = k

        inv_bcc = T[ci > 0 ? 1 / max(ci, tiny) : zero(T) for ci in bcc]
        spll = (spl .* (aleth' * inv_bcc)) .* inv_ss
        spll = T[(spll[j] < tiny || spl[j] <= zero(T)) ? zero(T) : spll[j] for j in 1:n]

        # Vectorized 3-point smoothing (bins 0 and 1 kept verbatim)
        new_spl = copy(spll)
        if n > 2
            new_spl[2:n-1] = (s .* spll[1:n-2] .+ spll[2:n-1] .+ s .* spll[3:n]) ./ denom_s
        end

        bcc = aleth * new_spl

        rel = norm(new_spl .- spl) / (norm(spl) + T(1e-12))
        spl = new_spl

        if rel < tol
            converged = true
            break
        end
    end

    spectrum = spl .* x0_safe
    residual_norm = T(norm(b .- W * spectrum))
    return UnfoldResult(spectrum, iterations, converged, residual_norm)
end
