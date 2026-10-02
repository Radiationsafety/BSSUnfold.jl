"""
OSEM-ANLM — Ordered-Subsets Expectation Maximisation with Asymptotic
Non-Local-Means regularisation. Faithful port of
`bssunfold/core/unfold_osem_anlm.py` (Jamaati et al., 2026).

* `estimate_noise_1d` — Immerkaer-style MAD on second differences.
* `anlm_filter_1d` — two-stage NLM filter, `h1 = σ/2`, then `h2(i) = σ·‖w1(i,·)‖₂`.
* `solve_osem_anlm` — OSEM update `x ← x·A_sᵀ(b_s / (A_s x + ε)) / (colsum(A_s) + ε)`,
  ANLM applied either after every subset (`anlm_mode=:subset`, article pseudo-code)
  or once at the end (`:post`).
"""

const _ANLM_MODES = (:subset, :post)

function estimate_noise_1d(x::AbstractVector{T}) where T<:AbstractFloat
    n = length(x)
    n < 3 && return T(0)
    d2 = @views x[3:end] .- 2.0 .* x[2:end-1] .+ x[1:end-2]
    mad = _median(abs.(d2))
    sigma = mad / (T(0.6745) * sqrt(T(6)))
    floor = T(1e-12) * maximum(abs, x) + T(1e-300)
    max(sigma, floor)
end

function _median(v::AbstractVector{T}) where T<:AbstractFloat
    s = sort(Vector{T}(v))
    n = length(s)
    n == 0 && return T(0)
    isodd(n) && return s[(n+1) ÷ 2]
    T(0.5) * (s[n ÷ 2] + s[n ÷ 2 + 1])
end

function _reflect_indices(offsets::AbstractVector{Int}, n::Int)
    idx = [(i .+ offsets) for i in 1:n]
    n == 1 && return [[1 for _ in offsets] for _ in 1:n]
    period = 2 * (n - 1)
    out = Vector{Vector{Int}}(undef, n)
    for i in 1:n
        row = Vector{Int}(undef, length(offsets))
        for (t, o) in enumerate(offsets)
            k = (i + o) - 1
            k = abs(k) % period
            row[t] = (k >= n ? period - k : k) + 1
        end
        out[i] = row
    end
    out
end

function anlm_filter_1d(x::AbstractVector{T};
                        h::Union{Real,Nothing}=nothing,
                        search_window::Integer=11,
                        similarity_window::Integer=3,
                        alpha::Real=T(1.0),
                        log_space::Bool=true) where T<:AbstractFloat
    n = length(x)
    n == 0 && throw(ArgumentError("input spectrum must be non-empty"))
    search_window >= 1 || throw(ArgumentError("search_window must be >= 1, got $search_window"))
    similarity_window >= 1 || throw(ArgumentError("similarity_window must be >= 1, got $similarity_window"))
    alpha > 0 || throw(ArgumentError("alpha must be positive, got $alpha"))
    if h !== nothing
        T(h) > 0 || throw(ArgumentError("h must be a positive noise level, got $h"))
    end
    n == 1 && return copy(x)
    div(search_window, 2) == 0 && return copy(x)

    sigma = h === nothing ? T(estimate_noise_1d(x)) : T(h)
    sigma = max(sigma, T(1e-12) * maximum(abs, x) + T(1e-300))

    work = if log_space
        # Log domain: add a tiny relative floor so zero bins stay finite.
        floor = T(1e-12) * maximum(x) + T(1e-300)
        log.(x .+ floor)
    else
        Vector{T}(x)
    end

    search_r = div(search_window, 2)
    half_v = div(similarity_window, 2)
    offsets = collect(-half_v:half_v)
    gauss = exp.(-T(0.5) .* (offsets ./ T(alpha)) .^ 2)
    gauss = gauss ./ sum(gauss)
    refl = _reflect_indices(offsets, n)
    patches = [x[idx] for idx in refl]

    window(i) = (max(1, i - search_r), min(n, i + search_r))

    function nlm_pass(signal::AbstractVector{T}, sigmas::AbstractVector{T})
        sig_patches = [signal[idx] for idx in refl]
        out = Vector{T}(undef, n)
        for i in 1:n
            lo, hi = window(i)
            ws = Vector{T}(undef, hi - lo + 1)
            for (kk, k) in enumerate(lo:hi)
                dd = sig_patches[i] .- sig_patches[k]
                dst = sum(dd[j]^2 * gauss[j] for j in eachindex(dd))
                rt = min(sqrt(dst) / sigmas[i], T(1e150))
                ws[kk] = exp(-(rt * rt))
            end
            z = sum(ws)
            num = T(0)
            for (kk, k) in enumerate(lo:hi)
                num += ws[kk] * signal[k]
            end
            out[i] = num / z
        end
        out
    end

    # Stage-1 uses uniform h1 = 0.5·σ.
    h1 = T(0.5) * sigma
    intermediate = nlm_pass(work, fill(h1, n))

    # Stage-2 parameters: h2(i) = σ · sqrt(Σ w1(i,·)²) — weights from the
    # *original* (unfiltered) patches, matching Python exactly.
    w2_sq = Vector{T}(undef, n)
    for i in 1:n
        lo, hi = window(i)
        wsum = T(0)
        for k in lo:hi
            d = patches[i] .- patches[k]
            dst = sum(d[j]^2 * gauss[j] for j in eachindex(d))
            rt = min(sqrt(dst) / h1, T(1e150))
            wsum += exp(-(rt * rt))
        end
        total = wsum
        s2 = T(0)
        for k in lo:hi
            d = patches[i] .- patches[k]
            dst = sum(d[j]^2 * gauss[j] for j in eachindex(d))
            rt = min(sqrt(dst) / h1, T(1e150))
            w = exp(-(rt * rt)) / total
            s2 += w * w
        end
        w2_sq[i] = s2
    end
    h2 = sigma .* sqrt.(w2_sq)

    filtered = nlm_pass(intermediate, h2)
    log_space ? exp.(filtered) : filtered
end

function solve_osem_anlm(A::AbstractMatrix{T}, b::AbstractVector{T}, x0::AbstractVector{T};
                         max_iterations::Integer=50,
                         n_subsets::Integer=1,
                         tolerance::T=T(1e-6),
                         h::Union{Real,Nothing}=nothing,
                         search_window::Integer=11,
                         similarity_window::Integer=3,
                         alpha::Real=T(1.0),
                         anlm_mode::Union{Symbol,String}=:subset,
                         log_space::Bool=true) where T<:AbstractFloat
    m, n = size(A)
    n_subsets >= 1 || throw(ArgumentError("n_subsets must be >= 1"))
    n_subsets <= m || throw(ArgumentError("n_subsets ($n_subsets) exceeds m ($m)"))
    mode = Symbol(anlm_mode)
    mode in _ANLM_MODES || throw(ArgumentError("anlm_mode must be one of $_ANLM_MODES, got $anlm_mode"))
    if h !== nothing
        T(h) > 0 || throw(ArgumentError("h must be positive, got $h"))
    end
    search_window >= 1 || throw(ArgumentError("search_window must be >= 1"))
    similarity_window >= 1 || throw(ArgumentError("similarity_window must be >= 1"))
    alpha > 0 || throw(ArgumentError("alpha must be positive"))

    eps = T(1e-11)
    subsets = _split_indices(m, Int(n_subsets))
    x = max.(Vector{T}(x0), T(0))
    converged = false
    iters = 0

    apply_anlm(v) = anlm_filter_1d(v; h=h, search_window=search_window,
                                    similarity_window=similarity_window,
                                    alpha=alpha, log_space=log_space)

    for it in 1:max_iterations
        iters = it
        x_old = copy(x)
        for idx in subsets
            A_s = A[idx, :]
            b_s = b[idx]
            colsum = vec(sum(A_s; dims=1))
            fwd = A_s * x
            ratio = b_s ./ (fwd .+ eps)
            corr = A_s' * ratio
            x = max.(x .* corr ./ (colsum .+ eps), T(0))
            mode === :subset && (x = apply_anlm(x))
        end
        rel = norm(x - x_old) / (norm(x_old) + eps)
        rel < tolerance && (converged = true; break)
    end

    mode === :post && (x = apply_anlm(x))
    x = max.(x, T(0))
    UnfoldResult(x, iters, converged, norm(b .- A * x))
end

function _split_indices(m::Int, k::Int)
    k = min(max(k, 1), m)
    base = m ÷ k
    rem  = m % k
    out  = Vector{UnitRange{Int}}()
    start = 1
    for s in 1:k
        size_s = base + (s <= rem ? 1 : 0)
        push!(out, start:(start + size_s - 1))
        start += size_s
    end
    out
end
