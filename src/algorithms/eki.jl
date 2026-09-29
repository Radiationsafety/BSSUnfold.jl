"""
    solve_eki(A, b, x0; n_ensemble=50, n_iterations=50, regularization=1e-4,
              inflation=1.02, noise_std=nothing, random_state=nothing)

Ensemble Kalman Inversion (Iglesias et al., 2013) for approximate
Bayesian unfolding without MCMC. Faithful port of `unfold_eki.solve_eki`.

    x_e <- x_e + C_md * C_dd^{-1} * (b + noise_e - A x_e)

where `C_dd` is the covariance of the predictions (with added noise and
regularization on the diagonal) and `C_md` the cross-covariance of state
and predictions. Covariance inflation and projection onto the nonnegative
orthant are applied after every update.

`random_state` (an integer in 0..2^32-1) selects a bit-exact port of NumPy's
legacy `RandomState` (MT19937 + polar Box-Muller `legacy_gauss`), so seeded
runs reproduce the Python reference stream element-for-element. Without a
seed, Julia's default RNG is used (statistically equivalent).
"""

# ─── NumPy legacy RandomState (MT19937) port ────────────────────────────────

mutable struct _EkiNumpyRNG
    mt::Vector{UInt32}
    idx::Int
    has_gauss::Bool
    gauss::Float64
    function _EkiNumpyRNG()
        s = new(zeros(UInt32, 624), 625, false, 0.0)
        return s
    end
end

function _eki_mt_seed!(r::_EkiNumpyRNG, seed::Integer)
    s = r.mt
    s[1] = UInt32(seed % 0x1_0000_0000)
    @inbounds for i in 2:624
        prev = UInt64(s[i - 1])
        x = prev ⊻ (prev >> 30)
        s[i] = UInt32((0x6c078965 * x + (i - 1)) % 0x1_0000_0000)
    end
    r.idx = 625
    r.has_gauss = false
    return r
end

@inline function _eki_mt_next32!(r::_EkiNumpyRNG)
    if r.idx > 624
        s = r.mt
        upper, lower = 0x8000_0000, 0x7fff_ffff
        n_len, m_len = 624, 397
        @inbounds for i in 1:(n_len - m_len)
            y = (s[i] & upper) | (s[i + 1] & lower)
            s[i] = s[i + m_len] ⊻ (y >> 1) ⊻ (isodd(y) ? 0x9908_b0df : UInt32(0))
        end
        @inbounds for i in (n_len - m_len + 1):n_len
            y = (s[i] & upper) | (s[mod1(i + 1, n_len)] & lower)
            s[i] = s[((i - 1 + m_len) % n_len) + 1] ⊻ (y >> 1) ⊻ (isodd(y) ? 0x9908_b0df : UInt32(0))
        end
        r.idx = 1
    end
    y = @inbounds r.mt[r.idx]
    r.idx += 1
    y ⊻= y >> 11
    y ⊻= (y << 7) & 0x9d2c_5680
    y ⊻= (y << 15) & 0xefc6_0000
    y ⊻= y >> 18
    return y
end

@inline function _eki_next_double!(r::_EkiNumpyRNG)
    a = UInt64(_eki_mt_next32!(r)) >> 5
    b = UInt64(_eki_mt_next32!(r)) >> 6
    return (a * 67108864.0 + b) / 9007199254740992.0
end

# NumPy legacy_gauss: polar (Marsaglia) Box-Muller with one cached value.
function _eki_gauss!(r::_EkiNumpyRNG)
    if r.has_gauss
        r.has_gauss = false
        return r.gauss
    end
    while true
        m1 = 2.0 * _eki_next_double!(r) - 1.0
        m2 = 2.0 * _eki_next_double!(r) - 1.0
        r2 = m1 * m1 + m2 * m2
        if r2 == 0.0 || r2 >= 1.0
            continue
        end
        fac = sqrt(-2.0 * log(r2) / r2)
        r.gauss = m1 * fac
        r.has_gauss = true
        return m2 * fac
    end
end

_eki_randn_numpy!(r::_EkiNumpyRNG) = _eki_gauss!(r)

# ─── Main solver ─────────────────────────────────────────────────────────────

function solve_eki(A::AbstractMatrix{T}, b::AbstractVector{T}, x0::AbstractVector{T};
                   n_ensemble::Integer=50,
                   n_iterations::Integer=50,
                   regularization::Real=1e-4,
                   inflation::Real=1.02,
                   noise_std::Union{Nothing,Real}=nothing,
                   random_state::Union{Nothing,Integer}=nothing) where T<:AbstractFloat
    m, n = size(A)
    n_ensemble = Int(n_ensemble)
    n_iterations = Int(n_iterations)

    AF = Float64.(Matrix(A))
    bf = Float64.(vec(collect(b)))
    x0f = Float64.(vec(collect(x0)))

    use_numpy = random_state !== nothing && 0 <= random_state < typemax(UInt32)
    rng_jl = random_state === nothing ? Random.default_rng() :
             MersenneTwister(Int(random_state))
    rng_np = use_numpy ? _eki_mt_seed!(_EkiNumpyRNG(), Int(random_state)) : nothing
    gauss = if use_numpy
        () -> _eki_randn_numpy!(rng_np)
    else
        () -> randn(rng_jl)
    end

    reg = Float64(regularization)
    infl = Float64(inflation)

    effective_noise_std = noise_std === nothing ?
        (m > 0 ? 0.05 * norm(bf) / sqrt(m) : 1e-6) : Float64(noise_std)
    noise_var = effective_noise_std^2

    sigma_prior = abs.(x0f) .+ 1e-6
    ensemble = Matrix{Float64}(undef, n, n_ensemble)
    @inbounds for i in 1:n            # NumPy fills row-major (C order)
        loc, sc = x0f[i], sigma_prior[i]
        for e in 1:n_ensemble
            ensemble[i, e] = loc + sc * gauss()
        end
    end

    I_m = Matrix{Float64}(I, m, m)
    @inbounds for iteration in 1:n_iterations
        predictions = AF * ensemble
        pred_mean = vec(mean(predictions; dims=2))
        state_mean = vec(mean(ensemble; dims=2))

        pred_pert = predictions .- pred_mean
        state_pert = ensemble .- state_mean

        C_dd = (pred_pert * pred_pert') / max(n_ensemble - 1, 1)
        C_dd .+= (noise_var + reg) .* I_m

        C_md = (state_pert * pred_pert') / max(n_ensemble - 1, 1)

        C_d_inv = try
            C_dd \ I_m
        catch
            pinv(C_dd)
        end

        innovation = Matrix{Float64}(undef, m, n_ensemble)
        @inbounds for i in 1:m
            for e in 1:n_ensemble
                innovation[i, e] = bf[i] + effective_noise_std * gauss() - predictions[i, e]
            end
        end
        ensemble = ensemble + C_md * (C_d_inv * innovation)

        ensemble .*= infl
        ensemble .= max.(ensemble, 0.0)
    end

    mean_spectrum = max.(vec(mean(ensemble; dims=2)), 0.0)

    residual = bf .- AF * mean_spectrum
    return UnfoldResult(T.(mean_spectrum), n_iterations, true, T(norm(residual)),
                        Dict{String,Any}("n_ensemble" => n_ensemble,
                                         "regularization" => reg,
                                         "inflation" => infl))
end
