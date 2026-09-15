"""
Monte-Carlo estimation of unfolding uncertainty (port of _montecarlo.py).
"""

"""
    monte_carlo_uncertainty(solve_func, A, b, x0, noise_level, n_samples;
                            random_state=nothing, kwargs...)

Estimate unfolding uncertainty via Monte Carlo: for each sample,
add Gaussian noise to `b`, perform the unfolding, collect statistics.

# Arguments
- `solve_func::Function` — function with signature `(A, b, x0; kwargs...) -> UnfoldResult`
- `A, b, x0` — system
- `noise_level::Real` — relative noise level (e.g. 0.01 = 1%)
- `n_samples::Int` — number of MC samples
- `random_state::Union{Int,Nothing}` — seed for reproducibility
- `kwargs...` — forwarded to `solve_func`

# Returns
NamedTuple with fields `mean`, `std`, `median`, `p5`, `p95`, `all` (matrix n_samples × n).
"""
function monte_carlo_uncertainty(solve_func::Function,
                                A::AbstractMatrix{T},
                                b::AbstractVector{T},
                                x0::AbstractVector{T},
                                noise_level::Real,
                                n_samples::Integer;
                                random_state::Union{Integer,Nothing}=nothing,
                                kwargs...) where T<:AbstractFloat
    n = length(x0)
    rng = random_state === nothing ? MersenneTwister() : MersenneTwister(random_state)
    σ = T(noise_level)

    # Pre-allocate
    spectra = zeros(T, n_samples, n)

    # Pre-generate noise (vectorised)
    noise_factors = T(1) .+ randn(rng, T, n_samples, length(b)) .* σ

    for i in 1:n_samples
        b_noisy = b .* view(noise_factors, i, :)
        result = solve_func(A, b_noisy, x0; kwargs...)
        spectra[i, :] .= result.spectrum
    end

    return (
        mean    = vec(mean(spectra, dims=1)),
        std     = vec(std(spectra, dims=1)),
        median  = vec(median(spectra, dims=1)),
        p5      = vec(mapslices(col -> quantile(col, 0.05), spectra, dims=1)),
        p95     = vec(mapslices(col -> quantile(col, 0.95), spectra, dims=1)),
        min     = vec(minimum(spectra, dims=1)),
        max     = vec(maximum(spectra, dims=1)),
        all     = spectra,
    )
end


"""
    add_noise(readings::Dict{String,T}, noise_level::Real, rng::AbstractRNG)

Add Gaussian noise to a readings dictionary. Returns a new Dict.
"""
function add_noise(readings::Dict{String,T}, noise_level::Real,
                  rng::AbstractRNG) where T<:AbstractFloat
    σ = T(noise_level)
    return Dict{String,T}(k => v * (T(1) + σ * randn(rng)) for (k, v) in readings)
end
