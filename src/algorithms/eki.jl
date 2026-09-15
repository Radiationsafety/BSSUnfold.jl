"""
    solve_eki(A, b, x0; n_ensemble=50, n_iterations=50, regularization=1e-4,
              inflation=1.02, noise_std=nothing, random_state=nothing)

Ensemble Kalman Inversion (Iglesias et al., 2013) для приближённой
байесовской развёртки без MCMC.

Ансамбль частиц распространяется через прямую модель; обновление
выполняется уравнением калмановского усиления:

    x_e <- x_e + C_md * C_dd⁻¹ * (b + noise_e - A x_e)

где `C_dd` — ковариация предсказаний (с добавкой шума и регуляризации),
`C_md` — кросс-ковариация состояния и предсказаний. После каждого шага
применяется инфляция ковариации и проекция на неотрицательный ортант.

# Возвращает
`UnfoldResult` со средним спектром ансамбля; `random_state` задаёт
воспроизводимость (MersenneTwister).
"""
function solve_eki(A::AbstractMatrix{T}, b::AbstractVector{T}, x0::AbstractVector{T};
                   n_ensemble::Integer=50,
                   n_iterations::Integer=50,
                   regularization::T=T(1e-4),
                   inflation::T=T(1.02),
                   noise_std::Union{Nothing,T}=nothing,
                   random_state::Union{Nothing,Integer}=nothing) where T<:AbstractFloat
    m, n = size(A)

    rng = random_state === nothing ? Random.default_rng() : MersenneTwister(random_state)

    effective_noise_std = noise_std === nothing ?
        (m > 0 ? T(0.05) * norm(b) / sqrt(m) : T(1e-6)) : noise_std
    noise_var = effective_noise_std^2

    sigma_prior = abs.(x0) .+ T(1e-6)
    ensemble = x0 .+ sigma_prior .* randn(rng, (n, n_ensemble))

    iterations = 0
    for iteration in 1:n_iterations
        iterations = iteration
        predictions = A * ensemble
        pred_mean = BSSUnfold.Statistics.mean(predictions, dims=2)
        state_mean = BSSUnfold.Statistics.mean(ensemble, dims=2)

        pred_pert = predictions .- pred_mean
        state_pert = ensemble .- state_mean

        C_dd = (pred_pert * pred_pert') / max(n_ensemble - 1, 1)
        C_dd .+= (noise_var + regularization) .* Matrix{T}(I, m, m)

        C_md = (state_pert * pred_pert') / max(n_ensemble - 1, 1)

        C_d_inv = try
            C_dd \ Matrix{T}(I, m, m)
        catch
            pinv(C_dd)
        end

        innovation = b .+ effective_noise_std .* randn(rng, (m, n_ensemble)) .- predictions
        ensemble .+= C_md * (C_d_inv * innovation)

        ensemble .*= inflation
        ensemble .= max.(ensemble, T(0))
    end

    mean_spectrum = vec(BSSUnfold.Statistics.mean(ensemble, dims=2))
    mean_spectrum = max.(mean_spectrum, T(0))

    residual = b .- A * mean_spectrum
    return UnfoldResult(mean_spectrum, iterations, true, norm(residual),
                        Dict{String,Any}("n_ensemble" => n_ensemble,
                                         "regularization" => regularization,
                                         "inflation" => inflation))
end
