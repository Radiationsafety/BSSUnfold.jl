"""
Bayesian parametric unfolding — Metropolis-Hastings sampling of the
5-parameter Maxwellian + 1/E + evaporation spectrum. Faithful port of
`bssunfold/core/unfold_bayesian_parametric.py`.

Parameters (`initial_params`): `A_th`, `T_th`, `A_epi`, `A_f`, `T_ev`, with
uniform priors:
* `A_th`, `A_epi`, `A_f ∈ [0, 1e-3]`
* `T_th ∈ [1e-9, 1e-3]`
* `T_ev ∈ [0.1, 20]`

Proposal: random-walk Gaussian, `std = proposal_scale · |x| + 1e-15`,
truncated to the same bounds. Returns the posterior mean parameters
evaluated on `parametric_model_fp` (the FRUIT-like model), scaled by
`log_steps`. RNG: `MersenneTwister(random_state)` — Julia has no PCG64
equivalent, so we don't bit-match Python; reproducibility is checked
Julia-vs-Julia only.
"""

function _maxwellian_fp(E::AbstractVector{T}, T_th::Real, A::Real) where T<:Real
    A * sqrt.(E) .* exp.(-E ./ T_th)
end
function _one_over_e_fp(E::AbstractVector{T}, A::Real) where T<:Real
    A ./ (E .+ T(1e-15))
end
function _evaporation_fp(E::AbstractVector{T}, T_ev::Real, A::Real) where T<:Real
    A .* exp.(-E ./ T_ev)
end

"""
    parametric_model_fp(E, A_th, T_th, A_epi, A_f, T_ev; epi_max=0.1)

5-parameter FRUIT-like model (thermal / epithermal / evaporation), shared
between the Bayesian sampler and the (deferred) `fruit_like` NLS port.
"""
function parametric_model_fp(E::AbstractVector{<:Real},
                             A_th::Real, T_th::Real, A_epi::Real, A_f::Real, T_ev::Real;
                             epi_max::Real=0.1)
    Ef = Float64.(collect(E))
    spec = zeros(length(Ef))
    th_mask  = Ef .< 0.4e-6
    epi_mask = (Ef .>= 0.4e-6) .& (Ef .< Float64(epi_max))
    fast     = Ef .>= Float64(epi_max)
    if any(th_mask)
        spec[th_mask] .+= _maxwellian_fp(Ef[th_mask], T_th, A_th)
    end
    if any(epi_mask)
        spec[epi_mask] .+= _one_over_e_fp(Ef[epi_mask], A_epi)
    end
    if any(fast)
        spec[fast] .+= _evaporation_fp(Ef[fast], T_ev, A_f)
    end
    spec
end

const _BAYES_PARAM_KEYS = (:A_th, :T_th, :A_epi, :A_f, :T_ev)

function _bayes_log_prior(p::AbstractDict{Symbol,<:Real})::Float64
    lp = 0.0
    p[:A_th]  < 0.0 && return typemin(Float64)
    p[:A_th]  > 1e-3 && return typemin(Float64)
    p[:T_th]  < 1e-9 && return typemin(Float64)
    p[:T_th]  > 1e-3 && return typemin(Float64)
    p[:A_epi] < 0.0  && return typemin(Float64)
    p[:A_epi] > 1e-3 && return typemin(Float64)
    p[:A_f]   < 0.0  && return typemin(Float64)
    p[:A_f]   > 1e-3 && return typemin(Float64)
    p[:T_ev]  < 0.1  && return typemin(Float64)
    p[:T_ev]  > 20.0 && return typemin(Float64)
    lp += -log(1e-3)
    lp += -log(1e-3 - 1e-9)
    lp += -log(1e-3)
    lp += -log(1e-3)
    lp += -log(19.9)
    lp
end

function _bayes_log_likelihood(p::AbstractDict{Symbol,<:Real},
                               A::AbstractMatrix{<:Real}, b::AbstractVector{<:Real},
                               E::AbstractVector{<:Real}, log_steps::AbstractVector{<:Real},
                               sigma::Real)::Float64
    spec = parametric_model_fp(E, p[:A_th], p[:T_th], p[:A_epi], p[:A_f], p[:T_ev])
    spec_with = spec .* log_steps
    resid = b .- A * spec_with
    -0.5 * sum((resid ./ sigma) .^ 2)
end

function _bayes_log_posterior(p::AbstractDict{Symbol,<:Real},
                              A, b, E, log_steps, sigma)
    lp = _bayes_log_prior(p)
    isfinite(lp) || return typemin(Float64)
    lp + _bayes_log_likelihood(p, A, b, E, log_steps, sigma)
end

function solve_bayesian_parametric(A_matrix::AbstractMatrix{T}, b_readings::AbstractVector{T},
                                   E::AbstractVector{T}, log_steps::AbstractVector{T};
                                   sigma::Real=T(0.02),
                                   initial_params::Union{AbstractDict{Symbol},Nothing}=nothing,
                                   n_samples::Integer=1000,
                                   burn_in::Integer=200,
                                   proposal_scale::Real=T(0.1),
                                   random_state::Union{Integer,Nothing}=nothing) where T<:AbstractFloat
    params = if initial_params === nothing
        Dict{Symbol,Float64}(:A_th => 1e-6, :T_th => 0.025e-6,
                             :A_epi => 1e-6, :A_f => 1e-6, :T_ev => 2.0)
    else
        Dict{Symbol,Float64}(k => Float64(v) for (k, v) in pairs(initial_params))
    end

    rng = random_state === nothing ? Random.MersenneTwister() :
                                     Random.MersenneTwister(Int(random_state))
    scale = Float64(proposal_scale)

    try
        current = copy(params)
        current_lp = _bayes_log_posterior(current, A_matrix, b_readings, E, log_steps, sigma)
        accepted = 0
        sums = Dict{Symbol,Float64}(k => 0.0 for k in keys(current))
        n_kept = 0
        for i in 1:(n_samples + burn_in)
            proposed = Dict{Symbol,Float64}()
            for (k, v) in current
                nv = v + randn(rng) * (scale * abs(v) + 1e-15)
                if k === :T_th || k === :T_ev
                    nv = max(nv, 1e-9)
                elseif startswith(String(k), "A_")
                    nv = max(nv, 0.0)
                end
                proposed[k] = nv
            end
            proposed_lp = _bayes_log_posterior(proposed, A_matrix, b_readings, E, log_steps, sigma)
            if log(rand(rng)) < proposed_lp - current_lp
                current = proposed
                current_lp = proposed_lp
                accepted += 1
            end
            if i > burn_in
                n_kept += 1
                for k in keys(sums)
                    sums[k] += current[k]
                end
            end
        end
        mean_params = Dict{Symbol,Float64}(k => v / max(n_kept, 1) for (k, v) in sums)

        spec_shape = parametric_model_fp(E, mean_params[:A_th], mean_params[:T_th],
                                          mean_params[:A_epi], mean_params[:A_f],
                                          mean_params[:T_ev])
        spectrum = spec_shape .* log_steps
        extra = Dict{String,Any}(
            "mean_params" => mean_params,
            "sigma" => Float64(sigma),
            "n_samples" => Int(n_samples),
            "burn_in" => Int(burn_in),
            "proposal_scale" => Float64(proposal_scale),
            "accepted" => accepted,
        )
        UnfoldResult(max.(spectrum, T(0)), Int(n_samples), true,
                     norm(b_readings .- A_matrix * spectrum), extra)
    catch err
        @warn "solve_bayesian_parametric: sampling failed ($err); returning zero spectrum."
        UnfoldResult(zeros(T, length(E)), 0, false, T(0),
                     Dict{String,Any}("error" => sprint(showerror, err)))
    end
end
