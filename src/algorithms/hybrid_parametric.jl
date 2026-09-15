"""
Hybrid parametric-nonparametric unfolding.

A two-stage pipeline: (1) physically motivated initial
approximation from the parametric FRUIT model (coarse grid scan
`find_initial_params` + `parametric_model`), (2) nonparametric
refinement by Landweber or MLEM iterations (with
non-negativity clipping).  If the parametric model fails — fallback to
a flat/mean scale.  The Python wrappers `unfold_*` are not ported here
(the Julia package uses the common `run_unfolding`).
"""
function _hyparam_landweber_iteration(spectrum::Vector{Float64}, A::Matrix{Float64},
                                     b::Vector{Float64}, step_size::Float64,
                                     max_iter::Int, tolerance::Float64)
    x = copy(spectrum)
    for i in 1:max_iter
        residual = b .- A * x
        gradient = A' * residual
        x_new = max.(x .+ step_size .* gradient, 0.0)
        if norm(x_new .- x) < tolerance
            return x_new, i
        end
        x = x_new
    end
    return x, max_iter
end

function _hyparam_mlem_iteration(spectrum::Vector{Float64}, A::Matrix{Float64},
                                 b::Vector{Float64}, max_iter::Int, tolerance::Float64)
    x = max.(copy(spectrum), 1e-15)
    for i in 1:max_iter
        computed = max.(A * x, 1e-15)
        ratio = b ./ computed
        correction = A' * ratio
        x_new = max.(x .* correction, 0.0)
        if norm(x_new .- x) / (norm(x) + 1e-15) < tolerance
            return x_new, i
        end
        x = x_new
    end
    return x, max_iter
end

"""
    solve_hybrid_parametric(A, b, x0=nothing; E=nothing, refinement_method="landweber",
                            max_iterations=100, tolerance=1e-6, step_size=0.01)
      -> UnfoldResult

Mixed unfolding: parametric FRUIT initialization (grid scan over
`P_th`/`P_epi`), then refinement by `refinement_method`
("landweber" with step `step_size` or "mlem").  `x0` is used
only as a fallback when the parametric model fails.  Returns
an `UnfoldResult` with `extra["message"]`.
"""
function solve_hybrid_parametric(A::AbstractMatrix, b::AbstractVector, x0::Union{Nothing,AbstractVector}=nothing;
                                 E::Union{Nothing,AbstractVector}=nothing,
                                 refinement_method::String="landweber",
                                 max_iterations::Integer=100,
                                 tolerance::Real=1e-6,
                                 step_size::Real=0.01)
    AF = Matrix{Float64}(A)
    bf = Vector{Float64}(b)
    n_energy = size(AF, 2)
    E_f = E === nothing ? collect(10.0 .^ range(-9, 2, length=n_energy)) : Float64.(collect(E))

    p = try
        find_initial_params(AF, bf, E_f, compute_log_steps(E_f) .* log(10); n_grid=5, n_restarts=1)
    catch
        nothing
    end

    if p !== nothing && p isa Vector{Float64}
        ss = compute_log_steps(E_f) .* log(10)
        parametric_guess = max.(parametric_model(E_f, p[1], p[2], p[3], p[4], p[5], p[6]) .* ss, 1e-30)
    else
        parametric_guess = fill(mean(bf) / max(mean(sum(AF, dims=2)), 1e-300), n_energy)
    end

    n_iter = 0
    success = false
    message = ""
    if refinement_method == "landweber"
        refined, n_iter = _hyparam_landweber_iteration(parametric_guess, AF, bf,
                                                        Float64(step_size), Int(max_iterations),
                                                        Float64(tolerance))
        success = n_iter < max_iterations
        message = success ? "Converged in $n_iter iterations" : "Max iterations reached"
    elseif refinement_method == "mlem"
        refined, n_iter = _hyparam_mlem_iteration(parametric_guess, AF, bf,
                                                   Int(max_iterations), Float64(tolerance))
        success = n_iter < max_iterations
        message = success ? "Converged in $n_iter iterations" : "Max iterations reached"
    else
        throw(ArgumentError("Unknown refinement method: $refinement_method"))
    end

    extra = Dict{String,Any}(
        "message" => message,
        "refinement_method" => refinement_method,
        "step_size" => Float64(step_size),
    )
    return UnfoldResult(max.(refined, 0.0), n_iter, success, norm(AF * refined .- bf), extra)
end
