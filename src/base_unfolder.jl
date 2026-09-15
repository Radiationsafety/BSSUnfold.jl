"""
Base framework for running unfolding (port of _base_unfolder.py).

All `unfold_*` algorithm functions are reduced to a call of `run_unfolding`
with the corresponding `solve_func` and `solve_kwargs`.
"""

"""
    make_solve_wrapper(solve_func; fixed_params...)

Create a wrapper compatible with the `run_unfolding` interface:
wrapper(A, b; kwargs...) -> UnfoldResult, forwarding x0 and fixed_params.
"""
function make_solve_wrapper(solve_func::Function; fixed_params...)
    function wrapper(A, b; kwargs...)
        x0 = pop!(Dict(kwargs), :x0, nothing)
        return solve_func(A, b, x0; merge(NamedTuple(fixed_params), kwargs)...)
    end
    return wrapper
end


"""
    run_unfolding(solve_func, detector_names, n_energy_bins, E_MeV,
                  sensitivities, cc_icrp116, readings;
                  initial_spectrum=nothing, default_initial=nothing,
                  solve_kwargs=NamedTuple(), method_name="",
                  calculate_errors=false, noise_level=0.01,
                  n_montecarlo=100, random_state=nothing, save_result=nothing)

Universal unfolding pipeline:

1. Validate inputs
2. Build system (A, b)
3. Normalize initial spectrum
4. Call `solve_func(A, b, x0=x0; solve_kwargs...)`
5. Standardize output
6. (optional) Monte-Carlo uncertainty
7. (optional) Save result

# Returns
`Dict{String,Any}` with standard keys.
"""
function run_unfolding(solve_func::Function,
                      detector_names::Vector{String},
                      n_energy_bins::Integer,
                      E_MeV::Vector{Float64},
                      sensitivities::Dict{String,Vector{T}},
                      cc_icrp116::Dict{String,Vector{T}},
                      readings::Dict{String,T};
                      initial_spectrum::Union{Nothing,Vector{T}}=nothing,
                      default_initial::Union{Nothing,Vector{T}}=nothing,
                      solve_kwargs::NamedTuple=NamedTuple(),
                      method_name::AbstractString="",
                      calculate_errors::Bool=false,
                      noise_level::Real=T(0.01),
                      n_montecarlo::Integer=100,
                      random_state::Union{Integer,Nothing}=nothing,
                      save_result::Union{Nothing,Function}=nothing) where T<:AbstractFloat

    # 0. Validate
    isempty(readings) && throw(ArgumentError("readings must be non-empty"))
    isempty(detector_names) && throw(ArgumentError("detector_names must be non-empty"))
    n_energy_bins > 0 || throw(ArgumentError("n_energy_bins must be positive"))
    length(E_MeV) == n_energy_bins || throw(ArgumentError("E_MeV length must match n_energy_bins"))

    # 1. Build system
    A, b, selected = build_system(readings, detector_names, sensitivities)

    # 2. Normalize initial
    if default_initial === nothing
        default_initial = ones(T, n_energy_bins) * T(0.5)
    end
    x0 = normalize_initial(initial_spectrum, default_initial, n_energy_bins)

    # 3. Solve
    # solve_func has the signature: (A, b, x0; kwargs...) -> UnfoldResult
    # solve_kwargs: NamedTuple with algorithm parameters
    result = solve_func(A, b, x0; solve_kwargs...)

    # 4. Standardize output
    output = standardize_output(
        result.spectrum, A, b, E_MeV, selected, cc_icrp116, method_name,
        Dict("iterations" => result.iterations,
             "converged"   => result.converged))

    # 5. Monte-Carlo uncertainty
    if calculate_errors
        @info "Calculating uncertainty with $n_montecarlo Monte-Carlo samples..."
        mc = monte_carlo_uncertainty(
            (A, b, x) -> solve_func(A, b, x; solve_kwargs...),
            A, b, x0, noise_level, n_montecarlo,
            random_state=random_state)
        output["spectrum_uncert_mean"]    = mc.mean
        output["spectrum_uncert_std"]     = mc.std
        output["spectrum_uncert_median"] = mc.median
        output["spectrum_uncert_p5"]     = mc.p5
        output["spectrum_uncert_p95"]     = mc.p95
        output["spectrum_uncert_all"]     = mc.all
        output["montecarlo_samples"]      = n_montecarlo
        output["noise_level"]             = Float64(noise_level)
        @info "...uncertainty calculation completed."
    end

    # 6. Save
    if save_result !== nothing
        save_result(output)
    end

    return output
end
