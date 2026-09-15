"""
Iterative refinement — two-pass unfolding.

Port from `bssunfold/src/bssunfold/core/unfold_iterative_refinement.py`.

Algorithm:
1. First pass: fast EM method (MLEM by default) with a small number of iterations —
   rough structure of the spectrum.
2. The residual r = b - A*x1 is computed.
3. Second pass: gradient method (Landweber by default) on the residual —
   corrects systematic errors of EM methods.
4. Combination: x_final = x1 + alpha * x2, where alpha is chosen via
   a linear search minimizing ||A*x_final - b||.

Combines the convergence speed of EM methods with the accuracy
of gradient methods.
"""

"""
    solve_iterative_refinement(A, b, x0; first_pass_solver, second_pass_solver,
                              first_pass_kwargs, second_pass_kwargs,
                              alpha, max_alpha_search)

Two-pass unfolding: EM method + gradient correction.

# Arguments
- `A::AbstractMatrix{T}`: response matrix (m × n)
- `b::AbstractVector{T}`: measurements (m,)
- `x0::AbstractVector{T}`: initial spectrum (n,)
- `first_pass_solver`: function `(A, b, x0; kwargs...) -> UnfoldResult` (default `solve_mlem`)
- `second_pass_solver`: likewise (default `solve_landweber`)
- `first_pass_kwargs`, `second_pass_kwargs`: `NamedTuple` of parameters
- `alpha::Union{T,Nothing}`: if `nothing` — linear search
- `max_alpha_search`: number of alpha candidates in the line search

# Returns
- `UnfoldResult{T}` with the spectrum; `extra` contains diagnostics
"""
function solve_iterative_refinement(A::AbstractMatrix{T}, b::AbstractVector{T}, x0::AbstractVector{T};
                                   first_pass_solver::Function=solve_mlem,
                                   second_pass_solver::Function=solve_landweber,
                                   first_pass_kwargs::NamedTuple=(max_iterations=150, tolerance=T(1e-4)),
                                   second_pass_kwargs::NamedTuple=(max_iterations=100, tolerance=T(1e-5)),
                                   alpha::Union{T,Nothing}=nothing,
                                   max_alpha_search::Integer=20,
                                   eps::T=T(1e-30)) where T<:AbstractFloat
    m, n = size(A)

    # --- First pass ---
    res1 = first_pass_solver(A, b, x0; first_pass_kwargs...)
    x1 = max.(res1.spectrum, T(0))

    # --- Residual ---
    r = b .- A * x1

    # --- Second pass on the residual ---
    x0_zero = zeros(T, n)
    res2 = second_pass_solver(A, r, x0_zero; second_pass_kwargs...)
    x2 = res2.spectrum

    # --- Combination ---
    if alpha !== nothing
        best_alpha = alpha
    else
        # Linear search: minimize ||A*(x1 + a*x2) - b||
        candidates = range(T(0), T(2); length=max_alpha_search)
        best_alpha = T(0)
        best_res = norm(A * x1 .- b)
        for a in candidates
            x_cand = x1 .+ a .* x2
            res_cand = norm(A * x_cand .- b)
            if res_cand < best_res
                best_res = res_cand
                best_alpha = a
            end
        end
    end

    spectrum = max.(x1 .+ best_alpha .* x2, T(0))
    residual = b .- A * spectrum

    return UnfoldResult(
        spectrum, res1.iterations + res2.iterations,
        res1.converged && res2.converged,
        norm(residual),
        Dict{String,Any}(
            "first_pass_residual"          => norm(r),
            "second_pass_correction_norm"  => norm(x2),
            "alpha"                        => best_alpha,
            "final_residual"               => norm(residual),
            "first_pass_iterations"        => res1.iterations,
            "second_pass_iterations"       => res2.iterations,
        )
    )
end
