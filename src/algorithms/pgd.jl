"""
PGD — Projected Gradient Descent unfolding for neutron spectra.

Solves: min_x (1/2)||Ax - b||² + (reg/2)||x||²   s.t.  x ∈ C

where C is the nonnegative orthant, a box [0, x_max], or the simplex
{x >= 0, sum x = total_fluence}. Each iteration takes a gradient step
followed by the Euclidean projection onto C. The simplex constraint keeps
the total fluence fixed at a physically meaningful value, which a plain
Landweber iteration (PGD on the orthant with fixed step) cannot enforce.

Reference: bssunfold, core/unfold_pgd.py.
"""
function _pgd_project_simplex(v::AbstractVector{T}, total::T) where T<:AbstractFloat
    n = length(v)
    u = sort(v; rev=true)
    css = cumsum(u)
    idx = findlast(j -> u[j] * j + (total - css[j]) > 0, 1:n)
    idx === nothing && error("simplex projection failed")
    theta = (css[idx] - total) / T(idx)
    return max.(v .- theta, T(0))
end


function _pgd_project_onto_set(x::AbstractVector{T},
                              constraint::AbstractString,
                              total_fluence::Union{Nothing,T},
                              x_max::T) where T<:AbstractFloat
    c = lowercase(constraint)
    if c == "nonnegative"
        return max.(x, T(0))
    end
    if c == "box"
        return min.(max.(x, T(0)), x_max)
    end
    if c == "simplex"
        if total_fluence === nothing || total_fluence <= 0
            throw(ArgumentError("simplex projection requires total_fluence > 0"))
        end
        return _pgd_project_simplex(x, T(total_fluence))
    end
    throw(ArgumentError("Unknown constraint '$c'; expected 'nonnegative', 'box' or 'simplex'"))
end


function solve_pgd(A::AbstractMatrix{T}, b::AbstractVector{T}, x0::AbstractVector{T};
                   max_iterations::Integer=1000,
                   tolerance::T=T(1e-6),
                   step_size::Union{Nothing,T}=nothing,
                   regularization::T=T(0.0),
                   constraint::AbstractString="nonnegative",
                   total_fluence::Union{Nothing,T}=nothing,
                   x_max::T=T(Inf),
                   backtracking::Bool=false) where T<:AbstractFloat
    validate_system(A, b; x0=x0, max_iterations=max_iterations, tolerance=tolerance)

    x = _pgd_project_onto_set(copy(x0), constraint, total_fluence, x_max)

    L = opnorm(A)^2 + max(regularization, T(0))
    if L <= 0
        return UnfoldResult(x, 0, false, norm(b .- A * x))
    end
    t = step_size === nothing ? T(1) / L : T(step_size)

    function _pgd_objective(z)
        r = A * z .- b
        return T(0.5) * dot(r, r) + T(0.5) * regularization * dot(z, z)
    end

    converged = false
    iters = 0

    for k in 1:Int(max_iterations)
        gradient = A' * (A * x .- b)
        if regularization != 0
            gradient .+= regularization .* x
        end

        x_new = x .- t .* gradient
        if backtracking
            # Armijo condition on the projected step; shrink trial until accepted
            direction = x_new .- x
            f0 = _pgd_objective(x)
            slope = dot(gradient, direction)
            trial = T(1)
            while trial > T(1e-12) && _pgd_objective(x .+ trial .* direction) > f0 + T(1e-4) * trial * slope
                trial *= T(0.5)
            end
            x_new = x .+ trial .* direction
        end
        x_new = _pgd_project_onto_set(x_new, constraint, total_fluence, x_max)

        rel_change = norm(x_new .- x) / max(norm(x), T(1e-30))
        x = x_new
        iters = k
        if rel_change < tolerance
            converged = true
            break
        end
    end

    return UnfoldResult(x, iters, converged, norm(b .- A * x))
end
