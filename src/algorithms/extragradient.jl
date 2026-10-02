"""
Korpelevich's extragradient method (Korpelevich, 1976) for neutron spectra.

Solves the robust unfolding saddle problem

    min_{x >= 0}  max_{||y||_2 <= 1}  1/2 ||A x - b||^2 + delta * y^T (A x - b)
    = min_{x >= 0} 1/2 ||A x - b||^2 + delta * ||A x - b||_2,

with delta = noise_level * ||b||_2, via the two-step (predict/correct)
extragradient scheme on the bilinear saddle form. The extra gradient
evaluation restores convergence for monotone operators where a plain
gradient step oscillates. (MIPT optimization course, lecture 13, homework 20.)
"""
function _extragradient_project_ball(v::AbstractVector{T}, radius::T) where T<:AbstractFloat
    n = norm(v)
    if n > radius
        return v .* (radius / max(n, T(1e-300)))
    end
    return copy(v)
end


function solve_extragradient(A::AbstractMatrix{T}, b::AbstractVector{T}, x0::AbstractVector{T};
                             max_iterations::Integer=2000,
                             tolerance::T=T(1e-8),
                             noise_level::T=T(0.02),
                             step_size::Union{Nothing,T}=nothing) where T<:AbstractFloat
    validate_system(A, b, x0=x0, max_iterations=max_iterations, tolerance=tolerance)

    nl = max(T(noise_level), T(0))
    delta = nl * norm(b)

    x = max.(copy(x0), T(0))
    y = zeros(T, size(A, 1))

    norm_A = opnorm(A)
    if step_size === nothing
        L = norm_A^2 + delta * norm_A + T(1)
        eta = T(0.9) / L
    else
        eta = T(step_size)
    end

    function _extragradient_F_x(xv::AbstractVector{T}, yv::AbstractVector{T})
        return A' * (A * xv .- b) .+ delta .* (A' * yv)
    end

    converged = false
    iters = 0
    @inbounds for k in 1:max_iterations
        residual = A * x .- b

        x_tilde = max.(x .- eta .* _extragradient_F_x(x, y), T(0))
        y_tilde = _extragradient_project_ball(y .+ eta .* residual, T(1))

        residual_t = A * x_tilde .- b
        x_new = max.(x .- eta .* _extragradient_F_x(x_tilde, y_tilde), T(0))
        y_new = _extragradient_project_ball(y .+ eta .* residual_t, T(1))

        rel_change = norm(x_new .- x) / max(norm(x), T(1e-30))
        x = x_new
        y = y_new
        iters = k
        if rel_change < tolerance
            converged = true
            break
        end
    end

    residual = b .- A * x
    return UnfoldResult(x, iters, converged, norm(residual))
end
