"""FISTA — Fast Iterative Shrinkage-Thresholding Algorithm — faithful
port of `bssunfold.core.unfold_fista.unfold_fista` (IRtools IRfista.m by
Silvia Gazzola et al.).

Accelerated proximal gradient for

    min_x 0.5‖Ax − b‖² + 0.5·regularization‖x‖² + l1‖x‖₁ + tv‖Dx‖₁

with nonnegativity projection and optional box constraints.  The step
size is `1/L` with `L = ‖A‖₂²` (exact spectral norm when the number of
energy bins is below 100, else a 20-step power iteration — seeded, and
therefore deterministic, unlike the unseeded NumPy version).
"""
_soft_threshold(x::AbstractVector{T}, threshold::T) where T<:AbstractFloat =
    sign.(x) .* max.(abs.(x) .- threshold, T(0))

function solve_fista(A::AbstractMatrix{T}, b::AbstractVector{T}, x0::AbstractVector{T};
                     max_iterations::Integer=500,
                     tolerance::Real=T(1e-8),
                     regularization::Real=T(0.0),
                     l1_penalty::Real=T(0.0),
                     tv_penalty::Real=T(0.0),
                     nonnegativity::Bool=true,
                     x_min::Real=T(0.0),
                     x_max::Real=T(Inf),
                     noise_level::Union{Nothing,Real}=nothing,
                     eta::Real=T(1.01)) where T<:AbstractFloat
    A = Matrix{T}(A)
    b = Vector{T}(b)
    m, n_energy = size(A)

    x = Vector{T}(x0)
    length(x) == n_energy ||
        throw(DimensionMismatch("x0 length must match the number of energy bins"))

    # Ensure nonnegativity of the starting point if required
    nonnegativity && (x = max.(x, T(0)))

    # FISTA variables
    y = copy(x)
    t = T(1)

    # Lipschitz constant L = ||A||^2 (spectral norm via exact SVD below
    # 100 bins, else a seeded power iteration)
    if n_energy < 100
        L = opnorm(A, 2)^2
    else
        rng = Random.MersenneTwister(20240607)
        v = randn(rng, n_energy)
        v = v ./ norm(v)
        for _ in 1:20
            u = A * v
            v_new = A' * u
            v = v_new ./ norm(v_new)
        end
        L = norm(A * v)^2 / norm(v)^2
    end
    L = max(L, T(1e-10))
    step_size = T(1) / L

    # TV difference matrix
    local D::Union{Nothing,Matrix{T}}
    if tv_penalty > 0
        D = zeros(T, n_energy - 1, n_energy)
        for i in 1:(n_energy-1)
            D[i, i] = -T(1)
            D[i, i+1] = T(1)
        end
    else
        D = nothing
    end

    # Discrepancy principle threshold
    discrepancy_threshold = noise_level !== nothing ?
        T(eta) * T(noise_level) * norm(b) : nothing

    converged = false
    iters = 0

    for k in 1:max_iterations
        iters = k
        x_old = copy(x)

        # Gradient step: y - (1/L) * Aᵀ (A y - b)
        residual = A * y .- b
        gradient = A' * residual

        # Add Tikhonov regularization gradient if needed
        regularization > 0 && (gradient .+= T(regularization) .* y)

        # Add TV regularization gradient if needed
        if D !== nothing
            gradient .+= T(tv_penalty) .* (D' * (D * y))
        end

        x_temp = y .- step_size .* gradient

        # Proximal operator (L1 soft thresholding)
        l1_penalty > 0 &&
            (x_temp = _soft_threshold(x_temp, step_size * T(l1_penalty)))

        # Apply constraints
        nonnegativity && (x_temp = max.(x_temp, T(0)))
        x_max < T(Inf) && (x_temp = clamp.(x_temp, T(x_min), T(x_max)))

        x = x_temp

        # FISTA acceleration step
        t_new = (T(1) + sqrt(T(1) + T(4) * t * t)) / T(2)
        y = x .+ ((t - T(1)) / t_new) .* (x .- x_old)
        t = t_new

        # Convergence monitoring
        current_residual = norm(A * x .- b)
        rel_change = norm(x .- x_old) / max(norm(x_old), T(1e-10))
        rel_change < T(tolerance) && (converged = true; break)

        # Discrepancy principle stopping
        if discrepancy_threshold !== nothing &&
           current_residual <= discrepancy_threshold
            converged = true
            break
        end
    end

    # Ensure nonnegativity of the final result
    spectrum = max.(x, T(0))
    residual = b .- A * spectrum
    return UnfoldResult(spectrum, iters, converged, norm(residual))
end
