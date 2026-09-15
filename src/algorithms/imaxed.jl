"""
    solve_imaxed(A, b, x0; sigma_factor=0.1, max_iterations=5000,
                 tolerance=1e-8, line_search_tol=1e-6)

IMAXED (Improved MAXED, Wong 2024): метод Ньютона в phi-пространстве с
Armijo backtracking line search. Минимизируется
`f(phi) = 0.5*(A phi - b)ᵀ S_b (A phi - b) + Σ phi_i*log(phi_i/phi0_i) - phi_i + phi0_i`,
где `S_b = diag(1/sigma²)`, `sigma = sigma_factor * max(b, eps)`.
`line_search_tol` — константа Armijo `c1` (обрезается в (0, 1)).

Возвращает `UnfoldResult(spectrum, iterations, converged, residual_norm)`.
"""
function solve_imaxed(A::AbstractMatrix{T}, b::AbstractVector{T}, x0::AbstractVector{T};
                      sigma_factor::Real=0.1,
                      max_iterations::Integer=5000,
                      tolerance::Real=1e-8,
                      line_search_tol::Real=1e-6) where T<:AbstractFloat
    m, n = size(A)

    phi_floor = T(1e-12)

    b_arr = T.(b)
    b_safe = max.(b_arr, T(1e-300))
    sigma = T(sigma_factor) .* b_safe
    sv = one(T) ./ sigma .^ 2

    phi_0 = max.(T.(x0), T(1e-300))
    length(phi_0) == n || throw(DimensionMismatch("x0 length $(length(phi_0)) != n $n"))
    log_phi_0 = log.(phi_0)

    At_Sb_A = A' * (sv .* A)

    objective(phi) = begin
        p = max.(phi, phi_floor)
        residual = A * p .- b_arr
        chi2 = T(0.5) * dot(residual, sv .* residual)
        kl = sum(p .* (log.(p .+ T(1e-300)) .- log_phi_0) .- p .+ phi_0)
        return chi2 + kl
    end

    gradient(phi) = begin
        p = max.(phi, phi_floor)
        residual = A * p .- b_arr
        return A' * (sv .* residual) .+ log.(p .+ T(1e-300)) .- log_phi_0
    end

    hessian(phi) = At_Sb_A + Diagonal(one(T) ./ (max.(phi, phi_floor) .+ T(1e-300)))

    phi = copy(phi_0)
    c1 = clamp(T(line_search_tol), T(1e-12), T(0.5))

    grad_norm = T(Inf)
    iters = 0

    for k in 1:max_iterations
        iters = k
        grad = gradient(phi)
        grad_norm = norm(grad)
        if grad_norm < tolerance
            break
        end

        H = hessian(phi)
        delta = try
            H \ (.-grad)
        catch
            reg = T(1e-6) * maximum(abs.(diag(H)))
            reg = reg == 0 ? T(1e-12) : reg
            (H + reg * I) \ (.-grad)
        end

        slope = dot(grad, delta)
        if slope >= 0
            delta = .-grad
            slope = dot(grad, delta)
        end

        base = objective(phi)
        beta = one(T)
        accepted = false
        for _ in 1:30
            trial = max.(phi .+ beta .* delta, phi_floor)
            if objective(trial) <= base + c1 * beta * slope
                accepted = true
                break
            end
            beta /= 2
        end
        if !accepted
            beta = T(0.01)
        end

        phi = max.(phi .+ beta .* delta, phi_floor)
    end

    if iters == max_iterations && grad_norm >= tolerance
        grad_norm = norm(gradient(phi))
    end

    residual = b .- A * phi
    extra = Dict{String,Any}("sigma_factor" => sigma_factor)
    return UnfoldResult(phi, iters, grad_norm < tolerance, norm(residual), extra)
end
