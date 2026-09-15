"""
    solve_amaxed_regularization(A, b, x0; sigma_factor=0.1, tau=1.0,
                                max_iterations=5000, tolerance=1e-8, line_search_tol=1e-6)

AMAXED-Regularization (Wong 2024): совместная минимизация хи-квадрата и
дивергенции Кульбака-Лейблера `tau * D_KL(phi || phi0)` без фиксированного
target chi-squared. Метод Ньютона с Armijo backtracking line search.
`line_search_tol` принят для совместимости с Python-интерфейсом.

Возвращает `UnfoldResult(spectrum, iterations, converged, residual_norm)`.
"""
function solve_amaxed_regularization(A::AbstractMatrix{T}, b::AbstractVector{T}, x0::AbstractVector{T};
                                     sigma_factor::Real=0.1,
                                     tau::Real=1.0,
                                     max_iterations::Integer=5000,
                                     tolerance::Real=1e-8,
                                     line_search_tol::Real=1e-6) where T<:AbstractFloat
    m, n = size(A)

    phi_floor = T(1e-12)

    phi_0 = max.(T.(x0), T(1e-300))
    phi_0_sum = sum(phi_0)
    phi_0_norm = phi_0 ./ phi_0_sum

    b_work = T.(b) ./ phi_0_sum
    b_safe = max.(b_work, T(1e-300))
    sigma = T(sigma_factor) .* b_safe
    sv = one(T) ./ sigma .^ 2
    AtSbA = A' * (sv .* A)

    function objective(phi_sol)
        p = max.(phi_sol, phi_floor)
        residual = A * p .- b_work
        kl = sum(p .* (log.(p ./ phi_0_norm .+ T(1e-300))) .- p .+ phi_0_norm)
        return tau * kl + dot(residual, sv .* residual)
    end

    function gradient(phi_sol)
        p = max.(phi_sol, phi_floor)
        residual = A * p .- b_work
        kl_grad = log.(p ./ phi_0_norm .+ T(1e-300))
        chi2_grad = T(2) .* (A' * (sv .* residual))
        return tau .* kl_grad .+ chi2_grad
    end

    function hessian(phi_sol)
        p = max.(phi_sol, phi_floor)
        kl_hess = Diagonal(one(T) ./ (p .+ T(1e-300)))
        chi2_hess = T(2) .* AtSbA
        return tau .* kl_hess .+ chi2_hess
    end

    phi_sol = copy(phi_0_norm)
    grad_norm = T(Inf)
    iters = 0

    for k in 1:max_iterations
        iters = k
        grad = gradient(phi_sol)

        grad_norm = norm(grad)
        if grad_norm < tolerance
            break
        end

        H = hessian(phi_sol)

        delta_phi = try
            H \ (.-grad)
        catch
            reg_param = T(1e-6) * maximum(abs.(diag(H)))
            (H + reg_param * I) \ (.-grad)
        end

        base_obj = objective(phi_sol)
        slope = dot(grad, delta_phi)
        beta = one(T)
        accepted = false
        for _ in 1:30
            new_phi = max.(phi_sol .+ beta .* delta_phi, phi_floor)
            if objective(new_phi) <= base_obj + T(1e-4) * beta * slope
                accepted = true
                break
            end
            beta /= 2
        end
        if !accepted
            beta = T(0.01)
        end

        phi_sol = max.(phi_sol .+ beta .* delta_phi, phi_floor)
    end

    x = phi_sol .* phi_0_sum
    residual = b .- A * x
    extra = Dict{String,Any}("sigma_factor" => sigma_factor, "tau" => tau)
    return UnfoldResult(x, iters, grad_norm < tolerance, norm(residual), extra)
end
