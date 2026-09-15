"""
    solve_amaxed(A, b, x0; sigma_factor=0.1, target_chi2=nothing,
                 max_iterations=5000, tolerance=1e-8, line_search_tol=1e-6)

AMAXED (Alternative MAXED, Wong 2024) with an inverted cross-entropy definition.

Minimizes the Kullback-Leibler divergence under a chi-squared constraint
via a Lagrange multiplier `mu` and Newton's method with backtracking line search
on the norm of the KKT residual. `target_chi2 = nothing` means automatic selection
(equal to the number of measurements `m`). `line_search_tol` is accepted for
compatibility with the Python interface (it is not used in the scheme itself).

Returns `UnfoldResult(spectrum, iterations, converged, residual_norm)`.
"""
function solve_amaxed(A::AbstractMatrix{T}, b::AbstractVector{T}, x0::AbstractVector{T};
                      sigma_factor::Real=0.1,
                      target_chi2::Union{Real,Nothing}=nothing,
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
    ASbA = sv .* A
    AtSbA = A' * ASbA

    Omega = target_chi2 === nothing ? T(m) : T(target_chi2)

    phi_sol = copy(phi_0_norm)
    mu = one(T)

    function lagrangian_gradients(phi_sol, mu)
        phi_safe = max.(phi_sol, phi_floor)
        phi_sum = sum(phi_safe)

        a = ASbA' * b_work
        residual = A * phi_safe .- b_work
        b_term = AtSbA' * phi_safe

        grad_phi = ones(T, n) ./ phi_sum .- phi_0_norm ./ phi_safe .+
                   T(2) * mu .* (b_term .- a)
        grad_mu = dot(residual, sv .* residual) .- Omega
        return grad_phi, grad_mu
    end

    function hessian(phi_sol, mu)
        phi_safe = max.(phi_sol, phi_floor)
        phi_sum = sum(phi_safe)

        a = ASbA' * b_work
        b_term = AtSbA' * phi_safe

        H_phi_phi = -(ones(T, n, n) ./ phi_sum^2) .+
                    Diagonal(phi_0_norm ./ phi_safe .^ 2) .+
                    T(2) * mu .* AtSbA
        H_phi_mu = T(2) .* (b_term .- a)

        H = Matrix{T}(undef, n + 1, n + 1)
        H[1:n, 1:n] .= H_phi_phi
        H[1:n, n+1] .= H_phi_mu
        H[n+1, 1:n] .= H_phi_mu
        H[n+1, n+1] = zero(T)
        return H
    end

    grad_norm = T(Inf)
    iters = 0

    for k in 1:max_iterations
        iters = k
        grad_phi, grad_mu = lagrangian_gradients(phi_sol, mu)
        state_grad = vcat(grad_phi, grad_mu)

        grad_norm = norm(state_grad)
        if grad_norm < tolerance
            break
        end

        H = hessian(phi_sol, mu)

        delta_state = try
            H \ (.-state_grad)
        catch
            reg_param = T(1e-6) * maximum(abs.(diag(H)))
            (H + reg_param * I) \ (.-state_grad)
        end

        delta_phi = delta_state[1:n]
        delta_mu = delta_state[n+1]

        beta = one(T)
        accepted = false
        for _ in 1:30
            new_phi = max.(phi_sol .+ beta .* delta_phi, phi_floor)
            new_mu = mu + beta * delta_mu
            g_phi, g_mu = lagrangian_gradients(new_phi, new_mu)
            kkt_residual = sqrt(dot(g_phi, g_phi) + g_mu^2)
            if kkt_residual <= (one(T) - T(1e-4) * beta) * grad_norm
                accepted = true
                break
            end
            beta /= 2
        end
        if !accepted
            beta = T(0.01)
        end

        phi_sol = max.(phi_sol .+ beta .* delta_phi, phi_floor)
        mu = mu + beta * delta_mu
    end

    x = phi_sol .* phi_0_sum
    residual = b .- A * x
    extra = Dict{String,Any}("sigma_factor" => sigma_factor,
                             "target_chi2" => target_chi2)
    return UnfoldResult(x, iters, grad_norm < tolerance, norm(residual), extra)
end
