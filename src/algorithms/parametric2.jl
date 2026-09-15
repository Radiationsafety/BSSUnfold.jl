"""
BON95-based parametric unfolding (Sannikov, GSF 1995; Babintsev et al.,
2022; Sannikov et al., Apparatus No.1, 2009).

The lethargy spectrum `E * Phi(E)` is a linear combination of four components:

    Thermal      (E < 0.1 MeV):  Fth  = Xth^(3/2) * exp(-Xth)
    Epithermal   (E < 10 MeV):   Fepi = E^(-b) * (1 - exp(-Xth))
    Intermediate (E < 10 MeV):   Fint = (1 - exp(-Xth))
    Fast                         Ff   = Xf^(3/2) * exp(-Xf),  Xf = (E/Tf)^c

with `Tth = 3.5e-8` MeV.  The free shape parameters `(b, Tf, c)` are found
by a grid scan with weighted NLS for the linear coefficients a1..a4
(`solve_bon95_parametric`); the SQP variants (cvxpy/qpsolvers in python)
are replaced by the same regularized Newton substep as in `parametric.jl`.
After the parametric fit — a refinement with multiplicative
directed-divergence (I-divergence / Itakura-Saito / Csiszar-Tusnady)
iterations (`directed_divergence_iteration`).
"""
const BON95_TTH = 3.5e-8

const BON95_DEFAULT_B_RANGE = (0.5, 2.0, 5)
const BON95_DEFAULT_TF_RANGE = (0.5, 10.0, 5)
const BON95_DEFAULT_C_RANGE = (0.5, 3.0, 4)

function _bon95_Fth(E::Vector{Float64})
    return @. (E / BON95_TTH)^1.5 * exp(-E / BON95_TTH)
end

function _bon95_Fepi(E::Vector{Float64}, b::Float64)
    return @. max(E, 1e-300)^(-b) * (1.0 - exp(-E / BON95_TTH))
end

function _bon95_Fint(E::Vector{Float64})
    return @. 1.0 - exp(-E / BON95_TTH)
end

function _bon95_Ff(E::Vector{Float64}, Tf::Float64, c::Float64)
    Xf = @. (E / Tf)^c
    return @. Xf^1.5 * exp(-Xf)
end

function bon95_lethargy(E::Vector{Float64}, b::Float64, Tf::Float64, c::Float64,
                        a1::Float64, a2::Float64, a3::Float64, a4::Float64)
    return a1 .* _bon95_Fth(E) .+ a2 .* _bon95_Fepi(E, b) .+
           a3 .* _bon95_Fint(E) .+ a4 .* _bon95_Ff(E, Tf, c)
end

function bon95_spectrum(E::Vector{Float64}, b::Float64, Tf::Float64, c::Float64,
                        a1::Float64, a2::Float64, a3::Float64, a4::Float64)
    lethargy = bon95_lethargy(E, b, Tf, c, a1, a2, a3, a4)
    return [E_j > 0 ? lethargy[j] / E_j : 0.0 for (j, E_j) in enumerate(E)]
end

function bon95_linear_coefficients(A::Matrix{Float64}, b::Vector{Float64}, E::Vector{Float64},
                                   ln_steps::Vector{Float64}, b_val::Float64,
                                   Tf::Float64, c_val::Float64,
                                   weights::Union{Nothing,Vector{Float64}})
    F = hcat(_bon95_Fth(E), _bon95_Fepi(E, b_val), _bon95_Fint(E),
             _bon95_Ff(E, Tf, c_val))
    weighted_F = F ./ reshape(max.(E, 1.0), :, 1) .* reshape(ln_steps, :, 1)
    B = A * weighted_F

    w = weights === nothing ? ones(length(b)) : weights
    sw = sqrt.(max.(w, 0.0))
    Bw = B .* sw
    bw = b .* sw

    a = max.(Bw \ bw, 0.0)
    residual = B * a .- b
    chi2 = sum(abs2, residual .* sqrt.(max.(w, 0.0))) / max(length(residual), 1)
    return a, chi2
end

function bon95_clean_edge_bins(phi::Vector{Float64}; factor::Float64=10.0)
    y = copy(phi)
    n = length(y)
    n < 3 && return y
    nm = sum(y[2:3]) / 2
    if nm > 0 && y[1] > factor * nm
        y[1] = 0.0
    end
    nm2 = sum(y[(n-2):(n-1)]) / 2
    if nm2 > 0 && y[n] > factor * nm2
        y[n] = 0.0
    end
    return y
end

function bon95_solve_shape(A::Matrix{Float64}, b::Vector{Float64}, E::Vector{Float64},
                           ln_steps::Vector{Float64}, b_val::Float64,
                           Tf::Float64, c_val::Float64,
                           weights::Union{Nothing,Vector{Float64}})
    a, chi2 = bon95_linear_coefficients(A, b, E, ln_steps, b_val, Tf, c_val, weights)
    phi = max.(bon95_spectrum(E, b_val, Tf, c_val, a[1], a[2], a[3], a[4]), 0.0)
    phi = bon95_clean_edge_bins(phi)
    spectrum = phi .* ln_steps
    return spectrum, chi2, a
end

function bon95_measurement_weights(b_meas::Union{Nothing,AbstractVector}, n::Int)
    if b_meas === nothing
        return ones(n)
    end
    return [Float64(x) > 0 ? 1.0 / Float64(x)^2 : 1.0 for x in b_meas]
end

"""
    solve_bon95_parametric(A, b, E, ln_steps; b_range=(0.5,2.0,5),
                           Tf_range=(0.5,10.0,5), c_range=(0.5,3.0,4),
                           b_meas=nothing, top_n=5)
      -> (best_params::Dict{String,Float64}, best_chi2, top_candidates)

Grid scan over (b, Tf, c) with the linear coefficients a1..a4 solved by
weighted NLS at each grid point; candidates are sorted by chi2.
`b_meas` — measured sigma (weights = 1/sigma^2); ranges are
`(min, max, n_points)`.
"""
function solve_bon95_parametric(A::AbstractMatrix, b::AbstractVector, E::AbstractVector,
                                ln_steps::AbstractVector;
                                b_range::Tuple{Real,Real,Integer}=BON95_DEFAULT_B_RANGE,
                                Tf_range::Tuple{Real,Real,Integer}=BON95_DEFAULT_TF_RANGE,
                                c_range::Tuple{Real,Real,Integer}=BON95_DEFAULT_C_RANGE,
                                b_meas::Union{Nothing,AbstractVector}=nothing,
                                top_n::Integer=5)
    AF = Matrix{Float64}(A)
    bf = Vector{Float64}(b)
    E_f = Float64.(collect(E))
    ln = Float64.(collect(ln_steps))
    weights = bon95_measurement_weights(b_meas, length(bf))

    b_vals = collect(range(Float64(b_range[1]), Float64(b_range[2]), length=Int(b_range[3])))
    Tf_vals = collect(range(Float64(Tf_range[1]), Float64(Tf_range[2]), length=Int(Tf_range[3])))
    c_vals = collect(range(Float64(c_range[1]), Float64(c_range[2]), length=Int(c_range[3])))

    params_list = Dict{String,Float64}[]
    chi_list = Float64[]
    for b_val in b_vals
        for Tf_val in Tf_vals
            for c_val in c_vals
                _, chi2, a = bon95_solve_shape(AF, bf, E_f, ln, b_val, Tf_val, c_val, weights)
                push!(params_list, Dict{String,Float64}(
                    "b" => b_val, "Tf" => Tf_val, "c" => c_val,
                    "a1" => a[1], "a2" => a[2], "a3" => a[3], "a4" => a[4]))
                push!(chi_list, chi2)
            end
        end
    end
    order = sortperm(chi_list)
    best = params_list[order[1]]
    top = params_list[order[1:min(end, Int(top_n))]]
    return best, chi_list[order[1]], top
end

const BON95_SHAPE_NAMES = ["b", "Tf", "c"]
const BON95_SHAPE_BOUNDS = Dict{String,Tuple{Float64,Float64}}(
    "b" => (0.5, 2.0), "Tf" => (0.5, 10.0), "c" => (0.5, 3.0))

function bon95_shape_clamp!(p::Dict{String,Float64})
    for (name, (lo, hi)) in BON95_SHAPE_BOUNDS
        p[name] = clamp(p[name], lo, hi)
    end
    return p
end

function bon95_pert_spectrum(A::Matrix{Float64}, b::Vector{Float64}, E::Vector{Float64},
                             ln::Vector{Float64}, p::Dict{String,Float64},
                             weights::Union{Nothing,Vector{Float64}})
    spectrum, _, _ = bon95_solve_shape(A, b, E, ln, p["b"], p["Tf"], p["c"], weights)
    return spectrum
end

function bon95_shape_jacobian(A::Matrix{Float64}, b::Vector{Float64}, E::Vector{Float64},
                              ln::Vector{Float64}, p::Dict{String,Float64},
                              weights::Union{Nothing,Vector{Float64}}, delta=1e-6)
    s0 = bon95_pert_spectrum(A, b, E, ln, p, weights)
    residual = A * s0 .- b
    J = zeros(length(E), 3)
    for (i, name) in enumerate(BON95_SHAPE_NAMES)
        lo, hi = BON95_SHAPE_BOUNDS[name]
        d = delta
        p_val = p[name]
        if p_val + d > hi
            d = max(0.0, hi - p_val) * 0.5
        end
        if d < 1e-15
            d = delta
            if p_val - d >= lo
                p_pert = copy(p)
                p_pert[name] = p_val - d
                s_pert = bon95_pert_spectrum(A, b, E, ln, p_pert, weights)
                J[:, i] .= (s0 .- s_pert) ./ d
            else
                J[:, i] .= 0.0
            end
            continue
        end
        p_pert = copy(p)
        p_pert[name] = p_val + d
        s_pert = bon95_pert_spectrum(A, b, E, ln, p_pert, weights)
        J[:, i] .= (s_pert .- s0) ./ d
    end
    return J, residual
end

"""
    solve_bon95_sqp(A, b, E, ln_steps; b_meas=nothing, initial_shape=nothing,
                    alpha=1e-4, max_iter=50, tol=1e-6)
      -> (spectrum, success, message, nfev)

SQP refinement of the BON95 shape parameters (b, Tf, c): at each iteration
a1..a4 are recomputed (NLS), then a constrained substep is solved

    min ||A_eff*delta + residual||^2 + alpha*||delta||^2,
    lo - p <= delta <= hi - p

via regularized normal equations (Newton) with clamping to the bounds
(the python port used cvxpy/qpsolvers for the same QP).
"""
function solve_bon95_sqp(A::AbstractMatrix, b::AbstractVector, E::AbstractVector,
                         ln_steps::AbstractVector;
                         b_meas=nothing,
                         initial_shape=nothing,
                         alpha::Real=1e-4,
                         max_iter::Integer=50,
                         tol::Real=1e-6)
    AF = Matrix{Float64}(A)
    bf = Vector{Float64}(b)
    E_f = Float64.(collect(E))
    ln = Float64.(collect(ln_steps))
    weights = bon95_measurement_weights(b_meas, length(bf))

    p = Dict{String,Float64}()
    if initial_shape !== nothing
        for name in BON95_SHAPE_NAMES
            p[name] = Float64(initial_shape[name])
        end
    else
        best, _, _ = solve_bon95_parametric(AF, bf, E_f, ln; b_meas=b_meas, top_n=1)
        for name in BON95_SHAPE_NAMES
            p[name] = best[name]
        end
    end
    bon95_shape_clamp!(p)

    message = ""
    nfev = 0
    success = false
    spectrum_final = nothing
    for k in 0:max_iter
        spectrum_k, _, _ = bon95_solve_shape(AF, bf, E_f, ln, p["b"], p["Tf"], p["c"], weights)
        nfev += 1
        residual = AF * spectrum_k .- bf
        if norm(residual) < tol
            success = true
            message = "Converged in $k iterations"
            spectrum_final = spectrum_k
            break
        end

        J, r = bon95_shape_jacobian(AF, bf, E_f, ln, p, weights)
        nfev += 2 * 3
        A_eff = AF * J
        n_p = 3
        P = A_eff' * A_eff .+ (max(Float64(alpha), 0.0)) .* Matrix{Float64}(I, n_p, n_p)
        q = A_eff' * r
        delta_val = -(P \ q)
        for (i, name) in enumerate(BON95_SHAPE_NAMES)
            lo, hi = BON95_SHAPE_BOUNDS[name]
            delta_val[i] = clamp(p[name] + delta_val[i], lo, hi) - p[name]
        end
        for (i, name) in enumerate(BON95_SHAPE_NAMES)
            p[name] += delta_val[i]
        end
        bon95_shape_clamp!(p)

        if norm(delta_val) < tol
            spectrum_final, _, _ = bon95_solve_shape(AF, bf, E_f, ln, p["b"], p["Tf"], p["c"], weights)
            success = true
            message = "Converged in $(k+1) iterations"
            break
        end
    end

    if spectrum_final === nothing
        spectrum_final, _, _ = bon95_solve_shape(AF, bf, E_f, ln, p["b"], p["Tf"], p["c"], weights)
    end
    isempty(message) && (message = "Max iterations ($max_iter) reached")
    return spectrum_final, success, message, nfev
end

"""
    solve_bon95_combined(A, b, E, ln_steps; b_meas=nothing, alpha=1e-4,
                         max_iter_qp=50, tol_qp=1e-6) -> (spectrum, success, message, nfev)

Grid scan (`solve_bon95_parametric`) as a starting point + SQP refinement
(`solve_bon95_sqp`).
"""
function solve_bon95_combined(A::AbstractMatrix, b::AbstractVector, E::AbstractVector,
                              ln_steps::AbstractVector;
                              b_meas=nothing, alpha::Real=1e-4,
                              max_iter_qp::Integer=50, tol_qp::Real=1e-6)
    AF = Matrix{Float64}(A)
    bf = Vector{Float64}(b)
    E_f = Float64.(collect(E))
    ln = Float64.(collect(ln_steps))
    best, _, _ = solve_bon95_parametric(AF, bf, E_f, ln; b_meas=b_meas, top_n=1)
    init = Dict{String,Float64}("b" => best["b"], "Tf" => best["Tf"], "c" => best["c"])
    spectrum, success, msg, nfev = solve_bon95_sqp(AF, bf, E_f, ln;
        b_meas=b_meas, initial_shape=init, alpha=alpha, max_iter=max_iter_qp, tol=tol_qp)
    return spectrum, success, "grid + SQP ($(msg))", 1 + nfev
end

"""
    directed_divergence_iteration(A, b, E, ln_steps, phi0; b_meas=nothing,
                                  max_iter=200, tol_chi2=1.0, tol_rel=1e-6)
      -> (spectrum, n_iter, chi2, converged)

Multiplicative refinement (Itakura-Saito / Csiszar-Tusnady):

    phi_{k+1}(E_j) = phi_k(E_j) * numerator_j / denominator_j,

where `numerator_j = sum_i A_ij * M_i / M_p_i`, `denominator_j = sum_i A_ij`,
`M_p_i = sum_j A_ij phi_j d(ln E)_j`.  Stop when chi2 < `tol_chi2`
or the relative change of the spectrum is < `tol_rel`.
"""
function directed_divergence_iteration(A::AbstractMatrix, b::AbstractVector, E::AbstractVector,
                                       ln_steps::AbstractVector, phi0::AbstractVector;
                                       b_meas::Union{Nothing,AbstractVector}=nothing,
                                       max_iter::Integer=200,
                                       tol_chi2::Real=1.0,
                                       tol_rel::Real=1e-6)
    AF = Matrix{Float64}(A)
    bf = Vector{Float64}(b)
    ln = Float64.(collect(ln_steps))
    phi = max.(Vector{Float64}(phi0), 1e-30)

    weights = b_meas === nothing ? ones(length(bf)) :
        [Float64(x) > 0 ? 1.0 / Float64(x)^2 : 1.0 for x in b_meas]

    denom = max.(vec(sum(AF, dims=1)), 1e-30)

    for iteration in 1:Int(max_iter)
        M_p = AF * (phi .* ln)
        M_p_safe = max.(M_p, 1e-30)

        residual = M_p .- bf
        chi2 = sum(abs2, residual .* sqrt.(max.(weights, 0.0))) / max(length(bf), 1)

        if chi2 < tol_chi2
            return phi, iteration, chi2, true
        end

        ratios = bf ./ M_p_safe
        numerator = AF' * ratios
        phi_new = max.(phi .* numerator ./ denom, 1e-30)

        rel_change = maximum(abs.(phi_new .- phi)) / (maximum(phi) + 1e-30)
        phi = phi_new

        if rel_change < tol_rel
            phi = bon95_clean_edge_bins(phi)
            M_p_final = AF * (phi .* ln)
            chi2_final = sum(abs2, (M_p_final .- bf) .* sqrt.(max.(weights, 0.0))) / max(length(bf), 1)
            return phi, iteration, chi2_final, true
        end
    end

    phi = bon95_clean_edge_bins(phi)
    M_p_final = AF * (phi .* ln)
    chi2_final = sum(abs2, (M_p_final .- bf) .* sqrt.(max.(weights, 0.0))) / max(length(bf), 1)
    return phi, Int(max_iter), chi2_final, chi2_final < tol_chi2
end

"""
    solve_parametric2(A, b, x0=nothing; E=nothing, b_meas=nothing,
                      optimizer="grid", b_range, Tf_range, c_range, alpha=1e-4,
                      max_iter_qp=50, tol_qp=1e-6, max_iter=200, tol_chi2=1.0)
      -> UnfoldResult

Full BON95 pipeline: (1) parametric fit with the chosen zone
optimizer (`"grid"`, `"cvxpy"`, `"qpsolvers"`, `"combined"` — all
SQP variants are solved by the in-house regularized Newton), (2)
directed-divergence refinement.  The final spectrum = `phi * ln_steps`.
`x0` — kept for API compatibility (optional).
"""
function solve_parametric2(A::AbstractMatrix, b::AbstractVector, x0::Union{Nothing,AbstractVector}=nothing;
                           E::Union{Nothing,AbstractVector}=nothing,
                           b_meas::Union{Nothing,AbstractVector}=nothing,
                           optimizer::String="grid",
                           b_range::Tuple{Real,Real,Integer}=BON95_DEFAULT_B_RANGE,
                           Tf_range::Tuple{Real,Real,Integer}=BON95_DEFAULT_TF_RANGE,
                           c_range::Tuple{Real,Real,Integer}=BON95_DEFAULT_C_RANGE,
                           alpha::Real=1e-4,
                           max_iter_qp::Integer=50,
                           tol_qp::Real=1e-6,
                           max_iter::Integer=200,
                           tol_chi2::Real=1.0)
    AF = Matrix{Float64}(A)
    bf = Vector{Float64}(b)
    n_energy = size(AF, 2)
    E_f = E === nothing ? collect(10.0 .^ range(-9, 2, length=n_energy)) : Float64.(collect(E))
    ln = compute_log_steps(E_f) .* log(10)
    nfev = 0
    best_chi2 = 0.0

    if optimizer == "grid"
        best_params, best_chi2, top = solve_bon95_parametric(AF, bf, E_f, ln;
            b_range=b_range, Tf_range=Tf_range, c_range=c_range, b_meas=b_meas, top_n=5)
        nfev = length(top)
        phi_param = bon95_spectrum(E_f, best_params["b"], best_params["Tf"], best_params["c"],
                                   best_params["a1"], best_params["a2"],
                                   best_params["a3"], best_params["a4"])
    elseif optimizer in ("cvxpy", "qpsolvers")
        spectrum_fit, _, _, nfev = solve_bon95_sqp(AF, bf, E_f, ln; b_meas=b_meas,
            alpha=Float64(alpha), max_iter=Int(max_iter_qp), tol=Float64(tol_qp))
        phi_param = spectrum_fit ./ max.(ln, 1e-30)
        best_chi2 = 0.0
    elseif optimizer == "combined"
        spectrum_fit, _, _, nfev = solve_bon95_combined(AF, bf, E_f, ln;
            b_meas=b_meas, alpha=Float64(alpha), max_iter_qp=Int(max_iter_qp), tol_qp=Float64(tol_qp))
        phi_param = spectrum_fit ./ max.(ln, 1e-30)
        best_chi2 = 0.0
    else
        throw(ArgumentError("Unknown optimizer: '$optimizer'. Choose from 'grid', 'cvxpy', 'qpsolvers', 'combined'."))
    end

    phi_param = max.(phi_param, 0.0)
    phi_param = bon95_clean_edge_bins(phi_param)

    phi_refined, n_iter, chi2_final, converged = directed_divergence_iteration(
        AF, bf, E_f, ln, phi_param; b_meas=b_meas, max_iter=Int(max_iter),
        tol_chi2=Float64(tol_chi2))
    phi_refined = bon95_clean_edge_bins(phi_refined)
    spectrum = phi_refined .* ln

    residual = norm(AF * spectrum .- bf)
    extra = Dict{String,Any}(
        "optimizer" => optimizer,
        "message" => "BON95 $optimizer fit + DD iteration ($n_iter iters, chi2=$chi2_final)",
        "chi2_dd" => chi2_final,
        "chi2_parametric" => best_chi2,
        "Tth" => BON95_TTH,
    )
    return UnfoldResult(max.(spectrum, 0.0), nfev + n_iter, converged, residual, extra)
end
