"""
BON95-based parametric unfolding (Sannikov, GSF 1995; Babintsev et al.,
2022; Sannikov et al., Apparatus No.1, 2009).
Faithful port of `bssunfold.core.unfold_parametric2` + `_bon95`.

The lethargy spectrum `E * Phi(E)` is a linear combination of four components:

    Thermal      (E < 0.1 MeV):  Fth  = Xth^(3/2) * exp(-Xth)
    Epithermal   (E < 10 MeV):   Fepi = E^(-b) * (1 - exp(-Xth))
    Intermediate (E < 10 MeV):   Fint = (1 - exp(-Xth))
    Fast                         Ff   = Xf^(3/2) * exp(-Xf),  Xf = (E/Tf)^c

with `Tth = 3.5e-8` MeV.  Shape parameters `(b, Tf, c)` are found by grid
search (`solve_bon95_parametric`), linear coefficients `a1..a4` by weighted
NLS; SQP variants (`solve_bon95_sqp`, used for optimizers "cvxpy",
"qpsolvers", "combined") refine the shape parameters with a box-constrained
regularized QP substep.  After the parametric fit the spectrum is refined by
multiplicative directed-divergence (I-divergence / Itakura-Saito) iterations.
"""
const BON95_TTH = 3.5e-8

const BON95_DEFAULT_B_RANGE = (0.5, 2.0, 5)
const BON95_DEFAULT_TF_RANGE = (0.5, 10.0, 5)
const BON95_DEFAULT_C_RANGE = (0.5, 3.0, 4)

function _bon95_Fth(E::Vector{Float64})
    Xth = E ./ BON95_TTH
    return Xth .^ 1.5 .* exp.(-Xth)
end

function _bon95_Fepi(E::Vector{Float64}, b::Float64)
    Xth = E ./ BON95_TTH
    return E .^ (-b) .* (1.0 .- exp.(-Xth))
end

function _bon95_Fint(E::Vector{Float64})
    Xth = E ./ BON95_TTH
    return 1.0 .- exp.(-Xth)
end

function _bon95_Ff(E::Vector{Float64}, Tf::Float64, c::Float64)
    Xf = (E ./ Tf) .^ c
    return Xf .^ 1.5 .* exp.(-Xf)
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

# numpy.linalg.lstsq(rcond=None): SVD with cutoff max(m,n)*eps*s_max
function _bon95_lstsq(M::Matrix{Float64}, y::Vector{Float64})
    F = svd(M)
    s = F.S
    cutoff = max(size(M)...) * eps(Float64) * s[1]
    c = F.U' * y
    for i in eachindex(c)
        c[i] = s[i] > cutoff ? c[i] / s[i] : 0.0
    end
    return F.V * c
end

function _bon95_clean_edge_bins(phi::Vector{Float64}; factor::Float64=10.0)
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

const bon95_clean_edge_bins = _bon95_clean_edge_bins

"""
    bon95_linear_coefficients(A, b, E, ln_steps, b_val, Tf, c_val, weights)

Port of `_solve_linear_coefficients`: basis columns
`B_ik = sum_j A_ij * F_k(E_j) / E_safe_j * ln_steps_j`, weighted least
squares via numpy-style lstsq, coefficients clamped non-negative, chi2 =
`mean(weights * residual^2)` with `residual = B*a - b`.
"""
function bon95_linear_coefficients(A::Matrix{Float64}, b::Vector{Float64}, E::Vector{Float64},
                                   ln_steps::Vector{Float64}, b_val::Float64,
                                   Tf::Float64, c_val::Float64,
                                   weights::Union{Nothing,Vector{Float64}})
    n_det = length(b)
    E_safe = [e > 0 ? e : 1.0 for e in E]

    F = hcat(_bon95_Fth(E), _bon95_Fepi(E, b_val), _bon95_Fint(E),
             _bon95_Ff(E, Tf, c_val))
    weighted_F = F ./ E_safe .* ln_steps
    B = A * weighted_F

    w = weights === nothing ? ones(n_det) : weights
    sw = sqrt.(w)
    Bw = B .* sw
    bw = b .* sw

    result = _bon95_lstsq(Bw, bw)
    a = max.(result, 0.0)

    residual = B * a .- b
    chi2 = all(w .> 0) ? sum(residual .^ 2 .* w) / n_det : sum(residual .^ 2) / n_det
    return a, chi2
end

function bon95_solve_shape(A::Matrix{Float64}, b::Vector{Float64}, E::Vector{Float64},
                           ln_steps::Vector{Float64}, b_val::Float64,
                           Tf::Float64, c_val::Float64,
                           weights::Union{Nothing,Vector{Float64}})
    a, chi2 = bon95_linear_coefficients(A, b, E, ln_steps, b_val, Tf, c_val, weights)
    phi = max.(bon95_spectrum(E, b_val, Tf, c_val, a[1], a[2], a[3], a[4]), 0.0)
    phi = _bon95_clean_edge_bins(phi)
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

Grid scan over (b, Tf, c) in the order `b` (outer), `Tf`, `c` (inner);
a1..a4 solved by weighted NLS at each grid point; candidates sorted by chi2
(stable).  `b_meas` — measured sigma (weights = 1/sigma^2).
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

    candidates = Dict{String,Float64}[]
    for b_val in b_vals
        for Tf_val in Tf_vals
            for c_val in c_vals
                a, chi2 = bon95_linear_coefficients(AF, bf, E_f, ln, b_val, Tf_val, c_val, weights)
                push!(candidates, Dict{String,Float64}(
                    "b" => b_val, "Tf" => Tf_val, "c" => c_val,
                    "a1" => a[1], "a2" => a[2], "a3" => a[3], "a4" => a[4],
                    "chi2" => chi2))
            end
        end
    end
    order = sortperm([c["chi2"] for c in candidates]; alg=MergeSort)
    best = candidates[order[1]]
    top = [candidates[order[i]] for i in 1:min(length(order), Int(top_n))]
    return best, best["chi2"], top
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

# min ||A_eff*delta + residual||^2 + alpha*||delta||^2  s.t.  lo <= delta <= hi
# (exact optimum via 3-var active-set enumeration; cvxpy/qpsolvers port)
function _bon95_box_qp(A_eff::Matrix{Float64}, residual::Vector{Float64},
                       alpha::Float64, lo::Vector{Float64}, hi::Vector{Float64})
    P = A_eff' * A_eff .+ alpha .* Matrix{Float64}(I, 3, 3)
    q = A_eff' * residual
    best = fill(0.0, 3)
    best_val = Inf
    for state in Iterators.product((0, 1, 2), (0, 1, 2), (0, 1, 2))
        d = fill(0.0, 3)
        free = Int[]
        for i in 1:3
            if state[i] == 1
                d[i] = lo[i]
            elseif state[i] == 2
                d[i] = hi[i]
            else
                push!(free, i)
            end
        end
        if isempty(free)
            delta = d
        else
            Pf = P[free, free]
            rhs = .- q[free] .- P[free, setdiff(1:3, free)] * d[setdiff(1:3, free)]
            df = Pf \ rhs
            any(x -> !isfinite(x), df) && continue
            delta = copy(d)
            delta[free] = df
            any(delta[free] .< lo[free] .- 1e-9) && continue
            any(delta[free] .> hi[free] .+ 1e-9) && continue
            delta[free] = clamp.(delta[free], lo[free], hi[free])
        end
        val = sum(abs2, A_eff * delta .+ residual) + alpha * sum(abs2, delta)
        if val < best_val
            best_val = val
            best = delta
        end
    end
    return best
end

"""
    solve_bon95_sqp(A, b, E, ln_steps; b_meas=nothing, initial_shape=nothing,
                    alpha=1e-4, max_iter=50, tol=1e-6)
      -> (spectrum, success, message, nfev)

Port of `solve_bon95_cvxpy` / `solve_bon95_qpsolvers` (same algorithm):
SQP over shape params (b, Tf, c); at each iteration a1..a4 are re-solved by
NLS, the spectrum is linearized via finite differences and the box-
constrained regularized QP substep is solved exactly.
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

    bounds = BON95_SHAPE_BOUNDS
    message = ""
    nfev = 0
    success = false
    spectrum_final = nothing
    for k in 0:(Int(max_iter) - 1)
        spectrum_k, _, _ = bon95_solve_shape(AF, bf, E_f, ln, p["b"], p["Tf"], p["c"], weights)
        nfev += 1
        residual = AF * spectrum_k .- bf
        if norm(residual) < tol
            return spectrum_k, true, "Converged in $k iterations", nfev
        end

        J, r = bon95_shape_jacobian(AF, bf, E_f, ln, p, weights)
        nfev += 2 * 3
        A_eff = AF * J

        lo = [bounds[BON95_SHAPE_NAMES[i]][1] - p[BON95_SHAPE_NAMES[i]] for i in 1:3]
        hi = [bounds[BON95_SHAPE_NAMES[i]][2] - p[BON95_SHAPE_NAMES[i]] for i in 1:3]
        delta_val = _bon95_box_qp(A_eff, r, Float64(alpha), lo, hi)

        for (i, name) in enumerate(BON95_SHAPE_NAMES)
            p[name] += delta_val[i]
        end
        bon95_shape_clamp!(p)

        if norm(delta_val) < Float64(tol)
            spectrum_final, _, _ = bon95_solve_shape(AF, bf, E_f, ln, p["b"], p["Tf"], p["c"], weights)
            return spectrum_final, true, "Converged in $(k + 1) iterations", nfev
        end
    end

    if spectrum_final === nothing
        spectrum_final, _, _ = bon95_solve_shape(AF, bf, E_f, ln, p["b"], p["Tf"], p["c"], weights)
    end
    isempty(message) && (message = "Max iterations ($max_iter) reached")
    return spectrum_final, false, message, nfev
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

    weights = bon95_measurement_weights(b_meas, length(bf))

    denom = vec(sum(AF, dims=1))
    denom = max.(denom, 1e-30)

    for iteration in 1:Int(max_iter)
        M_p = AF * (phi .* ln)
        M_p_safe = max.(M_p, 1e-30)

        residual = M_p .- bf
        chi2 = sum(residual .^ 2 .* weights) / length(bf)

        if chi2 < Float64(tol_chi2)
            return phi, iteration, chi2, true
        end

        ratios = bf ./ M_p_safe
        numerator = AF' * ratios
        phi_new = phi .* numerator ./ denom
        phi_new = max.(phi_new, 1e-30)

        rel_change = maximum(abs.(phi_new .- phi)) / (maximum(phi) + 1e-30)
        phi = phi_new

        if rel_change < Float64(tol_rel)
            phi = _bon95_clean_edge_bins(phi)
            M_p_final = AF * (phi .* ln)
            chi2_final = sum((M_p_final .- bf) .^ 2 .* weights) / length(bf)
            return phi, iteration, chi2_final, true
        end
    end

    phi = _bon95_clean_edge_bins(phi)
    M_p_final = AF * (phi .* ln)
    chi2_final = sum((M_p_final .- bf) .^ 2 .* weights) / length(bf)
    converged = chi2_final < Float64(tol_chi2)
    return phi, Int(max_iter), chi2_final, converged
end

"""
    solve_parametric2(A, b, x0=nothing; E=nothing, ln_steps=nothing,
                      b_meas=nothing, optimizer="grid", b_range, Tf_range,
                      c_range, alpha=1e-4, max_iter_qp=50, tol_qp=1e-6,
                      max_iter=200, tol_chi2=1.0) -> UnfoldResult

Full BON95 pipeline: (1) parametric fit with the chosen optimizer
(`"grid"`, `"cvxpy"`, `"qpsolvers"`, `"combined"`), (2) directed-divergence
refinement.  Final spectrum = `phi * ln_steps`.  `x0` is kept for API
compatibility (unused, as in python).
"""
function solve_parametric2(A::AbstractMatrix, b::AbstractVector, x0::Union{Nothing,AbstractVector}=nothing;
                           E::Union{Nothing,AbstractVector}=nothing,
                           ln_steps::Union{Nothing,AbstractVector}=nothing,
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
    E_f = E === nothing ? collect(10.0 .^ range(-9.0, 2.0, length=n_energy)) : Float64.(collect(E))
    ln = ln_steps === nothing ? compute_log_steps(E_f) .* log(10) : Float64.(collect(ln_steps))
    nfev = 0
    best_chi2 = 0.0
    message = ""

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
        throw(ArgumentError("Unknown optimizer: '$optimizer'. " *
                            "Choose from 'grid', 'cvxpy', 'qpsolvers', 'combined'."))
    end

    phi_param = max.(phi_param, 0.0)
    phi_param = _bon95_clean_edge_bins(phi_param)

    phi_refined, n_iter, chi2_final, converged = directed_divergence_iteration(
        AF, bf, E_f, ln, phi_param; b_meas=b_meas, max_iter=Int(max_iter),
        tol_chi2=Float64(tol_chi2))

    phi_refined = _bon95_clean_edge_bins(phi_refined)
    spectrum = phi_refined .* ln

    residual = norm(AF * spectrum .- bf)
    if optimizer == "grid"
        message = @sprintf("BON95 grid fit (chi2=%.4f) + DD iteration (%d iters, chi2=%.4f)",
                           best_chi2, n_iter, chi2_final)
    else
        message = @sprintf("BON95 %s fit + DD iteration (%d iters, chi2=%.4f)",
                           optimizer, n_iter, chi2_final)
    end

    extra = Dict{String,Any}(
        "optimizer" => optimizer,
        "message" => message,
        "chi2_dd" => chi2_final,
        "chi2_parametric" => best_chi2,
        "Tth" => BON95_TTH,
    )
    return UnfoldResult(spectrum, nfev + n_iter, converged, residual, extra)
end
