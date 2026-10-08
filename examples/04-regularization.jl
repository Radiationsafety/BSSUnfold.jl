### A Pluto.jl notebook ###
# v1.0.3

using Markdown
using InteractiveUtils

# ╔═╡ b3100000-0001-4000-8000-000000000001
begin
    using Pkg
    Pkg.activate(Base.current_project() !== nothing ? Base.current_project() : "..")
    using BSSUnfold
    using LinearAlgebra
    using Random
    using Statistics
    using Plots
    using Printf
    gr()
end

# ╔═╡ b3100000-0002-4000-8000-000000000002
md"""
# Tikhonov regularization and parameter selection

**Tikhonov regularization** is the classic method for regularizing ill-posed problems:

$$\min_x \|Ax - b\|^2 + \lambda \|Lx\|^2$$

where $\lambda$ is the regularization parameter and $L$ is an operator (usually an identity matrix
or a finite-difference operator for smoothness).

**Problem:** how to choose $\lambda$?

BSSUnfold.jl supports 3 selection methods:
1. **L-curve** — balance between ||Ax-b|| and ||Lx||
2. **GCV** — Generalized Cross-Validation
3. **Discrepancy principle** — if the noise level is known
"""

# ╔═╡ b3100000-0003-4000-8000-000000000003
begin
    Random.seed!(42)
    n = 100
    m = 14
    E_MeV = 10 .^ range(-9, log10(20), length=n)
    detector_names = ["sphere_$(d)in" for d in (0, 2, 3, 5, 8, 10, 12, 15, 18, 20, 22, 25, 28, 30)]
    sensitivities = Dict{String,Vector{Float64}}()
    for name in detector_names
        d = parse(Float64, replace(name, "sphere_" => "", "in" => ""))
        rf = 0.5 * exp.(-((log10.(E_MeV) .- log10(max(d * 0.3, 1e-9))) .* 2) .^ 2)
        rf .+= 0.1 ./ (E_MeV .+ 1e-9)
        sensitivities[name] = rf
    end
    x_true = exp.(-E_MeV ./ 1.5); x_true ./= sum(x_true)
    rng = MersenneTwister(42)
    A = Matrix(hcat([sensitivities[name] for name in detector_names]...)')
    noise_level = 0.01
    b = A * x_true .+ noise_level .* randn(rng, m)
    x0 = ones(n) .* 0.5
    println("Problem ready: noise level $(noise_level*100)%")
end

# ╔═╡ b3100000-0004-4000-8000-000000000004
md"""
## 1. Comparison of λ selection methods
"""

# ╔═╡ b3100000-0005-4000-8000-000000000005
begin
    res_lcurve = select_regularization_parameter(A, b, x0, method=:lcurve)
    res_gcv    = select_regularization_parameter(A, b, x0, method=:gcv)
    res_disc   = select_regularization_parameter(A, b, x0, method=:discrepancy,
                                                noise_level=noise_level)

    @printf("Method               | λ (selected)\n")
    @printf("---------------------|---------------\n")
    @printf("L-curve              | %.6e\n", res_lcurve.lambda)
    @printf("GCV                  | %.6e\n", res_gcv.lambda)
    @printf("Discrepancy (σ=%.3f) | %.6e\n", noise_level, res_disc.lambda)
end

# ╔═╡ b3100000-0006-4000-8000-000000000006
md"""
## 2. L-curve visualization

The L-curve is a log-log plot of the residual vs the regularizer norm.
The optimal λ is the point of maximum curvature.
"""

# ╔═╡ b3100000-0008-4000-8000-000000000008
md"""
## 3. Comparison of unfolded spectra

Tikhonov with different λ selection methods:
"""

# ╔═╡ b3100000-000a-4000-8000-00000000000a
md"""
## 4. Comparison with TSVD

TSVD (Truncated SVD) is a different regularization: we discard small singular values.
"""

# ╔═╡ b3100000-000c-4000-8000-00000000000c
md"""
## 5. Summary

- Tikhonov with **GCV** usually gives good results without knowledge of the noise level
- **Discrepancy principle** works best if the noise is known from measurements
- **L-curve** is robust but may give biased estimates at very low noise
- TSVD with an appropriate `k` is a simple alternative to Tikhonov
"""

# ╔═╡ b3100000-0007-4000-8000-000000000007
begin
    λ_range = 10.0 .^ range(-6, 2, length=30)
    residuals = Float64[]
    reg_norms = Float64[]
    for λ in λ_range
        res = solve_tikhonov(A, b, x0, regularization=λ)
        push!(residuals, res.residual_norm)
        push!(reg_norms, norm(res.spectrum))
    end

    p = plot(residuals, reg_norms, xscale=:log10, yscale=:log10,
             lw=2, color=:darkblue, label="L-curve",
             xlabel="||Ax - b|| (residual norm)",
             ylabel="||x|| (solution norm)",
             title="L-curve: λ selection",
             legend=:topright, size=(600, 400))

    # Mark the selected λ
    res_lcurve_selected = solve_tikhonov(A, b, x0, regularization=res_lcurve.lambda)
    scatter!(p, [res_lcurve_selected.residual_norm],
             [norm(res_lcurve_selected.spectrum)],
             markersize=8, color=:red, label="L-curve: λ=$(round(res_lcurve.lambda, digits=4))")

    res_gcv_selected = solve_tikhonov(A, b, x0, regularization=res_gcv.lambda)
    scatter!(p, [res_gcv_selected.residual_norm],
             [norm(res_gcv_selected.spectrum)],
             markersize=8, color=:green, label="GCV: λ=$(round(res_gcv.lambda, digits=4))")

    res_disc_selected = solve_tikhonov(A, b, x0, regularization=res_disc.lambda)
    scatter!(p, [res_disc_selected.residual_norm],
             [norm(res_disc_selected.spectrum)],
             markersize=8, color=:orange, label="Discrepancy: λ=$(round(res_disc.lambda, digits=4))")
end

# ╔═╡ b3100000-000b-4000-8000-00000000000b
begin
    p = plot(E_MeV, x_true, xscale=:log10, yscale=:log10,
             lw=3, color=:black, label="Truth",
             xlabel="Energy, MeV", ylabel="Φ(E)",
             title="Comparison of regularization methods",
             legend=:topright, size=(700, 400))

    # Tikhonov with GCV
    res_tik = solve_tikhonov(A, b, x0, regularization=res_gcv.lambda)
    cos_tik = dot(res_tik.spectrum, x_true) / (norm(res_tik.spectrum) * norm(x_true) + 1f-30)
    plot!(p, E_MeV, res_tik.spectrum, lw=1.5, color=:red,
          label="Tikhonov (cos=$(round(cos_tik, digits=3)))")

    # TSVD with different truncation ranks
    for (k, color) in [(3, :blue), (5, :green), (8, :orange), (10, :purple)]
        res_tsvd = solve_tsvd(A, b, x0, truncation_rank=k)
        cos_tsvd = dot(res_tsvd.spectrum, x_true) / (norm(res_tsvd.spectrum) * norm(x_true) + 1f-30)
        plot!(p, E_MeV, res_tsvd.spectrum, lw=1.2, ls=:dash, color=color,
              label="TSVD k=$k (cos=$(round(cos_tsvd, digits=3)))")
    end
    p
end

# ╔═╡ b3100000-0009-4000-8000-000000000009
begin
    p = plot(E_MeV, x_true, xscale=:log10, yscale=:log10,
             lw=3, color=:black, label="Truth",
             xlabel="Energy, MeV", ylabel="Φ(E)",
             title="Tikhonov: different λ selection methods",
             legend=:topright, size=(700, 400))

    for (label, λ, color) in [("L-curve", res_lcurve.lambda, :red),
                              ("GCV", res_gcv.lambda, :green),
                              ("Discrepancy", res_disc.lambda, :orange)]
        res = solve_tikhonov(A, b, x0, regularization=λ)
        cos = dot(res.spectrum, x_true) / (norm(res.spectrum) * norm(x_true) + 1f-30)
        plot!(p, E_MeV, res.spectrum, lw=1.5, color=color,
              label="$label (λ=$(round(λ, digits=4)), cos=$(round(cos, digits=3)))")
    end
    p
end

# ╔═╡ Cell order:
# ╟─b3100000-0001-4000-8000-000000000001
# ╟─b3100000-0002-4000-8000-000000000002
# ╟─b3100000-0003-4000-8000-000000000003
# ╟─b3100000-0004-4000-8000-000000000004
# ╟─b3100000-0005-4000-8000-000000000005
# ╟─b3100000-0006-4000-8000-000000000006
# ╟─b3100000-0007-4000-8000-000000000007
# ╟─b3100000-0008-4000-8000-000000000008
# ╟─b3100000-0009-4000-8000-000000000009
# ╟─b3100000-000a-4000-8000-00000000000a
# ╟─b3100000-000b-4000-8000-00000000000b
# ╟─b3100000-000c-4000-8000-00000000000c
