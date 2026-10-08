### A Pluto.jl notebook ###
# v0.20.x — Robustness analysis

using Markdown
using InteractiveUtils

# ╔═╡ d3100000-0001-4000-8000-000000000001
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

# ╔═╡ d3100000-0002-4000-8000-000000000002
md"""
# Robustness analysis

**Robustness** — the stability of an algorithm under variations of the input data:

1. **Noise robustness**: how does the result change with different noise levels?
2. **Initial approximation**: sensitivity to $x_0$
3. **Random seed**: difference between runs with different RNG

A good unfolding algorithm should:
- Give a stable result under 1-5% noise
- Not depend critically on the initial spectrum
- Work reproducibly across different random seeds
"""

# ╔═╡ d3100000-0003-4000-8000-000000000003
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
    A = Matrix(hcat([sensitivities[name] for name in detector_names]...)')
    b_clean = A * x_true
    x0 = ones(n) .* 0.5
    println("Problem ready: clean b, $(length(detector_names)) spheres")
end

# ╔═╡ d3100000-0004-4000-8000-000000000004
md"""
## 1. Noise robustness

Let us run the methods at different noise levels (0.1% — 10%) and measure
how stable their spectrum reconstructions are.
"""

# ╔═╡ d3100000-0005-4000-8000-000000000005
begin
    noise_levels = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
    methods_to_test = [
        ("MLEM",     () -> solve_mlem(A, b_noisy, x0, max_iterations=500, tolerance=1e-8)),
        ("GRAVEL",   () -> solve_gravel(A, b_noisy, x0, max_iterations=300, tolerance=1e-8)),
        ("OSEM",     () -> solve_osem(A, b_noisy, x0, max_iterations=30, n_subsets=4)),
        ("Tikhonov", () -> solve_tikhonov(A, b_noisy, x0, regularization=1e-3)),
    ]

    noise_results = Dict{String, Vector{Float64}}()  # method => cos_sims

    for (name, fn) in methods_to_test
        cos_sims = Float64[]
        for σ in noise_levels
            global b_noisy = b_clean .+ σ .* randn(MersenneTwister(42), m)
            res = fn()
            cos = dot(res.spectrum, x_true) / (norm(res.spectrum) * norm(x_true) + 1f-30)
            push!(cos_sims, cos)
        end
        noise_results[name] = cos_sims
    end

    @printf("%-10s | ", "Method")
    for σ in noise_levels
        @printf("%7.2f%%  ", σ*100)
    end
    println()
    println("-"^80)
    for (name, cos_sims) in noise_results
        @printf("%-10s | ", name)
        for c in cos_sims
            @printf(" %.4f   ", c)
        end
        println()
    end
end

# ╔═╡ d3100000-0006-4000-8000-000000000006
begin
    p = plot(xlabel="Noise level (%)", ylabel="Cosine similarity",
             title="Noise robustness",
             legend=:topright, size=(700, 400))
    colors = [:darkblue, :red, :green, :purple]
    for (i, (name, _)) in enumerate(methods_to_test)
        plot!(p, noise_levels .* 100, noise_results[name],
              lw=2, color=colors[i], marker=:circle, label=name)
    end
    hline!(p, [0.5], ls=:dash, color=:gray, label="cos=0.5")
    p
end

# ╔═╡ d3100000-0007-4000-8000-000000000007
md"""
## 2. Robustness to initial approximation

Let us run GRAVEL with different x₀ values and check how stable the result is.
"""

# ╔═╡ d3100000-0008-4000-8000-000000000008
begin
    x0_options = [
        ("Flat=0.5",       ones(n) .* 0.5),
        ("Flat=0.1",       ones(n) .* 0.1),
        ("Flat=1.0",       ones(n) .* 1.0),
        ("1/E",            1.0 ./ (E_MeV .+ 1e-9)),
        ("Decay",          exp.(-E_MeV ./ 5.0)),
        ("Random",         rand(MersenneTwister(99), n)),
    ]

    cos_for_x0 = Float64[]
    for (label, x0_test) in x0_options
        res = solve_gravel(A, b_clean, x0_test, max_iterations=500, tolerance=1e-10)
        cos = dot(res.spectrum, x_true) / (norm(res.spectrum) * norm(x_true) + 1f-30)
        push!(cos_for_x0, cos)
    end

    bar(first.(x0_options), cos_for_x0,
        legend=false, xrotation=30,
        ylabel="Cosine similarity",
        title="GRAVEL: robustness to x₀",
        color=:darkblue, size=(700, 400))
    hline!([0.9], ls=:dash, color=:green, label="cos=0.9 (good)")
end

# ╔═╡ d3100000-0009-4000-8000-000000000009
md"""
## 3. Statistics across different random seeds

Let us run the unfolding 10 times with different noise (σ=1%) and look at the variance.
"""

# ╔═╡ d3100000-000a-4000-8000-00000000000a
begin
    n_seeds = 20
    seed_results = Dict{String, Vector{Float64}}()

    for (name, fn) in methods_to_test
        cos_sims = Float64[]
        for seed in 1:n_seeds
            global b_noisy = b_clean .+ 0.01 .* randn(MersenneTwister(seed), m)
            try
                res = fn()
                cos = dot(res.spectrum, x_true) /
                      (norm(res.spectrum) * norm(x_true) + 1f-30)
                push!(cos_sims, cos)
            catch
                push!(cos_sims, NaN)
            end
        end
        seed_results[name] = cos_sims
    end

    # Box plot statistics
    names_methods = collect(keys(seed_results))
    stats_table = []
    for name in names_methods
        vals = filter(!isnan, seed_results[name])
        push!(stats_table, (
            name=name,
            mean=mean(vals),
            std=std(vals),
            min=minimum(vals),
            max=maximum(vals)
        ))
    end

    @printf("%-10s | %8s | %8s | %8s | %8s\n",
            "Method", "mean", "std", "min", "max")
    @printf("%s\n", "-"^60)
    for r in stats_table
        @printf("%-10s | %.4f   | %.4f  | %.4f  | %.4f\n",
                r.name, r.mean, r.std, r.min, r.max)
    end
end

# ╔═╡ d3100000-000b-4000-8000-00000000000b
begin
    # Box plot — using scatter with jitter
    p = plot(xlabel="Method", ylabel="Cosine similarity",
             title="Variance over $n_seeds runs",
             legend=false, size=(700, 400))
    for (i, name) in enumerate(names_methods)
        scatter!(p, fill(i, length(seed_results[name])),
                 seed_results[name], markersize=4, alpha=0.6, color=:darkblue)
    end
    xticks!(p, 1:length(names_methods), names_methods, xrotation=30)
    hline!(p, [0.5], ls=:dash, color=:red, label="")
    p
end

# ╔═╡ d3100000-000c-4000-8000-00000000000c
md"""
## 4. Robustness summary

- **GRAVEL** — the most robust to noise and initial approximations
- **Tikhonov** — stable, but may lose details at large λ
- **MLEM** — sensitive to x₀ with a small number of iterations
- **OSEM** — accelerates MLEM, but the variance between runs is higher

### Practical recommendations for BSS unfolding

1. Use GRAVEL with x₀ = 1/E or flat 0.5
2. Run 30–50 MC samples for uncertainty estimation
3. Compare 2-3 methods for critical applications
4. If prior information is available — use Staysl
"""

# ╔═╡ Cell order:
# ╟─d3100000-0001-4000-8000-000000000001
# ╟─d3100000-0002-4000-8000-000000000002
# ╟─d3100000-0003-4000-8000-000000000003
# ╟─d3100000-0004-4000-8000-000000000004
# ╟─d3100000-0005-4000-8000-000000000005
# ╟─d3100000-0006-4000-8000-000000000006
# ╟─d3100000-0007-4000-8000-000000000007
# ╟─d3100000-0008-4000-8000-000000000008
# ╟─d3100000-0009-4000-8000-000000000009
# ╟─d3100000-000a-4000-8000-00000000000a
# ╟─d3100000-000b-4000-8000-00000000000b
# ╟─d3100000-000c-4000-8000-00000000000c
