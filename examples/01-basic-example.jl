### A Pluto.jl notebook ###
# v1.0.3

using Markdown
using InteractiveUtils

# ╔═╡ 005c4e30-3c8a-4a90-b3df-7e15e08a32e0
begin
    using Pkg
    Pkg.activate(Base.current_project() !== nothing ? Base.current_project() : "..")
    using BSSUnfold
    using LinearAlgebra
    using Random
    using Statistics
    using Plots
    gr()
end

# ╔═╡ c1000000-0001-4000-8000-000000000001
md"""
# BSSUnfold.jl — Basic unfolding example

This notebook demonstrates the classic BSSUnfold workflow:

1. Build a synthetic **response matrix** (simulating a Bonner sphere spectrometer)
2. Generate a "true" neutron spectrum
3. Produce synthetic detector readings
4. Unfold the spectrum with the **GRAVEL** algorithm
5. Compare the unfolded spectrum with the truth

> **Installation:** `using Pkg; Pkg.add("BSSUnfold")`
"""

# ╔═╡ c1000000-0002-4000-8000-000000000002
md"""
## 1. Data preparation

A Bonner sphere spectrometer (BSS) consists of 14 spheres of different diameters (0", 2", 3", 5",
8", 10", 12", etc.). Each sphere has its own **response function** —
sensitivity to neutrons of different energies.

The energy grid typically covers the range from thermal neutrons (10⁻⁹ MeV)
to fast ones (20 MeV), with a logarithmic spacing — usually 100–640 bins.
"""

# ╔═╡ c1000000-0003-4000-8000-000000000003
begin
    Random.seed!(42)

    # Configuration: 14 spheres, 100 energy bins
    m = 14  # number of spheres
    n = 100 # number of bins

    # Energy grid: logarithmic from 1e-9 to 20 MeV
    E_MeV = 10 .^ range(-9, log10(20), length=n)

    # Detector names
    detector_names = ["sphere_$(d)in" for d in (0, 2, 3, 4.2, 5, 6, 7, 8, 9, 10, 11, 12, 15, 18)]

    # Imitation of response functions: peak shifts with sphere diameter
    sensitivities = Dict{String,Vector{Float64}}()
    for (i, name) in enumerate(detector_names)
        d = parse(Float64, replace(name, "sphere_" => "", "in" => ""))
        # Thermal component + peak at ~d MeV
        rf = 0.5 * exp.(-((log10.(E_MeV) .- log10(max(d * 0.3, 1e-9))) .* 2) .^ 2)
        rf .+= 0.1 ./ (E_MeV .+ 1e-9)  # 1/E component (thermal)
        sensitivities[name] = rf
    end

    # ICRP-116 dose coefficients (simplified)
    cc_icrp116 = interpolate_coefficients(get_coefficients("ICRP116"), E_MeV)

    println("Done: $(length(detector_names)) spheres, $n energy bins")
    println("Energy range: $(round(E_MeV[1], digits=2)) – $(round(E_MeV[end], digits=2)) MeV")
end

# ╔═╡ c1000000-0004-4000-8000-000000000004
begin
    # True spectrum: typical fission spectrum + thermal component
    x_true = exp.(-E_MeV ./ 1.5) .* (1.0 .+ 0.3 .* sin.(E_MeV .* 2))
    x_true ./= sum(x_true)  # normalize

    # Synthetic readings: b = A * x_true + noise
    rng = MersenneTwister(42)
    A = Matrix(hcat([sensitivities[name] for name in detector_names]...)')
    b_true = A * x_true
    b_noisy = b_true .+ 0.01 .* randn(rng, m)

    readings = Dict(name => b_noisy[i] for (i, name) in enumerate(detector_names))

    # Visualization
    p1 = plot(E_MeV, x_true, xscale=:log10, yscale=:log10,
              label="True spectrum", lw=2, color=:darkblue,
              xlabel="Energy, MeV", ylabel="Φ(E), a.u.",
              title="True neutron spectrum",
              legend=:topright)
    p2 = bar(detector_names, b_noisy,
             label="Detector readings",
             color=:darkred, alpha=0.7,
             xlabel="Detector", ylabel="Count",
             title="Synthetic readings",
             xrotation=45, legend=false)
    plot(p1, p2, layout=(1, 2), size=(900, 400))
end

# ╔═╡ c1000000-0005-4000-8000-000000000005
md"""
## 2. Unfolding with GRAVEL

We create a `Detector` object and call `unfold_gravel`.

GRAVEL is an iterative algorithm based on weighted log-likelihood:

$$x_{k+1}[j] = x_k[j] \cdot \exp\left(\frac{\sum_i W_{ij} \ln(b_i / (Ax_k)_i)}{\sum_i W_{ij}}\right)$$

where $W_{ij} = b_i \cdot A_{ij} \cdot x_k[j] / (Ax_k)_i$.
"""

# ╔═╡ c1000000-0006-4000-8000-000000000006
begin
    detector = Detector(detector_names, E_MeV, sensitivities, cc_icrp116)

    result = unfold_gravel(detector, readings,
                         max_iterations=500,
                         tolerance=1e-8)

    println("Method:      $(result["method"])")
    println("Iterations:  $(result["iterations"])")
    println("Converged:   $(result["converged"])")
    println("||b - Ax||: $(round(result["residual_norm"], digits=6))")
end

# ╔═╡ c1000000-0007-4000-8000-000000000007
begin
    # Compare the unfolded spectrum with the truth
    cos_sim = dot(result["spectrum"], x_true) /
              (norm(result["spectrum"]) * norm(x_true) + 1e-30)
    rel_err = norm(result["spectrum"] .- x_true) / (norm(x_true) + 1e-30)

    println("Cosine similarity: $(round(cos_sim, digits=4))")
    println("Relative error: $(round(rel_err, digits=4))")

    plot(E_MeV, x_true, xscale=:log10, yscale=:log10,
         label="Truth", lw=3, color=:darkblue,
         xlabel="Energy, MeV", ylabel="Φ(E)",
         title="Spectrum comparison",
         legend=:topright, size=(700, 400))
    plot!(E_MeV, result["spectrum"], lw=2, color=:red,
          label="GRAVEL unfolded")
end

# ╔═╡ c1000000-0008-4000-8000-000000000008
md"""
## 3. Dose quantities

BSSUnfold automatically computes **dose conversion coefficients** per ICRP-116:
effective dose and operational quantities (H*(10), H_p(10), etc.).
"""

# ╔═╡ c1000000-0009-4000-8000-000000000009
begin
    if haskey(result, "doserates")
        println("Dose coefficients (per sphere, as a demonstration):")
        for (name, dr) in sort(collect(result["doserates"]), by=x->x[1])[1:5]
            println("  $name: $(round(dr, digits=6))")
        end
    end
end

# ╔═╡ c1000000-000a-4000-8000-00000000000a
md"""
## 4. Summary

In this notebook we:

- ✅ Built a synthetic BSS problem (14 spheres × 100 energy bins)
- ✅ Ran GRAVEL unfolding with a single `unfold_gravel` call
- ✅ Obtained the spectrum, dose coefficients and quality metrics
- ✅ Visually compared the unfolded spectrum with the truth

### What next?

- Notebook **03-uncertainty** — Monte-Carlo uncertainty estimation
- Notebook **05-mlem_example** — MLEM algorithm
- Notebook **33-methods_comparison** — comparison of all 15 algorithms
"""

# ╔═╡ Cell order:
# ╟─005c4e30-3c8a-4a90-b3df-7e15e08a32e0
# ╟─c1000000-0001-4000-8000-000000000001
# ╟─c1000000-0002-4000-8000-000000000002
# ╟─c1000000-0003-4000-8000-000000000003
# ╟─c1000000-0004-4000-8000-000000000004
# ╟─c1000000-0005-4000-8000-000000000005
# ╟─c1000000-0006-4000-8000-000000000006
# ╟─c1000000-0007-4000-8000-000000000007
# ╟─c1000000-0008-4000-8000-000000000008
# ╟─c1000000-0009-4000-8000-000000000009
# ╟─c1000000-000a-4000-8000-00000000000a
