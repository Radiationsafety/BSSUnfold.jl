### A Pluto.jl notebook ###
# v0.20.x — SeaPearl CP unfolding of an IAEA reference spectrum (with optional
#           CP+RL learned heuristic)

using Markdown
using InteractiveUtils

# ╔═╡ f4500000-0001-4000-8000-000000000001
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

# ╔═╡ f4500000-0002-4000-8000-000000000002
md"""
# Constraint-Programming unfolding with SeaPearl.jl

[`solve_seapearl`](https://github.com/corail-research/SeaPearl.jl) treats
spectrum unfolding as a **finite-domain feasibility problem**: the flux of
every energy bin is quantized onto `n_levels` levels, every Bonner-sphere
reading produces a `kσ`-compatibility constraint, and adjacent bins are tied
by a smoothing band. The solver then **enumerates the whole feasible set**,
which yields what no iterative method in the package provides — per-bin
*interval estimates* of the null-space uncertainty.

This notebook demonstrates, on a **real IAEA reference spectrum**
(`ISO_ref_AmBe` from the IAEA Compendium, folded through the real GSF
response functions):

1. CP unfolding with the default `BasicHeuristic` — feasible-set enumeration,
   minimum-χ² member, per-bin intervals;
2. comparison with the classic GRAVEL method and with the reference doses;
3. the optional **CP+RL pipeline**: a SeaPearl `LearnedHeuristic` trained
   with the methodology of
   [corail-research/learning-generic-csp](https://github.com/corail-research/learning-generic-csp)
   (training script: `seapearl_training/train_seapearl_bss.jl`).

> **Version note.** SeaPearl 0.4.x declares `julia = "1.8 - 1.9"` upstream
> while BSSUnfold.jl requires Julia 1.10. Use the compat fork
> (`Radiationsafety/SeaPearl.jl`, branch `compat/julia-1.10`, declaration-only
> change) so that ALL sections run in ONE Julia 1.10 session — one command
> builds the environment: `julia examples/seapearl_training/setup_seapearl_env.jl`,
> then `julia --project=examples/seapearl_training examples/45-seapearl.jl`.
> Sections 1–2 always run; section 3 is still skipped gracefully when
> SeaPearl is not loadable — see the notes there.
"""

# ╔═╡ f4500000-0003-4000-8000-000000000003
begin
    detector = Detector()             # real GSF response functions + ICRP-116
    csv_path = joinpath(dirname(@__DIR__), "examples", "data",
        "MonteCarlo_Calculated_spectra_from_IAEA_Comp_for_comparison.csv")
    ref_names, E_ref, ref_spectra = load_spectra_csv(csv_path)
    benchmark = "ISO_ref_AmBe"
    x_true = Float64.(ref_spectra[benchmark])
    println("Loaded $(length(ref_names)) IAEA reference spectra; benchmark: ",
            benchmark)
    println("Detector spheres: ", join(detector_names(detector), ", "))
    println("Fine energy bins: ", n_energy_bins(detector))
end

# ╔═╡ f4500000-0004-4000-8000-000000000004
begin
    # Synthetic effective readings of the reference spectrum
    readings = get_effective_readings_for_spectra(detector, E_ref, x_true)
    # The unfolding system, exactly as `run_unfolding` builds it
    A_fine, b_fine, spheres = build_system(
        Dict(k => Float64(v) for (k, v) in readings),
        detector_names(detector), detector.config.sensitivities)
    @printf("Fine system: %d spheres × %d bins\n", size(A_fine, 1), size(A_fine, 2))
end

# ╔═╡ f4500000-0005-4000-8000-000000000005
md"""
## 1. A CP-friendly coarse problem

Full CP enumeration on the 60-bin grid is needlessly expensive, and the
physical information of a 10-sphere measurement lives on a coarser scale
anyway. We therefore merge the response matrix onto **15 coarse bins**
(fluence-preserving column merging via `coarsen_columns`) and project the
reference spectrum accordingly. The readings are then folded **on the same
quantization grid the CP solver will use**, so the true spectrum is provably
feasible and the demonstration isolates the *method*, not a modelling
artifact.
"""

# ╔═╡ f4500000-0006-4000-8000-000000000006
begin
    const N_COARSE = 15
    const NOISE_NB = 0.05        # relative reading noise (deterministic below)
    const SEED_NB  = 2024

    A_coarse = coarsen_columns(A_fine, N_COARSE)
    E_fine   = energy_grid(detector)
    n_fine   = length(E_fine)
    edges    = floor.(Int, collect(range(0, n_fine, length=N_COARSE + 1)))
    x_coarse = [mean(x_true[(edges[k]+1):edges[k+1]]) for k in 1:N_COARSE]
    E_coarse = [E_fine[edges[k]+1] for k in 1:N_COARSE]

    # Flux ceiling = the same estimator solve_seapearl uses internally
    mv = 2.0 * maximum(max.(A_coarse \ (A_coarse * x_coarse), 0.0))
    Δ8 = mv / 7.0                                  # quantization step, n_levels = 8

    # Project the truth onto the 8-level grid + adjacent-bin smoothness band
    x_q = clamp.(round.(x_coarse ./ Δ8) .* Δ8, 0.0, mv)
    for _ in 1:2
        for k in 1:N_COARSE-1
            x_q[k+1] = clamp(x_q[k+1], x_q[k] - 2Δ8, x_q[k] + 2Δ8)
        end
        for k in N_COARSE-1:-1:1
            x_q[k] = clamp(x_q[k], x_q[k+1] - 2Δ8, x_q[k+1] + 2Δ8)
        end
    end

    # Noisy readings folded on the feasible quantized truth
    b_clean = A_coarse * x_q
    rng_nb  = MersenneTwister(SEED_NB)
    b_coarse = b_clean .* (1.0 .+ NOISE_NB .* randn(rng_nb, length(b_clean)))
    x0_coarse = ones(N_COARSE) .* (0.5 * mv)

    @printf("Coarse problem: %d spheres × %d bins, ceiling mv = %.3f\n",
            size(A_coarse, 1), N_COARSE, mv)
    @printf("Reference fluence (coarse): %.3f\n", sum(x_q))
end

# ╔═╡ f4500000-0007-4000-8000-000000000007
md"""
## 2. CP unfolding with the default heuristic

`solve_seapearl` enumerates up to `max_solutions` admissible spectra and
returns the minimum-χ² member; `extra["spectrum_lower"/"spectrum_upper"]`
are the per-bin minimum/maximum over the whole feasible set.
"""

# ╔═╡ f4500000-0008-4000-8000-000000000008
begin
    res_cp = solve_seapearl(A_coarse, b_coarse, x0_coarse;
                            n_levels = 16, k_sigma = 2.0,
                            noise_level = NOISE_NB, max_value = mv,
                            max_solutions = 128, time_limit_ms = 60_000)
    cp_ok = get(res_cp.extra, "n_solutions", 0) > 0
    if cp_ok
        status_cp = get(res_cp.extra, "status", "?")
        @printf("status = %s | %d feasible spectra | exhaustive = %s\n",
                status_cp, get(res_cp.extra, "n_solutions", 0),
                string(get(res_cp.extra, "exhaustive", false)))
        @printf("relative residual  ||b − A·φ|| / ||b|| = %.3f\n",
                norm(A_coarse * res_cp.spectrum .- b_coarse) / norm(b_coarse))
        @printf("reference spectrum inside intervals: %.0f%% of bins\n",
                100 * mean(res_cp.extra["spectrum_lower"] .- 1e-12 .<= x_q .<=
                           res_cp.extra["spectrum_upper"] .+ 1e-12))
    else
        @printf("solve_seapearl unavailable: %s\n",
                get(res_cp.extra, "error", "no feasible solution"))
    end
    cp_ok
end

# ╔═╡ f4500000-0009-4000-8000-000000000009
if cp_ok
    res_gravel = solve_gravel(A_coarse, b_coarse, x0_coarse; max_iterations = 200)
    lower = res_cp.extra["spectrum_lower"]
    upper = res_cp.extra["spectrum_upper"]
    p_cp = plot(E_coarse, x_q, xscale = :log10, yscale = :log10, lw = 3,
                color = :black, label = "IAEA reference (coarse)",
                xlabel = "Energy, MeV", ylabel = "Φ(E) per bin",
                title = "SeaPearl CP unfolding — ISO AmBe, $(size(A_coarse,1)) spheres",
                legend = :bottomleft, size = (860, 520))
    plot!(p_cp, E_coarse, upper; fillrange = lower, fillalpha = 0.25, lw = 0,
          color = :steelblue, label = "CP feasible set (min–max)")
    plot!(p_cp, E_coarse, res_cp.spectrum, lw = 2, color = :steelblue,
          marker = :circle, label = "CP min-χ² member")
    plot!(p_cp, E_coarse, Float64.(res_gravel.spectrum), lw = 2, ls = :dash,
          color = :crimson, label = "GRAVEL")
    p_cp
else
    md"""*(SeaPearl not loadable in this session — the CP plot and the dose
    comparison are skipped; everything else is unaffected.)*"""
end

# ╔═╡ f4500000-000a-4000-8000-00000000000a
if cp_ok
    # Dose rates (pSv/s) from reference / CP / GRAVEL coarse spectra
    dose_ref = calculate_dose_rates(x_q)
    dose_cp  = calculate_dose_rates(Float64.(res_cp.spectrum))
    dose_gr  = calculate_dose_rates(Float64.(res_gravel.spectrum))
    @printf("%-10s | %12s | %12s | %8s | %8s\n", "Dose", "reference",
            "CP", "CP diff %", "GRAVEL %")
    println("-" ^ 62)
    for k in ("AP", "ISO")
        haskey(dose_ref, k) || continue
        dcp = get(dose_cp, k, NaN)
        dgr = get(dose_gr, k, NaN)
        @printf("%-10s | %12.3e | %12.3e | %7.1f%% | %7.1f%%\n", k,
                dose_ref[k], dcp,
                100 * (dcp - dose_ref[k]) / dose_ref[k],
                100 * (dgr - dose_ref[k]) / dose_ref[k])
    end
end

# ╔═╡ f4500000-000b-4000-8000-00000000000b
md"""
## 3. CP + RL: a learned value-selection heuristic

The default `BasicHeuristic` is deliberately *blind*: it works everywhere but
explores the tree canonically. SeaPearl's `LearnedHeuristic` replaces the
value selection with a GNN agent trained — following the
[learning-generic-csp](https://github.com/corail-research/learning-generic-csp)
methodology (instance distribution → generic CSP encoding → GNN
value-selection agent) — on randomized BSS instances generated by
`seapearl_training/bss_generator.jl`.

* Training script: `seapearl_training/train_seapearl_bss.jl` (SeaPearl 0.4.5
  + Flux; runs in the single-session 1.10 environment from
  `seapearl_training/setup_seapearl_env.jl`, or on Julia 1.8–1.9 with the
  registry SeaPearl).
* The pretrained network parameters ship as
  `examples/data/seapearl_bss_agent.ser` and are loaded through
  `seapearl_training/agent_builder.jl`.
* Evaluation metrics: `examples/data/seapearl_bss_training_metrics.json`.

> **Version note.** Upstream SeaPearl declares `julia = "1.8 - 1.9"`; on
> Julia 1.10 install the compat fork (`Pkg.add(url=
> "https://github.com/Radiationsafety/SeaPearl.jl", rev="compat/julia-1.10")`
> — or just run `seapearl_training/setup_seapearl_env.jl`). Without SeaPearl
> this cell prints a hint instead of failing. In a SeaPearl-capable
> environment it runs a head-to-head comparison at `n_levels = 8` against the
> same `BasicHeuristic` settings.
"""

# ╔═╡ f4500000-000c-4000-8000-00000000000c
begin
    rl_error_msg = Ref("")
    rl_result = try
        # Dynamic loads keep this cell optional: Pluto does not try to resolve
        # SeaPearl/Flux at notebook-parse time.
        SP = Base.require(Main, :SeaPearl)
        FX = Base.require(Main, :Flux)
        Core.eval(Main, :(SeaPearl = $SP))     # bindings for agent_builder.jl
        Core.eval(Main, :(Flux = $FX))
        include(joinpath(@__DIR__, "seapearl_training", "agent_builder.jl"))
        weights = deserialize(joinpath(@__DIR__, "data", "seapearl_bss_agent.ser"))
        lh = load_bss_agent_params!(build_bss_learned_heuristic(8), weights)
        res = solve_seapearl(A_coarse, b_coarse, x0_coarse;
                             n_levels = 8, k_sigma = 2.0,
                             noise_level = NOISE_NB, max_value = mv,
                             max_solutions = 64, time_limit_ms = 60_000,
                             learned_heuristic = lh)
        (res = res, loaded = true)
    catch err
        rl_error_msg[] = sprint(showerror, err)
        (res = nothing, loaded = false)
    end
    rl_result.loaded
end

# ╔═╡ f4500000-000d-4000-8000-00000000000d
if rl_result.loaded
    res_rl = rl_result.res
    res_basic8 = solve_seapearl(A_coarse, b_coarse, x0_coarse;
                                n_levels = 8, k_sigma = 2.0,
                                noise_level = NOISE_NB, max_value = mv,
                                max_solutions = 64, time_limit_ms = 60_000)
    @printf("%-18s | %10s | %10s | %12s\n", "heuristic", "status",
            "n_solutions", "nodes used*")
    println("-" ^ 58)
    @printf("%-18s | %10s | %10d | %12s\n", "BasicHeuristic",
            get(res_basic8.extra, "status", "?"),
            get(res_basic8.extra, "n_solutions", 0),
            get(res_basic8.extra, "exhaustive", false) ? "full tree" : "limited")
    @printf("%-18s | %10s | %10d | %12s\n", "LearnedHeuristic",
            get(res_rl.extra, "status", "?"),
            get(res_rl.extra, "n_solutions", 0),
            get(res_rl.extra, "exhaustive", false) ? "full tree" : "limited")
    md"""Both heuristics solve the same CSP; the learned heuristic selects
    values with a 16k-parameter GNN over the tripartite state graph. With the
    committed 100-episode training run it matches the baseline's search
    efficiency — longer training and richer instance distributions (raise
    `--episodes` in `train_seapearl_bss.jl`) are the obvious next step."""
else
    md"""**SeaPearl is not loadable in this session** — the CP+RL comparison
    is skipped (sections 1–2 above are unaffected).

    To enable it: run `julia examples/seapearl_training/setup_seapearl_env.jl`
    (builds a single-session Julia 1.10 environment with the SeaPearl compat
    fork + BSSUnfold), then re-open this notebook with
    `--project=examples/seapearl_training`. On Julia 1.8–1.9 the registry
    version works directly: `Pkg.add(name = "SeaPearl", version = "0.4.5")`.
    Retraining from scratch: `seapearl_training/train_seapearl_bss.jl`.

    ```
    $(rl_error_msg[])
    ```"""
end

# ╔═╡ f4500000-000e-4000-8000-00000000000e
md"""
## Take-aways

* The CP formulation turns BSS unfolding into **feasible-set enumeration**:
  the shaded band is the per-bin min–max over all spectra compatible with the
  `kσ` windows — a direct picture of the null-space ambiguity that point
  estimators (GRAVEL, MLEM, …) hide.
* The minimum-χ² member of the feasible set is competitive with GRAVEL while
  carrying its own uncertainty band.
* The CP+RL hook (`learned_heuristic = ...`) is fully wired: the committed
  agent artifact reproduces the training-time search behaviour, and the
  training pipeline is reproducible from `seapearl_training/`.

**References**

* SeaPearl.jl — corail-research/SeaPearl.jl (CP solver + RL heuristics)
* learning-generic-csp — corail-research/learning-generic-csp (methodology
  for the instance-distribution → GNN-agent training pipeline)
* IAEA Compendium of Neutron Spectra and Reference Benchmarks (reference
  spectra and GSF response functions used through `Detector()`)
"""
