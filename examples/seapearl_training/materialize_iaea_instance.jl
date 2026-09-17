# materialize_iaea_instance.jl — export the IAEA reference unfolding problem
# used by examples/45-seapearl.jl as a self-contained JSON file, so that the
# CP+RL training pipeline (examples/seapearl_training/train_seapearl_bss.jl)
# can evaluate the trained heuristic on the SAME problem even when it runs in
# a separate Julia 1.8–1.9 side environment that cannot load BSSUnfold.jl.
# (With the single-session 1.10 environment from setup_seapearl_env.jl this
# bridge is optional — the pipeline loads BSSUnfold directly.)
#
# Run from the repository root on Julia ≥ 1.10:
#   julia --project=. examples/seapearl_training/materialize_iaea_instance.jl
#
# Output: examples/data/seapearl_iaea_instance.json

using BSSUnfold
using Random
using Statistics
using JSON
using Printf

const SEED = 2024
const N_COARSE = 15          # coarse energy bins for the CP problem
const NOISE = 0.02           # relative reading noise (deterministic via SEED)

# ─── 1. Detector + IAEA reference spectrum ───────────────────────────────────
detector = Detector()                      # real GSF response functions + ICRP-116
csv_path = joinpath(@__DIR__, "..", "data",
    "MonteCarlo_Calculated_spectra_from_IAEA_Comp_for_comparison.csv")
ref_names, E_ref, ref_spectra = load_spectra_csv(csv_path)
benchmark = "ISO_ref_AmBe"
benchmark in ref_names || error("benchmark $benchmark not found in $ref_names")
x_true = Float64.(ref_spectra[benchmark])

# ─── 2. Synthetic effective readings of the reference spectrum ──────────────
readings = get_effective_readings_for_spectra(detector, E_ref, x_true)

# ─── 3. Build the unfolding system (identical to run_unfolding) ─────────────
A_fine, b_fine, selected = build_system(
    Dict(k => Float64(v) for (k, v) in readings),
    detector_names(detector),
    detector.config.sensitivities)
E_fine = energy_grid(detector)
n_fine = length(E_fine)
@info "Fine system" size_A=size(A_fine) selected

# ─── 4. Coarsen to the CP-friendly grid (fluence-preserving) ────────────────
A_coarse = coarsen_columns(A_fine, N_COARSE)

edges = floor.(Int, collect(range(0, n_fine, length=N_COARSE + 1)))
x_coarse = [mean(x_true[(edges[k]+1):edges[k+1]]) for k in 1:N_COARSE]
E_coarse = [E_fine[edges[k]+1] for k in 1:N_COARSE]

# Grid-consistent fold: quantize the coarse reference spectrum onto the very
# grid `solve_seapearl` will build from this data (ceiling = 2 × max of the
# clipped minimum-norm LS solution), enforce the adjacent-bin smoothness
# band, and fold EXACTLY on the quantized spectrum. This makes the instance
# provably feasible for the CP encoding (up to the noise term) so that a
# benchmark of value-selection heuristics on the fixed model is meaningful.
const SMOOTH_BINS = 2                      # = max(1, n_levels ÷ 3) at n_levels=8
b_prov = A_coarse * x_coarse               # provisional clean fold
mv = 2.0 * maximum(max.(A_coarse \ b_prov, 0.0))   # solver's ceiling estimator
delta = mv / 7.0                           # n_levels = 8
x_q = clamp.(round.(x_coarse ./ delta) .* delta, 0.0, mv)
for _ in 1:2
    for k in 1:N_COARSE-1
        x_q[k+1] = clamp(x_q[k+1], x_q[k] - SMOOTH_BINS * delta,
                         x_q[k] + SMOOTH_BINS * delta)
    end
    for k in N_COARSE-1:-1:1
        x_q[k] = clamp(x_q[k], x_q[k+1] - SMOOTH_BINS * delta,
                       x_q[k+1] + SMOOTH_BINS * delta)
    end
end

b_clean = A_coarse * x_q
rng = MersenneTwister(SEED)
b = b_clean .* (1.0 .+ NOISE .* randn(rng, length(b_clean)))
sigma = NOISE .* abs.(b)

# ─── 5. Export ───────────────────────────────────────────────────────────────
inst = Dict(
    "meta" => Dict(
        "benchmark" => benchmark,
        "source" => "IAEA Compendium reference spectra (examples/data CSV)",
        "detector" => "GSF response functions, spheres: $(join(selected, ", "))",
        "n_fine_bins" => n_fine,
        "n_coarse_bins" => N_COARSE,
        "noise_level" => NOISE,
        "seed" => SEED,
        "note" => "b = A_coarse * x_q * (1 + noise); x_q = reference " *
                  "spectrum projected onto the solve_seapearl quantization " *
                  "grid (n_levels=8, smooth_bins=2, ceiling mv)",
    ),
    "E_grid" => E_coarse,
    "A" => [Vector{Float64}(A_coarse[i, :]) for i in 1:size(A_coarse, 1)],
    "b" => b,
    "sigma" => sigma,
    "mv" => mv,
    "x_true_coarse" => x_q,
    "sphere_names" => selected,
)
out_path = joinpath(@__DIR__, "..", "data", "seapearl_iaea_instance.json")
open(out_path, "w") do io
    JSON.print(io, inst, 2)
end
@info "Written" path=out_path
@printf("Fluence check: coarse ∫φ dE = %.4e, reference (uniform-log) = %.4e\n",
        sum(x_coarse), sum(x_true))
