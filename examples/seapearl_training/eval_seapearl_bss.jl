# eval_seapearl_bss.jl — benchmark a trained BSS value-selection heuristic
# against the BasicHeuristic baseline.
#
# Loads the network parameters saved by train_seapearl_bss.jl
# (seapearl_bss_agent.ser) and runs two benchmarks:
#   1. held-out random instances from the training distribution;
#   2. the materialized IAEA reference instance
#      (examples/data/seapearl_iaea_instance.json, produced by
#      materialize_iaea_instance.jl on Julia 1.10).
#
# Run in the same environment as the training script — prefer the
# single-session Julia 1.10 env from setup_seapearl_env.jl:
#   julia --project=examples/seapearl_training eval_seapearl_bss.jl \
#       [--episodes 100]   # only used to label the report
#
# Updates seapearl_bss_training_metrics.json in place (adds the "eval_*"
# sections).

using SeaPearl
using Flux
using Random
using Statistics
using Serialization
using JSON
using Printf

include(joinpath(@__DIR__, "bss_generator.jl"))
include(joinpath(@__DIR__, "agent_builder.jl"))

const CFG = Dict(
    "n_spheres"     => 6,
    "n_bins"        => 15,
    "n_levels"      => 8,
    "k_sigma"       => 3.0,
    "noise_level"   => 0.08,
    "eval_instances"=> 6,
)
function cli_get(key::String, default::String)
    for i in 1:length(ARGS)-1
        ARGS[i] == "--$key" && return ARGS[i+1]
    end
    return default
end
const OUT_DIR   = cli_get("out", joinpath(@__DIR__, "..", "data"))
const N_LABEL    = cli_get("episodes", "100")
const IAEA_JSON  = cli_get("iaea", joinpath(OUT_DIR, "seapearl_iaea_instance.json"))
const AGENT_PATH = cli_get("agent", joinpath(OUT_DIR, "seapearl_bss_agent.ser"))
const N_LEVELS   = Int(CFG["n_levels"])

# ─── Load trained parameters into a fresh agent ─────────────────────────────
learned_heuristic = build_bss_learned_heuristic(N_LEVELS)
weights = deserialize(AGENT_PATH)
load_bss_agent_params!(learned_heuristic, weights)   # params + inference mode
println("Loaded parameters from $AGENT_PATH")

# ─── Baseline: same value semantics as solve_seapearl's enumeration ────────
basic_heuristic = SeaPearl.BasicHeuristic(
    (x; cpmodel=nothing) -> minimum(x.domain))

function run_episode(value_selection, generator; rng)
    model = SeaPearl.CPModel(SeaPearl.Trailer())
    SeaPearl.fill_with_generator!(model, generator; rng=rng)
    dt = @elapsed SeaPearl.search!(model, SeaPearl.DFSearch(),
                                   SeaPearl.MinDomainVariableSelection{true}(),
                                   value_selection)
    return Dict(
        "nodes"     => model.statistics.numberOfNodesBeforeRestart,
        "solutions" => model.statistics.numberOfSolutions,
        "time_s"    => dt,
    )
end

function benchmark(value_selection, generator; n_eval::Int, seed::Int)
    runs = [run_episode(value_selection, generator; rng=MersenneTwister(seed + k))
            for k in 1:n_eval]
    return Dict(
        "nodes_mean"     => mean(r["nodes"] for r in runs),
        "nodes_median"   => median(r["nodes"] for r in runs),
        "solutions_mean" => mean(r["solutions"] for r in runs),
        "time_s_mean"    => mean(r["time_s"] for r in runs),
        "runs"           => runs,
    )
end

# ─── 1. Held-out random instances ────────────────────────────────────────────
println("\nBenchmark on held-out random instances ...")
eval_gen = BSSUnfoldingGenerator(Int(CFG["n_spheres"]), Int(CFG["n_bins"]),
                                 N_LEVELS;
                                 k_sigma = Float64(CFG["k_sigma"]),
                                 noise_level = Float64(CFG["noise_level"]),
                                 max_solutions = 32, time_limit_s = 60)
ev_basic   = benchmark(basic_heuristic, eval_gen;
                       n_eval=Int(CFG["eval_instances"]), seed=10_000)
ev_learned = benchmark(learned_heuristic, eval_gen;
                       n_eval=Int(CFG["eval_instances"]), seed=10_000)
@printf("  Basic  : %8.1f nodes (median %6.1f), %5.2f solutions, %6.3f s\n",
        ev_basic["nodes_mean"], ev_basic["nodes_median"],
        ev_basic["solutions_mean"], ev_basic["time_s_mean"])
@printf("  Learned: %8.1f nodes (median %6.1f), %5.2f solutions, %6.3f s\n",
        ev_learned["nodes_mean"], ev_learned["nodes_median"],
        ev_learned["solutions_mean"], ev_learned["time_s_mean"])

# ─── 2. Materialized IAEA reference instance ────────────────────────────────
ev_iaea_basic = ev_iaea_learned = nothing
if isfile(IAEA_JSON)
    println("\nBenchmark on the IAEA reference instance ($IAEA_JSON) ...")
    inst = JSON.parsefile(IAEA_JSON)
    A = permutedims(reduce(hcat, [Float64.(r) for r in inst["A"]]))
    b = Float64.(inst["b"])
    sigma = Float64.(inst["sigma"])
    mv = haskey(inst, "mv") ? Float64(inst["mv"]) : flux_ceiling(A, b, nothing)
    iaea_gen = FixedInstanceGenerator(A, b, sigma, mv, N_LEVELS,
                                      Float64(CFG["k_sigma"]), 2, 1e8,
                                      32, 120)
    ev_iaea_basic   = run_episode(basic_heuristic, iaea_gen;
                                  rng=MersenneTwister(1))
    ev_iaea_learned = run_episode(learned_heuristic, iaea_gen;
                                  rng=MersenneTwister(1))
    @printf("  Basic  : %6d nodes, %3d solutions, %.3f s\n",
            ev_iaea_basic["nodes"], ev_iaea_basic["solutions"],
            ev_iaea_basic["time_s"])
    @printf("  Learned: %6d nodes, %3d solutions, %.3f s\n",
            ev_iaea_learned["nodes"], ev_iaea_learned["solutions"],
            ev_iaea_learned["time_s"])
else
    println("\n(IAEA instance JSON not found at $IAEA_JSON — skipping. " *
            "Generate it with materialize_iaea_instance.jl on Julia 1.10.)")
end

# ─── Update the metrics report in place ─────────────────────────────────────
metrics_path = joinpath(OUT_DIR, "seapearl_bss_training_metrics.json")
report = isfile(metrics_path) ? JSON.parsefile(metrics_path) : Dict("config" => CFG)
report["trained_episodes_label"] = N_LABEL
report["eval_random_instances"] = Dict("basic" => ev_basic, "learned" => ev_learned)
report["eval_iaea_instance"] = Dict("basic" => ev_iaea_basic,
                                    "learned" => ev_iaea_learned)
open(metrics_path, "w") do io
    JSON.print(io, report, 2)
end
println("\nUpdated $metrics_path")
