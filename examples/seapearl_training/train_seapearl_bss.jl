# train_seapearl_bss.jl — CP+RL training of a SeaPearl value-selection
# heuristic for Bonner-sphere spectrum unfolding.
#
# Methodology follows corail-research/learning-generic-csp ("instance
# distribution → generic CSP encoding → GNN value-selection agent"),
# implemented natively on SeaPearl.jl so the trained `LearnedHeuristic`
# plugs directly into `BSSUnfold.solve_seapearl(learned_heuristic=...)`
# (see examples/45-seapearl.jl).
#
# ─── Environment requirements ────────────────────────────────────────────────────
# SeaPearl 0.4.x declares `julia = "1.8 - 1.9"` upstream, while BSSUnfold.jl
# requires 1.10. Preferred: the single-session Julia 1.10 environment built by
# `seapearl_training/setup_seapearl_env.jl` (SeaPearl compat fork + BSSUnfold):
#
#   julia examples/seapearl_training/setup_seapearl_env.jl
#   julia --project=examples/seapearl_training train_seapearl_bss.jl \
#       [--episodes 40] [--timeout 900] [--out ../data]
#
# Alternative (Julia 1.8–1.9 side environment with the registry SeaPearl):
#
#   juliaup add 1.9.4                        # once
#   julia-1.9 --project=<training-env> -e 'using Pkg;
#       Pkg.add(name="SeaPearl", version="0.4.5");
#       Pkg.add(["Flux", "JSON"])'   # Flux/JSON: needed to build the agent
#   julia-1.9 --project=<env-with-SeaPearl> train_seapearl_bss.jl \
#       [--episodes 40] [--timeout 900] [--out ../data]
#
# Outputs (written to `--out`, default: ../data):
#   seapearl_bss_agent.ser   — serialized network parameters of the RL agent
#                              (load in examples/45-seapearl.jl via agent_builder.jl)
#   seapearl_bss_training_metrics.json — training/evaluation metrics
#
# ⚠ The parameter artifact is a plain serialization of the network weights
# (no SeaPearl/agent objects inside): the shipped file (trained on 1.9 /
# SeaPearl 0.4.5 / Flux 0.12) loads and runs identically on Julia 1.10.
# Regenerate it with this script if your SeaPearl/Flux versions differ.

using SeaPearl
using Flux
using Random
using Statistics
using Serialization
using JSON
using Printf

include(joinpath(@__DIR__, "bss_generator.jl"))
include(joinpath(@__DIR__, "agent_builder.jl"))

# ─── Configuration ───────────────────────────────────────────────────────────
const CFG = Dict(
    "n_spheres"     => 6,      # Bonner spheres per instance (6 → nontrivial
                               #   search trees; 10 over-constrains the CSP)
    "n_bins"        => 15,     # coarse energy bins per instance
    "n_levels"      => 8,      # flux-quantization levels (= CP domain size)
    "k_sigma"       => 3.0,    # compatibility window in σ units
    "noise_level"   => 0.08,   # relative measurement noise of instances
    "episodes"      => 100,
    "eval_instances"=> 6,
    "seed"          => 42,
)

# Shared featurization flags — MUST match the flags used when loading the
# agent in examples/45-seapearl.jl (they are part of the state
# representation the network was trained on).
const CHOSEN_FEATURES = Dict{String,Bool}(
    "constraint_activity"          => true,
    "variable_initial_domain_size" => true,
    "variable_domain_size"         => true,
    "variable_is_bound"            => true,
    "variable_is_branchable"       => true,
    "node_number_of_neighbors"     => true,
)
# Default featurization has 3 node-type one-hot features + the 6 above:
const NUM_IN_FEATURES = 3 + length(CHOSEN_FEATURES)

# Parse simple CLI overrides
function parse_args()
    opts = Dict{String,Any}()
    args = ARGS
    i = 1
    while i <= length(args)
        if startswith(args[i], "--") && i < length(args)
            opts[args[i][3:end]] = args[i+1]; i += 2
        else
            i += 1
        end
    end
    return opts
end
const OPTS = parse_args()
const N_EPISODES  = parse(Int, get(OPTS, "episodes", string(CFG["episodes"])))
const TIMEOUT_S   = parse(Int, get(OPTS, "timeout", "900"))
const OUT_DIR     = get(OPTS, "out", joinpath(@__DIR__, "..", "data"))
const IAEA_JSON   = get(OPTS, "iaea", joinpath(OUT_DIR, "seapearl_iaea_instance.json"))

# ─── Agent construction (shared builder: agent_builder.jl) ──────────────
n_levels = Int(CFG["n_levels"])

learned_heuristic = build_bss_learned_heuristic(n_levels)

println("Agent built. Trainable parameters: ",
        sum(length, Flux.params(learned_heuristic.agent.policy.learner.approximator.model)))

# ─── Training ────────────────────────────────────────────────────────────────
gen = BSSUnfoldingGenerator(Int(CFG["n_spheres"]), Int(CFG["n_bins"]),
                            n_levels;
                            k_sigma = Float64(CFG["k_sigma"]),
                            noise_level = Float64(CFG["noise_level"]),
                            max_solutions = 32, time_limit_s = 20)

println("Training for $N_EPISODES episodes (timeout: $(TIMEOUT_S)s) ...")

# NOTE on chunking: SeaPearl 0.4.x has a race between the DQN replay-buffer
# batch sampling and the search backtracking, which can throw a BoundsError
# mid-episode (`CircularVectorBuffer ... BoundsError`). We therefore train in
# *chunks*: each chunk runs inside a try/catch; on a crash the weights
# learned so far are kept, the replay buffer is rebuilt fresh, and training
# continues with the next chunk.
const CHUNK_EPISODES = 25
t_train = 0.0
episodes_done = 0
last_metrics = nothing
weights = Flux.params(learned_heuristic.agent.policy.learner.approximator.model)

function run_training!()
    global t_train, episodes_done, last_metrics, learned_heuristic, weights
    while episodes_done < N_EPISODES
        n_here = min(CHUNK_EPISODES, N_EPISODES - episodes_done)
        learned_heuristic = build_bss_learned_heuristic(n_levels)   # fresh buffer
        Flux.loadparams!(learned_heuristic.agent.policy.learner.approximator.model, weights)
        chunk_metrics = nothing
        ok = true
        t0 = @elapsed try
            chunk_metrics, _ = SeaPearl.train!(
                valueSelectionArray = learned_heuristic,
                generator = gen,
                nbEpisodes = Int(n_here),
                strategy = SeaPearl.DFSearch(),
                variableHeuristic = SeaPearl.MinDomainVariableSelection{true}(),
                out_solver = false,
                verbose = true,
                evaluator = nothing,
                training_timeout = TIMEOUT_S,
                rngTraining = MersenneTwister(CFG["seed"] + episodes_done),
            )
        catch err
            ok = false
            @warn "Training chunk crashed (known SeaPearl 0.4.x DQN/backtracking " *
                  "buffer race); keeping the weights learned so far." err
        end
        t_train += t0
        episodes_done += n_here
        weights = Flux.params(learned_heuristic.agent.policy.learner.approximator.model)
        last_metrics = chunk_metrics === nothing ? last_metrics : chunk_metrics[1]
        @printf("  ... %d/%d episodes done (chunk %s, %.1f s)\n",
                episodes_done, N_EPISODES, ok ? "ok" : "crashed", t0)
    end
end
run_training!()
metrics = (last_metrics === nothing ? nothing : (last_metrics,))
@printf("Training done in %.1f s\n", t_train)
# --- Persist the training artifact IMMEDIATELY (before any evaluation) ----
isdir(OUT_DIR) || mkpath(OUT_DIR)
agent_path = joinpath(OUT_DIR, "seapearl_bss_agent.ser")
serialize(agent_path,
          Flux.params(learned_heuristic.agent.policy.learner.approximator.model))
@printf("\nNetwork parameters saved to %s (%.1f KiB)\n", agent_path,
        stat(agent_path).size / 1024)

train_summary = if metrics[1] === nothing
    Dict("episodes_recorded" => 0, "note" => "all chunks crashed")
else
    m = metrics[1]
    Dict(
        "episodes_recorded" => length(m.nodeVisited),
        "node_visited_last" => isempty(m.nodeVisited) ? nothing : last(m.nodeVisited),
        "total_reward_last" => isempty(m.totalReward) ? nothing : last(m.totalReward),
        "time_needed_last"  => isempty(m.timeNeeded)  ? nothing : last(m.timeNeeded),
        "loss_last"         => isempty(m.loss)        ? nothing : last(m.loss),
    )
end
report = Dict(
    "config" => merge(CFG, Dict("episodes" => N_EPISODES,
                                "num_in_features" => BSS_NUM_IN_FEATURES)),
    "chosen_features" => BSS_CHOSEN_FEATURES,
    "train_time_s" => t_train,
    "train" => train_summary,
)
metrics_path = joinpath(OUT_DIR, "seapearl_bss_training_metrics.json")
open(metrics_path, "w") do io
    JSON.print(io, report, 2)
end
println("Training metrics saved to $metrics_path")
println("\nNext step: run eval_seapearl_bss.jl (separate process) to benchmark")
println("the learned heuristic against the BasicHeuristic.")
