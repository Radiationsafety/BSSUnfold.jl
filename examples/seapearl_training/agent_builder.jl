# agent_builder.jl — SeaPearl CP+RL agent construction for BSS unfolding.
#
# Shared by:
#   * examples/seapearl_training/train_seapearl_bss.jl  (training, Julia 1.9
#     side-environment with SeaPearl 0.4.5)
#   * examples/45-seapearl.jl                           (loading the pretrained
#     agent and running the CP+RL pipeline)
#
# Requires: SeaPearl (and Flux) loadable in the calling scope. The RL stack
# is accessed as `SeaPearl.RL` (ReinforcementLearning.jl, a SeaPearl
# dependency), so only SeaPearl itself has to be `using`-able.

# Featurization flags — part of the state representation the network was
# trained on; MUST stay identical between training and inference.
const BSS_CHOSEN_FEATURES = Dict{String,Bool}(
    "constraint_activity"          => true,
    "variable_initial_domain_size" => true,
    "variable_domain_size"         => true,
    "variable_is_bound"            => true,
    "variable_is_branchable"       => true,
    "node_number_of_neighbors"     => true,
)
# Default featurization has 3 node-type one-hot features + the 6 above:
const BSS_NUM_IN_FEATURES = 3 + length(BSS_CHOSEN_FEATURES)

"""
    build_bss_cpnn(n_levels) -> SeaPearl.CPNN

The GNN value network for the BSS unfolding CSP: 2 graph-convolution layers
over the tripartite (constraint–variable–value) state graph, a per-variable
MLP, and a linear head with one score per quantization level (the action
space = domain `0..n_levels-1`).
"""
function build_bss_cpnn(n_levels::Int)
    SeaPearl.CPNN(
        graphChain = Flux.Chain(
            SeaPearl.GraphConv(BSS_NUM_IN_FEATURES => 64, Flux.leakyrelu),
            SeaPearl.GraphConv(64 => 64, Flux.leakyrelu),
        ),
        nodeChain = Flux.Chain(
            Flux.Dense(64, 64, Flux.leakyrelu),
            Flux.Dense(64, 32, Flux.leakyrelu),
        ),
        outputChain = Flux.Dense(32, n_levels),
    )
end

"""
    build_bss_agent(n_levels; rng=MersenneTwister(42)) -> RL.Agent

Canonical SeaPearl 0.4.x DQN agent (double approximator, ε-greedy explorer,
SLART replay buffer with a legal-action mask of size `n_levels`).
"""
function build_bss_agent(n_levels::Int; rng::AbstractRNG=MersenneTwister(42))
    RL = SeaPearl.RL
    RL.Agent(
        policy = RL.QBasedPolicy(
            learner = RL.DQNLearner(
                approximator = RL.NeuralNetworkApproximator(
                    model = build_bss_cpnn(n_levels),
                    optimizer = Flux.ADAM(0.0005f0),
                ),
                target_approximator = RL.NeuralNetworkApproximator(
                    model = build_bss_cpnn(n_levels),
                    optimizer = Flux.ADAM(0.0005f0),
                ),
                loss_func = Flux.Losses.huber_loss,
                stack_size = nothing,
                γ = 0.99f0,
                batch_size = 16,
                update_horizon = 4,
                min_replay_history = 16,
                update_freq = 2,
                target_update_freq = 20,
            ),
            explorer = RL.EpsilonGreedyExplorer(
                ϵ_stable = 0.05,
                kind = :exp,
                ϵ_init = 1.0,
                warmup_steps = 0,
                decay_steps = 600.0,
                step = 1,
                is_break_tie = false,
                rng = rng,
            ),
        ),
        trajectory = RL.CircularArraySLARTTrajectory(
            capacity = 300,
            state = SeaPearl.DefaultTrajectoryState[] => (),
            legal_actions_mask = Vector{Bool} => (n_levels,),
        ),
    )
end

"""
    build_bss_learned_heuristic(n_levels; agent=build_bss_agent(n_levels))

Wrap the agent into the `SimpleLearnedHeuristic` accepted by
`BSSUnfold.solve_seapearl(learned_heuristic=...)`.
"""
function build_bss_learned_heuristic(n_levels::Int;
                                     agent = build_bss_agent(n_levels))
    SeaPearl.SimpleLearnedHeuristic{
        SeaPearl.DefaultStateRepresentation{SeaPearl.DefaultFeaturization,
                                            SeaPearl.DefaultTrajectoryState},
        SeaPearl.DefaultReward, SeaPearl.FixedOutput}(
            agent; chosen_features = copy(BSS_CHOSEN_FEATURES))
end

"""
    load_bss_agent_params!(lh, weights) -> lh

Load serialized network parameters (`Flux.params(...)` saved by
`train_seapearl_bss.jl`) into the heuristic's approximator and switch the
heuristic to **inference mode**.

⚠ The testmode! call is REQUIRED before solving with a freshly built
`SimpleLearnedHeuristic`: a new heuristic defaults to `trainMode=true`,
which pushes transitions into the replay buffer and triggers DQN updates
inside the search — this both corrupts the search and can throw
BoundsErrors. `SeaPearl.train!` handles this automatically for its own
heuristic, but a manually constructed one must be switched explicitly.
"""
function load_bss_agent_params!(lh, weights)
    Flux.loadparams!(lh.agent.policy.learner.approximator.model, weights)
    Flux.loadparams!(lh.agent.policy.learner.target_approximator.model, weights)
    Flux.testmode!(lh)          # inference mode: no replay-buffer pushes
    return lh
end
