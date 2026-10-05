# Batch 7 — Gnowee hybrid metaheuristic (vendored port of unfold_gnowee.py
# plus its optimizer _gnowee.py): Lévy flights + golden-ratio crossover +
# scatter search + DE-style mutation with elitist population update.

using Test, BSSUnfold, LinearAlgebra, Random, Statistics

# Under-determined Bonner-sphere-like problem (more bins than detectors)
function _make_gnowee_problem(m::Int=14, n::Int=40; seed::Int=42)
    rng = MersenneTwister(seed)
    A = rand(rng, m, n) .+ 0.3
    x_true = exp.(-collect(range(0.0, 4.0; length=n)))
    b = A * x_true .+ 0.005 .* randn(rng, m)
    x0 = fill(0.5, n)
    return A, b, x0, x_true
end

@testset "Batch7 — Gnowee" begin
    A, b, x0, x_true = _make_gnowee_problem()
    n = length(x0)

    @testset "Basic solve" begin
        res = solve_gnowee(A, b, x0; random_state=1)
        @test res isa UnfoldResult
        @test length(res.spectrum) == n
        @test all(isfinite.(res.spectrum))
        @test all(res.spectrum .>= 0)
        @test res.iterations > 0
        @test res.residual_norm ≥ 0
        # the objective is scale-consistent, so the reported best cost and the
        # raw residual stay on the order of the data
        @test res.extra["best_fitness"] > 0
        @test norm(A * res.spectrum - b) < 2 * norm(b)
    end

    @testset "Cost decreases with the evaluation budget" begin
        # Same seed, larger budget: the loop is elitist and the random stream
        # only depends on the seed, so the best cost must be non-increasing.
        for seed in (1, 2, 3)
            costs = Float64[]
            for (fevals, stall) in ((600, 500), (1500, 1500), (3000, 3000))
                res = solve_gnowee(A, b, x0; random_state=seed,
                                   max_fevals=fevals, stall_limit=stall, max_gens=400)
                push!(costs, res.extra["best_fitness"])
            end
            @test costs[1] ≥ costs[2] ≥ costs[3]
            @test costs[3] < costs[1]
        end
    end

    @testset "Reproducibility" begin
        r1 = solve_gnowee(A, b, x0; random_state=7, max_fevals=1500, stall_limit=1500)
        r2 = solve_gnowee(A, b, x0; random_state=7, max_fevals=1500, stall_limit=1500)
        @test r1.spectrum == r2.spectrum
        @test r1.extra["best_fitness"] == r2.extra["best_fitness"]
        @test r1.iterations == r2.iterations
        # a different seed gives a different search trajectory
        r3 = solve_gnowee(A, b, x0; random_state=8, max_fevals=1500, stall_limit=1500)
        @test r3.spectrum != r1.spectrum
    end

    @testset "Extra diagnostics" begin
        res = solve_gnowee(A, b, x0; random_state=1)
        for key in ("population", "max_gens", "max_fevals", "stall_limit", "conv_tol",
                    "opt_conv_tol", "frac_elite", "frac_levy", "frac_mutation",
                    "alpha_levy", "gamma_levy", "n_levy", "scaling_factor",
                    "init_sampling", "regularization", "norm", "smoothness_order",
                    "smoothness_weight", "entropy_weight", "half_range",
                    "best_fitness", "evaluations", "generations", "timeline_len")
            @test haskey(res.extra, key)
        end
        @test res.extra["population"] == 25
        @test res.extra["max_gens"] == 200
        @test res.extra["max_fevals"] == 5000
        @test res.extra["stall_limit"] == 200
        @test res.extra["frac_elite"] == 0.2
        @test res.extra["frac_levy"] == 1.0
        @test res.extra["frac_mutation"] == 0.2
        @test res.extra["alpha_levy"] == 1.5
        @test res.extra["gamma_levy"] == 1.0
        @test res.extra["n_levy"] == 1
        @test res.extra["scaling_factor"] == 10.0
        @test res.extra["init_sampling"] == "lhc"
        @test res.extra["regularization"] == 1e-2
        @test res.extra["norm"] == 2
        @test res.extra["smoothness_order"] == 2
        @test res.extra["smoothness_weight"] == 1.0
        @test res.extra["entropy_weight"] == 0.0
        @test res.extra["half_range"] == 2.0
        @test res.extra["evaluations"] == res.iterations
        @test res.extra["timeline_len"] ≥ 1
        @test length(res.extra["timeline"]) == res.extra["timeline_len"]
        # improvement history: fitness non-increasing along the timeline
        tl = res.extra["timeline"]
        @test all(tl[i][3] ≥ tl[i + 1][3] for i in 1:length(tl)-1)
    end

    @testset "Convergence flag and caps" begin
        # a tiny evaluation cap is hit, so the run did not "converge"
        res = solve_gnowee(A, b, x0; random_state=1, max_fevals=300, stall_limit=100_000)
        @test !res.converged
        @test res.extra["evaluations"] ≥ 300
        # a generation cap is likewise reported as non-converged
        res_g = solve_gnowee(A, b, x0; random_state=1, max_gens=1, max_fevals=100_000,
                            stall_limit=100_000)
        @test !res_g.converged
        @test res_g.extra["generations"] ≥ 1
    end

    @testset "Option paths" begin
        # uniform (non-LHC) initial sampling, 'lhs' aliases 'lhc'
        r_rand = solve_gnowee(A, b, x0; random_state=1, init_sampling="random",
                              max_fevals=1200, stall_limit=1200)
        @test all(isfinite.(r_rand.spectrum)) && all(r_rand.spectrum .>= 0)
        r_lhs = solve_gnowee(A, b, x0; random_state=1, init_sampling="lhs",
                             max_fevals=1200, stall_limit=1200)
        r_lhc = solve_gnowee(A, b, x0; random_state=1, init_sampling="lhc",
                             max_fevals=1200, stall_limit=1200)
        @test r_lhs.extra["best_fitness"] == r_lhc.extra["best_fitness"]

        # norm 1 / first-order smoothness / entropy term
        r1 = solve_gnowee(A, b, x0; random_state=2, norm=1, smoothness_order=1,
                          max_fevals=1200, stall_limit=1200)
        @test all(r1.spectrum .>= 0) && all(isfinite.(r1.spectrum))
        r0 = solve_gnowee(A, b, x0; random_state=2, smoothness_order=0, entropy_weight=0.01,
                          max_fevals=1200, stall_limit=1200)
        @test all(r0.spectrum .>= 0) && all(isfinite.(r0.spectrum))

        # degenerate knobs: no elitism / no mutation / single Lévy parent
        rd = solve_gnowee(A, b, x0; random_state=3, frac_elite=0.0, frac_mutation=0.0,
                          frac_levy=0.0, regularization=0.0, smoothness_weight=0.0,
                          max_fevals=800, stall_limit=800)
        @test all(isfinite.(rd.spectrum)) && all(rd.spectrum .>= 0)

        # multi-sample Lévy (n_levy > 1) and a wider log-space window
        rn = solve_gnowee(A, b, x0; random_state=3, n_levy=3, half_range=3.0,
                          max_fevals=1200, stall_limit=1200)
        @test all(isfinite.(rn.spectrum)) && all(rn.spectrum .> 0)

        # a non-zero initial spectrum is used as the warm start directly
        warm = solve_gnowee(A, b, max.(x_true, 1e-9); random_state=4,
                            max_fevals=1200, stall_limit=1200)
        @test length(warm.spectrum) == n
        # all-zero x0 falls back to the Landweber warm start (same as `nothing`)
        z = zeros(n)
        lw = solve_gnowee(A, b, z; random_state=4, max_fevals=1200, stall_limit=1200)
        nn = solve_gnowee(A, b, nothing; random_state=4, max_fevals=1200, stall_limit=1200)
        @test lw.spectrum == nn.spectrum
        @test all(isfinite.(lw.spectrum)) && all(lw.spectrum .>= 0)

        # Float32 inputs keep the element type
        r32 = solve_gnowee(Float32.(A), Float32.(b), Float32.(x0); random_state=1)
        @test r32.spectrum isa Vector{Float32}
        @test length(r32.spectrum) == n
    end

    @testset "Validation" begin
        @test_throws ArgumentError solve_gnowee(A, b, x0; norm=3)
        @test_throws ArgumentError solve_gnowee(A, b, x0; norm=0)
        @test_throws ArgumentError solve_gnowee(A, b, x0; smoothness_order=3)
        @test_throws ArgumentError solve_gnowee(A, b, x0; init_sampling="grid")
        @test_throws ArgumentError solve_gnowee(A, b[1:5], x0)
        @test_throws ArgumentError solve_gnowee(A, b, x0; random_state=1, frac_elite=1.5)
        @test_throws ArgumentError solve_gnowee(A, b, x0; random_state=1, alpha_levy=2.5)
    end
end
