# Tests for the cascade (sequential multi-method) unfolding module:
# port of bssunfold `core/unfold_cascade.py`.

using Test
using BSSUnfold
using LinearAlgebra
using Random
using Statistics

# Same synthetic problem as the other batches (m detectors, n energy bins)
function _make_cascade_problem(m::Int=12, n::Int=40; seed::Int=42)
    rng = MersenneTwister(seed)
    A = rand(rng, m, n) .+ 0.3
    A ./= sum(A, dims=2)
    x_true = exp.(-collect(range(0, 4, length=n)))
    b = A * x_true .+ 0.005 .* randn(rng, m)
    x0 = ones(n) .* 0.5
    return A, b, x0, x_true
end

const _E_cascade = collect(range(1e-3, 20.0; length=40))

@testset "Batch7 — Cascade" begin
    A, b, x0, x_true = _make_cascade_problem()
    n = length(x0)

    @testset "CascadeStage defaults" begin
        stage = CascadeStage("tsvd")
        @test stage.method == "tsvd"
        @test isempty(stage.params)
        @test stage.use_as_initial
        @test !stage.use_as_prior
        @test !stage.store_intermediate
        @test stage.quality_threshold === nothing
        @test stage.max_iterations === nothing
        @test stage.timeout == 60.0
        @test !stage.coarse
        @test stage.coarse_bins === nothing

        full = CascadeStage("mlem"; params=Dict(:max_iterations => 10),
                            use_as_initial=false, use_as_prior=true,
                            store_intermediate=true, quality_threshold=0.25,
                            max_iterations=20, timeout=5.0, coarse=true,
                            coarse_bins=8)
        @test full.params[:max_iterations] == 10
        @test !full.use_as_initial && full.use_as_prior && full.store_intermediate
        @test full.quality_threshold == 0.25 && full.max_iterations == 20
        @test full.timeout == 5.0 && full.coarse && full.coarse_bins == 8
    end

    @testset "create_default_cascade kinds" begin
        for kind in ("general", "soft", "hard", "fast_refinement", "unknown_kind")
            stages = create_default_cascade(kind)
            @test stages isa Vector{<:CascadeStage}
            @test length(stages) ≥ 2
            @test all(isa(s, CascadeStage) for s in stages)
        end
        @test create_default_cascade("general")[1].method == "tsvd"
        @test create_default_cascade("general")[1].quality_threshold == 0.3
        @test create_default_cascade("soft")[1].use_as_initial === false
    end

    @testset "compute_quality_metrics" begin
        rng = MersenneTwister(1)
        spec = abs.(randn(rng, n)) .* 0.2 .+ 1.0 .+ 1e-3
        metrics = compute_quality_metrics(spec, A * spec, b, _E_cascade)
        for key in ("chi_square", "smoothness", "flux_error", "negativity_count",
                    "hardness_ratio", "peak_count", "overall_quality")
            @test haskey(metrics, key)
            @test isfinite(metrics[key])
        end
        @test metrics["negativity_count"] == 0
        # A perfect reconstruction has a vanishing chi-square.
        exact = compute_quality_metrics(x_true, A * x_true, A * x_true, _E_cascade)
        @test exact["chi_square"] < 1e-20
        @test 0 < exact["smoothness"] ≤ 1
    end

    @testset "select_next_method" begin
        available = ["tsvd", "mlem", "gravel", "bayes_spline"]
        @test select_next_method(Dict{String,Any}("smoothness" => 0.2,
                                                  "chi_square" => 1.0,
                                                  "flux_error" => 0.05), available, 0) == "tsvd"
        @test select_next_method(Dict{String,Any}("smoothness" => 0.5,
                                                  "chi_square" => 8.0,
                                                  "flux_error" => 0.05), available, 1) == "mlem"
        @test select_next_method(Dict{String,Any}("smoothness" => 0.5,
                                                  "chi_square" => 1.0,
                                                  "flux_error" => 0.5), available, 2) == "gravel"
        @test select_next_method(Dict{String,Any}("smoothness" => 0.5,
                                                  "chi_square" => 1.0,
                                                  "flux_error" => 0.05), available, 3) == "bayes_spline"
        # nothing preferred is available → rotation over the defaults
        @test select_next_method(Dict{String,Any}(), ["nobody"], 0) == "landweber"
        @test select_next_method(Dict{String,Any}(), ["nobody"], 1) == "mlem"
    end

    @testset "Output shape, finiteness, non-negativity" begin
        for kind in ("general", "soft", "hard", "fast_refinement")
            res = solve_cascade(A, b, x0; stages=create_default_cascade(kind),
                                verbose=false, E_MeV=_E_cascade)
            @test res isa UnfoldResult
            @test length(res.spectrum) == n
            @test all(isfinite.(res.spectrum))
            @test all(res.spectrum .≥ 0)
            @test res.converged
            @test res.residual_norm ≥ 0
            @test res.extra["status"] == "OK"
            @test res.extra["stages_run"] ≥ 1
        end
    end

    @testset "Single-stage cascade equals the member solver" begin
        for (name, params) in (("mlem", Dict(:max_iterations => 50)),
                               ("landweber", Dict(:max_iterations => 50)),
                               ("tsvd", Dict(:truncation_rank => 8)),
                               ("gravel", Dict(:max_iterations => 50)))
            member = getfield(BSSUnfold, Symbol("solve_", name))(A, b, copy(x0); params...)
            one_stage = solve_cascade(A, b, x0;
                                      stages=[CascadeStage(name; params=params)],
                                      verbose=false, E_MeV=_E_cascade)
            @test one_stage.extra["stages_run"] == 1
            @test one_stage.extra["method_sequence"] == [name]
            @test length(one_stage.spectrum) == n
            @test isapprox(one_stage.spectrum, max.(member.spectrum, 0.0), rtol=1e-9)
        end
    end

    @testset "Two- vs three-stage cascades differ" begin
        two = [CascadeStage("tsvd"; params=Dict(:truncation_rank => 15),
                            use_as_initial=false, store_intermediate=true),
               CascadeStage("mlem"; params=Dict(:max_iterations => 150),
                            use_as_initial=true, store_intermediate=true)]
        three = [two; CascadeStage("bayes_spline"; params=Dict(:spline_smooth => 0.3),
                                   use_as_initial=true, use_as_prior=true,
                                   store_intermediate=true)]
        r2 = solve_cascade(A, b, x0; stages=two, verbose=false, E_MeV=_E_cascade)
        r3 = solve_cascade(A, b, x0; stages=three, verbose=false, E_MeV=_E_cascade)
        @test r2.extra["stages_run"] == 2
        @test r3.extra["stages_run"] == 3
        @test length(r2.extra["convergence_history"]) == 2
        @test length(r3.extra["convergence_history"]) == 3
        @test norm(r2.spectrum - r3.spectrum) > 1e-6 * norm(r2.spectrum)
        @test !(r2.spectrum ≈ r3.spectrum)
        # the first two stages of the three-stage chain reproduce the two-stage run
        @test isapprox(r2.extra["intermediate_results"]["stage_1_mlem"]["spectrum"],
                       r3.extra["intermediate_results"]["stage_1_mlem"]["spectrum"],
                       rtol=1e-12)
    end

    @testset "use_as_initial=false restarts from the cascade x0" begin
        stages = [CascadeStage("mlem"; params=Dict(:max_iterations => 30)),
                  CascadeStage("landweber"; params=Dict(:max_iterations => 5),
                               use_as_initial=false)]
        res = solve_cascade(A, b, x0; stages=stages, verbose=false, E_MeV=_E_cascade)
        alone = solve_landweber(A, b, copy(x0); max_iterations=5)
        @test isapprox(res.spectrum, max.(alone.spectrum, 0.0), rtol=1e-9)
    end

    @testset "Unknown method throws, failing stage does not abort" begin
        @test_throws ErrorException solve_cascade(A, b, x0;
                                                 stages=[CascadeStage("does_not_exist")],
                                                 verbose=false, E_MeV=_E_cascade)
        # a stage that is unavailable is skipped, the rest of the cascade runs
        mixed = [CascadeStage("does_not_exist"),
                 CascadeStage("mlem"; params=Dict(:max_iterations => 40))]
        res = solve_cascade(A, b, x0; stages=mixed, verbose=false, E_MeV=_E_cascade)
        @test res.extra["stages_run"] == 1
        @test all(isfinite.(res.spectrum))

        # a stage that raises inside the solver is caught and the cascade continues
        broken = [CascadeStage("mlem"; params=Dict(:no_such_keyword => 1)),
                  CascadeStage("landweber"; params=Dict(:max_iterations => 40))]
        res2 = solve_cascade(A, b, x0; stages=broken, verbose=false, E_MeV=_E_cascade)
        @test res2.extra["stages_run"] == 1
        @test res2.extra["status"] == "OK"
        @test length(res2.extra["convergence_history"]) == 1
        @test res2.extra["convergence_history"][1]["method"] == "landweber"
        alone = solve_landweber(A, b, copy(x0); max_iterations=40)
        @test isapprox(res2.spectrum, max.(alone.spectrum, 0.0), rtol=1e-9)

        # every stage failing behaves like the Python "no successful stages" result
        @test_throws ErrorException solve_cascade(A, b, x0;
                                                 stages=[CascadeStage("no_such_method_a"),
                                                         CascadeStage("no_such_method_b")],
                                                 verbose=false, E_MeV=_E_cascade)
    end

    @testset "Extra bookkeeping" begin
        stages = [CascadeStage("tsvd"; params=Dict(:truncation_rank => 12),
                               use_as_initial=false, store_intermediate=true),
                  CascadeStage("mlem"; params=Dict(:max_iterations => 60),
                               use_as_initial=true, store_intermediate=true)]
        res = solve_cascade(A, b, x0; stages=stages, verbose=false, E_MeV=_E_cascade)
        for key in ("stages_run", "total_time", "intermediate_results", "quality_metrics",
                    "method_sequence", "convergence_history", "status", "message",
                    "cascade_result")
            @test haskey(res.extra, key)
        end
        @test res.extra["method_sequence"] == ["tsvd", "mlem"]
        @test res.extra["total_time"] ≥ 0
        @test res.extra["message"] == "Successfully completed 2 cascade stages"
        @test res.extra["quality_metrics"]["chi_square"] ≥ 0
        @test haskey(res.extra["intermediate_results"], "stage_0_tsvd")
        @test haskey(res.extra["intermediate_results"], "stage_1_mlem")
        stored = res.extra["intermediate_results"]["stage_0_tsvd"]
        @test length(stored["spectrum"]) == n
        @test isfinite(stored["metrics"]["smoothness"])
        @test stored["result"] isa UnfoldResult
        history = res.extra["convergence_history"]
        @test length(history) == 2
        @test history[1]["stage"] == 0 && history[2]["stage"] == 1
        @test history[1]["method"] == "tsvd"
        @test haskey(history[2], "overall_quality")
        cascade_result = res.extra["cascade_result"]
        @test cascade_result isa CascadeResult
        @test cascade_result.status == "OK"
        @test cascade_result.stages_run == 2
        @test cascade_result.spectrum !== nothing
        @test length(cascade_result.spectrum) == n
    end

    @testset "Quality threshold stops the cascade" begin
        stages = [CascadeStage("tsvd"; params=Dict(:truncation_rank => 12),
                               use_as_initial=false, store_intermediate=true,
                               quality_threshold=0.0),
                  CascadeStage("mlem"; params=Dict(:max_iterations => 60)),
                  CascadeStage("gravel"; params=Dict(:max_iterations => 60))]
        res = solve_cascade(A, b, x0; stages=stages, verbose=false, E_MeV=_E_cascade)
        @test res.extra["stages_run"] == 1
        @test res.extra["method_sequence"] == ["tsvd"]
        @test length(res.extra["convergence_history"]) == 1

        # max_iterations caps the stage's own budget
        capped = solve_cascade(A, b, x0;
                               stages=[CascadeStage("mlem"; params=Dict(:max_iterations => 500),
                                                    max_iterations=20)],
                               verbose=false, E_MeV=_E_cascade)
        expected = solve_mlem(A, b, copy(x0); max_iterations=20)
        @test isapprox(capped.spectrum, max.(expected.spectrum, 0.0), rtol=1e-9)
    end

    @testset "Coarse stages and multi_resolution" begin
        coarse = [CascadeStage("tsvd"; params=Dict(:truncation_rank => 6),
                               use_as_initial=false, store_intermediate=true,
                               coarse=true, coarse_bins=10),
                  CascadeStage("mlem"; params=Dict(:max_iterations => 40),
                               use_as_initial=true, store_intermediate=true)]
        res = solve_cascade(A, b, x0; stages=coarse, verbose=false, E_MeV=_E_cascade)
        @test res.extra["stages_run"] == 2
        @test length(res.spectrum) == n
        @test all(isfinite.(res.spectrum)) && all(res.spectrum .≥ 0)

        # multi_resolution toggles the first stage without mutating the caller's list
        original = [CascadeStage("tsvd"; params=Dict(:truncation_rank => 12),
                                 use_as_initial=false),
                    CascadeStage("mlem"; params=Dict(:max_iterations => 40))]
        res_mr = solve_cascade(A, b, x0; stages=original, verbose=false,
                               multi_resolution=true, coarse_bins=8, E_MeV=_E_cascade)
        @test length(res_mr.spectrum) == n
        @test res_mr.extra["stages_run"] == 2
        @test !original[1].coarse
        @test original[1].coarse_bins === nothing

        # prolongated coarse spectrum keeps the fluence of the coarse solve
        @test length(res.extra["intermediate_results"]) == 2
    end

    @testset "Adaptive cascade" begin
        res = solve_adaptive_cascade(A, b, x0; max_stages=2, initial_method="tsvd",
                                     verbose=false, E_MeV=_E_cascade)
        @test res isa UnfoldResult
        @test length(res.spectrum) == n
        @test all(isfinite.(res.spectrum)) && all(res.spectrum .≥ 0)
        @test res.extra["status"] == "OK"
        @test res.extra["method_sequence"][1] == "tsvd"
        @test length(Set(res.extra["method_sequence"])) == length(res.extra["method_sequence"])
    end
end
