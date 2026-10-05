# Tests for the two multi-method orchestration modules:
# unfold_combined.jl (sequential pipeline) and unfold_composite.jl
# (adaptive hardness-classified ensemble).

using Test
using BSSUnfold
using LinearAlgebra
using Random
using Statistics

# Same synthetic problem as the other batch test files
function _make_orch_problem(m::Int=12, n::Int=40; seed::Int=42)
    rng = MersenneTwister(seed)
    A = rand(rng, m, n) .+ 0.3
    A ./= sum(A, dims=2)
    x_true = exp.(-collect(range(0, 4, length=n)))
    b = A * x_true .+ 0.005 .* randn(rng, m)
    x0 = ones(n) .* 0.5
    return A, b, x0, x_true
end

function _stage(method::AbstractString; params=Dict{String,Any}(), kwargs...)
    return Dict{String,Any}(merge(Dict{String,Any}("method" => method, "params" => params),
                                  Dict{String,Any}(string(k) => v for (k, v) in kwargs)))
end


@testset "Batch7 — Combined" begin
    A, b, x0, x_true = _make_orch_problem()
    n = size(A, 2)

    @testset "Output shape and physicality" begin
        res = solve_combined(A, b, x0; verbose=false,
                             pipeline=[_stage("mlem"; params=Dict("max_iterations" => 200)),
                                       _stage("landweber"; params=Dict("max_iterations" => 200))])
        @test length(res.spectrum) == n
        @test all(isfinite.(res.spectrum))
        @test all(res.spectrum .≥ 0)
        @test res.residual_norm ≈ norm(b .- A * res.spectrum)
        @test res.iterations ≥ 0
        # data fidelity of a chained pipeline
        @test norm(A * res.spectrum .- b) < 0.5 * norm(b)
    end

    @testset "Single-member pipeline reduces to that member" begin
        member = _stage("mlem"; params=Dict("max_iterations" => 300, "tolerance" => 1e-8))
        res = solve_combined(A, b, x0; pipeline=[member], verbose=false)
        ref = solve_mlem(A, b, x0; max_iterations=300, tolerance=1e-8)
        @test res.spectrum ≈ max.(ref.spectrum, 0.0) rtol = 0.0 atol = 1e-12
        @test res.extra["pipeline_info"]["stages"] == ["mlem"]
        @test length(res.extra["stage_spectra"]) == 1
    end

    @testset "Chaining feeds the previous result into the next solver" begin
        pipe = [_stage("mlem"; params=Dict("max_iterations" => 200, "tolerance" => 1e-8)),
                _stage("landweber"; params=Dict("max_iterations" => 200, "tolerance" => 1e-8))]
        res = solve_combined(A, b, x0; pipeline=pipe, verbose=false)
        x1 = max.(collect(Float64, solve_mlem(A, b, x0; max_iterations=200, tolerance=1e-8).spectrum), 0.0)
        x2 = max.(collect(Float64, solve_landweber(A, b, x1; max_iterations=200, tolerance=1e-8).spectrum), 0.0)
        @test res.extra["stage_spectra"][1] ≈ x1 rtol = 0.0 atol = 1e-12
        @test res.spectrum ≈ x2 rtol = 0.0 atol = 1e-12
    end

    @testset "Different orderings give different spectra" begin
        kw = Dict("max_iterations" => 200, "tolerance" => 1e-8)
        ab = solve_combined(A, b, x0; verbose=false,
                            pipeline=[_stage("mlem"; params=kw), _stage("landweber"; params=kw)])
        ba = solve_combined(A, b, x0; verbose=false,
                            pipeline=[_stage("landweber"; params=kw), _stage("mlem"; params=kw)])
        @test ab.extra["pipeline_info"]["stages"] == ["mlem", "landweber"]
        @test ba.extra["pipeline_info"]["stages"] == ["landweber", "mlem"]
        @test maximum(abs.(ab.spectrum .- ba.spectrum)) > 1e-6
    end

    @testset "use_as_initial=false breaks the chain" begin
        kw = Dict("max_iterations" => 200, "tolerance" => 1e-8)
        res = solve_combined(A, b, x0; verbose=false,
                             pipeline=[_stage("mlem"; params=kw),
                                       _stage("landweber"; params=kw, use_as_initial=false)])
        direct = solve_landweber(A, b, x0; max_iterations=200, tolerance=1e-8)
        @test res.spectrum ≈ max.(direct.spectrum, 0.0) rtol = 0.0 atol = 1e-12
    end

    @testset "Stage params dict is recorded verbatim" begin
        kw = Dict("max_iterations" => 137, "tolerance" => 1e-7)
        res = solve_combined(A, b, x0; verbose=false, pipeline=[_stage("cgls"; params=kw)])
        @test res.extra["pipeline_info"]["params"][1]["max_iterations"] == 137
    end

    @testset "store_intermediate keeps per-stage results" begin
        kw = Dict("max_iterations" => 150)
        res = solve_combined(A, b, x0; verbose=false,
                             pipeline=[_stage("mlem"; params=kw, store_intermediate=true),
                                       _stage("gravel"; params=kw, store_intermediate=true)])
        inter = res.extra["intermediate_results"]
        @test Set(keys(inter)) == Set(["stage_1_mlem", "stage_2_gravel"])
        @test length(inter["stage_1_mlem"].spectrum) == n
        @test length(inter["stage_2_gravel"].spectrum) == n
    end

    @testset "Unknown method name raises" begin
        @test_throws ArgumentError solve_combined(A, b, x0; verbose=false,
                                                  pipeline=[_stage("no_such_method")])
        @test_throws ArgumentError solve_combined(A, b, x0; verbose=false,
                                                  pipeline=[_stage("mlem"), _stage("nope")])
    end
end


@testset "Batch7 — Composite" begin
    A, b, x0, x_true = _make_orch_problem()
    n = size(A, 2)
    cheap = ["mlem", "landweber", "gravel", "cgls", "tsvd"]

    @testset "Output shape and physicality" begin
        res = solve_composite(A, b, x0; n_methods=4, method_names=cheap)
        @test length(res.spectrum) == n
        @test all(isfinite.(res.spectrum))
        @test all(res.spectrum .≥ 0)
        @test res.residual_norm ≈ norm(b .- A * res.spectrum)
        @test norm(A * res.spectrum .- b) < 0.9 * norm(b)
    end

    @testset "Ensemble is at least as good as its worst member" begin
        res = solve_composite(A, b, x0; n_methods=4, method_names=cheap)
        spectra = res.extra["individual_spectra"]
        @test !isempty(spectra)
        worst = maximum(norm(A * s .- b) for s in values(spectra))
        # convex combination => ||A x_ens - b|| <= sum(w_i) ||A x_i - b|| <= worst
        @test res.residual_norm ≤ worst + 1e-9
    end

    @testset "Extra carries the per-member bookkeeping" begin
        res = solve_composite(A, b, x0; n_methods=3,
                              method_names=["mlem", "landweber", "bogus_method"])
        ex = res.extra
        @test ex["status"] == "OK"
        @test ex["candidates"] == ["mlem", "landweber", "bogus_method"]
        @test ex["successful_methods"] == ["mlem", "landweber"]
        @test haskey(ex["messages"], "bogus_method")
        @test ex["messages"]["bogus_method"] == "unknown method"
        @test ex["method_order"] == ["mlem", "landweber"]
        @test Set(keys(ex["individual_spectra"])) == Set(["mlem", "landweber"])
        @test all(haskey(ex["weights"], nm) for nm in ex["successful_methods"])
        @test 0.0 ≤ ex["consistency"] ≤ 1.0
        @test occursin("Combined", ex["message"])
    end

    @testset "Weights are base weight times confidence in [0, 1]" begin
        w = Dict{String,Float64}("mlem" => 1.0, "landweber" => 0.5, "gravel" => 2.0)
        res = solve_composite(A, b, x0; n_methods=3,
                              method_names=["mlem", "landweber", "gravel"],
                              ensemble_weights=w)
        used = res.extra["weights"]
        @test all(used[nm] ≥ 0.0 for nm in keys(used))
        for nm in keys(used)
            @test used[nm] ≤ w[nm] + 1e-12
        end
        @test sum(values(used)) > 0.0
        # the combination is the weighted average of the member spectra
        combined = sum(used[nm] .* res.extra["individual_spectra"][nm] for nm in keys(used))
        @test res.spectrum ≈ combined / sum(values(used)) rtol = 1e-12
    end

    @testset "Single-member ensemble reduces to that member" begin
        res = solve_composite(A, b, x0; n_methods=1, method_names=["gravel"])
        ref = solve_gravel(A, b, x0; max_iterations=1000, tolerance=1e-8)
        @test res.spectrum ≈ max.(ref.spectrum, 0.0) rtol = 0.0 atol = 1e-12
        @test res.extra["weights"]["gravel"] == 1.0
        @test res.extra["consistency"] == 0.0
    end

    @testset "n_methods truncates the pool" begin
        res = solve_composite(A, b, x0; n_methods=2, method_names=cheap)
        @test res.extra["candidates"] == cheap[1:2]
        @test length(res.extra["successful_methods"]) ≤ 2
    end

    @testset "No method succeeds" begin
        @test_throws ErrorException solve_composite(A, b, x0; n_methods=3,
                                                   method_names=["no_such_a", "no_such_b"])
    end

    @testset "Spectrum features" begin
        E = collect(10.0 .^ range(-9.0, 2.0, length=n))
        f = compute_spectrum_features(x_true, E)
        @test f["hardness_ratio"] ≈ sum(x_true .* E) / sum(x_true) rtol = 1e-10
        @test f["total_flux"] ≈ sum(x_true) rtol = 1e-10
        @test f["peak"] == maximum(x_true)
        @test 0.0 ≤ f["entropy"] ≤ 1.0
        # a flat spectrum is maximally entropic
        ff = compute_spectrum_features(ones(n), E)
        @test ff["entropy"] ≈ 1.0 rtol = 1e-10
    end

    @testset "Hardness classification thresholds" begin
        bin(hr) = classify_spectrum_by_hardness(Dict{String,Float64}("hardness_ratio" => hr))
        @test bin(0.05) == "very_soft"
        @test bin(0.15) == "soft"
        @test bin(0.35) == "intermediate"
        @test bin(0.7) == "hard"
        @test bin(2.0) == "very_hard"
        # missing feature falls back to 0.5, which is not < 0.5 => hard
        # (exact Python behaviour: hr = features.get("hardness_ratio", 0.5))
        @test classify_spectrum_by_hardness(Dict{String,Float64}()) == "hard"
        @test classify_spectrum_by_hardness(Dict{String,Float64}()) == bin(0.5)
    end

    @testset "Hardness-selected pool and general fallback" begin
        # a spectrum living in the first energy bin has a very small mean energy
        soft = zeros(n); soft[1] = 1.0
        E = collect(10.0 .^ range(-9.0, 2.0, length=n))
        res = solve_composite(A, b, x0; n_methods=1, spectrum=soft, energy=E)
        @test res.extra["hardness_bin"] == "very_soft"
        @test res.extra["candidates"] == DEFAULT_BIN_METHODS["very_soft"][1:1]

        # no spectrum and no method_names => curated general pool
        res_gen = solve_composite(A, b, x0; n_methods=2)
        @test res_gen.extra["hardness_bin"] === nothing
        @test res_gen.extra["candidates"] == GENERAL_METHODS[1:2]
        @test length(res_gen.extra["successful_methods"]) ≥ 1
        @test !isempty(DEFAULT_ENSEMBLE_WEIGHTS)
    end
end
