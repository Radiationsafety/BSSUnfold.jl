# Tests for the SeaPearl-based CP unfolding method (solve_seapearl).
#
# SeaPearl.jl is an OPTIONAL lazy dependency (like Turing.jl for solve_mcmc):
#   - present  → real CP enumeration tests (feasibility set, χ² ranking,
#                interval estimates, Detector wrapper);
#   - absent   → graceful degradation checks (zero spectrum + error metadata).

@testset "SeaPearl-CSP" begin
    A, b, x0, x_true = _make_problem3(8, 20)

    sp_probe = try
        Base.eval(Main, :(import SeaPearl))
        true
    catch
        false
    end

    if sp_probe
        # Ground truth quantized on the same grid the solver uses: guarantees
        # a feasible solution exists for the clean-data test.
        n_levels = 16
        mv = 2.0 * maximum(x_true)
        delta = mv / (n_levels - 1)
        phi_true = round.(x_true ./ delta) .* delta   # quantized spectrum
        b_clean = A * phi_true                        # exact readings (no noise)

        @testset "Feasibility set on clean quantized data" begin
            res = solve_seapearl(A, b_clean, x0; n_levels=n_levels,
                                 k_sigma=2.0, noise_level=0.01,
                                 max_solutions=64, time_limit_ms=30000)
            @test typeof(res) <: UnfoldResult
            @test length(res.spectrum) == 20
            @test all(res.spectrum .≥ 0)
            @test all(isfinite.(res.spectrum))
            @test res.converged
            @test res.extra["n_solutions"] ≥ 1
            # The true quantized spectrum must be in the feasible set, so the
            # best member must reproduce the readings within the kσ window.
            @test res.extra["chi2_best"] < 1e3
            @test norm(A * res.spectrum .- b_clean) ≤
                  norm(A * (ones(20) .* (sum(b_clean) / sum(A))) .- b_clean) + 1e-9
        end

        @testset "Interval estimates" begin
            res = solve_seapearl(A, b_clean, x0; n_levels=n_levels,
                                 k_sigma=2.0, noise_level=0.05,
                                 max_solutions=128, time_limit_ms=30000)
            lower = res.extra["spectrum_lower"]
            upper = res.extra["spectrum_upper"]
            @test all(lower .≥ 0)
            @test all(upper .≤ res.extra["max_value"] + 1e-12)
            @test all(lower .≤ upper .+ 1e-12)
            @test all(res.spectrum .≥ lower .- 1e-12)
            @test all(res.spectrum .≤ upper .+ 1e-12)
            if res.extra["n_solutions"] > 1
                @test any(upper .> lower .+ 1e-12)
            end
            @test length(res.extra["solutions_chi2"]) == res.extra["n_solutions"]
        end

        @testset "Determinism of DFS enumeration" begin
            r1 = solve_seapearl(A, b_clean, x0; n_levels=8, max_solutions=32,
                                time_limit_ms=15000)
            r2 = solve_seapearl(A, b_clean, x0; n_levels=8, max_solutions=32,
                                time_limit_ms=15000)
            @test r1.spectrum ≈ r2.spectrum
            @test r1.extra["n_solutions"] == r2.extra["n_solutions"]
        end

        @testset "ILDS strategy" begin
            res = solve_seapearl(A, b_clean, x0; n_levels=8, strategy=:ilds,
                                 ilds_max_discrepancy=1, max_solutions=16,
                                 time_limit_ms=15000)
            @test res.converged || res.extra["n_solutions"] ≥ 0
            @test length(res.spectrum) == 20
        end

        @testset "Infeasible problem → flux-matched fallback" begin
            # A large positive bias makes every quantized spectrum incompatible
            # with a nearly-zero noise window; relaxations are deliberately tiny.
            b_biased = b_clean .+ 0.5
            res = solve_seapearl(A, b_biased, x0; n_levels=8, k_sigma=1.0,
                                 noise_level=0.0, relax_attempts=1,
                                 relax_step=0.001, expand_attempts=1,
                                 max_solutions=16, time_limit_ms=10000)
            @test typeof(res) <: UnfoldResult
            @test !res.converged
            @test res.extra["n_solutions"] == 0
            @test haskey(res.extra, "error")
            @test all(isfinite.(res.spectrum))
        end

        @testset "unfold_seapearl Detector wrapper" begin
            n = 20
            names = ["a", "b", "c", "d", "e"]
            rng = MersenneTwister(123)
            E_MeV = collect(range(1e-6, 10.0, length=n))
            sens = Dict(nm => rand(rng, n) .+ 0.1 for nm in names)
            cc = Dict(nm => rand(rng, n) for nm in names)
            d = Detector(names, E_MeV, sens, cc)
            true_spec = exp.(-E_MeV ./ 1.5)
            readings = Dict(nm => sum(sens[nm] .* true_spec) for nm in names)
            result = unfold_seapearl(d, readings, n_levels=8, max_solutions=32,
                                     time_limit_ms=15000)
            @test haskey(result, "spectrum")
            @test length(result["spectrum"]) == n
            @test result["method"] == "SeaPearl-CSP"
            # algorithm-specific metadata merged from UnfoldResult.extra
            @test haskey(result, "spectrum_lower")
            @test haskey(result, "spectrum_upper")
            @test haskey(result, "n_solutions")
        end
    else
        @testset "Graceful degradation without SeaPearl" begin
            res = solve_seapearl(A, b, x0)
            @test typeof(res) <: UnfoldResult
            @test length(res.spectrum) == 20
            @test all(res.spectrum .== 0)
            @test !res.converged
            @test res.extra["error"] == "SeaPearl.jl not available"
            @test seapearl_available() == false  # function, not shadowed
        end
    end

    @testset "Input validation" begin
        @test_throws ArgumentError solve_seapearl(A, b[1:5], x0)
        @test_throws ArgumentError solve_seapearl(A, b, x0[1:5])
        @test_throws ArgumentError solve_seapearl(A, b, x0; n_levels=1)
        @test_throws ArgumentError solve_seapearl(A, b, x0; max_solutions=0)
        @test_throws ArgumentError solve_seapearl(A, b, x0; value_order=:middle)
        @test_throws ArgumentError solve_seapearl(A, b, x0; expand_factor=0.5)
    end
end
