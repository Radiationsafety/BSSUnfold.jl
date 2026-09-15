# Tests for batch 3: NSDUAZ, NSpline, MCMC (Turing), Genetic, QUBO

using Test
using BSSUnfold
using LinearAlgebra
using Random
using Statistics

# Shared fixture
function _make_problem3(m::Int=14, n::Int=100; seed::Int=42)
    rng = MersenneTwister(seed)
    A = rand(rng, m, n) .+ 0.3
    A ./= sum(A, dims=2)
    x_true = exp.(-collect(range(0, 5, length=n)))
    b = A * x_true .+ 0.01 .* randn(rng, m)
    x0 = ones(n) .* 0.1
    return A, b, x0, x_true
end


@testset "Batch3 — NSDUAZ" begin
    A, b, x0, x_true = _make_problem3(14, 100)

    @testset "Basic solve (SPUNIT/Bunki iteration)" begin
        res = solve_nsduaz(A, b, x0)
        @test length(res.spectrum) == 100
        @test all(res.spectrum .≥ 0)
        @test all(isfinite.(res.spectrum))
        @test res.iterations > 0
    end

    @testset "Default tolerance is NSDUAZ's ~1%" begin
        # solve_nsduaz — a thin wrapper over solve_bunki with tolerance=0.01
        res1 = solve_nsduaz(A, b, x0)
        res2 = solve_bunki(A, b, x0, tolerance=0.01)
        @test res1.spectrum ≈ res2.spectrum
    end

    @testset "Max iterations respected" begin
        res = solve_nsduaz(A, b, x0, max_iterations=5)
        @test res.iterations ≤ 5
    end

    @testset "builtin_catalogue" begin
        E = collect(range(1e-9, 20.0, length=100))
        cat = builtin_catalogue(E)
        @test Set(keys(cat)) == Set(["ambe", "cf252", "reactor"])
        for (k, v) in cat
            @test length(v) == 100
            @test all(v .≥ 0)
            @test sum(v) ≈ 1.0 atol = 1e-9
        end
    end

    @testset "select_catalogue_initial picks matching entry" begin
        E = collect(range(1e-9, 20.0, length=100))
        cat = builtin_catalogue(E)
        names = ["0_in", "2_in", "5_in", "10_in", "20.32_cm"]
        rng = MersenneTwister(7)
        sens = Dict(nm => rand(rng, 100) .+ 0.1 for nm in names)
        # Synthetic measurements from the cf252 spectrum
        x_cf = cat["cf252"] .* 1000.0
        readings = Dict(nm => sum(sens[nm] .* x_cf) for nm in names)
        spec, label = select_catalogue_initial(readings, names, sens;
                                               catalogue=cat, E_MeV=E)
        @test label == "cf252"
        @test length(spec) == 100
        @test all(spec .≥ 0)
        # Reading ratios must be reproduced
        A_sel = Matrix(hcat([sens[nm] for nm in names]...)')
        b_pred = A_sel * spec
        r_meas = [readings[nm] for nm in names] ./ readings["20.32_cm"]
        r_pred = b_pred ./ b_pred[findfirst(==("20.32_cm"), names)]
        @test r_meas ≈ r_pred rtol = 1e-8
    end

    @testset "unfold_nsduaz Detector wrapper" begin
        n = 50
        names = ["0_in", "2_in", "5_in", "20.32_cm"]
        rng = MersenneTwister(123)
        E_MeV = collect(range(1e-7, 10.0, length=n))
        sens = Dict(nm => rand(rng, n) .+ 0.1 for nm in names)
        cc = Dict(nm => rand(rng, n) for nm in names)
        d = Detector(names, E_MeV, sens, cc)
        true_spec = exp.(-E_MeV ./ 1.5)
        readings = Dict(nm => sum(sens[nm] .* true_spec) for nm in names)
        result = unfold_nsduaz(d, readings, max_iterations=100)
        @test haskey(result, "spectrum")
        @test length(result["spectrum"]) == n
        @test result["method"] == "NSDUAZ"
        @test haskey(result, "catalogue")
        @test result["catalogue"] in ("ambe", "cf252", "reactor")
        # With explicit initial_spectrum the catalogue is not used
        result2 = unfold_nsduaz(d, readings, initial_spectrum=ones(n) * 0.5)
        @test !haskey(result2, "catalogue")
    end
end


@testset "Batch3 — NSpline" begin
    A, b, x0, x_true = _make_problem3(14, 100)
    E = collect(range(1e-9, 20.0, length=100))

    @testset "auto_knots" begin
        kn = auto_knots(E; n_segments=8)
        @test length(kn) == 9
        @test issorted(kn)
        @test all(>(0), diff(kn))
        @test kn[1] ≈ 1e-9
        @test kn[end] ≈ 20.0
        @test_throws ArgumentError auto_knots([-1.0, -2.0, -3.0])
    end

    @testset "Knot presets" begin
        @test haskey(NSPLINE_KNOT_PRESETS, "BARS5_channel")
        @test haskey(NSPLINE_KNOT_PRESETS, "IGRIK_channel")
        @test haskey(NSPLINE_KNOT_PRESETS, "IGRIK_surface")
        @test haskey(NSPLINE_KNOT_PRESETS, "YAGUAR_channel")
        @test_throws ArgumentError solve_nspline(A, b, x0, E_MeV=E,
                                                knots="nonexistent_preset")
    end

    @testset "build_continuity_matrix" begin
        kn = auto_knots(E; n_segments=5)
        M = length(kn) - 1
        D_c0c1 = build_continuity_matrix(kn; continuity="C0C1")
        @test size(D_c0c1) == (2 * (M - 1), 3 * M)
        D_c0 = build_continuity_matrix(kn; continuity="C0")
        @test size(D_c0) == (M - 1, 3 * M)
        D_none = build_continuity_matrix(kn; continuity="none")
        @test size(D_none) == (0, 3 * M)
        @test_throws ArgumentError build_continuity_matrix(kn; continuity="bogus")
    end

    @testset "nspline_eval positive" begin
        kn = auto_knots(E; n_segments=4)
        M = length(kn) - 1
        a = fill(1.0, M); q = fill(0.5, M); r = fill(-0.01, M)
        vals = nspline_eval(E, a, q, r, kn)
        @test all(vals .> 0)
        @test all(isfinite.(vals))
    end

    @testset "directed_divergence" begin
        @test directed_divergence([0.5, 0.5], [0.5, 0.5]) ≈ 0.0 atol = 1e-12
        @test directed_divergence([0.7, 0.3], [0.5, 0.5]) > 0
    end

    @testset "fit_nspline recovers smooth spectrum" begin
        # A 1/E-like spectrum fits the N-spline well
        phi = 1.0 ./ E
        phi ./= sum(phi)
        N_E, info = fit_nspline(E, phi)
        @test length(N_E) == 100
        @test all(N_E .> 0)
        @test info["knots_source"] == "auto"
        @test haskey(info, "a") && haskey(info, "q") && haskey(info, "r")
        # The shape must be close (cosine similarity)
        cos_sim = dot(N_E, phi) / (norm(N_E) * norm(phi))
        @test cos_sim > 0.99
    end

    @testset "solve_nspline basic" begin
        # The N-spline needs an energy grid — E_MeV is a mandatory keyword
        @test_throws ArgumentError solve_nspline(A, b, x0)
        res = solve_nspline(A, b, x0, E_MeV=E, max_iterations=30)
        @test length(res.spectrum) == 100
        @test all(res.spectrum .≥ 0)
        @test all(isfinite.(res.spectrum))
        @test res.iterations ≥ 0
        @test haskey(res.extra, "H")
        @test haskey(res.extra, "nev")
        @test haskey(res.extra, "fluence")
        @test res.extra["H"] ≥ 0  # directed divergence is non-negative
    end

    @testset "solve_nspline_full diagnostics" begin
        full = solve_nspline_full(A, b, x0, E; max_iterations=20)
        for key in ("spectrum", "iterations", "converged", "stop_reason", "H",
                    "H_history", "H_target", "nev", "nev_limit", "acceptable",
                    "Qr", "relative_residuals", "fluence", "mean_energy",
                    "knots", "knots_source", "continuity", "params")
            @test haskey(full, key)
        end
        @test length(full["H_history"]) == full["iterations"] + 1
        @test full["nev_limit"] ≈ 1.0 + 2.0 / sqrt(count(b .> 0))
        # H_history must be non-increasing (divergence minimization)
        @test issorted(full["H_history"]; rev=true) ||
              length(full["H_history"]) ≤ 2
    end

    @testset "Continuity options" begin
        for cont in ("C0C1", "C0", "none")
            res = solve_nspline(A, b, x0, E_MeV=E, max_iterations=5,
                                continuity=cont)
            @test all(isfinite.(res.spectrum))
        end
    end

    @testset "unfold_nspline Detector wrapper" begin
        n = 50
        names = ["a", "b", "c", "d"]
        rng = MersenneTwister(123)
        E_MeV = collect(range(1e-6, 10.0, length=n))
        sens = Dict(nm => rand(rng, n) .+ 0.1 for nm in names)
        cc = Dict(nm => rand(rng, n) for nm in names)
        d = Detector(names, E_MeV, sens, cc)
        true_spec = exp.(-E_MeV ./ 1.5)
        readings = Dict(nm => sum(sens[nm] .* true_spec) for nm in names)
        result = unfold_nspline(d, readings, max_iterations=10)
        @test haskey(result, "spectrum")
        @test length(result["spectrum"]) == n
        @test result["method"] == "NSpline"
    end
end


@testset "Batch3 — Genetic" begin
    A, b, x0, x_true = _make_problem3(14, 60)

    @testset "PSO basic" begin
        res = solve_genetic(A, b, x0, solver=:pso, epoch=10, pop_size=20,
                            random_state=42)
        @test length(res.spectrum) == 60
        @test all(res.spectrum .≥ 0)
        @test all(isfinite.(res.spectrum))
        @test res.extra["solver"] == "pso"
        @test isfinite(res.extra["fitness"])
    end

    @testset "GA (TGASU) basic" begin
        res = solve_genetic(A, b, x0, solver=:ga, epoch=10, pop_size=20,
                            crossover=:arithmetic, mutation=:iterative,
                            random_state=42)
        @test all(res.spectrum .≥ 0)
        @test res.extra["solver"] == "ga"
    end

    @testset "DE basic" begin
        res = solve_genetic(A, b, x0, solver=:de, epoch=10, pop_size=20,
                            random_state=42)
        @test all(res.spectrum .≥ 0)
    end

    @testset "GWO basic" begin
        res = solve_genetic(A, b, x0, solver=:gwo, epoch=10, pop_size=20,
                            random_state=42)
        @test all(res.spectrum .≥ 0)
    end

    @testset "NSGA-II basic" begin
        res = solve_genetic(A, b, x0, solver=:nsga2, epoch=10, pop_size=20,
                            random_state=42)
        @test all(res.spectrum .≥ 0)
        diag = res.extra["diagnostics"]
        @test haskey(diag, "pareto_front_size")
        @test diag["pareto_front_size"] ≥ 1
        @test isfinite(diag["pareto_min_residual"])
    end

    @testset "Solver aliases" begin
        res = solve_genetic(A, b, x0, solver=:differential_evolution,
                            epoch=5, pop_size=10, random_state=1)
        @test all(res.spectrum .≥ 0)
        res2 = solve_genetic(A, b, x0, solver=:pareto, epoch=5, pop_size=10,
                             random_state=1)
        @test all(res2.spectrum .≥ 0)
    end

    @testset "Reproducibility with seed" begin
        r1 = solve_genetic(A, b, x0, solver=:pso, epoch=15, pop_size=20,
                           random_state=7)
        r2 = solve_genetic(A, b, x0, solver=:pso, epoch=15, pop_size=20,
                           random_state=7)
        @test r1.spectrum ≈ r2.spectrum
    end

    @testset "Landweber warm-start (x0 = zeros)" begin
        res = solve_genetic(A, b, zeros(60), solver=:pso, epoch=10,
                            pop_size=20, random_state=42)
        @test all(res.spectrum .≥ 0)
    end

    @testset "two_step mode" begin
        res = solve_genetic(A, b, x0, solver=:pso, epoch=10, pop_size=20,
                            two_step=true, random_state=42)
        @test length(res.spectrum) == 60
        @test all(res.spectrum .≥ 0)
    end

    @testset "Smoother" begin
        res = solve_genetic(A, b, x0, solver=:pso, epoch=10, pop_size=20,
                            smoother="gaussian", random_state=42)
        @test all(res.spectrum .≥ 0)
        res2 = solve_genetic(A, b, x0, solver=:pso, epoch=10, pop_size=20,
                             smoother="second_difference", random_state=42)
        @test all(res2.spectrum .≥ 0)
    end

    @testset "n_runs averaging" begin
        res = solve_genetic(A, b, x0, solver=:pso, epoch=5, pop_size=10,
                            n_runs=2, random_state=42)
        @test all(res.spectrum .≥ 0)
    end

    @testset "Validation errors" begin
        @test_throws ArgumentError solve_genetic(A, b, x0, solver=:cmaes)
        @test_throws ArgumentError solve_genetic(A, b, x0, norm=3)
        @test_throws ArgumentError solve_genetic(A, b, x0, smoothness_order=5)
        @test_throws ArgumentError solve_genetic(A, b, x0, crossover=:xyz)
        @test_throws ArgumentError solve_genetic(A, b, x0, mutation=:xyz)
        @test_throws ArgumentError solve_genetic(A, b, x0, pareto_select=:xyz)
    end

    @testset "coarsen/split roundtrip" begin
        xs = collect(range(1.0, 60.0, length=60))
        Ac = coarsen_columns(reshape(xs, 1, :), 10)
        @test size(Ac) == (1, 10)
        # The sum of values is preserved
        @test sum(Ac) ≈ sum(xs) rtol = 1e-12
        x_back = split_coarse(vec(Ac), 60)
        @test sum(x_back) ≈ sum(xs) rtol = 1e-12
    end

    @testset "apply_smoother preserves fluence" begin
        x = abs.(randn(MersenneTwister(3), 50)) .+ 0.1
        for sm in ("gaussian", "mbc", "gaussian_mbc", "second_difference")
            s = apply_smoother(x, sm)
            @test length(s) == 50
            @test all(s .≥ 0)
            @test sum(s) ≈ sum(x) rtol = 1e-9
        end
    end

    @testset "unfold_genetic Detector wrapper" begin
        n = 40
        names = ["a", "b", "c", "d"]
        rng = MersenneTwister(123)
        E_MeV = collect(range(1e-6, 10.0, length=n))
        sens = Dict(nm => rand(rng, n) .+ 0.1 for nm in names)
        cc = Dict(nm => rand(rng, n) for nm in names)
        d = Detector(names, E_MeV, sens, cc)
        true_spec = exp.(-E_MeV ./ 1.5)
        readings = Dict(nm => sum(sens[nm] .* true_spec) for nm in names)
        result = unfold_genetic(d, readings, solver=:pso, epoch=5,
                                pop_size=10, random_state=42)
        @test haskey(result, "spectrum")
        @test length(result["spectrum"]) == n
        @test result["method"] == "Genetic"
    end
end


@testset "Batch3 — QUBO" begin
    A, b, x0, x_true = _make_problem3(8, 20)

    @testset "Basic solve" begin
        res = solve_qubo(A, b, x0, n_bits=4, annealing_time=100,
                         num_reads=3, random_state=42)
        @test length(res.spectrum) == 20
        @test all(res.spectrum .≥ 0)
        @test all(isfinite.(res.spectrum))
        @test res.extra["n_bits"] == 4
        @test isfinite(res.extra["energy"])
    end

    @testset "Binary roundtrip" begin
        x = abs.(randn(MersenneTwister(5), 10))
        mv = maximum(x) * 1.5   # ensures val < 1 → strictly binary bits
        binary = spectrum_to_binary(x; n_bits=6, max_value=mv)
        @test length(binary) == 60
        @test all(b -> b in (0, 1), binary)
        x_back = binary_to_spectrum(binary, 10; n_bits=6, max_value=mv)
        # Quantization (truncation) error ≤ max_value / 2^n_bits per bin
        @test all(abs.(x_back .- x) .≤ mv / 2^6 + 1e-12)
        # Edge case val = 1.0: the Python-compatible "digit 2",
        # decoded exactly to max_value
        edge = spectrum_to_binary([1.0]; n_bits=4, max_value=1.0)
        @test edge[1] == 2
        @test binary_to_spectrum(edge, 1; n_bits=4, max_value=1.0)[1] ≈ 1.0
    end

    @testset "Reproducibility with seed" begin
        r1 = solve_qubo(A, b, x0, n_bits=3, annealing_time=50, num_reads=2,
                        random_state=99)
        r2 = solve_qubo(A, b, x0, n_bits=3, annealing_time=50, num_reads=2,
                        random_state=99)
        @test r1.spectrum ≈ r2.spectrum
    end

    @testset "Explicit max_value bounds spectrum" begin
        res = solve_qubo(A, b, x0, n_bits=4, max_value=0.5,
                         annealing_time=50, num_reads=2, random_state=1)
        @test all(res.spectrum .≤ 0.5 + 1e-12)
    end

    @testset "Validation" begin
        @test_throws ArgumentError solve_qubo(A, b, x0, n_bits=0)
    end

    @testset "unfold_qubo Detector wrapper" begin
        n = 20
        names = ["a", "b", "c", "d", "e"]
        rng = MersenneTwister(123)
        E_MeV = collect(range(1e-6, 10.0, length=n))
        sens = Dict(nm => rand(rng, n) .+ 0.1 for nm in names)
        cc = Dict(nm => rand(rng, n) for nm in names)
        d = Detector(names, E_MeV, sens, cc)
        true_spec = exp.(-E_MeV ./ 1.5)
        readings = Dict(nm => sum(sens[nm] .* true_spec) for nm in names)
        result = unfold_qubo(d, readings, n_bits=3, annealing_time=50,
                             num_reads=2, random_state=42)
        @test haskey(result, "spectrum")
        @test length(result["spectrum"]) == n
        @test result["method"] == "QUBO-Annealing"
    end
end


@testset "Batch3 — MCMC (Turing.jl)" begin
    A, b, x0, x_true = _make_problem3(8, 20)

    # Determine the availability of Turing in the current environment
    turing_available = try
        Base.eval(Main, :(using Turing))
        true
    catch
        false
    end

    if turing_available
        @testset "Real sampling (small problem)" begin
            res = solve_mcmc(A, b, x0, n_samples=200, tune=100, chains=1,
                             random_state=42)
            @test length(res.spectrum) == 20
            @test all(res.spectrum .≥ 0)   # exp(theta) > 0 by construction
            @test all(isfinite.(res.spectrum))
            @test res.converged
            for key in ("samples", "mean", "median", "std", "hpd_lower",
                        "hpd_upper", "rhat_max")
                @test haskey(res.extra, key)
            end
            @test size(res.extra["samples"]) == (200, 20)
            @test all(res.extra["hpd_lower"] .≤ res.extra["hpd_upper"])
            @test res.extra["hpd_lower"] ≈ res.extra["hpd_lower"]  # sanity
        end

        @testset "Reproducibility with seed" begin
            r1 = solve_mcmc(A, b, x0, n_samples=100, tune=50, chains=1,
                            random_state=7)
            r2 = solve_mcmc(A, b, x0, n_samples=100, tune=50, chains=1,
                            random_state=7)
            @test r1.spectrum ≈ r2.spectrum
        end

        @testset "Hierarchical noise option" begin
            res = solve_mcmc(A, b, x0, n_samples=100, tune=50, chains=1,
                             use_hierarchical=true, random_state=3)
            @test all(res.spectrum .≥ 0)
        end
    else
        @testset "Graceful degradation without Turing" begin
            res = solve_mcmc(A, b, x0, n_samples=10, tune=10, chains=1)
            @test typeof(res) <: UnfoldResult
            @test length(res.spectrum) == 20
            @test all(res.spectrum .== 0)
            @test !res.converged
            @test haskey(res.extra, "error")
        end
    end

    @testset "Input validation" begin
        @test_throws ArgumentError solve_mcmc(A, b[1:5], x0)
    end

    @testset "HPD interval helper" begin
        samples = randn(MersenneTwister(11), 1000, 3)
        lo, hi = BSSUnfold._hpd_interval(samples, 0.95)
        @test all(lo .< hi)
        # HPD for the standard normal ≈ ±1.96
        @test all(abs.(hi .+ lo) .< 0.3)
    end

    @testset "Split R-hat helper" begin
        # Identical chains → R̂ ≈ 1
        rng = MersenneTwister(12)
        samples = repeat(randn(rng, 100, 2), 2, 1)
        rhat = BSSUnfold._split_rhat(samples, 2)
        @test all(rhat .< 1.5)
        # Diverged chains → R̂ >> 1 (shift only the second chain)
        shifted = copy(samples)
        shifted[101:200, 2] .+= 10.0
        rhat2 = BSSUnfold._split_rhat(shifted, 2)
        @test rhat2[2] > 2
    end
end
