# Tests for Detector (port of tests/test_detector.py)
using Test
using BSSUnfold
using LinearAlgebra
using Random

@testset "Detector — initialization" begin
    n = 100
    names = ["sphere_0", "sphere_2", "sphere_3", "sphere_5",
                      "sphere_8", "sphere_10", "sphere_12"]
    m = length(names)
    E_MeV = collect(range(1e-6, 20.0, length=n))
    sensitivities = Dict(name => rand(n) for name in names)
    cc_icrp116 = Dict(name => rand(n) for name in names)

    d = Detector(names, E_MeV, sensitivities, cc_icrp116)

    @test length(d) == m
    @test n_energy_bins(d) == n
    @test length(energy_grid(d)) == n
    @test length(d.config.detector_names) == m
    @test isempty(d.results_history)
end

@testset "Detector — wrong dimensions" begin
    n = 100
    detector_names = ["a", "b"]
    E_MeV = collect(range(1e-6, 20.0, length=n))
    # Sensitivities wrong length
    sensitivities = Dict("a" => rand(50), "b" => rand(50))
    cc_icrp116 = Dict("a" => rand(n), "b" => rand(n))
    @test_throws AssertionError Detector(detector_names, E_MeV, sensitivities, cc_icrp116)
end

@testset "Detector — unfold_mlem returns proper output" begin
    n = 100
    detector_names = ["sphere_0", "sphere_2", "sphere_3", "sphere_5",
                      "sphere_8", "sphere_10", "sphere_12"]
    rng = MersenneTwister(42)
    E_MeV = collect(range(1e-6, 20.0, length=n))
    sensitivities = Dict(name => rand(rng, n) for name in detector_names)
    cc_icrp116 = Dict(name => rand(rng, n) for name in detector_names)

    d = Detector(detector_names, E_MeV, sensitivities, cc_icrp116)

    # Create realistic readings
    true_spectrum = exp.(-E_MeV ./ 2.0)
    readings = Dict(name => sum(sensitivities[name] .* true_spectrum)
                   for name in detector_names)

    result = unfold_mlem(d, readings, max_iterations=500)

    @test result isa Dict{String,Any}
    @test haskey(result, "spectrum")
    @test haskey(result, "method")
    @test haskey(result, "energy")
    @test haskey(result, "residual")
    @test haskey(result, "residual_norm")
    @test haskey(result, "iterations")
    @test haskey(result, "converged")
    @test haskey(result, "effective_readings")
    @test haskey(result, "doserates")
    @test result["method"] == "MLEM"
    @test length(result["spectrum"]) == n
    @test all(result["spectrum"] .≥ 0)
end

@testset "Detector — all unfold_* methods" begin
    n = 50
    detector_names = ["a", "b", "c", "d"]
    rng = MersenneTwister(123)
    E_MeV = collect(range(1e-6, 10.0, length=n))
    sensitivities = Dict(name => rand(rng, n) .+ 0.1 for name in detector_names)
    cc_icrp116 = Dict(name => rand(rng, n) for name in detector_names)

    d = Detector(detector_names, E_MeV, sensitivities, cc_icrp116)

    true_spectrum = exp.(-E_MeV ./ 1.5)
    readings = Dict(name => sum(sensitivities[name] .* true_spectrum)
                   for name in detector_names)

    # All methods must return a Dict with the right keys
    for unfold_fn in [unfold_mlem, unfold_gravel, unfold_landweber,
                     unfold_maxed, unfold_tikhonov, unfold_tsvd,
                     unfold_sandii, unfold_bunki, unfold_kaczmarz,
                     unfold_cgls, unfold_fista, unfold_bsrem,
                     unfold_osem, unfold_staysl, unfold_doroshenko]
        result = unfold_fn(d, readings, max_iterations=100)
        @test result isa Dict{String,Any}
        @test haskey(result, "spectrum")
        @test length(result["spectrum"]) == n
        @test all(isfinite.(result["spectrum"]))
    end
end

@testset "Detector — save_result callback" begin
    n = 20
    detector_names = ["a", "b", "c"]
    rng = MersenneTwister(99)
    E_MeV = collect(range(1e-6, 10.0, length=n))
    sensitivities = Dict(name => rand(rng, n) for name in detector_names)
    cc_icrp116 = Dict(name => rand(rng, n) for name in detector_names)
    d = Detector(detector_names, E_MeV, sensitivities, cc_icrp116)

    readings = Dict(name => rand() for name in detector_names)
    result = unfold_mlem(d, readings, max_iterations=10,
                        save_result=r -> save_result!(d, r))
    @test length(d.results_history) == 1
    @test d.results_history[1]["method"] == "MLEM"
end
