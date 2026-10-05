# Tests for the ODL advanced unfolding ports: Chambolle–Pock PDHG and
# Douglas–Rachford splitting, both with optional 1-D TV regularization.
# Port of bssunfold/core/unfold_odl_advanced.py (pure-NumPy ODL scheme).

using Test
using BSSUnfold
using LinearAlgebra
using Random
using Statistics

# Small well-conditioned synthetic unfolding problem (b floored at 1e-3 to
# match the parity harness; x0 is the framework default uniform 0.5 guess).
function _make_odl_problem(m::Int=12, n::Int=40; seed::Int=42)
    rng = MersenneTwister(seed)
    A = rand(rng, m, n) .+ 0.3
    A ./= sum(A, dims=2)
    x_true = exp.(-collect(range(0, 4, length=n)))
    b = A * x_true
    b = max.(b, 1e-3)
    x0 = 0.5 .* ones(n)
    return A, b, x0, x_true
end


@testset "Batch7 — ODL PDHG" begin
    A, b, x0, x_true = _make_odl_problem()

    @testset "Basic solve" begin
        res = solve_odl_pdhg(A, b, x0; max_iterations=100)
        @test length(res.spectrum) == 40
        @test all(res.spectrum .≥ 0)
        @test all(isfinite.(res.spectrum))
        @test res.iterations == 100
        # PDHG on the augmented quadratic must beat the uniform start
        @test norm(A * res.spectrum .- b) < norm(A * x0 .- b)
    end

    @testset "Extra diagnostics" begin
        res = solve_odl_pdhg(A, b, x0)
        @test haskey(res.extra, "use_tv")
        @test haskey(res.extra, "tv_weight")
        @test haskey(res.extra, "tau")
        @test haskey(res.extra, "sigma")
        @test res.extra["max_iterations"] == 100
        @test res.extra["use_tv"]
        @test res.extra["tv_weight"] == 0.1
        @test res.extra["tau"] > 0 && res.extra["sigma"] > 0
        # automatic steps: tau = sigma = 0.99 / op_norm
        @test res.extra["tau"] ≈ res.extra["sigma"]
    end

    @testset "TV changes the solution" begin
        res_tv = solve_odl_pdhg(A, b, x0; use_tv=true)
        res_no = solve_odl_pdhg(A, b, x0; use_tv=false)
        @test !(res_no.spectrum ≈ res_tv.spectrum)
    end

    @testset "tv_weight = 0 behaves like plain PDHG" begin
        res_no = solve_odl_pdhg(A, b, x0; use_tv=false)
        res_w0 = solve_odl_pdhg(A, b, x0; tv_weight=0.0)
        @test res_w0.spectrum ≈ res_no.spectrum atol = 1e-12
    end

    @testset "tau / sigma overrides are honoured" begin
        res_auto = solve_odl_pdhg(A, b, x0)
        auto = res_auto.extra["tau"]
        res_ov = solve_odl_pdhg(A, b, x0; tau=0.5 * auto, sigma=0.5 * auto)
        @test res_ov.extra["tau"] ≈ 0.5 * auto
        @test res_ov.extra["sigma"] ≈ 0.5 * auto
        # a smaller fixed step slows the 100-iteration trajectory
        @test !(res_ov.spectrum ≈ res_auto.spectrum)
    end

    @testset "nonnegativity off keeps a finite signed spectrum" begin
        res = solve_odl_pdhg(A, b, x0; nonnegativity=false, use_tv=false)
        @test all(isfinite.(res.spectrum))
    end
end


@testset "Batch7 — ODL Douglas-Rachford" begin
    A, b, x0, x_true = _make_odl_problem()

    @testset "Basic solve" begin
        res = solve_odl_douglas_rachford(A, b, x0; max_iterations=100)
        @test length(res.spectrum) == 40
        @test all(res.spectrum .≥ 0)
        @test all(isfinite.(res.spectrum))
        @test res.iterations == 100
        @test norm(A * res.spectrum .- b) < norm(A * x0 .- b)
    end

    @testset "Extra diagnostics" begin
        res = solve_odl_douglas_rachford(A, b, x0)
        @test res.extra["gamma"] == 1.0
        @test res.extra["max_iterations"] == 100
        @test res.extra["use_tv"]
        @test res.extra["tv_weight"] == 0.1
    end

    @testset "TV changes the solution" begin
        res_tv = solve_odl_douglas_rachford(A, b, x0; use_tv=true)
        res_no = solve_odl_douglas_rachford(A, b, x0; use_tv=false)
        @test !(res_no.spectrum ≈ res_tv.spectrum)
    end

    @testset "tv_weight = 0 disables the TV prox" begin
        res_no = solve_odl_douglas_rachford(A, b, x0; use_tv=false)
        res_w0 = solve_odl_douglas_rachford(A, b, x0; tv_weight=0.0)
        @test res_w0.spectrum ≈ res_no.spectrum atol = 1e-12
    end

    @testset "nonnegativity off keeps a finite signed spectrum" begin
        res = solve_odl_douglas_rachford(A, b, x0; nonnegativity=false)
        @test all(isfinite.(res.spectrum))
    end
end
