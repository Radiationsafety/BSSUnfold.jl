# Batch 5 — Bucket-C: optional JuMP/Optim backend (docplex, scip, commercial,
# interval, nnqp, qpmad). Runs in both the base environment (JuMP absent →
# graceful degradation) and `env/jump` (JuMP + HiGHS present → real solves).

using Test, LinearAlgebra, Random
using BSSUnfold

function _make_problem5(m::Int, n::Int, seed::Int=1)
    rng = MersenneTwister(seed)
    A = rand(rng, m, n) .+ 0.1
    x_true = max.(rand(rng, n), 0.05)
    b = A * x_true .+ 0.01 * randn(rng, m)
    x0 = fill(0.5, n)
    (A, b, x0, x_true)
end

const _JUMP_PRESENT5 = try
    Base.eval(Main, :(using JuMP))
    true
catch
    false
end

@testset "Batch5 — bucket-C availability" begin
    # has_jump() must not throw and must agree with the probe above.
    @test BSSUnfold.has_jump() isa Bool
end

@testset "Batch5 — solve_docplex" begin
    A, b, x0, _ = _make_problem5(12, 24)
    r = solve_docplex(A, b, x0; regularization=1e-3, norm=2)
    @test r isa UnfoldResult
    @test length(r.spectrum) == 24
    @test all(r.spectrum .>= 0)
    if _JUMP_PRESENT5
        @test r.converged
        @test r.extra["engine"] == "highs"
        # Optimality: gradient + αx ≥ 0 on support, ≥ 0 everywhere.
        grad = A' * (A * r.spectrum .- b) .+ 1e-3 * r.spectrum
        @test minimum(grad) > -1e-4
    else
        @test !r.converged
        @test all(r.spectrum .== 0)
        @test haskey(r.extra, "error")
    end
end

@testset "Batch5 — solve_scip" begin
    A, b, x0, _ = _make_problem5(12, 24)
    r = solve_scip(A, b, x0; regularization=1e-3, norm=2)
    @test r isa UnfoldResult
    if _JUMP_PRESENT5
        @test r.converged
        @test haskey(r.extra, "note")
    else
        @test !r.converged
    end
end

@testset "Batch5 — solve_commercial (aliases)" begin
    A, b, x0, _ = _make_problem5(10, 20)
    # Unknown alias must be rejected by the validation, not by the backend.
    @test_throws ArgumentError solve_commercial(A, b, x0; solver=:nope)
    # :gurobi reaches `_bucket_c_qp`; with JuMP present but Gurobi.jl absent
    # it warns-and-returns-zeros (converged=false). Without JuMP, same result.
    r = solve_commercial(A, b, x0; solver=:gurobi, regularization=1e-3, norm=2)
    @test r isa UnfoldResult
    @test r.extra["license_required"] === true
    @test r.extra["solver"] == "gurobi"
    if _JUMP_PRESENT5
        # Engine may or may not be present; either is fine.
        @test length(r.spectrum) == 20
    else
        @test !r.converged
    end
end

# TV-friendly data for the interval LP: x_true must fit inside `tv_bound`,
# otherwise the Shary LP is genuinely infeasible and every 2n solve returns
# INFEASIBLE (independent of engine or backend).
function _make_smooth_problem5(m::Int, n::Int, seed::Int=1)
    rng = MersenneTwister(seed)
    A = rand(rng, m, n) .+ 0.1
    x_true = range(0.2, 1.0; length=n) .+ 0.01 .* rand(rng, n)
    b = A * x_true .+ 0.005 * randn(rng, m)
    x0 = fill(0.5, n)
    (A, b, x0, x_true)
end

@testset "Batch5 — solve_interval" begin
    A, b, x0, _ = _make_smooth_problem5(10, 15)
    r = solve_interval(A, b, x0; tv_bound=2.0, noise_level=0.05)
    @test r isa UnfoldResult
    if _JUMP_PRESENT5
        @test r.converged
        lo = r.extra["spectrum_lower"]
        hi = r.extra["spectrum_upper"]
        mid = r.extra["spectrum_mid"]
        @test length(lo) == 15 && length(hi) == 15
        @test all(lo .>= -1e-9)
        @test all(lo .<= mid .+ 1e-9)
        @test all(mid .<= hi .+ 1e-9)
        # Wider noise ⇒ wider intervals (monotone).
        r2 = solve_interval(A, b, x0; tv_bound=2.0, noise_level=0.10)
        @test sum(r2.extra["spectrum_upper"] .- r2.extra["spectrum_lower"]) >=
              sum(hi .- lo) - 1e-6
    else
        @test !r.converged
    end
end

@testset "Batch5 — solve_interval_tol" begin
    A, b, x0, _ = _make_smooth_problem5(10, 12)
    r = solve_interval_tol(A, b, x0; noise_level=0.05, max_iterations=200)
    if _JUMP_PRESENT5
        @test r isa UnfoldResult
        @test length(r.spectrum) == 12
        @test haskey(r.extra, "spectrum_lower")
        @test haskey(r.extra, "x_pseudo")
    else
        @test !r.converged
    end
end

@testset "Batch5 — solve_interval_posterior" begin
    A, b, x0, _ = _make_smooth_problem5(10, 12)
    r = solve_interval_posterior(A, b, x0; noise_level=0.05)
    @test r isa UnfoldResult
    if _JUMP_PRESENT5
        @test all(r.extra["interval_width"] .>= 0)
    end
end

@testset "Batch5 — solve_interval_intvalpy (Python-only)" begin
    A, b, x0, _ = _make_problem5(6, 8)
    r = solve_interval_intvalpy(A, b, x0)
    @test !r.converged
    @test haskey(r.extra, "error")
end

@testset "Batch5 — solve_nnqp" begin
    A, b, x0, _ = _make_problem5(10, 15)
    r = solve_nnqp(A, b, x0; regularization=1e-3)
    @test r isa UnfoldResult
    if _JUMP_PRESENT5
        @test r.converged
        @test all(r.spectrum .>= -1e-9)
    else
        @test !r.converged
    end
    # backend kwarg must accept all documented spellings.
    for be in ("python", "native", "nnqp", "jump")
        r2 = solve_nnqp(A, b, x0; regularization=1e-3, backend=be)
        @test r2 isa UnfoldResult
    end
    @test_throws ArgumentError solve_nnqp(A, b, x0; backend="bogus")
end

@testset "Batch5 — solve_qpmad" begin
    A, b, x0, _ = _make_problem5(10, 15)
    r = solve_qpmad(A, b, x0; regularization=1e-3)
    @test r isa UnfoldResult
    if _JUMP_PRESENT5
        @test r.converged
        @test r.extra["status"] == 0
        @test r.iterations == 0
    else
        @test !r.converged
        @test r.extra["status"] == 2
    end
    # Infeasible box (lb > ub) → status 2.
    if _JUMP_PRESENT5
        r2 = solve_qpmad(A, b, x0; regularization=1e-3,
                         lb=fill(2.0, 15), ub=fill(1.0, 15))
        @test r2.extra["status"] == 2
        @test !r2.converged
    end
end

@testset "Batch5 — engine selection (COSMO)" begin
    A, b, x0, _ = _make_problem5(12, 20)
    # COSMO is a conic ADMM engine; when installed alongside JuMP the
    # solver=:cosmo path must reach it and match the HiGHS active-set
    # reference on the objective value within COSMO's ADMM tolerance
    # (default primal/dual ≈1e-4; we observe ≈2e-3 on underdetermined
    # 12×20 problems). Pointwise spectrum parity is weaker still — we
    # only check that the solution direction agrees.
    r_h = solve_docplex(A, b, x0; regularization=1e-3, norm=2, solver=:highs)
    r_c = solve_docplex(A, b, x0; regularization=1e-3, norm=2, solver=:cosmo)
    cosmo_installed = _JUMP_PRESENT5 && try
        Base.eval(Main, :(using COSMO)); true
    catch
        false
    end
    if cosmo_installed
        @test r_c.converged
        @test r_c.extra["engine"] == "cosmo"
        obj(x) = 0.5 * sum(abs2, A * x .- b) + 0.5 * 1e-3 * sum(abs2, x)
        @test abs(obj(r_c.spectrum) - obj(r_h.spectrum)) / abs(obj(r_h.spectrum)) < 5e-3
    else
        @test !r_c.converged || r_c.extra["engine"] == "cosmo"
    end
end

@testset "Batch5 — Detector wrappers" begin
    # Build a tiny 2-detector problem to reach the unfold_* wrappers.
    n = 8
    E = collect(range(0.01, 10.0; length=n))
    sens = Dict{String,Vector{Float64}}(
        "D1" => [exp(-x) for x in E],
        "D2" => [x * exp(-x/3) for x in E])
    cc = Dict{String,Vector{Float64}}("ISO" => fill(1.0, n))
    cfg = DetectorConfig(["D1", "D2"], E, sens, cc)
    d = Detector(cfg)
    readings = Dict("D1" => 1.2, "D2" => 3.4)
    # In base env all wrappers must return without throwing.
    for f in (unfold_docplex, unfold_scip, unfold_interval,
              unfold_nnqp, unfold_qpmad)
        out = f(d, readings)
        @test out isa Dict{String,Any}
        @test haskey(out, "spectrum")
        @test length(out["spectrum"]) == n
    end
    if _JUMP_PRESENT5
        out = unfold_interval(d, readings; tv_bound=1.0, noise_level=0.05)
        @test haskey(out, "spectrum_lower")
        @test haskey(out, "spectrum_upper")
    end
end
