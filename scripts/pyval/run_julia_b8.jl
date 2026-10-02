# Julia side of batch 8 (bucket-C: docplex / scip / commercial / interval /
# nnqp / qpmad — optional JuMP backend). Requires the JuMP+HiGHS environment.
# Usage: julia --project=env/jump run_julia_b8.jl <problem.json> <outdir>

using BSSUnfold
using JSON
using LinearAlgebra
using Printf

const problem = JSON.parse(read(open(ARGS[1]), String))
const DIR = ARGS[2]
const A = hcat((Float64.(r) for r in problem["A"])...)'
const b = Float64.(problem["b"])
const x0 = Float64.(problem["x0"])

mkpath(DIR)

function dump_result(name::String, r)
    chi2 = norm(A * r.spectrum - b)^2 / norm(b)^2
    @printf("%-22s iters=%5d converged=%-5s chi2=%.6g\n",
            name, r.iterations, string(r.converged), chi2)
    payload = Dict("spectrum" => Vector{Float64}(r.spectrum),
                   "chi2" => chi2,
                   "iterations" => r.iterations,
                   "converged" => r.converged)
    for k in ("status", "engine", "spectrum_lower", "spectrum_upper",
              "spectrum_mid", "interval_width")
        haskey(r.extra, k) || continue
        v = r.extra[k]
        payload[k] = v isa AbstractVector ? Vector{Float64}(v) : v
    end
    JSON.print(open(joinpath(DIR, name * ".json"), "w"), payload)
end

function run(name::String, f; kwargs...)
    payload = try
        r = f(A, b, copy(x0); kwargs...)
        dump_result(name, r)
    catch err
        @printf("%-22s ERROR %s\n", name, err)
        JSON.print(open(joinpath(DIR, name * ".json"), "w"),
                   Dict("error" => sprint(showerror, err)))
    end
end

# Docplex / SCIP / commercial all describe the same canonical QP.
run("docplex_l2",   solve_docplex;   regularization=1e-4, norm=2)
run("docplex_l1",   solve_docplex;   regularization=1e-4, norm=1)
run("docplex_sm1",  solve_docplex;   regularization=1e-3, norm=2, smoothness_order=1, smoothness_weight=1e-2)
run("docplex_sm2",  solve_docplex;   regularization=1e-3, norm=2, smoothness_order=2, smoothness_weight=1e-2)
run("scip_l2",      solve_scip;      regularization=1e-4, norm=2)
run("scip_l1",      solve_scip;      regularization=1e-4, norm=1)
run("commercial_highs", solve_commercial; solver=:gurobi, regularization=1e-4, norm=2)

# nnqp / qpmad share the box-QP kernel.
run("nnqp",         solve_nnqp;      regularization=1e-4)
run("nnqp_sm1",     solve_nnqp;      regularization=1e-3, smoothness_order=1, smoothness_weight=1e-2)
run("qpmad",        solve_qpmad;     regularization=1e-4)
run("qpmad_box",    solve_qpmad;     regularization=1e-4, lb=fill(0.0, length(x0)), ub=fill(5.0, length(x0)))

# Interval LP family (n ≤ 80 for runtime).
if length(x0) <= 80
    run("interval_tv",   solve_interval;        tv_bound=1.0, noise_level=0.02)
    run("interval_flat", solve_interval;        tv_bound=1e6, noise_level=0.02)
    run("interval_tol",  solve_interval_tol;    noise_level=0.02, max_iterations=500)
    run("interval_post", solve_interval_posterior; noise_level=0.02)
end

println("julia b8 done")
