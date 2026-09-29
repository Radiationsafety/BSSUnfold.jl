# Run BSSUnfold.jl solve_eki (seeded) on the shared problem.
# Usage: julia --project=. run_julia_b1.jl <problem.json> <outdir> [seed]

using BSSUnfold
using JSON
using LinearAlgebra
using Random

const DIR = ARGS[2]
SEED = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 7

problem = JSON.parse(read(open(ARGS[1]), String))
A = hcat((Float64.(r) for r in problem["A"])...)'
b = Float64.(problem["b"])
x0 = Float64.(problem["x0"])

res = solve_eki(A, b, x0; random_state=SEED)
mkpath(DIR)
open(joinpath(DIR, "eki.json"), "w") do io
    JSON.print(io, Dict(
        "spectrum" => Vector{Float64}(res.spectrum),
        "iterations" => res.iterations,
        "converged" => res.converged,
    ))
end
println("julia b1: eki ok (seed $SEED)")
