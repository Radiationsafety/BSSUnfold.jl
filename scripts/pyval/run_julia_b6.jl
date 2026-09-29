# Run BSSUnfold.jl QUBO solver (port of solve_qubo_unfold, defaults).
# Usage: julia --project=. run_julia_b6.jl <problem.json> <outdir> [seed]

using BSSUnfold
using JSON
using LinearAlgebra

function save(name::String, res::UnfoldResult, chi2::Float64)
    mkpath(DIR)
    open(joinpath(DIR, name * ".json"), "w") do io
        JSON.print(io, Dict("spectrum" => Vector{Float64}(res.spectrum),
                            "chi2" => chi2))
    end
end

problem = JSON.parse(read(open(ARGS[1]), String))
A = hcat((Float64.(r) for r in problem["A"])...)'
b = Float64.(problem["b"])
x0 = Float64.(problem["x0"])
const DIR = ARGS[2]
seed = length(ARGS) > 2 ? parse(Int, ARGS[3]) : nothing

res = solve_qubo(A, b, x0; random_state=seed)
chi2 = norm(A * res.spectrum - b)^2 / norm(b)^2
println("qubo seed=$seed chi2=", chi2, " energy=", res.extra["energy"])
save("qubo", res, chi2)
if seed !== nothing
    save("qubo_s$seed", res, chi2)
end
println("julia b6 done (seed=$seed)")
