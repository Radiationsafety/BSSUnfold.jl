# Run BSSUnfold.jl stochastic solvers (eki/genetic/maeo/maeo_ensemble/qubo)
# Usage: julia --project=. run_julia_a6.jl <problem.json> <outdir> [seed]

using BSSUnfold
using JSON
using LinearAlgebra
using Random

const DIR = ARGS[2]
const SEED = length(ARGS) > 2 ? parse(Int, ARGS[3]) : 7

function save(name::String, x::AbstractVector)
    mkpath(DIR)
    open(joinpath(DIR, name * ".json"), "w") do io
        JSON.print(io, Dict("spectrum" => Vector{Float64}(x)))
    end
end

function save_err(name::String, err)
    mkpath(DIR)
    open(joinpath(DIR, name * ".json"), "w") do io
        JSON.print(io, Dict("error" => sprint(showerror, err)))
    end
end

function run_one(name::String, f)
    Random.seed!(SEED)
    try
        res = f()
        x = res isa UnfoldResult ? res.spectrum :
            res isa NamedTuple ? first(res) :
            res isa Tuple ? first(res) :
            res isa AbstractDict ? get(res, "spectrum", get(res, :spectrum, first(values(res)))) :
            res
        save(name, vec(Float64.(x)))
        println("ok   $name")
    catch err
        save_err(name, err)
        println("FAIL $name: ", sprint(showerror, err)[1:min(end, 300)])
    end
end

problem = JSON.parse(read(open(ARGS[1]), String))
A = hcat((Float64.(r) for r in problem["A"])...)'
b = Float64.(problem["b"])
x0 = Float64.(problem["x0"])

specs = [
    "eki" => () -> solve_eki(A, b, x0),
    "genetic" => () -> solve_genetic(A, b, x0),
    "maeo" => () -> solve_maeo(A, b, x0),
    "maeo_ensemble" => () -> solve_maeo_ensemble(A, b, x0),
    "qubo" => () -> solve_qubo(A, b, x0),
]

for (name, f) in specs
    run_one(name, f)
end
println("julia a6 done (seed=$SEED)")
