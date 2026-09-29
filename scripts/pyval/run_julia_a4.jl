# Run BSSUnfold.jl solvers (agent a4: maxed, doroshenko, cgls, lanczos)
# Usage: julia --project=. run_julia_a4.jl <problem.json> <outdir>

using BSSUnfold
using JSON
using LinearAlgebra
using Random

const DIR = ARGS[2]

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
    Random.seed!(7)
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
        println("FAIL $name: ", sprint(showerror, err)[1:min(end, 160)])
    end
end

problem = JSON.parse(read(open(ARGS[1]), String))
A = hcat((Float64.(r) for r in problem["A"])...)'
b = Float64.(problem["b"])
x0 = Float64.(problem["x0"])

specs = [
    "maxed" => () -> solve_maxed(A, b, x0),
    "doroshenko" => () -> solve_doroshenko(A, b, x0),
    "cgls" => () -> solve_cgls(A, b, x0),
    "lanczos" => () -> solve_lanczos(A, b, x0),
]

for (name, f) in specs
    run_one(name, f)
end
println("julia done: ", length(specs), " methods")
