# Run BSSUnfold.jl solvers (bunki, nsduaz) on the shared problem.
# Usage: julia --project=. run_julia_a1.jl <problem.json> <outdir>

using BSSUnfold
using JSON
using LinearAlgebra

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
    try
        res = f()
        x = res isa UnfoldResult ? res.spectrum : first(res)
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
    "bunki" => () -> solve_bunki(A, b, x0),
    "nsduaz" => () -> solve_nsduaz(A, b, x0),
]

for (name, f) in specs
    run_one(name, f)
end
println("julia done: ", length(specs), " methods")
