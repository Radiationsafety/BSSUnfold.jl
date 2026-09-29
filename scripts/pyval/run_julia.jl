# Run BSSUnfold.jl solvers on the shared problem.
# Usage: julia --project=. run_julia.jl <problem.json> <outdir>

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
E = Float64.(problem["E_MeV"])
le = log10.(E)
log_steps = Vector{Float64}(undef, length(E))
log_steps[1] = le[2] - le[1]
log_steps[end] = le[end] - le[end-1]
for j in 2:length(E)-1
    log_steps[j] = (le[j+1] - le[j-1]) / 2
end
ln_steps = log_steps .* log(10.0)

specs = [
    "mlem" => () -> solve_mlem(A, b, x0),
    "gravel" => () -> solve_gravel(A, b, x0),
    "landweber" => () -> solve_landweber(A, b, x0),
    "maxed" => () -> solve_maxed(A, b, x0),
    "bunki" => () -> solve_bunki(A, b, x0),
    "bunkiut" => () -> solve_bunkiut(A, b, x0),
    "rebunki" => () -> solve_rebunki(A, b, x0),
    "nsduaz" => () -> solve_nsduaz(A, b, x0),
    "sandii" => () -> solve_sandii(A, b, x0),
    "osem" => () -> solve_osem(A, b, x0),
    "bsrem" => () -> solve_bsrem(A, b, x0),
    "cgls" => () -> solve_cgls(A, b, x0),
    "kaczmarz" => () -> solve_kaczmarz(A, b, x0),
    "randomized_kaczmarz" => () -> solve_randomized_kaczmarz(A, b, x0),
    "iterative_refinement" => () -> solve_iterative_refinement(A, b, x0),
    "lanczos" => () -> solve_lanczos(A, b, x0),
    "doroshenko" => () -> solve_doroshenko(A, b, x0),
    "staysl" => () -> solve_staysl(A, b, x0),
    "sart" => () -> solve_sart(A, b, x0),
    "mapem" => () -> solve_mapem(A, b, x0),
    "mlem_stop" => () -> solve_mlem_stop(A, b, x0),
    "amaxed" => () -> solve_amaxed(A, b, x0),
    "amaxed_regularization" => () -> solve_amaxed_regularization(A, b, x0),
    "imaxed" => () -> solve_imaxed(A, b, x0),
    "bayes" => () -> solve_bayes(A, b, x0),
    "bayes_spline" => () -> solve_bayes_spline(A, b, x0),
    "directed_divergence" => () -> solve_directed_divergence(A, b, x0),
    "ferdor" => () -> solve_ferdor(A, b, x0),
    "gks" => () -> solve_gks(A, b, x0),
    "tikhonov_tv" => () -> solve_tikhonov_tv(A, b, x0),
    "tikhonov_legendre" => () -> solve_tikhonov_legendre(A, b, x0),
    "tsvd" => () -> solve_tsvd(A, b, x0),
    "scipy_direct" => () -> solve_scipy_direct(A, b, x0),
    "statreg" => () -> solve_statreg(A, b, x0; E_MeV=E),
    "reconst" => () -> solve_reconst(A, b, x0; E_MeV=E),
    "express" => () -> solve_express(A, b, x0),
    "eki" => () -> solve_eki(A, b, x0),
    "ensemble" => () -> solve_ensemble(A, b, x0),
    "crystal_ball" => () -> solve_crystal_ball(A, b, x0),
    "cvxpy" => () -> solve_cvxpy(A, b, x0; regularization=0.01),
    "qpsolvers" => () -> solve_qpsolvers(A, b, x0; regularization=0.01),
    "cs" => () -> solve_cs(A, b, x0),
    "nspline" => () -> solve_nspline(A, b, x0; E_MeV=E),
    "nspline_full" => () -> solve_nspline_full(A, b, x0, E),
    "hybrid_parametric" => () -> solve_hybrid_parametric(A, b, x0),
    "parametric" => () -> solve_parametric(A, b, x0; E_MeV=E),
    "parametric2" => () -> solve_parametric2(A, b; E=E, ln_steps=ln_steps),
    "genetic" => () -> solve_genetic(A, b, x0),
    "maeo" => () -> solve_maeo(A, b, x0),
    "maeo_ensemble" => () -> solve_maeo_ensemble(A, b, x0),
    "qubo" => () -> solve_qubo(A, b, x0),
    "nnksvd" => () -> solve_nnksvd(A, b, x0),
    "omp" => () -> solve_omp(A, b, 5),
    "nn_omp" => () -> solve_nn_omp(A, b, 5),
    "nnls_topk" => () -> solve_nnls_topk(A, b, 5),
    "tikhonov_nnls" => () -> solve_tikhonov_nnls(A, b),
    "sl0" => () -> solve_sl0(A, b),
    # Julia-only methods (no matrix-level Python entry point in 0.28.0):
    "fista" => () -> solve_fista(A, b, x0),
    "tikhonov" => () -> solve_tikhonov(A, b, x0),
    "hybrid_gmres" => () -> solve_hybrid_gmres(A, b, x0),
    "direct" => () -> solve_direct(A, b, x0),
]

for (name, f) in specs
    run_one(name, f)
end
println("julia done: ", length(specs), " methods")
