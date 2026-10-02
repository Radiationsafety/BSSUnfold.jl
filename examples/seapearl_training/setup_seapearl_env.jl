# One-command setup of a single-session Julia 1.10 environment for the
# SeaPearl CP + RL pipeline (BSSUnfold + SeaPearl + the trained heuristic).
#
# Why a script instead of a checked-in Project.toml:
#   * SeaPearl upstream (0.4.x) declares `julia = "1.8 - 1.9"`, so the plain
#     registry version cannot install on Julia 1.10. We add the
#     Radiationsafety compat fork (branch `compat/julia-1.10`, a
#     declaration-only change) via a git URL.
#   * The registry metadata for GPUCompiler 0.17.3 says `julia = "1.6.0-1.9"`,
#     but the actual v0.17.3 tag declares `julia = "1.6"` (caret). Julia 1.10
#     resolves git sources from the repo declaration, so pinning the upstream
#     tag lets CUDA 3.13.1 + Flux 0.12 resolve exactly as they did on 1.9.
#   * Julia 1.10 `Pkg.instantiate` does not honour `[sources]` in a checked-in
#     Project.toml, so the pins are applied programmatically here. The script
#     works on Julia 1.10 and newer.
#
# Usage (from anywhere):
#
#   julia examples/seapearl_training/setup_seapearl_env.jl
#
# After it finishes, run the CP+RL pipeline in the same session:
#
#   julia --project=examples/seapearl_training examples/45-seapearl.jl
#   julia --project=examples/seapearl_training \
#       examples/seapearl_training/eval_seapearl_bss.jl

using Pkg

const ENV_DIR = @__DIR__              # examples/seapearl_training/
const REPO_ROOT = joinpath(ENV_DIR, "..", "..")

Pkg.activate(ENV_DIR)

# Both git pins (and the plain deps) in a single resolve call, so the
# resolver sees the whole constraint set at once.
Pkg.add([
    # Registry says julia "1.6.0-1.9" for 0.17.3; the tag itself says "1.6".
    PackageSpec(url = "https://github.com/JuliaGPU/GPUCompiler.jl.git",
                rev = "v0.17.3"),
    # Compat fork: only the julia declaration is relaxed (1.8 - 1.10),
    # no source changes vs upstream 0.4.5.
    PackageSpec(url = "https://github.com/Radiationsafety/SeaPearl.jl.git",
                rev = "compat/julia-1.10"),
    PackageSpec(name = "Flux"),
    PackageSpec(name = "JSON"),
])

# BSSUnfold itself, from the repository checkout (its JSON compat accepts the
# 0.21.x that SeaPearl pins, so both live in this one environment).
Pkg.develop(path = REPO_ROOT)

Pkg.instantiate()
Pkg.precompile()

println("""
SeaPearl CP+RL environment ready at: $ENV_DIR

Verify in one session:
  julia --project=$ENV_DIR -e 'using BSSUnfold, SeaPearl; \\
      println("CP+RL single session OK: ", seapearl_available())'
""")
