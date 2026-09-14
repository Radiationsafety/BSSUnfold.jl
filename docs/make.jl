using Documenter
using BSSUnfold

makedocs(
    sitename = "BSSUnfold.jl",
    authors  = "Konstantin Chizhov, Alexei Chizhov, Dmitry Borschev, Maria Akimochkina, Z.ai",
    modules  = [BSSUnfold],
    format   = Documenter.HTML(
        canonical = "https://radiationsafety.github.io/BSSUnfold.jl/stable/",
        edit_link = "main",
        assets    = ["assets/favicon.ico"],
    ),
    pages = [
        "Home"          => "index.md",
        "Tutorial"      => "tutorial.md",
        "Algorithms"    => "algorithms.md",
        "API Reference" => "api.md",
        "Comparison with bssunfold (Python)" => "comparison.md",
        "License"       => "license.md",
    ],
    warnonly = [:missing_docs],
)

deploydocs(
    repo = "github.com/Radiationsafety/BSSUnfold.jl",
    devbranch = "main",
    push_preview = true,
)
