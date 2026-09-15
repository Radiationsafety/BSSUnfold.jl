# BSSUnfold.jl documentation

This directory contains the documentation sources built with
[Documenter.jl](https://documenter.juliadocs.org/).

## Local build

```bash
cd docs
julia --project=. -e 'using Pkg; Pkg.develop(path=".."); Pkg.instantiate()'
julia --project=. make.jl
```

After the build, open `docs/build/index.html` in a browser.

## Structure

```
docs/
├── make.jl         ← Documenter entry point
├── Project.toml    ← docs dependencies
└── src/
    ├── index.md    ← landing page
    ├── api.md      ← auto-generated API reference
    ├── tutorial.md ← short tutorial
    └── assets/     ← images, etc.
```

### Note

The docs no longer list a CI badge: the GitHub Actions CI workflow was
removed from the repository history; documentation is built and deployed
locally or via Documenter's `deploydocs`.

## Publishing to GitHub Pages

Documentation is published automatically via GitHub Actions on push to
`main` (if the workflow is present).
URL: `https://radiationsafety.github.io/BSSUnfold.jl/stable/`
