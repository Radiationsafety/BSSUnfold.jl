# Документация BSSUnfold.jl

Эта директория содержит исходники документации, генерируемой через
[Documenter.jl](https://documenter.juliadocs.org/).

## Локальная сборка

```bash
cd docs
julia --project=. -e 'using Pkg; Pkg.develop(path=".."); Pkg.instantiate()'
julia --project=. make.jl
```

После сборки откройте `docs/build/index.html` в браузере.

## Структура

```
docs/
├── make.jl         ← точка входа Documenter
├── Project.toml    ← зависимости docs
└── src/
    ├── index.md    ← главная страница
    ├── api.md      ← автогенерируемый API
    ├── tutorial.md ← краткий tutorial
    └── assets/     ← изображения и пр.
```

## Публикация на GitHub Pages

Документация автоматически публикуется через GitHub Actions при push в `main`.
URL: `https://radiationsafety.github.io/BSSUnfold.jl/stable/`
