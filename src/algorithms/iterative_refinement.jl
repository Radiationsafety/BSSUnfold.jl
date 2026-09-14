"""
Iterative refinement — двухпроходная развёртка.

Порт из `bssunfold/src/bssunfold/core/unfold_iterative_refinement.py`.

Алгоритм:
1. Первый проход: быстрый EM-метод (MLEM по умолчанию) с малым числом итераций —
   грубая структура спектра.
2. Вычисляется невязка r = b - A*x1.
3. Второй проход: градиентный метод (Landweber по умолчанию) на невязке —
   корректирует систематические ошибки EM-методов.
4. Комбинирование: x_final = x1 + alpha * x2, где alpha выбирается через
   линейный поиск для минимизации ||A*x_final - b||.

Позволяет объединить скорость сходимости EM-методов с точностью
градиентных методов.
"""

"""
    solve_iterative_refinement(A, b, x0; first_pass_solver, second_pass_solver,
                              first_pass_kwargs, second_pass_kwargs,
                              alpha, max_alpha_search)

Двухпроходная развёртка: EM-метод + градиентная коррекция.

# Аргументы
- `A::AbstractMatrix{T}`: response matrix (m × n)
- `b::AbstractVector{T}`: измерения (m,)
- `x0::AbstractVector{T}`: начальный спектр (n,)
- `first_pass_solver`: функция `(A, b, x0; kwargs...) -> UnfoldResult` (по умолч. `solve_mlem`)
- `second_pass_solver`: аналогично (по умолч. `solve_landweber`)
- `first_pass_kwargs`, `second_pass_kwargs`: `NamedTuple` параметров
- `alpha::Union{T,Nothing}`: если `nothing` — линейный поиск
- `max_alpha_search`: число кандидатов alpha в line search

# Возвращает
- `UnfoldResult{T}` со спектром; `extra` содержит диагностику
"""
function solve_iterative_refinement(A::AbstractMatrix{T}, b::AbstractVector{T}, x0::AbstractVector{T};
                                   first_pass_solver::Function=solve_mlem,
                                   second_pass_solver::Function=solve_landweber,
                                   first_pass_kwargs::NamedTuple=(max_iterations=150, tolerance=T(1e-4)),
                                   second_pass_kwargs::NamedTuple=(max_iterations=100, tolerance=T(1e-5)),
                                   alpha::Union{T,Nothing}=nothing,
                                   max_alpha_search::Integer=20,
                                   eps::T=T(1e-30)) where T<:AbstractFloat
    m, n = size(A)

    # --- Первый проход ---
    res1 = first_pass_solver(A, b, x0; first_pass_kwargs...)
    x1 = max.(res1.spectrum, T(0))

    # --- Невязка ---
    r = b .- A * x1

    # --- Второй проход на невязке ---
    x0_zero = zeros(T, n)
    res2 = second_pass_solver(A, r, x0_zero; second_pass_kwargs...)
    x2 = res2.spectrum

    # --- Комбинирование ---
    if alpha !== nothing
        best_alpha = alpha
    else
        # Линейный поиск: минимизировать ||A*(x1 + a*x2) - b||
        candidates = range(T(0), T(2); length=max_alpha_search)
        best_alpha = T(0)
        best_res = norm(A * x1 .- b)
        for a in candidates
            x_cand = x1 .+ a .* x2
            res_cand = norm(A * x_cand .- b)
            if res_cand < best_res
                best_res = res_cand
                best_alpha = a
            end
        end
    end

    spectrum = max.(x1 .+ best_alpha .* x2, T(0))
    residual = b .- A * spectrum

    return UnfoldResult(
        spectrum, res1.iterations + res2.iterations,
        res1.converged && res2.converged,
        norm(residual),
        Dict{String,Any}(
            "first_pass_residual"          => norm(r),
            "second_pass_correction_norm"  => norm(x2),
            "alpha"                        => best_alpha,
            "final_residual"               => norm(residual),
            "first_pass_iterations"        => res1.iterations,
            "second_pass_iterations"       => res2.iterations,
        )
    )
end
