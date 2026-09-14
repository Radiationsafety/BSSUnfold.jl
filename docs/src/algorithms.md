# Algorithms

BSSUnfold.jl реализует 15 алгоритмов развёртки нейтронных спектров. Все они
следуют единому интерфейсу: `solve_<algorithm>(A, b, x0; kwargs...) -> UnfoldResult`.

## Итеративные EM-методы

### MLEM — Maximum Likelihood Expectation Maximization

```math
x_{k+1} = x_k \odot (A^T (b / (A x_k)))
```

- **Сохраняет** неотрицательность
- **Монотонно** увеличивает правдоподобие
- Сходится медленно; обычно 500–5000 итераций

```julia
result = solve_mlem(A, b, x0, max_iterations=2000, tolerance=1e-8)
```

### OSEM — Ordered Subset Expectation Maximization

Группирует измерения по подмножествам (subsets), обновляя $x$ по каждому
подмножеству. Ускоряет сходимость в `n_subsets` раз.

```julia
result = solve_osem(A, b, x0, max_iterations=50, n_subsets=4)
```

### BSREM — Block-Sequential Regularized EM

OSEM со встроенной регуляризацией (по умолчанию $L_2$):

```julia
result = solve_bsrem(A, b, x0, max_iterations=50, n_subsets=4, regularization=1e-3)
```

## Взвешенные методы

### GRAVEL

Взвешенный лог-правдоподобный метод. Популярный алгоритм для BSS.

```math
x_{k+1}[j] = x_k[j] \exp\left(\frac{\sum_i W_{ij} \ln(b_i/(Ax_k)_i)}{\sum_i W_{ij}}\right)
```

```julia
result = solve_gravel(A, b, x0, max_iterations=500, tolerance=1e-8)
```

### MAXED — Maximum Entropy Deconvolution

Максимизирует энтропию Шеннона при ограничениях $Ax = b$.

```julia
result = solve_maxed(A, b, x0, max_iterations=500)
```

## Прямые методы

### Tikhonov regularization

```math
\min_x \|Ax - b\|^2 + \lambda \|Lx\|^2
```

Решается через нормальные уравнения `(A^T A + λI) x = A^T b`.

```julia
result = solve_tikhonov(A, b, x0, regularization=1e-3)
```

### TSVD — Truncated Singular Value Decomposition

Отбрасывает сингулярные числа меньше порога.

```julia
result = solve_tsvd(A, b, x0, truncation_rank=8)
```

## Итеративные методы без регуляризации

### Landweber

```math
x_{k+1} = x_k + \omega A^T (b - A x_k)
```

```julia
result = solve_landweber(A, b, x0, max_iterations=500, omega=0.0)
```

### Kaczmarz

Row-action метод: обновляет $x$ по одной строке $A$ за раз.

```julia
result = solve_kaczmarz(A, b, x0, max_iterations=100)
```

### CGLS — Conjugate Gradient Least Squares

Применение CG к нормальным уравнениям без их явного формирования.

```julia
result = solve_cgls(A, b, x0, max_iterations=200)
```

### FISTA — Fast Iterative Shrinkage-Thresholding

Проксимальный градиент с ускорением $O(1/k^2)$.

```julia
result = solve_fista(A, b, x0, max_iterations=200, regularization=1e-4)
```

## Классические BSS-методы

### Sandii (1970)

Итеративный EM-подобный алгоритм с сохранением интеграла.

```julia
result = solve_sandii(A, b, x0, max_iterations=500)
```

### Bunki

Модифицированный MLEM с коэффициентом релаксации $\alpha$:

```math
x_{k+1} = x_k \cdot (1 + \alpha (A^T (b/Ax) - 1))
```

```julia
result = solve_bunki(A, b, x0, max_iterations=500, alpha=0.8)
```

### Staysl (1982)

Байесовский метод с априорным спектром:

```julia
result = solve_staysl(A, b, x0, max_iterations=500)
```

### Doroshenko (1986)

Итеративный метод с сохранением интеграла.

```julia
result = solve_doroshenko(A, b, x0, max_iterations=500)
```

## Сводная таблица

| Алгоритм  | Тип               | Регуляризация | Скорость  | Точность |
|-----------|-------------------|---------------|-----------|----------|
| MLEM      | EM итеративный    | Нет           | Медленно  | Высокая  |
| OSEM      | EM subset         | Нет           | Быстро    | Средняя  |
| BSREM     | EM subset + reg   | Да            | Быстро    | Высокая  |
| GRAVEL    | Взвешенный        | Слабая        | Средне    | Высокая  |
| MAXED     | Maximum entropy   | Встроенная    | Средне    | Средняя  |
| Tikhonov  | Прямой            | Сильная       | Очень быстро | Низкая |
| TSVD      | Прямой            | Сильная       | Очень быстро | Низкая |
| Landweber | Итеративный       | Нет           | Средне    | Средняя  |
| Kaczmarz  | Row-action        | Нет           | Быстро    | Средняя  |
| CGLS      | CG                | Слабая        | Быстро    | Высокая  |
| FISTA     | Proximal gradient | L1            | Быстро    | Средняя  |
| Sandii    | EM вариант        | Нет           | Средне    | Средняя  |
| Bunki     | EM relaxed        | Нет           | Средне    | Средняя  |
| Staysl   | Bayesian          | Prior         | Средне    | Высокая  |
| Doroshenko| Итеративный       | Нет           | Средне    | Средняя  |
