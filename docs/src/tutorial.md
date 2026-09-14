# Tutorial

## 1. Установка

```julia
using Pkg
Pkg.add("BSSUnfold")
```

## 2. Создание Detector

`Detector` — это структура, инкапсулирующая конфигурацию BSS:
имена сфер, энергетическую сетку, response functions и коэффициенты ICRP-116.

```julia
using BSSUnfold

# Энергетическая сетка (логарифмическая, 100 бинов от 1e-9 до 20 МэВ)
E_MeV = 10 .^ range(-9, log10(20), length=100)

# Имена сфер
detector_names = ["sphere_0in", "sphere_2in", "sphere_3in",
                  "sphere_5in", "sphere_8in", "sphere_10in", "sphere_12in"]

# Response functions (должны быть загружены из файла или вычислены)
sensitivities = Dict(name => load_response_function(name) for name in detector_names)

# ICRP-116 коэффициенты для дозы
cc_icrp116 = Dict(name => load_icrp116(name) for name in detector_names)

# Создание детектора
detector = Detector(detector_names, E_MeV, sensitivities, cc_icrp116)
```

## 3. Развёртка спектра

```julia
# Показания детекторов (из реальных измерений)
readings = Dict(
    "sphere_0in"  => 0.001,
    "sphere_2in"  => 0.012,
    "sphere_3in"  => 0.054,
    "sphere_5in"  => 0.184,
    "sphere_8in"  => 0.220,
    "sphere_10in" => 0.158,
    "sphere_12in" => 0.087,
)

# Запуск развёртки GRAVEL
result = unfold_gravel(detector, readings, max_iterations=500)

println("Метод:     \$(result["method"])")
println("Итераций:  \$(result["iterations"])")
println("Сходился:  \$(result["converged"])")
println("||b-Ax||:  \$(result["residual_norm"])")
```

## 4. Оценка неопределённости

```julia
result = unfold_gravel(detector, readings,
                     max_iterations=500,
                     calculate_errors=true,
                     noise_level=0.01,
                     n_montecarlo=100)

# Доступ к MC-результатам
mean_spectrum = result["spectrum_uncert_mean"]
std_spectrum  = result["spectrum_uncert_std"]
p5            = result["spectrum_uncert_p5"]
p95           = result["spectrum_uncert_p95"]
```

## 5. Сравнение алгоритмов

```julia
# Запуск нескольких методов на одной задаче
for unfold_fn in [unfold_mlem, unfold_gravel, unfold_tikhonov, unfold_cgls]
    result = unfold_fn(detector, readings, max_iterations=500)
    println("\$(result["method"]): cos=\$(cos_sim(result["spectrum"], x_true))")
end
```

## 6. Регуляризация

```julia
# Автоматический выбор λ через GCV
A, b, _ = build_system(readings, detector_names, sensitivities)
x0 = ones(length(E_MeV)) * 0.5

result = select_regularization_parameter(A, b, x0, method=:gcv)
λ = result.lambda
println("Оптимальная λ = \$λ")

# Решение Tikhonov с выбранной λ
res = solve_tikhonov(A, b, x0, regularization=λ)
```

## 7. Визуализация

```julia
using Plots

plot(E_MeV, result["spectrum"],
     xscale=:log10, yscale=:log10,
     label="GRAVEL",
     xlabel="Энергия, МэВ", ylabel="Φ(E)",
     title="Развёрнутый спектр")
```
