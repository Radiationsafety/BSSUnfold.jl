# Tutorial

## 1. Installation

```julia
using Pkg
Pkg.add("BSSUnfold")
# or, until the package is registered in General:
# Pkg.add(url="https://github.com/Radiationsafety/BSSUnfold.jl")
```

## 2. Creating Detector

`Detector` is a struct that encapsulates the BSS configuration: sphere names,
energy grid, response functions and ICRP-116 dose coefficients.

```julia
using BSSUnfold

# Energy grid (logarithmic, 100 bins from 1e-9 to 20 MeV)
E_MeV = 10 .^ range(-9, log10(20), length=100)

# Sphere names
detector_names = ["sphere_0in", "sphere_2in", "sphere_3in",
                  "sphere_5in", "sphere_8in", "sphere_10in", "sphere_12in"]

# Response functions (must be loaded from a file or computed)
sensitivities = Dict(name => load_response_function(name) for name in detector_names)

# ICRP-116 coefficients for dose
cc_icrp116 = Dict(name => load_icrp116(name) for name in detector_names)

# Create the detector
detector = Detector(detector_names, E_MeV, sensitivities, cc_icrp116)
```

For real measurements you can use the built-in response functions shipped
with the package (`RF_GSF`, `RF_PTB`, `RF_LANL`, `RF_JINR`, `RF_FERMILAB`,
`RF_EURADOS`, `RF_IHEP`):

```julia
detector = Detector(RF_GSF)             # ICRP-116 by default
detector = Detector(RF_PTB; cc_type="ICRP74_operational")
```

or the zero-argument default constructor (RF_GSF):

```julia
detector = Detector()
```

## 3. Unfolding a spectrum

```julia
# Detector readings (from real measurements)
readings = Dict(
    "sphere_0in"  => 0.001,
    "sphere_2in"  => 0.012,
    "sphere_3in"  => 0.054,
    "sphere_5in"  => 0.184,
    "sphere_8in"  => 0.220,
    "sphere_10in" => 0.158,
    "sphere_12in" => 0.087,
)

# Run GRAVEL unfolding
result = unfold_gravel(detector, readings, max_iterations=500)

println("Method:     \$(result["method"])")
println("Iterations: \$(result["iterations"])")
println("Converged:  \$(result["converged"])")
println("||b-Ax||:   \$(result["residual_norm"])")
```

## 4. Uncertainty estimation

```julia
result = unfold_gravel(detector, readings,
                     max_iterations=500,
                     calculate_errors=true,
                     noise_level=0.01,
                     n_montecarlo=100)

# Access MC results
mean_spectrum = result["spectrum_uncert_mean"]
std_spectrum  = result["spectrum_uncert_std"]
p5            = result["spectrum_uncert_p5"]
p95           = result["spectrum_uncert_p95"]
```

## 5. Comparing algorithms

```julia
# Run several methods on the same problem
for unfold_fn in [unfold_mlem, unfold_gravel, unfold_tikhonov, unfold_cgls]
    result = unfold_fn(detector, readings, max_iterations=500)
    println("\$(result["method"]): cos=\$(cos_sim(result["spectrum"], x_true))")
end
```

For a systematic comparison use `benchmark_unfold_methods`, which ranks
methods by the 52 metrics from `compare_spectra` (see
`examples/33-methods_comparison.jl`).

## 6. Regularization

```julia
# Automatic λ selection via GCV
A, b, _ = build_system(readings, detector_names, sensitivities)
x0 = ones(length(E_MeV)) * 0.5

result = select_regularization_parameter(A, b, x0, method=:gcv)
λ = result.lambda
println("Optimal λ = \$λ")

# Tikhonov solve with the selected λ
res = solve_tikhonov(A, b, x0, regularization=λ)
```

## 7. Dose rates

```julia
# Convert the unfolded spectrum to dose rates using ICRP-116 coefficients
dose = calculate_dose_rates(detector, readings)
```

## 8. Visualization

```julia
using Plots

plot(E_MeV, result["spectrum"],
     xscale=:log10, yscale=:log10,
     label="GRAVEL",
     xlabel="Energy, MeV", ylabel="Φ(E)",
     title="Unfolded spectrum")
```
