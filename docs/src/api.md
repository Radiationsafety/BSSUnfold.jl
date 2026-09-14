# API Reference

```@meta
CurrentModule = BSSUnfold
```

## Типы

```@docs
UnfoldResult
Detector
DetectorConfig
```

## Алгоритмы развёртки

```@docs
solve_mlem
solve_gravel
solve_landweber
solve_maxed
solve_tikhonov
solve_tsvd
solve_sandii
solve_bunki
solve_kaczmarz
solve_cgls
solve_fista
solve_bsrem
solve_osem
solve_staysl
solve_doroshenko
```

## Detector-методы (высокий уровень)

```@docs
unfold_mlem
unfold_gravel
unfold_landweber
unfold_maxed
unfold_tikhonov
unfold_tsvd
unfold_sandii
unfold_bunki
unfold_kaczmarz
unfold_cgls
unfold_fista
unfold_bsrem
unfold_osem
unfold_staysl
unfold_doroshenko
```

## Framework

```@docs
run_unfolding
make_solve_wrapper
build_system
normalize_initial
validate_system
standardize_output
```

## Monte-Carlo неопределённость

```@docs
monte_carlo_uncertainty
add_noise
```

## Регуляризация

```@docs
select_regularization_parameter
lcurve_selection
gcv_selection
```

## Detector utilities

```@docs
save_result!
n_energy_bins
energy_grid
detector_names
```
