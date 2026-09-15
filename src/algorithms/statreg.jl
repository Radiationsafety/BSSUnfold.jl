"""
Statistical regularization — метод Тургина (Turchin), порт `solve_statreg`.

    φ̂ = argmin { ½‖Σ⁻¹⸍²(Aφ−b)‖² + ½ α ‖D₂ φ‖² }

`D₂` — оператор второй конечной разности. Параметр `α` задаётся
пользователем (`unfoldermethod = "User"`) или подбирается автоматически
по L-кривой (максимальная кривизна границы log‖residual‖ vs log‖‖D₂φ‖‖;
`"EmpiricalBayes"`).
"""

function _statreg_d2(n::Integer)
    n <= 2 && return zeros(0, n)
    L = zeros(Float64, n - 2, n)
    for j in 1:n - 2
        L[j, j] = 1.0
        L[j, j + 1] = -2.0
        L[j, j + 2] = 1.0
    end
    return L
end

function _statreg_lcurve(A_tilde::Matrix{Float64}, b_tilde::Vector{Float64},
                         L::Matrix{Float64};
                         n_alphas::Int=50,
                         alpha_range::Tuple{Real,Real}=(1e-8, 1e3))
    alphas = 10.0 .^ range(log10(Float64(alpha_range[1])),
                           log10(Float64(alpha_range[2])); length=n_alphas)
    ATA = A_tilde' * A_tilde
    ATb = A_tilde' * b_tilde
    LTL = L' * L

    residuals = Float64[]
    norms = Float64[]
    used = Float64[]

    for alpha in alphas
        P = ATA .+ alpha .* LTL
        x = try
            max.(P \ ATb, 0.0)
        catch
            continue
        end
        push!(residuals, norm(A_tilde * x .- b_tilde))
        push!(norms, norm(L * x))
        push!(used, alpha)
    end

    length(residuals) < 3 && return 1.0, used, residuals, norms

    log_res = log.(max.(residuals, 1e-300))
    log_norm = log.(max.(norms, 1e-300))

    v = [log_res[end] - log_res[1], log_norm[end] - log_norm[1]]
    edge = norm(v)
    edge < 1e-300 && return used[div(length(used), 2) + 1], used, residuals, norms

    distances = Float64[]
    for i in eachindex(log_res)
        w = [log_res[i] - log_res[1], log_norm[i] - log_norm[1]]
        push!(distances, abs(v[1] * w[2] - v[2] * w[1]) / edge)
    end

    idx = argmax(collect(distances))
    return used[idx], used, residuals, norms
end

"""
    solve_statreg(A, b, x0=nothing; E_MeV=nothing, unfoldermethod="EmpiricalBayes",
                  regularization=nothing, basis_name="CubicSplines",
                  boundary=nothing, derivative_degree=2)

Развёртка методом статистической регуляризации Тургина (Turchin 1967).

- `unfoldermethod` = `"EmpiricalBayes"` (L-кривая, по умолчанию) либо
  `"User"` (фиксированный `regularization`; по умолчанию 1e-4).
- `derivative_degree` — порядок разностной регуляризации (реализован 2).
- `basis_name`, `boundary`, `E_MeV` — игнорируются (совместимость API).
"""
function solve_statreg(A::AbstractMatrix{Float64}, b::AbstractVector{Float64},
                       x0::Union{AbstractVector{Float64},Nothing}=nothing;
                       E_MeV::Union{AbstractVector{Float64},Nothing}=nothing,
                       unfoldermethod::Union{AbstractString,Symbol}="EmpiricalBayes",
                       regularization::Union{Real,Nothing}=nothing,
                       basis_name::Union{AbstractString,Symbol}="CubicSplines",
                       boundary::Union{AbstractString,Nothing}=nothing,
                       derivative_degree::Integer=2)
    derivative_degree == 2 || throw(ArgumentError(
        "Only derivative_degree=2 is implemented"))
    n_ene = size(A, 2)

    all(b .>= 0) || throw(ArgumentError("STREG requires strictly positive measurements"))
    if any(==(0.0), b)
        keep = b .> 0
        A = A[keep, :]
        b = b[keep]
        isempty(b) && throw(ArgumentError("STREG requires strictly positive measurements"))
    end

    L = _statreg_d2(n_ene)

    sigma = max.(b .* 0.05, 1e-300)
    sigma_inv = 1.0 ./ sigma
    A_tilde = A .* sigma_inv
    b_tilde = b .* sigma_inv

    if unfoldermethod in ("User", :User)
        alpha = regularization === nothing ? 1e-4 : Float64(regularization)
    elseif unfoldermethod in ("EmpiricalBayes", :EmpiricalBayes)
        alpha, used, residuals, norms = _statreg_lcurve(A_tilde, b_tilde, L)
    else
        throw(ArgumentError("Unknown method: $unfoldermethod"))
    end

    ATA = A_tilde' * A_tilde
    ATb = A_tilde' * b_tilde
    LTL = L' * L

    x = try
        (ATA .+ alpha .* LTL) \ ATb
    catch
        pinv(ATA .+ alpha .* LTL) * ATb
    end
    spectrum = max.(x, 0.0)
    resid = b .- A * spectrum
    return UnfoldResult(spectrum, 1, true, norm(resid),
        Dict{String,Any}("unfoldermethod" => String(unfoldermethod),
                         "regularization" => alpha,
                         "derivative_degree" => derivative_degree))
end
