#!/usr/bin/env julia
# Headless Pluto-notebook runner: parses `# ╔═╡` cells, follows the
# "Cell order:" trailer (or file order when absent), and evaluates every
# code cell in `Main` in sequence. Markdown-only cells (`md"""…"""`) and
# Pluto `@bind`/`Editor` widgets are skipped — the point is to catch load,
# syntax and runtime errors in the notebook's Julia code, not to render the
# UI.

const MARKER = r"^# ╔═╡ ([0-9a-f\-]+)"

function parse_notebook(path::AbstractString)
    lines = readlines(path)
    cells = Dict{String,Vector{String}}()
    order = String[]
    current_id = nothing
    preamble = String[]
    in_preamble = true
    for line in lines
        m = match(MARKER, line)
        if m !== nothing
            in_preamble = false
            current_id = m.captures[1]
            cells[current_id] = String[]
            push!(order, current_id)
        elseif in_preamble
            push!(preamble, line)
        elseif current_id !== nothing
            push!(cells[current_id], line)
        end
    end
    # Look for "# ╔═╡ Cell order:" trailer (Pluto stores the ordered UUIDs
    # there as `# ╠═<uuid>` lines).
    trailer_idx = findfirst(l -> occursin(r"^# ╔═╡ Cell order:", l), lines)
    if trailer_idx !== nothing
        ordered = String[]
        for l in lines[trailer_idx:end]
            m = match(r"^# ╠═([0-9a-f\-]+)", l)
            m !== nothing && push!(ordered, m.captures[1])
        end
        !isempty(ordered) && (order = ordered)
    end
    return cells, order, preamble
end

function cell_is_code(cell::Vector{String})
    first_line = ""
    for line in cell
        isempty(strip(line)) && continue
        first_line = line
        break
    end
    startswith(strip(first_line), "md\"\"\"") && return false
    startswith(strip(first_line), "raw\"\"\"") && return false
    startswith(strip(first_line), "r\"\"\"") && return false
    return true
end

function strip_begin_end(cell::Vector{String})
    # `begin` ... `end` cells: strip the wrapper so that locals stay in scope
    # (Top-level eval on a plain sequence of statements, not inside a block).
    nonblank_idx = findall(i -> !isempty(strip(cell[i])), eachindex(cell))
    isempty(nonblank_idx) && return ""
    first_i, last_i = first(nonblank_idx), last(nonblank_idx)
    first_line = strip(cell[first_i])
    last_line = strip(cell[last_i])
    if first_line == "begin" && last_line == "end"
        return join(cell[(first_i+1):(last_i-1)], "\n")
    end
    return join(cell, "\n")
end

function run_notebook(path::AbstractString; stop_on_error::Bool=false)
    Base.require(Base, :Markdown)
    cells, order, preamble = parse_notebook(path)
    # Pluto evaluates `@__DIR__`/`@__FILE__` inside a notebook as the
    # notebook's own directory, so pin cwd and rewrite those macros so
    # relative `joinpath(@__DIR__, ...)` lookups match.
    nb_dir  = abspath(dirname(path))
    nb_file = abspath(path)
    n_ok, n_err, n_skip = 0, 0, 0
    errs = Tuple{Int,String,String}[]
    prev = pwd()
    cd(nb_dir)
    try
        eval_in_main(join(preamble, "\n"), nb_file)
        for (i, id) in enumerate(order)
            cell = get(cells, id, String[])
            cell_is_code(cell) || (n_skip += 1; continue)
            body = strip_begin_end(cell)
            try
                eval_in_main(body, nb_file)
                n_ok += 1
            catch err
                n_err += 1
                push!(errs, (i, id, sprint(showerror, err)))
                if stop_on_error
                    println("      cell #$i ($id) FAILED: ", first(split(sprint(showerror, err), "\n")))
                    break
                end
            end
        end
    finally
        cd(prev)
    end
    return (ok=n_ok, err=n_err, skip=n_skip, errs=errs)
end

# Evaluate a fragment as if it lived inside `nb_file`, so `@__FILE__`,
# `@__DIR__` and `@__LINE__` expand to the notebook's location.
function eval_in_main(body::AbstractString, nb_file::AbstractString)
    ex = Meta.parseall(body, filename=nb_file)
    Core.eval(Main, ex)
end

function _main(paths)
    isempty(paths) && push!(paths, "examples/01-basic-example.jl")
    overall = true
    for p in paths
        println("\n── running ", p, " ──")
        r = run_notebook(p)
        println("   ok=", r.ok, " err=", r.err, " skip(md/widget)=", r.skip)
        for (i, id, msg) in r.errs
            overall = false
            println("   cell #", i, " (", id, "):")
            println("     ", replace(msg, "\n" => "\n     "))
        end
    end
    return overall
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(_main(ARGS) ? 0 : 1)
end
