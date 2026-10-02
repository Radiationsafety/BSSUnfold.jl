#!/usr/bin/env bash
# Verify every examples/*.jl notebook runs headless.  One Julia subprocess
# per notebook keeps globals (`detector_names`, `A`, `x_true`, …) from
# bleeding across notebooks and shadowing the exported API.

set -u
cd "$(dirname "$0")/.."

overall=0
for nb in examples/*.jl; do
    echo
    echo "── $nb ──"
    julia --project=. -e "include(\"scripts/run_pluto_notebook.jl\"); r = run_notebook(\"$nb\"); println(\"   ok=\", r.ok, \" err=\", r.err, \" skip=\", r.skip); for (i, id, msg) in r.errs; println(\"   cell #\", i, \" (\", id, \"):\"); println(\"     \", replace(msg, \"\\n\" => \"\\n     \")); end; exit(r.err == 0 ? 0 : 1)"
    if [ $? -ne 0 ]; then
        overall=1
    fi
done

exit $overall
