# PREPARED ONLY. One isolated process per existing script/backend/thread setting.
include("full_backend.jl")
length(ARGS)>=1 || error("legacy script name required")
script=popfirst!(ARGS)
basename(script)==script && endswith(script,".jl") || error("benchmark basename required")
println("LEGACY VARIANT: ",script,"; process request=",FULL_THREADS,
        "; this script may explicitly use fewer BLAS/Julia workers; see its own environment metadata.")
println("Legacy references are raw diagnostics; only full_worker records carry verified four-backend comparisons.")
include(joinpath(@__DIR__,script))
