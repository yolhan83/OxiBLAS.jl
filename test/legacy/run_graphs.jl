# Former campaign runner, kept for the archived campaign scripts: it now runs
# the "graphs" group of runtests.jl (see groups.jl).
ENV["OXIBLAS_TEST_GROUP"]="graphs"
include(joinpath(@__DIR__,"..","runtests.jl"))
