# Former campaign runner, kept for the archived campaign scripts: it now runs
# the "all" group of runtests.jl (see groups.jl).
ENV["OXIBLAS_TEST_GROUP"]="all"
include(joinpath(@__DIR__,"..","runtests.jl"))
