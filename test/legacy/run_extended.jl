# Former campaign runner, kept for the archived campaign scripts: it now runs
# the "generic" group of runtests.jl (see groups.jl).
ENV["OXIBLAS_TEST_GROUP"]="generic"
include(joinpath(@__DIR__,"..","runtests.jl"))
