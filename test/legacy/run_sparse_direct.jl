# Former campaign runner, kept for the archived campaign scripts: it now runs
# the sparse direct files through runtests.jl.
ENV["OXIBLAS_TEST_FILES"]="sparse/sparse_pivot_regression.jl,sparse/sparse_indefinite.jl,sparse/multifrontal_lu.jl,sparse/out_of_core.jl"
include(joinpath(@__DIR__,"..","runtests.jl"))
