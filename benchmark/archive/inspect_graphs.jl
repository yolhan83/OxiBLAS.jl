# PREPARED ONLY. LLVM/native generation requires explicit authorization.
using OxiBLAS, InteractiveUtils
length(ARGS)==1 || error("usage: inspect_graphs.jl NEW_DIRECTORY")
out=abspath(ARGS[1]); ispath(out) && error("refusing to overwrite diagnostics"); mkpath(out)
entries=(("hopcroft_karp",OxiBLAS._hopcroft_karp,Tuple{Vector{Vector{Int}},Int}),
         ("fm_pass",OxiBLAS._fm_refine_bisection!,Tuple{Vector{Int},OxiBLAS._PartitionGraph,Float64,Float64,Int}),
         ("column_elements",column_minimum_degree,Tuple{CSRMatrix{Float64,Int}}))
for (name,f,signature) in entries
    open(joinpath(out,name*".ll"),"w") do io
        code_llvm(io,f,signature;optimize=true,debuginfo=:none)
    end
    open(joinpath(out,name*".s"),"w") do io
        code_native(io,f,signature;debuginfo=:none)
    end
end
