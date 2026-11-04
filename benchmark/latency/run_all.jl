# Each path in a fresh process (so that no path benefits from another one's
# compilation). Prints: path, load s, first call s, second call s.
paths=["gemm_f64","gemm_f32","lu_dense","cholesky_dense","ldiv_dense","sparse_backslash_spd","sparse_backslash_lu","einsum"]
script=joinpath(@__DIR__,"first_call.jl"); project=dirname(dirname(@__DIR__))
println("path\tload_s\tfirst_s\tsecond_s")
for p in paths
    out=read(`$(Base.julia_cmd()) --startup-file=no --project=$project $script $p`,String)
    lines=split(strip(out),'\n'); load=split(lines[1],'\t')[2]; f=split(lines[end],'\t')
    println(p,"\t",load,"\t",f[2],"\t",f[3]); flush(stdout)
end
