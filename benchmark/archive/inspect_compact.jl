using OxiBLAS, InteractiveUtils
out=joinpath(@__DIR__,"..","..","reports")
open(joinpath(out,"compact-assembly-environment.txt"),"w") do io
    versioninfo(io)
    println(io,"Real contiguous compact AXPY packets; other paths use native column loops.")
end
for T in (Float32,Float64)
    w=kernel_info(T).lanes
    types=Tuple{Ptr{T},Ptr{T},Int,T,Val{w}}
    stem="compact-axpy-$(T)"
    open(joinpath(out,stem*"-native.s"),"w") do io
        code_native(io,OxiBLAS._compact_axpy_ptr!,types;debuginfo=:none)
    end
    open(joinpath(out,stem*"-llvm.ll"),"w") do io
        code_llvm(io,OxiBLAS._compact_axpy_ptr!,types;debuginfo=:none)
    end
end
