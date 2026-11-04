using OxiBLAS, InteractiveUtils
out=joinpath(@__DIR__,"..","..","reports")
open(joinpath(out,"level1-assembly-environment.txt"),"w") do io
    versioninfo(io)
    println(io,"ROTM uses compiler vectorization; widened dots preserve accumulation order.")
end
for (name,f,types) in (("rotm-Float32",rotm!,Tuple{Vector{Float32},Vector{Float32},RotmParams{Float32}}),
                       ("rotm-Float64",rotm!,Tuple{Vector{Float64},Vector{Float64},RotmParams{Float64}}),
                       ("dsdot",dsdot,Tuple{Vector{Float32},Vector{Float32}}),
                       ("sdsdot",sdsdot,Tuple{Float32,Vector{Float32},Vector{Float32}}))
    open(joinpath(out,"$(name)-native.s"),"w") do io
        code_native(io,f,types;debuginfo=:none)
    end
    open(joinpath(out,"$(name)-llvm.ll"),"w") do io
        code_llvm(io,f,types;debuginfo=:none)
    end
end
for T in (Float32,Float64), flag in (-1,0,1)
    width=kernel_info(T).lanes
    types=Tuple{Ptr{T},Ptr{T},Int,RotmParams{T},Val{width},Val{flag}}
    stem="rotm-packet-$(T)-flag$(flag)"
    open(joinpath(out,stem*"-native.s"),"w") do io
        code_native(io,OxiBLAS._rotm_simd!,types;debuginfo=:none)
    end
    open(joinpath(out,stem*"-llvm.ll"),"w") do io
        code_llvm(io,OxiBLAS._rotm_simd!,types;debuginfo=:none)
    end
end
