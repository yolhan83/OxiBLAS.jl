using OxiBLAS, InteractiveUtils
out=joinpath(@__DIR__,"..","..","reports")
mkpath(out)
for T in (Float32,Float64,ComplexF32,ComplexF64)
    info=kernel_info(T)
    info.kind===:scalar && continue
    w,r,nr=OxiBLAS._shape(T,info.kind)
    f=T<:Real ? OxiBLAS._simd_micro! : OxiBLAS._simd_complex_micro!
    args=Tuple{Ptr{T},Int,Ptr{T},Ptr{T},Int,T,T,Val{w},Val{r},Val{nr}}
    tag="$(info.kind)-$(lowercase(string(T)))"
    open(joinpath(out,tag*"-native.s"),"w") do io
        code_native(io,f,args;debuginfo=:none)
    end
    open(joinpath(out,tag*"-llvm.ll"),"w") do io
        code_llvm(io,f,args;debuginfo=:none)
    end
end
