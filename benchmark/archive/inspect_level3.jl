using OxiBLAS, InteractiveUtils
out=joinpath(@__DIR__,"..","..","reports")
open(joinpath(out,"level3-assembly-environment.txt"),"w") do io
    versioninfo(io)
    println(io,"Scalar/block leaves; ISA-specific GEMM microkernels are inspected separately.")
    println(io,"Matrix argument types, dynamic Bool side/triangle/unit flags; no timed execution.")
end
for T in (Float32,Float64,ComplexF32,ComplexF64), (name,f,types) in
    (("trmm",OxiBLAS._trmm_leaf!,Tuple{Matrix{T},Matrix{T},T,Bool,Bool,Bool}),
     ("trsm",OxiBLAS._trsm_leaf!,Tuple{Matrix{T},Matrix{T},Bool,Bool,Bool}))
    stem=joinpath(out,"$(name)-leaf-$(T)")
    open(stem*"-native.s","w") do io
        code_native(io,f,types;debuginfo=:none)
    end
    open(stem*"-llvm.ll","w") do io
        code_llvm(io,f,types;debuginfo=:none)
    end
end
