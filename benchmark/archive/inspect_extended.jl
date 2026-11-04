# PREPARED ONLY: no compilation/LLVM/native generation before authorization.
using OxiBLAS, InteractiveUtils, Test
include(joinpath(@__DIR__,"..","..","test","helpers","optional_float_types.jl"))
const INSPECT_OPTIONAL_TYPES=optional_float_types()
length(ARGS)==1 || error("usage: inspect_extended.jl NEW_DIRECTORY")
out=abspath(ARGS[1]);ispath(out) && error("refusing to overwrite diagnostics");mkpath(out)
open(joinpath(out,"ambiguities.txt"),"w") do io
    for pair in Test.detect_ambiguities(OxiBLAS;recursive=true)
        println(io,pair)
    end
end
setprecision(BigFloat,256) do
    for T in vcat(DataType[Float32,Float64,ComplexF32,ComplexF64,BigFloat,Complex{BigFloat}],INSPECT_OPTIONAL_TYPES)
        A=T[3 .125;.25 4];F=lu_factor(A;threaded=false)
        targets=(("gemm",gemm!,Tuple{Matrix{T},Matrix{T},Matrix{T},T,T}),
                 ("solve",solve,Tuple{typeof(F),Vector{T}}))
        for (name,f,signature) in targets
            stem=joinpath(out,"$(name)-$(T)")
            open(stem*".ll","w") do io
                code_llvm(io,f,signature;optimize=true,debuginfo=:none)
            end
            open(stem*".s","w") do io
                code_native(io,f,signature;debuginfo=:none)
            end
        end
    end
end
