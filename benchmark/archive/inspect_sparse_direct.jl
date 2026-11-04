# PREPARED ONLY. This script executes code and emits LLVM/native artifacts;
# running it requires its own explicit inclusion in an approved campaign.
using OxiBLAS, InteractiveUtils
length(ARGS)==1 || error("usage: inspect_sparse_direct.jl NEW_OUTPUT_DIR")
output=abspath(ARGS[1]); ispath(output) && error("refusing to overwrite diagnostics"); mkpath(output)
for T in (Float64,ComplexF64)
    C=CSRMatrix(T[4 1;1 3]); F=multifrontal_lu(C;threaded=false)
    signatures=(("block_inverse",OxiBLAS._ldl_block_scaled,Tuple{T,T,T,Bool,Int}),
                ("multifrontal_solve",solve,Tuple{typeof(F),Vector{T}}))
    for (name,f,signature) in signatures
        stem=joinpath(output,"$(name)-$(T)")
        open(stem*".ll","w") do io
            code_llvm(io,f,signature;optimize=true,debuginfo=:none)
        end
        open(stem*".s","w") do io
            code_native(io,f,signature;debuginfo=:none)
        end
    end
    D=out_of_core_lu(C;blocksize=1)
    try
        work=view(D.panel,1:2); signature=Tuple{typeof(work),typeof(D),Bool,Bool,Int}
        open(joinpath(output,"disk_sweep-$(T).ll"),"w") do io
            code_llvm(io,OxiBLAS._ooc_triangular!,signature;optimize=true,debuginfo=:none)
        end
        open(joinpath(output,"disk_sweep-$(T).s"),"w") do io
            code_native(io,OxiBLAS._ooc_triangular!,signature;debuginfo=:none)
        end
    finally
        close(D)
    end
end
