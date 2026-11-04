# Prepared diagnostic only: compilation/IR generation require separate approval.
using OxiBLAS, InteractiveUtils
length(ARGS)==1 || error("usage: inspect_block_spectral.jl NEW_OUTPUT_DIRECTORY")
out=abspath(ARGS[1]); ispath(out) && error("refusing to overwrite inspection artifacts")
mkpath(out)
# Concrete specialization used by block Krylov A*V. Fixing threaded=false keeps
# this inspection focused on SpMM dispatch and the inner sparse arithmetic.
function inspect_block_apply!(Y::Matrix{T},A::CSRMatrix{T,Int},X::Matrix{T}) where T
    OxiBLAS._block_apply!(Y,A,X;threaded=false)
end
open(joinpath(out,"environment.txt"),"w") do io
    versioninfo(io;verbose=true)
    println(io,"Question: concrete block SpMM dispatch, unexpected dynamic calls and complex arithmetic lowering.")
    println(io,"Signatures: Matrix{T}, CSRMatrix{T,Int}, Matrix{T}; T=Float64,ComplexF32.")
    println(io,"No performance or correctness claim follows from these artifacts alone.")
end
for T in (Float64,ComplexF32)
    signature=Tuple{Matrix{T},CSRMatrix{T,Int},Matrix{T}}
    open(joinpath(out,"block-spmm-$(T).ll"),"w") do io
        code_llvm(io,inspect_block_apply!,signature;optimize=true,debuginfo=:none)
    end
    open(joinpath(out,"block-spmm-$(T).s"),"w") do io
        code_native(io,inspect_block_apply!,signature;debuginfo=:none)
    end
end
