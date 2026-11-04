# Standalone targeted campaign, prepared only. Normal runtests.jl includes these too.
using Test, Random, OxiBLAS
import LinearAlgebra as LA
Random.seed!(0x0b1a5); LA.BLAS.set_num_threads(1)
const TYPES=(Float32,Float64,ComplexF32,ComplexF64)
tol(::Type{T}) where T = real(T)===Float32 ? 2f-4 : 3e-12
op(A,t)=t===:N ? A : t===:T ? transpose(A) : adjoint(A)
@testset "Sparse direct supervised campaign" begin
    include("sparse_pivot_regression.jl")
    include("sparse_indefinite.jl")
    include("multifrontal_lu.jl")
    include("out_of_core.jl")
end
