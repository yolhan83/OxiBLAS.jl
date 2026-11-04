# PREPARED ONLY. All numerical execution requires an approved named campaign.
using Test, Random, OxiBLAS
import LinearAlgebra as LA
Random.seed!(0x0b1a5); LA.BLAS.set_num_threads(1)
const TYPES=(Float32,Float64,ComplexF32,ComplexF64)
tol(::Type{T}) where T = real(T)===Float32 ? 2f-4 : 3e-12
op(A,t)=t===:N ? A : t===:T ? transpose(A) : adjoint(A)
@testset "Extended precision supervised campaign" begin
    include("extended_scalar.jl")
    include("extended_blas.jl")
    include("extended_lapack.jl")
    include("extended_spectral.jl")
    include("generic_contract.jl")
    include("generic_tensors.jl")
end
