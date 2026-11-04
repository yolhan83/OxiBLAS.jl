# PREPARED ONLY. Do not execute before approval of a named campaign.
using Test, Random, OxiBLAS
import LinearAlgebra as LA
Random.seed!(0x0b1a5); LA.BLAS.set_num_threads(1)
const TYPES=(Float32,Float64,ComplexF32,ComplexF64)
tol(::Type{T}) where T = real(T)===Float32 ? 2f-4 : 3e-12
op(A,t)=t===:N ? A : t===:T ? transpose(A) : adjoint(A)
@testset "Complete audited graph row" begin
    include("weighted_matching.jl")
    include("sparse_ordering_advanced.jl")
    include("multilevel_partition.jl")
    include("graph_matching.jl")
    include("graph_refinement.jl")
    include("graph_structure.jl")
end
