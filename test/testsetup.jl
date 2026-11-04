# Shared test prelude: loaded once by runtests.jl before any test file, so that
# every file runs on its own (no file depends on another one having run).
# LinearAlgebra and SparseArrays serve only as references in the tests.
using Test, Random, OxiBLAS, SparseArrays
import LinearAlgebra as LA
Random.seed!(0x0b1a5)
LA.BLAS.set_num_threads(1)

const TYPES=(Float32,Float64,ComplexF32,ComplexF64)
tol(::Type{T}) where T = real(T)===Float32 ? 2f-4 : 3e-12
op(A,t)=t===:N ? A : t===:T ? transpose(A) : adjoint(A)

# Reference implementations and optional floating-point providers.
include(joinpath(@__DIR__,"helpers","level1_references.jl"))
include(joinpath(@__DIR__,"helpers","compact_references.jl"))
include(joinpath(@__DIR__,"helpers","optional_float_types.jl"))
# Matrix and array types that fail on forbidden reads, and shared checks.
include(joinpath(@__DIR__,"helpers","probes.jl"))
include(joinpath(@__DIR__,"helpers","extended.jl"))
include(joinpath(@__DIR__,"helpers","checks.jl"))
