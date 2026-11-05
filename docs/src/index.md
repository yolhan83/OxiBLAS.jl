# OxiBLAS.jl

OxiBLAS.jl is a linear algebra library written entirely in Julia. It is an
adapted and extended port of the Rust library
[OxiBLAS](https://github.com/cool-japan/oxiblas) (reference pinned at commit
`8d82bb6`).

- **Native computation.** No computation is delegated to BLAS, LAPACK,
  Accelerate, SuiteSparse or the original Rust code. Those libraries are only
  references in the tests and benchmarks. `LinearAlgebra` provides the package
  with its wrapper types (`Transpose`, `Adjoint`, `Hermitian`…), its exceptions
  and the `Factorization` interface, never its computations.
- **SIMD micro-kernels.** They use LLVM vectors (`llvmcall`), without SIMD.jl
  or LoopVectorization.jl. NEON is validated on Apple Silicon; the AVX2 and
  AVX-512 variants are implemented but not validated on x86 hardware.
- **Types.** The optimized paths cover `Float32`, `Float64`, `ComplexF32` and
  `ComplexF64`. Many algorithms also accept generic floating-point types
  (`BigFloat`, `Float16`, extended-precision types), without the same
  performance.

## Installation

The package is not registered yet:

```julia
using Pkg
Pkg.add(url="https://github.com/yolhan83/OxiBLAS.jl")
```

## First examples

**Native matrix multiplication.** [`gemm!`](@ref) writes `C ← α·op(A)·op(B) + β·C`,
with an optional reusable workspace:

```@example home
using OxiBLAS
A = randn(513, 257); B = randn(257, 389); C = zeros(513, 389)
w = GemmWorkspace(Float64, 513, 389, 257)
gemm!(C, A, B, 0.7, 0.0; workspace=w)
C ≈ 0.7 * A * B
```

**High-level API.** It reuses the names of `LinearAlgebra`, but its functions
belong to OxiBLAS and are not exported:

```@example home
using LinearAlgebra      # only for I and the check
M = randn(50, 50) + 50I
F = OxiBLAS.lu(M)        # native LU (a LinearAlgebra.Factorization)
b = randn(50)
x = OxiBLAS.ldiv(F, b)   # reuses F; F \ b gives the same result
M * x ≈ b
```

**Sparse matrices.** The OxiBLAS formats ([`CSRMatrix`](@ref), [`CSCMatrix`](@ref),
[`COOMatrix`](@ref)…) support `\`, with a factorization chosen from the structure, as
in SparseArrays:

```@example home
using SparseArrays
S = sprand(200, 200, 0.02) + 10I
A = CSRMatrix(S)
x = A \ ones(200)
S * x ≈ ones(200)
```

**Tensors.** [`einsum`](@ref) contracts dense or sparse arrays:

```@example home
X = randn(8, 9, 10); Y = randn(10, 7)
Z = einsum("abc,cd->abd", X, Y)
size(Z)
```

## Organization of the documentation

- **[How do I…?](guide/howto.md)** gives task-oriented recipes: solving a
  system (direct or iterative), least squares, SVD, eigenvalues, products,
  sparse matrices, tensors.
- The **guides** present each family of features and its conventions.
- The **API reference** groups the docstrings by source family.

## Names shared with LinearAlgebra

`OxiBLAS.lu`, `OxiBLAS.qr`, `OxiBLAS.svd`, `OxiBLAS.ldiv`, `OxiBLAS.exp`, etc.
are functions distinct from those of `LinearAlgebra` and `Base`
(`OxiBLAS.lu !== LinearAlgebra.lu`). They are not exported, so `using
LinearAlgebra, OxiBLAS` creates no name conflict. Extensions of `LinearAlgebra`
and `Base` functions (for instance `A \ b` on a [`CSRMatrix`](@ref)) only concern
signatures that contain an OxiBLAS type: no method is added for `Matrix` or
`SparseMatrixCSC`.
