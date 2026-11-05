# BLAS 1, 2, 3 and matrix multiplication

Every function works on 1-based arrays. The optimized paths cover `Float32`,
`Float64`, `ComplexF32` and `ComplexF64`. Outputs must be mutable and free of
repeated indices.

## GEMM

```@example blas
using OxiBLAS
A = randn(Float32, 300, 200); B = randn(Float32, 200, 150); C = zeros(Float32, 300, 150)
gemm!(C, A, B)                           # C ← A·B
gemm!(C, A, B, 2f0, 1f0; transA=:N)      # C ← 2·A·B + C
G = zeros(Float32, 200, 200)
gemm!(G, A, A; transA=:C)                # G ← Aᴴ·A
kernel_info(Float32)
```

- `transA`/`transB` are `:N`, `:T` or `:C`. Non-unit-stride views,
  `transpose` and `adjoint` are accepted.
- An input that overlaps the output is copied before writing. `β = 0` does not
  read the old output; `α = 0` or `k = 0` does not read the inputs.
- `algorithm` selects a path: `:auto`, `:reference`, `:blocked`, `:packed`,
  `:register`, `:simd`. `kernel` is `:auto`, `:scalar`, `:neon`, `:avx2` or
  `:avx512`; an unavailable ISA raises an error before anything is written.
- **Workspace.** `GemmWorkspace(T, m, n, k; config, workers)` reserves
  reusable panels. Sharing it between calls is protected by a lock; for
  independent concurrent calls, use one workspace per task.
- **Parallelism.** It is controlled by `julia --threads=N`, `threaded=false`
  and `GemmConfig(parallel_threshold=...)`. The number of tasks follows the
  size of the product.
- Variants: [`gemm_cache_oblivious!`](@ref), [`gemm_strassen!`](@ref), [`gemm_winograd!`](@ref)
  (explicitly chosen), [`gemm_batched!`](@ref), [`gemm_strided_batched!`](@ref),
  [`gemm_extended!`](@ref) (generic types).

## `OxiBLAS.mul!` and `OxiBLAS.mul`

`OxiBLAS.mul!(C, A, B, α, β)` and `OxiBLAS.mul(A, B)` (not exported) dispatch
to the right kernel from the types: dense GEMM, sparse × dense, sparse ×
sparse, structured matrices. `A * B` on the OxiBLAS sparse formats goes
through the same layer.

## BLAS 1

| Function | Convention |
|---|---|
| `axpy!(y, x, α=1)`, `scal!(x, α)` | y ← y + αx ; x ← αx |
| `copy_vector!(y, x)`, `swap!(x, y)` | copy with snapshot; swap of disjoint vectors |
| `dotu(x, y)`, `dotc(x, y)` | sum without / with conjugation of the first operand |
| [`dot_kahan`](@ref), [`dot_pairwise`](@ref) | compensated and pairwise summation |
| [`dsdot`](@ref), [`sdsdot`](@ref) | `Float32` vectors, `Float64` accumulation |
| [`nrm2`](@ref), [`asum`](@ref), [`iamax`](@ref), [`iamin`](@ref) | scaled Euclidean norm; complex metric \|Re\|+\|Im\| |
| [`rotg`](@ref), [`rot!`](@ref), [`rotmg`](@ref), [`rotmg!`](@ref), [`rotm!`](@ref), [`RotmParams`](@ref) | Givens and modified rotations |
| [`axpy_parallel!`](@ref), [`scal_parallel!`](@ref), [`dot_parallel`](@ref), [`nrm2_parallel`](@ref), [`asum_parallel`](@ref) | explicit multithreaded variants |

## BLAS 2

| Function | Convention |
|---|---|
| `gemv!(y, A, x, α=1, β=0; trans=:N)` | y ← α·op(A)·x + β·y |
| `ger!(A, x, y, α)`, [`gerc!`](@ref) | rank-1 update, yᵀ / yᴴ |
| [`symv!`](@ref), [`hemv!`](@ref) | same order as [`gemv!`](@ref), `uplo=:U` or `:L` |
| [`syr!`](@ref), [`syr2!`](@ref), [`her!`](@ref), [`her2!`](@ref) | only the selected triangle is modified |
| `trmv!(x, A; uplo, trans, unit)`, [`trsv!`](@ref) | a unit diagonal is never read |

## BLAS 3

| Function | Convention |
|---|---|
| `symm!(C, A, B, α, β; side, uplo)`, [`hemm!`](@ref) | A is the structured matrix, side `:L`/`:R` |
| `trmm!(B, A, α; ...)`, [`trsm!`](@ref) | B is overwritten; `side`, `uplo`, `trans`, `unit` |
| `syrk!(C, A, α, β; ...)`, [`herk!`](@ref), [`syr2k!`](@ref), [`her2k!`](@ref) | only the triangle of C is updated |
| `gemmt!(C, A, B, α=1, β=0; uplo, transA, transB)` | triangle of α·op(A)·op(B) + β·C |
| [`triangular_workspace`](@ref) | reusable workspace for TRMM/TRSM |

The structured BLAS 3 routines reuse the GEMM engine and its options
(`algorithm`, `kernel`, `threaded`, `workspace`, `config`). SYRK, HERK, SYR2K,
HER2K and GEMMT can spread independent tiles across tasks.

Measurements and methodology: [Performance](performance.md).
