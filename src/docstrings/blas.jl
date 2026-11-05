# SPDX-License-Identifier: Apache-2.0
# Docstrings of the BLAS family (level 1, 2, 3) whose definitions are one-line
# methods or several specialized methods. Attached to the function bindings.

# ---- Level 1 -----------------------------------------------------------------

@doc raw"""
    axpy!(y, x, α=1) -> y

`y ← α·x + y` in place. `x` and `y` have the same length; `y` must not alias
`x` partially.
""" axpy!

@doc raw"""
    scal!(x, α) -> x

`x ← α·x` in place. With `α = 0` the result is exactly zero (NaN and Inf in `x`
are not propagated), as in reference BLAS.
""" scal!

@doc raw"""
    copy_vector!(y, x) -> y

BLAS `copy`: `y ← x`, with a length check.
""" copy_vector!

@doc raw"""
    swap!(x, y) -> (x, y)

BLAS `swap`: exchanges the contents of `x` and `y` (same length).
""" swap!

@doc raw"""
    dotu(x, y)

Unconjugated dot product `Σ x[i]·y[i]` (BLAS `dotu`). For real vectors it equals
[`dotc`](@ref).
""" dotu

@doc raw"""
    dotc(x, y)

Conjugated dot product `Σ conj(x[i])·y[i]` (BLAS `dotc`), the Euclidean inner
product of complex vectors.
""" dotc

@doc raw"""
    dot_kahan(x, y; conjugate=false)

Dot product with Kahan compensated summation: an error bound independent of the
length for well-scaled data, at about twice the cost of [`dotu`](@ref).
`conjugate=true` conjugates `x`.
""" dot_kahan

@doc raw"""
    dot_pairwise(x, y; conjugate=false)

Dot product with pairwise (cascade) summation: error growth in `O(log n)`
instead of `O(n)`. `conjugate=true` conjugates `x`.
""" dot_pairwise

@doc raw"""
    asum(x)

BLAS `asum`: `Σ |x[i]|` for real vectors, `Σ (|re x[i]| + |im x[i]|)` for
complex ones (the BLAS convention, not the 1-norm of complex numbers).
""" asum

@doc raw"""
    iamax(x) -> Int

Index of the first entry of largest magnitude (`|re| + |im|` for complex, as
BLAS `i?amax`). Returns 0 for an empty vector.
""" iamax

@doc raw"""
    iamin(x) -> Int

Index of the first entry of smallest magnitude, with the [`iamax`](@ref)
convention. Returns 0 for an empty vector.
""" iamin

@doc raw"""
    rot!(x, y, c, s) -> (x, y)

Applies the plane rotation `[c s; -conj(s) c]` to the pairs `(x[i], y[i])`
(BLAS `rot`), `c` real. The rotation itself comes from [`rotg`](@ref).
""" rot!

@doc raw"""
    rotmg(d1, d2, x1, y1) -> (; d1, d2, x1, params)

Constructs a modified Givens rotation (BLAS `rotmg`) that zeroes the second
component of `(√d1·x1, √d2·y1)`. Returns the updated scale factors `d1`, `d2`,
the updated `x1` and the rotation `params::RotmParams` (flag and `h` entries),
which [`rotm!`](@ref) applies.
""" rotmg

@doc raw"""
    axpy_parallel!(y, x, α=1; threaded=true, parallel_threshold=nothing) -> y

Multithreaded [`axpy!`](@ref): the vector is split into contiguous chunks
above `parallel_threshold` elements (default: a per-type threshold).
""" axpy_parallel!

@doc raw"""
    scal_parallel!(x, α; threaded=true, parallel_threshold=nothing) -> x

Multithreaded [`scal!`](@ref).
""" scal_parallel!

@doc raw"""
    dot_parallel(x, y; conjugate=false, threaded=true, parallel_threshold=nothing)

Multithreaded dot product: each task reduces one chunk with the sequential
kernel, then the partial sums are added. The result can differ from
[`dotu`](@ref)/[`dotc`](@ref) by rounding (different summation order).
""" dot_parallel

@doc raw"""
    nrm2_parallel(x; threaded=true, parallel_threshold=nothing)

Multithreaded Euclidean norm with overflow-safe scaling, like [`nrm2`](@ref).
""" nrm2_parallel

@doc raw"""
    asum_parallel(x; threaded=true, parallel_threshold=nothing)

Multithreaded [`asum`](@ref).
""" asum_parallel

# ---- Level 2 -----------------------------------------------------------------

@doc raw"""
    gemv!(y, A, x, α=1, β=0; trans=:N, threaded=true, parallel_threshold=nothing) -> y

General matrix–vector product `y ← α·op(A)·x + β·y`, with `op` given by
`trans`: `:N` (A), `:T` (transpose) or `:C` (adjoint). `β = 0` never reads `y`.
""" gemv!

@doc raw"""
    symv!(y, A, x, α=1, β=0; uplo=:U, threaded=true) -> y

Symmetric matrix–vector product `y ← α·A·x + β·y`, reading only the `uplo`
triangle (`:U` or `:L`) of `A`.
""" symv!

@doc raw"""
    hemv!(y, A, x, α=1, β=0; uplo=:U, threaded=true) -> y

Hermitian matrix–vector product `y ← α·A·x + β·y`, reading only the `uplo`
triangle of `A`; the imaginary part of the diagonal is ignored.
""" hemv!

@doc raw"""
    trmv!(x, A; uplo=:U, trans=:N, unit=false) -> x

Triangular matrix–vector product `x ← op(A)·x` in place, `A` upper (`uplo=:U`)
or lower triangular; `unit=true` assumes a unit diagonal without reading it.
""" trmv!

@doc raw"""
    trsv!(x, A; uplo=:U, trans=:N, unit=false) -> x

Triangular solve `op(A)·x = b` in place (`x` holds `b` on entry). A zero
diagonal entry raises `SingularException` (unless `unit=true`).
""" trsv!

@doc raw"""
    ger!(A, x, y, α=1) -> A

Rank-1 update `A ← α·x·yᵀ + A` (BLAS `geru`).
""" ger!

@doc raw"""
    gerc!(A, x, y, α=1) -> A

Conjugated rank-1 update `A ← α·x·yᴴ + A` (BLAS `gerc`).
""" gerc!

@doc raw"""
    syr!(A, x, α=1; uplo=:U) -> A

Symmetric rank-1 update `A ← α·x·xᵀ + A` of the `uplo` triangle.
""" syr!

@doc raw"""
    syr2!(A, x, y, α=1; uplo=:U) -> A

Symmetric rank-2 update `A ← α·x·yᵀ + α·y·xᵀ + A` of the `uplo` triangle.
""" syr2!

@doc raw"""
    her!(A, x, α=1; uplo=:U) -> A

Hermitian rank-1 update `A ← α·x·xᴴ + A` of the `uplo` triangle, `α` real.
""" her!

@doc raw"""
    her2!(A, x, y, α=1; uplo=:U) -> A

Hermitian rank-2 update `A ← α·x·yᴴ + conj(α)·y·xᴴ + A` of the `uplo` triangle.
""" her2!

# ---- Level 3 -----------------------------------------------------------------

@doc raw"""
    gemm_reference!(C, A, B, α=1, β=0; kwargs...) -> C

[`gemm!`](@ref) with `algorithm=:reference`: the plain triple loop, used as a
correctness reference.
""" gemm_reference!

@doc raw"""
    gemm_blocked!(C, A, B, α=1, β=0; kwargs...) -> C

[`gemm!`](@ref) with `algorithm=:blocked`: cache blocking without packing or
SIMD micro-kernels.
""" gemm_blocked!

@doc raw"""
    symm!(C, A, B, α=1, β=0; side=:L, uplo=:U, threaded=true) -> C

Symmetric matrix product: `C ← α·A·B + β·C` (`side=:L`) or `C ← α·B·A + β·C`
(`side=:R`), with `A` symmetric and only its `uplo` triangle read.
""" symm!

@doc raw"""
    hemm!(C, A, B, α=1, β=0; side=:L, uplo=:U, threaded=true) -> C

Hermitian counterpart of [`symm!`](@ref).
""" hemm!

@doc raw"""
    syrk!(C, A, α=1, β=0; uplo=:U, trans=:N, threaded=true) -> C

Symmetric rank-k update of the `uplo` triangle of `C`:
`C ← α·A·Aᵀ + β·C` (`trans=:N`) or `C ← α·Aᵀ·A + β·C` (`trans=:T`). The other
triangle is not written.
""" syrk!

@doc raw"""
    herk!(C, A, α=1, β=0; uplo=:U, trans=:N, threaded=true) -> C

Hermitian rank-k update `C ← α·A·Aᴴ + β·C` (`trans=:N`) or `α·Aᴴ·A + β·C`
(`trans=:C`) of the `uplo` triangle; `α` and `β` are real.
""" herk!

@doc raw"""
    syr2k!(C, A, B, α=1, β=0; uplo=:U, trans=:N, threaded=true) -> C

Symmetric rank-2k update `C ← α·A·Bᵀ + α·B·Aᵀ + β·C` (`trans=:N`) of the
`uplo` triangle.
""" syr2k!

@doc raw"""
    her2k!(C, A, B, α=1, β=0; uplo=:U, trans=:N, threaded=true) -> C

Hermitian rank-2k update `C ← α·A·Bᴴ + conj(α)·B·Aᴴ + β·C` of the `uplo`
triangle, `β` real.
""" her2k!
