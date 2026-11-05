# High-level API

The high-level API reuses the names and conventions of `LinearAlgebra`, on top
of the native engines.

- **Functions of OxiBLAS.** `OxiBLAS.lu !== LinearAlgebra.lu`. They are not
  exported: write `OxiBLAS.lu(A)`, `OxiBLAS.ldiv(F, b)`.
- **LinearAlgebra types.** The factorizations (`OxiBLAS.LU`,
  `OxiBLAS.Cholesky`, `OxiBLAS.QR`…) are subtypes of
  `LinearAlgebra.Factorization`: `F \ b`, `ldiv!(Y, F, b)`, `F'`, `inv(F)`,
  `det(F)` work as for the factorizations of `LinearAlgebra`.
- **No hidden fallback.** On a matrix that no OxiBLAS engine supports, the call
  raises a `MethodError`, never a silent BLAS/LAPACK computation.

```@example hl
using OxiBLAS, LinearAlgebra
A = randn(60, 60) + 60I; B = randn(60, 3)
F = OxiBLAS.lu(A)
X = F \ B                       # generic LinearAlgebra code, OxiBLAS primitives
F2 = OxiBLAS.factorize(Hermitian(A * A' + I))   # Cholesky, chosen from the structure
S = OxiBLAS.svd(randn(40, 20))
(typeof(F2), size(S.U), norm(A * X - B) < 1e-10)
```

## Factorizations and functions

| Function | Engine |
|---|---|
| `OxiBLAS.lu`, `lu!` | [`lu_factor`](@ref) (dense), [`multifrontal_lu`](@ref) (sparse) |
| `OxiBLAS.cholesky`, `cholesky!` | [`cholesky_factor`](@ref), [`multifrontal_cholesky`](@ref) |
| `OxiBLAS.ldlt`, `bunchkaufman` (and `!`) | [`ldlt_factor`](@ref), [`sparse_ldlt`](@ref) |
| `OxiBLAS.qr` (`NoPivot`, `ColumnNorm`, sparse), `qr!`, `lq`, `lq!` | [`qr_factor`](@ref), [`qr_pivoted`](@ref), [`sparse_qr`](@ref) |
| `OxiBLAS.svd`, `svdvals` (and `!`) | [`svd_factor`](@ref), [`singular_values`](@ref) |
| `OxiBLAS.eigen`, `eigvals`, `eigvecs`, `eigmax`, `eigmin` | [`eigen_hermitian`](@ref), [`eigen_general`](@ref), [`eigenvalues`](@ref) |
| `OxiBLAS.eigen(A, B)`, `eigvals(A, B)` | [`eigen_generalized`](@ref) or QZ ([`eigen_pencil`](@ref)) |
| `OxiBLAS.schur(A)`, `schur(A, B)`, `ordschur`, `hessenberg` | [`schur_factor`](@ref), [`qz_factor`](@ref), [`hessenberg_factor`](@ref) |
| `OxiBLAS.factorize` | choice from the structure (see below) |
| `OxiBLAS.det`, `logdet`, `logabsdet`, `inv`, `rank`, `cond`, `pinv`, `nullspace`, `isposdef` | through the engines; `inv` of a sparse matrix is refused |
| `OxiBLAS.norm`, `opnorm`, `kron`, `sylvester`, `lyap` | native engines |
| `OxiBLAS.exp`, `log`, `sqrt`, `sin`, `cos` | matrix functions; on a number, those of `Base` |
| `OxiBLAS.mul`, `mul!` | dense, sparse and structured products |
| `OxiBLAS.spdiagm`, `sparsevec` | sparse constructors in the OxiBLAS formats |

## Linear solves: `ldiv`, `ldiv!` and `\`

| Form | Semantics |
|---|---|
| `OxiBLAS.ldiv!(Y, F, B; workspace=nothing)` | reuses the factorization `F`; `Y` receives the solution, `B` is untouched |
| `OxiBLAS.ldiv!(Y, A, B; workspace=nothing)` | first factorizes with `OxiBLAS.factorize(A)` |
| `OxiBLAS.ldiv(F, B)`, `OxiBLAS.ldiv(A, B)` | allocating forms, `Vector` or `Matrix` like `B` |
| `A \ B`, `LinearAlgebra.ldiv!(Y, A, B)` | on the OxiBLAS sparse formats and their wrappers |
| `LinearAlgebra.ldiv!(A, B)` | square systems: `B` is overwritten with the solution |

```@example hl
using SparseArrays
S = sprand(300, 300, 0.01) + 10I; A = CSRMatrix(S); b = randn(300)
F = OxiBLAS.factorize(A)
x1 = A \ b; x2 = OxiBLAS.ldiv(F, b); y = similar(b); OxiBLAS.ldiv!(y, A, b)
x1 ≈ x2 ≈ y
```

- **Dimensions.** `B` has `size(F, 1)` rows and `Y` has `size(F, 2)`.
- **Rectangular systems.** A tall system (`m > n`) gives the least-squares
  solution; a wide dense system (`n > m`) gives the minimum-norm solution.
- **In-place solve.** `Y === B` is allowed for a square system. A partial
  overlap between `Y` and `B` is refused.
- **Right-hand side and destination.** A sparse `B` is read through a dense
  copy. A sparse destination is refused: the solution is dense.
- **Precision.** The solve runs in the element type of the factorization. A
  real factor and a complex `B` of the same precision are solved as two real
  systems.

### Zero-allocation rectangular solves and the `workspace` keyword

Unlike `LinearAlgebra.ldiv!(Y, F, B)`, `OxiBLAS.ldiv!` accepts an optional
`workspace` keyword argument:

```julia
OxiBLAS.ldiv!(Y, F, B; workspace=nothing)
OxiBLAS.ldiv!(Y, A, B; workspace=nothing)
```

- **Why `workspace` exists.** On a rectangular factorization `F` of size
  `m × n`, the underlying `LinearAlgebra.ldiv!(F, X)` primitive operates in
  place on a buffer `X` of `max(m, n)` rows (`Qᴴ` is applied to the `m` rows
  of `B`, and the `n` solution rows are written into the leading `n` rows of
  `X`). When `m > n` (tall system), `Y` only has `n` rows and `B` must not be
  modified, so an `m`-row buffer is needed.
- **When `workspace` is NOT needed (`workspace=nothing` already avoids allocating `X`):**
  1. **Square systems (`m == n`):** `B` is copied directly into `Y` and solved
     in place in `Y` (0 allocations for sparse `Cholesky` and dense `LU`/`Cholesky`).
  2. **Wide systems (`n > m`, e.g. underdetermined `LQ` or `QR`):** `Y` has
     `n > m` rows, so `OxiBLAS.ldiv!` uses `Y` itself as the working buffer.
  3. **`SVD` factorizations (any shape `m × n`):** `OxiBLAS.ldiv!` projects
     `Uᴴ B` into a small rank-sized vector/matrix and writes `V (Σ⁻¹ Uᴴ B)`
     directly into `Y`.
  4. **Sparse right-hand sides (`B` sparse, `m ≥ n`):** the dense copy of `B`
     is reused in place as the working buffer.
- **When and how to use `workspace`:** When solving repeated tall least-squares
  problems (`m > n`) in a loop with a dense `B` and `QR`, `QRPivoted`,
  `QRSparse`, or `LQ'`, preallocate a buffer of at least `m` rows once per
  task and pass it as `workspace`:

```@example hl
Tall = randn(200, 80); F_qr = OxiBLAS.qr(Tall)
b = randn(200); y = zeros(80)
ws = zeros(200)                              # allocated once per task (length >= m)
OxiBLAS.ldiv!(y, F_qr, b; workspace=ws)      # 0 allocations
y ≈ Tall \ b
```

  `workspace` must have 1-based indexing, size at least `(max(m, n), size(B, 2))`,
  must not alias `Y` or `B`, and must be exclusive to the calling task during
  the `ldiv!` call.

## Keyword extensions beyond `LinearAlgebra`

Where an OxiBLAS high-level function accepts keyword arguments beyond standard
`LinearAlgebra` (or unifies SuiteSparse/CHOLMOD keywords on native engines),
their meaning and usage are:

| Function | Keyword(s) | Purpose and when to use |
|---|---|---|
| `OxiBLAS.ldiv!(Y, F_or_A, B; workspace=nothing)` | `workspace` | Caller-owned buffer of size `≥ (max(m, n), size(B, 2))` for tall (`m > n`) `QR` / `QRPivoted` / `QRSparse` / `LQ'` solves with dense `B`, avoiding the `m`-row temporary allocation in hot loops. |
| `OxiBLAS.factorize(A; check=true)` | `check` | Forwarded across structured and sparse branches. When `check=false`, factorizations that support singular representations retain zero pivots with `issuccess(F) == false`. |
| `OxiBLAS.lu(A::sparse; check=true)` | `check` | When `check=false`, multifrontal and supernodal LU retain zero diagonal entries on singular matrices, returning `issuccess(F) == false` while allowing `det`/`logabsdet` evaluation. |
| `OxiBLAS.qr(A::sparse; ordering=:colamd)` | `ordering` | Fill-reducing column permutation for multifrontal sparse QR: `:colamd` (default), `:amd` (AMD of `AᵀA`), `:natural` (`1:n`), or a custom permutation vector of `1:n`. |
| `OxiBLAS.cholesky(A::sparse; check=true, perm=nothing, shift=0)` | `perm`, `shift` | `perm`: custom fill-reducing permutation vector (`nothing` uses AMD). `shift`: diagonal shift `σ` factoring `A + σI` (as in CHOLMOD). `cholesky!(F, A; check=true, shift=0)` also accepts `shift` while reusing `F.p`. |
| `OxiBLAS.ldlt(A::sparse; check=true, perm=nothing, shift=0)` | `check`, `perm`, `shift` | Same `perm` and `shift` options as `OxiBLAS.cholesky` for pivotless multifrontal `LDLᵀ`; `ldlt!(F, A; shift=0)` reuses `F.p`. |
| `OxiBLAS.bunchkaufman(A::sparse, rook=false; check=true, perm=nothing)` | `check`, `perm` | Custom fill-reducing permutation vector (`nothing` uses AMD) for sparse indefinite Bunch–Kaufman factorization. |

## Singular matrices, determinants and `check=false`

OxiBLAS enforces clear separation between mathematically singular matrices, numerical failures, and linear solves:
- **`check=false` in LU, LDLᵀ and Bunch–Kaufman:** When a matrix is structurally or numerically singular with genuinely zero pivots, setting `check=false` allows the engine to retain the singular factor (with zero diagonal entries) and flag `issuccess(F) == false`.
- **Determinants:** `det(A)`, `logabsdet(A)`, and `logdet(A)` automatically exploit `check=false` via structural factorization without swallowing arbitrary errors:
  - For a singular matrix $A$: `det(A) == zero(T)`, `logabsdet(A) == (-Inf, zero(T))`, and `logdet(A) == -Inf` (or complex logarithm of zero according to standard Julia conventions).
  - A numerical overflow or non-finite floating-point input throws an `ArgumentError`, not a zero determinant.
- **Linear solves (`F \ b`):** Solves that require an invertible matrix refuse singular factors and throw `SingularException`, ensuring mathematical safety.

## What `OxiBLAS.factorize` chooses

| Input | Factorization |
|---|---|
| dense square Hermitian | Cholesky, or Bunch–Kaufman if it is not positive definite |
| dense square | LU |
| dense rectangular | column-pivoted QR |
| sparse triangular (tested in O(nnz)) | native sparse substitution |
| sparse Hermitian | Cholesky, or Bunch–Kaufman |
| sparse complex symmetric | Bunch–Kaufman |
| sparse square | multifrontal LU |
| sparse tall | multifrontal QR |
| `Diagonal`, `Bidiagonal`, triangular, banded, packed | structured engines ([guide](structured.md)) |

## Limitations

- **Sparse systems.**
  - A wide (underdetermined) sparse system is not solved: the sparse QR
    requires m ≥ n.
  - The sparse QR is not rank-revealing: `F \ b` raises `SingularException`
    when R is numerically rank-deficient.
- **No failure state.** The sparse engines (LU, Cholesky) and the dense LDLᵀ
  raise the standard exception even with `check=false`.
- **Refused options.**
  - `bunchkaufman(A, true)` (rook pivoting) is not available.
  - An indefinite [`PackedMatrix`](@ref) has no indefinite factorization.
- **No solve.** `Eigen`, `Schur` and `Hessenberg` have no solve primitive, as
  in `LinearAlgebra`.
