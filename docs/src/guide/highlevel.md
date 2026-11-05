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
| `OxiBLAS.lu`, `lu!` | `lu_factor` (dense), `multifrontal_lu` (sparse) |
| `OxiBLAS.cholesky`, `cholesky!` | `cholesky_factor`, `multifrontal_cholesky` |
| `OxiBLAS.ldlt`, `bunchkaufman` (and `!`) | `ldlt_factor`, `sparse_ldlt` |
| `OxiBLAS.qr` (`NoPivot`, `ColumnNorm`, sparse), `qr!`, `lq`, `lq!` | `qr_factor`, `qr_pivoted`, `sparse_qr` |
| `OxiBLAS.svd`, `svdvals` (and `!`) | `svd_factor`, `singular_values` |
| `OxiBLAS.eigen`, `eigvals`, `eigvecs`, `eigmax`, `eigmin` | `eigen_hermitian`, `eigen_general`, `eigenvalues` |
| `OxiBLAS.eigen(A, B)`, `eigvals(A, B)` | `eigen_generalized` or QZ (`eigen_pencil`) |
| `OxiBLAS.schur(A)`, `schur(A, B)`, `ordschur`, `hessenberg` | `schur_factor`, `qz_factor`, `hessenberg_factor` |
| `OxiBLAS.factorize` | choice from the structure (see below) |
| `OxiBLAS.det`, `logdet`, `logabsdet`, `inv`, `rank`, `cond`, `pinv`, `nullspace`, `isposdef` | through the engines; `inv` of a sparse matrix is refused |
| `OxiBLAS.norm`, `opnorm`, `kron`, `sylvester`, `lyap` | native engines |
| `OxiBLAS.exp`, `log`, `sqrt`, `sin`, `cos` | matrix functions; on a number, those of `Base` |
| `OxiBLAS.mul`, `mul!` | dense, sparse and structured products |
| `OxiBLAS.spdiagm`, `sparsevec` | sparse constructors in the OxiBLAS formats |

## Linear solves: `ldiv`, `ldiv!` and `\`

| Form | Semantics |
|---|---|
| `OxiBLAS.ldiv!(Y, F, B)` | reuses the factorization `F`; `Y` receives the solution, `B` is untouched |
| `OxiBLAS.ldiv!(Y, A, B)` | first factorizes with `OxiBLAS.factorize(A)` |
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
- **Rectangular systems.** A tall system gives the least-squares solution.
- **In-place solve.** `Y === B` is allowed for a square system. A partial
  overlap between `Y` and `B` is refused.
- **Right-hand side and destination.** A sparse `B` is read through a dense
  copy. A sparse destination is refused: the solution is dense.
- **Precision.** The solve runs in the element type of the factorization. A
  real factor and a complex `B` of the same precision are solved as two real
  systems.

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
  - An indefinite `PackedMatrix` has no indefinite factorization.
- **No solve.** `Eigen`, `Schur` and `Hessenberg` have no solve primitive, as
  in `LinearAlgebra`.
