# Sparse matrices

## Formats

| Format | Constructor |
|---|---|
| [`CSRMatrix`](@ref), [`CSCMatrix`](@ref), [`COOMatrix`](@ref) | `CSRMatrix(m, n, rowptr, colind, values)`, or conversion of a matrix (`CSRMatrix(S)` for `S::SparseMatrixCSC`) |
| [`DIAMatrix`](@ref), [`ELLMatrix`](@ref), [`BSRMatrix`](@ref), [`BSCMatrix`](@ref), [`HYBMatrix`](@ref), [`SELLMatrix`](@ref) | extended formats, converted from CSR/CSC |
| [`SparseTensor`](@ref) | N-dimensional sparse array ([tensors](tensors.md)) |

These types are `SparseArrays.AbstractSparseMatrix`. They interoperate with
SparseArrays:

- conversions `SparseMatrixCSC(A)` and `CSCMatrix(S)` (no copy from a
  `SparseMatrixCSC`);
- `nnz`, `findnz`, `getindex`, broadcast;
- `hcat`, `vcat`, `kron`, `blockdiag`, `fkeep!`.

`OxiBLAS.spdiagm` and `OxiBLAS.sparsevec` build the OxiBLAS formats directly.

## Products

`sparse_mul!(y, A, x, α, β; trans)` computes the sparse × dense product,
`sparse_matmul(A, B)` the sparse × sparse product. `A * B`, `A * x` and
`OxiBLAS.mul!` dispatch to the same kernels.

## Solves: `\` and factorizations

```@example sp
using OxiBLAS, SparseArrays, LinearAlgebra
n = 400; S = sprand(n, n, 0.01) + 10I; A = CSRMatrix(S); b = randn(n)
x = A \ b                         # factorization chosen from the structure
F = OxiBLAS.lu(A)                 # multifrontal LU, F.L*F.U == A[F.p,F.q]
P = CSRMatrix(S * S' + I)
C = OxiBLAS.cholesky(Hermitian(P))
(norm(S * x - b) / norm(b), norm(S * (F \ b) - b) / norm(b), norm(P * (C \ b) - b) / norm(b))
```

`A \ B` on an OxiBLAS format follows the order of SparseArrays:

| Structure (tested in O(nnz)) | Factorization |
|---|---|
| triangular or diagonal | native sparse substitution ([`sparse_triangular_solve`](@ref)) |
| Hermitian | multifrontal Cholesky, or Bunch–Kaufman if not positive definite |
| complex symmetric | Bunch–Kaufman |
| other square | multifrontal LU |
| tall | multifrontal QR (least squares) |

| Engine | Functions |
|---|---|
| LU | [`multifrontal_lu`](@ref), [`sparse_lu`](@ref), [`sparse_lu_static`](@ref), [`sparse_lu_blocked`](@ref) |
| Cholesky | [`multifrontal_cholesky`](@ref), [`supernodal_cholesky`](@ref), [`sparse_cholesky`](@ref), [`symbolic_cholesky`](@ref) |
| indefinite LDLᵀ | [`sparse_ldlt`](@ref) (Bunch–Kaufman within fronts), [`ldl_inertia`](@ref) |
| QR | [`sparse_qr`](@ref), [`sparse_apply_q!`](@ref) |
| out of core | [`out_of_core_lu`](@ref), [`out_of_core_cholesky`](@ref), [`out_of_core_memory_usage`](@ref) |
| triangular | [`sparse_triangular_solve`](@ref) |

## Iterative solvers and preconditioners

Examples, the table of which solver suits which matrix, and the result
fields: [How do I… solve iteratively](howto.md#Iteratively-(Krylov-methods)).

- **Solvers.** [`cg`](@ref), [`pcg`](@ref), [`gmres`](@ref), [`fgmres`](@ref), [`bicgstab`](@ref), [`minres`](@ref), [`tfqmr`](@ref),
  [`qmr`](@ref), [`idrs`](@ref), and their block variants [`block_cg`](@ref) and [`block_gmres`](@ref). Each
  returns the solution, the convergence status, the iterations and the
  actually recomputed residual.
- **Preconditioners.** [`ilu0`](@ref), [`ilut`](@ref), [`ic0`](@ref), [`ict`](@ref), [`spai`](@ref), [`ainv`](@ref), [`amg`](@ref),
  [`relaxation_preconditioner`](@ref), [`polynomial_preconditioner`](@ref) and
  [`additive_schwarz`](@ref).

## Orderings and graphs

The available ordering families:

- **minimum degree:** [`amd_ordering`](@ref), [`colamd_ordering`](@ref),
  [`minimum_degree_ordering`](@ref), [`multiple_minimum_degree`](@ref),
  [`column_minimum_degree`](@ref);
- **profile and bandwidth:** [`reverse_cuthill_mckee`](@ref);
- **nested dissection:** [`nested_dissection`](@ref), [`multilevel_nested_dissection`](@ref).

The graph tools are:

- [`connected_components`](@ref), [`level_sets`](@ref), [`multilevel_partition`](@ref),
  [`refine_partition`](@ref);
- bipartite matchings ([`bipartite_matching`](@ref), [`weighted_bipartite_matching`](@ref));
- choice of an ordering: [`sparse_ordering`](@ref), [`select_sparse_ordering`](@ref),
  [`ordering_metrics`](@ref), [`compare_orderings`](@ref).

## Input and output

[`read_matrix_market`](@ref) and [`write_matrix_market`](@ref) read and write the Matrix
Market format.

## Limitations

- **Wide (underdetermined) sparse systems.** There is no solver:
  `OxiBLAS.qr` refuses a wide matrix with an `ArgumentError`, and so does
  `A \ b`.
- **Sparse QR without rank detection.** It is not rank-revealing:
  `SingularException` on a numerically rank-deficient matrix.
- **Sparse destination refused.** A solution is dense.
- **Inverse refused.** `OxiBLAS.inv` of a sparse matrix raises an
  `ArgumentError`: factorize and solve instead.
- **No failure state.** The sparse LU and Cholesky engines raise the standard
  exception even with `check=false`.
- **`SparseMatrixCSC` unchanged.** OxiBLAS adds no `LinearAlgebra`/
  `SparseArrays` method on this type: `S \ b` stays the one of SparseArrays
  (UMFPACK, CHOLMOD). `OxiBLAS.ldiv(S, b)` and `OxiBLAS.lu(S)` use the
  OxiBLAS engines.
