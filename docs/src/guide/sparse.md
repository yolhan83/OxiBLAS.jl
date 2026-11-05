# Sparse matrices

## Formats

| Format | Constructor |
|---|---|
| `CSRMatrix`, `CSCMatrix`, `COOMatrix` | `CSRMatrix(m, n, rowptr, colind, values)`, or conversion of a matrix (`CSRMatrix(S)` for `S::SparseMatrixCSC`) |
| `DIAMatrix`, `ELLMatrix`, `BSRMatrix`, `BSCMatrix`, `HYBMatrix`, `SELLMatrix` | extended formats, converted from CSR/CSC |
| `SparseTensor` | N-dimensional sparse array ([tensors](tensors.md)) |

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
| triangular or diagonal | native sparse substitution (`sparse_triangular_solve`) |
| Hermitian | multifrontal Cholesky, or Bunch–Kaufman if not positive definite |
| complex symmetric | Bunch–Kaufman |
| other square | multifrontal LU |
| tall | multifrontal QR (least squares) |

| Engine | Functions |
|---|---|
| LU | `multifrontal_lu`, `sparse_lu`, `sparse_lu_static`, `sparse_lu_blocked` |
| Cholesky | `multifrontal_cholesky`, `supernodal_cholesky`, `sparse_cholesky`, `symbolic_cholesky` |
| indefinite LDLᵀ | `sparse_ldlt` (Bunch–Kaufman within fronts), `ldl_inertia` |
| QR | `sparse_qr`, `sparse_apply_q!` |
| out of core | `out_of_core_lu`, `out_of_core_cholesky`, `out_of_core_memory_usage` |
| triangular | `sparse_triangular_solve` |

## Iterative solvers and preconditioners

- **Solvers.** `cg`, `pcg`, `gmres`, `fgmres`, `bicgstab`, `minres`, `tfqmr`,
  `qmr`, `idrs`, and their block variants `block_cg` and `block_gmres`. Each
  returns the solution, the convergence status, the iterations and the
  actually recomputed residual.
- **Preconditioners.** `ilu0`, `ilut`, `ic0`, `ict`, `spai`, `ainv`, `amg`,
  `relaxation_preconditioner`, `polynomial_preconditioner` and
  `additive_schwarz`.

## Orderings and graphs

The available ordering families:

- **minimum degree:** `amd_ordering`, `colamd_ordering`,
  `minimum_degree_ordering`, `multiple_minimum_degree`,
  `column_minimum_degree`;
- **profile and bandwidth:** `reverse_cuthill_mckee`;
- **nested dissection:** `nested_dissection`, `multilevel_nested_dissection`.

The graph tools are:

- `connected_components`, `level_sets`, `multilevel_partition`,
  `refine_partition`;
- bipartite matchings (`bipartite_matching`, `weighted_bipartite_matching`);
- choice of an ordering: `sparse_ordering`, `select_sparse_ordering`,
  `ordering_metrics`, `compare_orderings`.

## Input and output

`read_matrix_market` and `write_matrix_market` read and write the Matrix
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
