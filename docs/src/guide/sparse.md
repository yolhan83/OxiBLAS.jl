# Matrices creuses

## Formats

| Format | Constructeur |
|---|---|
| `CSRMatrix`, `CSCMatrix`, `COOMatrix` | `CSRMatrix(m, n, rowptr, colind, values)`, ou conversion d'une matrice (`CSRMatrix(S)` pour `S::SparseMatrixCSC`) |
| `DIAMatrix`, `ELLMatrix`, `BSRMatrix`, `BSCMatrix`, `HYBMatrix`, `SELLMatrix` | formats étendus, convertis depuis CSR/CSC |
| `SparseTensor` | tableau creux à N dimensions ([tenseurs](tensors.md)) |

Ces types sont des `SparseArrays.AbstractSparseMatrix`. Ils interopèrent avec
SparseArrays :
- conversions `SparseMatrixCSC(A)` et `CSCMatrix(S)` (sans copie depuis un
  `SparseMatrixCSC`) ;
- `nnz`, `findnz`, `getindex`, broadcast ;
- `hcat`, `vcat`, `kron`, `blockdiag`, `fkeep!`.

`OxiBLAS.spdiagm` et `OxiBLAS.sparsevec` construisent directement les formats
OxiBLAS.

## Produits

`sparse_mul!(y, A, x, α, β; trans)` calcule le produit creux × dense,
`sparse_matmul(A, B)` le produit creux × creux. `A * B`, `A * x` et
`OxiBLAS.mul!` dispatchent vers les mêmes noyaux.

## Résolution : `\` et factorisations

```@example sp
using OxiBLAS, SparseArrays, LinearAlgebra
n = 400; S = sprand(n, n, 0.01) + 10I; A = CSRMatrix(S); b = randn(n)
x = A \ b                         # choix de factorisation selon la structure
F = OxiBLAS.lu(A)                 # LU multifrontale, F.L*F.U == A[F.p,F.q]
P = CSRMatrix(S * S' + I)
C = OxiBLAS.cholesky(Hermitian(P))
(norm(S * x - b) / norm(b), norm(S * (F \ b) - b) / norm(b), norm(P * (C \ b) - b) / norm(b))
```

`A \ B` sur un format OxiBLAS suit l'ordre de SparseArrays :

| Structure (testée en O(nnz)) | Factorisation |
|---|---|
| triangulaire ou diagonale | substitution creuse native (`sparse_triangular_solve`) |
| hermitienne | Cholesky multifrontale, ou Bunch–Kaufman si elle n'est pas définie positive |
| symétrique complexe | Bunch–Kaufman |
| autre carrée | LU multifrontale |
| haute | QR multifrontale (moindres carrés) |

| Moteur | Fonctions |
|---|---|
| LU | `multifrontal_lu`, `sparse_lu`, `sparse_lu_static`, `sparse_lu_blocked` |
| Cholesky | `multifrontal_cholesky`, `supernodal_cholesky`, `sparse_cholesky`, `symbolic_cholesky` |
| LDLᵀ indéfini | `sparse_ldlt` (Bunch–Kaufman par fronts), `ldl_inertia` |
| QR | `sparse_qr`, `sparse_apply_q!` |
| sur disque | `out_of_core_lu`, `out_of_core_cholesky`, `out_of_core_memory_usage` |
| triangulaire | `sparse_triangular_solve` |

## Solveurs itératifs et préconditionneurs

- **Solveurs.** `cg`, `pcg`, `gmres`, `fgmres`, `bicgstab`, `minres`,
  `tfqmr`, `qmr`, `idrs`, et leurs variantes par blocs `block_cg` et
  `block_gmres`. Chacun renvoie la solution, le statut de convergence, les
  itérations et le résidu réellement recalculé.
- **Préconditionneurs.** `ilu0`, `ilut`, `ic0`, `ict`, `spai`, `ainv`, `amg`,
  `relaxation_preconditioner`, `polynomial_preconditioner` et
  `additive_schwarz`.

## Ordonnancements et graphes

Les familles d'ordonnancement disponibles :
- **degré minimum :** `amd_ordering`, `colamd_ordering`,
  `minimum_degree_ordering`, `multiple_minimum_degree`, `column_minimum_degree` ;
- **profil et bande :** `reverse_cuthill_mckee` ;
- **dissection emboîtée :** `nested_dissection`, `multilevel_nested_dissection`.

Les outils de graphes sont :
- `connected_components`, `level_sets`, `multilevel_partition`,
  `refine_partition` ;
- les couplages bipartis (`bipartite_matching`, `weighted_bipartite_matching`) ;
- la sélection d'un ordonnancement : `sparse_ordering`,
  `select_sparse_ordering`, `ordering_metrics`, `compare_orderings`.

## Entrées et sorties

`read_matrix_market` et `write_matrix_market` lisent et écrivent le format
Matrix Market.

## Limitations

- **Système creux large (sous-déterminé).** Il n'a pas de solveur :
  `OxiBLAS.qr` refuse une matrice large par une `ArgumentError`, et `A \ b`
  aussi.
- **QR creuse sans détection du rang.** Elle n'est pas révélatrice du rang :
  `SingularException` sur une matrice numériquement déficiente.
- **Destination creuse refusée.** Une solution est dense.
- **Inverse refusé.** `OxiBLAS.inv` d'une matrice creuse lève une
  `ArgumentError` : il faut factoriser puis résoudre.
- **Pas d'état d'échec.** Les moteurs LU et Cholesky creux lèvent l'exception
  standard même avec `check=false`.
- **`SparseMatrixCSC` non modifié.** OxiBLAS n'ajoute aucune méthode de
  `LinearAlgebra`/`SparseArrays` sur ce type : `S \ b` reste celui de
  SparseArrays (UMFPACK, CHOLMOD). `OxiBLAS.ldiv(S, b)` et `OxiBLAS.lu(S)`
  passent par les moteurs OxiBLAS.

Conception historique :  [LOT_SPARSE_DIRECT](../dev/LOT_SPARSE_DIRECT.md) et
[LOT_GRAPHES_CREUX](../dev/LOT_GRAPHES_CREUX.md).
