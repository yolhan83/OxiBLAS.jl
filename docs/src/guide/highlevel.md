# API haut niveau

L'API haut niveau reprend les noms et les conventions de `LinearAlgebra`, au-dessus
des moteurs natifs.

- **Des fonctions propres à OxiBLAS.** `OxiBLAS.lu !== LinearAlgebra.lu`.
  Elles ne sont pas exportées : on écrit `OxiBLAS.lu(A)`, `OxiBLAS.ldiv(F, b)`.
- **Les types LinearAlgebra.** Les factorisations (`OxiBLAS.LU`,
  `OxiBLAS.Cholesky`, `OxiBLAS.QR`…) sont des sous-types de
  `LinearAlgebra.Factorization` : `F \ b`, `ldiv!(Y, F, b)`, `F'`, `inv(F)`,
  `det(F)` fonctionnent comme pour les factorisations de `LinearAlgebra`.
- **Pas de repli caché.** Sur une matrice qu'aucun moteur OxiBLAS ne
  prend en charge, l'appel renvoie une `MethodError`, jamais un calcul
  BLAS/LAPACK silencieux.

```@example hl
using OxiBLAS, LinearAlgebra
A = randn(60, 60) + 60I; B = randn(60, 3)
F = OxiBLAS.lu(A)
X = F \ B                       # générique LinearAlgebra, primitives OxiBLAS
F2 = OxiBLAS.factorize(Hermitian(A * A' + I))   # Cholesky choisi par la structure
S = OxiBLAS.svd(randn(40, 20))
(typeof(F2), size(S.U), norm(A * X - B) < 1e-10)
```

## Factorisations et fonctions

| Fonction | Moteur |
|---|---|
| `OxiBLAS.lu`, `lu!` | `lu_factor` (dense), `multifrontal_lu` (creux) |
| `OxiBLAS.cholesky`, `cholesky!` | `cholesky_factor`, `multifrontal_cholesky` |
| `OxiBLAS.ldlt`, `bunchkaufman` (et `!`) | `ldlt_factor`, `sparse_ldlt` |
| `OxiBLAS.qr` (`NoPivot`, `ColumnNorm`, creux), `qr!`, `lq`, `lq!` | `qr_factor`, `qr_pivoted`, `sparse_qr` |
| `OxiBLAS.svd`, `svdvals` (et `!`) | `svd_factor`, `singular_values` |
| `OxiBLAS.eigen`, `eigvals`, `eigvecs`, `eigmax`, `eigmin` | `eigen_hermitian`, `eigen_general`, `eigenvalues` |
| `OxiBLAS.eigen(A, B)`, `eigvals(A, B)` | `eigen_generalized` ou QZ (`eigen_pencil`) |
| `OxiBLAS.schur(A)`, `schur(A, B)`, `ordschur`, `hessenberg` | `schur_factor`, `qz_factor`, `hessenberg_factor` |
| `OxiBLAS.factorize` | choix selon la structure (voir ci-dessous) |
| `OxiBLAS.det`, `logdet`, `logabsdet`, `inv`, `rank`, `cond`, `pinv`, `nullspace`, `isposdef` | via les moteurs ; `inv` d'une matrice creuse est refusé |
| `OxiBLAS.norm`, `opnorm`, `kron`, `sylvester`, `lyap` | moteurs natifs |
| `OxiBLAS.exp`, `log`, `sqrt`, `sin`, `cos` | fonctions matricielles ; sur un nombre, celles de `Base` |
| `OxiBLAS.mul`, `mul!` | produit dense, creux et structuré |
| `OxiBLAS.spdiagm`, `sparsevec` | constructeurs creux au format OxiBLAS |

## Résolution linéaire : `ldiv`, `ldiv!` et `\`

| Forme | Sémantique |
|---|---|
| `OxiBLAS.ldiv!(Y, F, B)` | réutilise la factorisation `F` ; `Y` reçoit la solution, `B` intact |
| `OxiBLAS.ldiv!(Y, A, B)` | factorise d'abord par `OxiBLAS.factorize(A)` |
| `OxiBLAS.ldiv(F, B)`, `OxiBLAS.ldiv(A, B)` | formes allouantes, `Vector` ou `Matrix` selon `B` |
| `A \ B`, `LinearAlgebra.ldiv!(Y, A, B)` | sur les formats creux OxiBLAS et leurs enveloppes |
| `LinearAlgebra.ldiv!(A, B)` | systèmes carrés : `B` est écrasé par la solution |

```@example hl
using SparseArrays
S = sprand(300, 300, 0.01) + 10I; A = CSRMatrix(S); b = randn(300)
F = OxiBLAS.factorize(A)
x1 = A \ b; x2 = OxiBLAS.ldiv(F, b); y = similar(b); OxiBLAS.ldiv!(y, A, b)
x1 ≈ x2 ≈ y
```

- **Dimensions.** `B` a `size(F, 1)` lignes et `Y` en a `size(F, 2)`.
- **Systèmes rectangulaires.** Un système haut donne la solution des moindres
  carrés.
- **Résolution en place.** `Y === B` est autorisé pour un système carré. Un
  chevauchement partiel entre `Y` et `B` est refusé.
- **Second membre et destination.** Un `B` creux est lu via une copie dense.
  Une destination creuse est refusée : la solution est dense.
- **Précision.** Le calcul se fait dans le type de la factorisation. Un
  facteur réel et un `B` complexe de même précision sont résolus comme deux
  systèmes réels.

## Choix de `OxiBLAS.factorize`

| Entrée | Factorisation |
|---|---|
| dense carrée hermitienne | Cholesky, ou Bunch–Kaufman si elle n'est pas définie positive |
| dense carrée | LU |
| dense rectangulaire | QR à pivot de colonnes |
| creuse triangulaire (testé en O(nnz)) | substitution creuse native |
| creuse hermitienne | Cholesky, ou Bunch–Kaufman |
| creuse symétrique complexe | Bunch–Kaufman |
| creuse carrée | LU multifrontale |
| creuse haute | QR multifrontale |
| `Diagonal`, `Bidiagonal`, triangulaires, bandes, compactes | moteurs structurés ([guide](structured.md)) |

## Limitations

- **Systèmes creux.**
  - Un système creux large (sous-déterminé) n'est pas résolu : la QR creuse
    exige m ≥ n.
  - La QR creuse n'est pas révélatrice du rang : `F \ b` lève
    `SingularException` si R est numériquement déficient.
- **Absence d'état d'échec.** Les moteurs creux (LU, Cholesky) et le LDLᵀ
  dense lèvent l'exception standard même avec `check=false`.
- **Options refusées.**
  - `bunchkaufman(A, true)` (pivot rook) n'est pas disponible.
  - Un `PackedMatrix` indéfini n'a pas de factorisation indéfinie.
- **Pas de solve.** `Eigen`, `Schur` et `Hessenberg` n'ont pas de
  primitive de résolution, comme dans `LinearAlgebra`.
