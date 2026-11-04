# Factorisations denses et LAPACK

Les moteurs denses sont écrits en Julia et réutilisent le GEMM natif pour
leurs mises à jour par blocs. Chacun renvoie un objet propre (`LUFactor`,
`QRFactor`…), utilisable avec `solve` ou par l'[API haut niveau](highlevel.md),
qui l'enveloppe dans une `LinearAlgebra.Factorization`.

```@example fact
using OxiBLAS, LinearAlgebra
A = randn(80, 80) + 80I; b = randn(80)
F = lu_factor(A)                 # F.L*F.U ≈ A[F.p,:]
x = solve(F, b)
H = Hermitian(A * A')
C = cholesky_factor(Matrix(H))   # triangle inférieur, F.L*F.L' ≈ A
Q = qr_factor(randn(100, 40))    # moindres carrés : least_squares(Q, y)
norm(A * x - b) / norm(b)
```

## Systèmes linéaires

| Fonction | Résultat |
|---|---|
| `lu_factor(A; blocksize, check, threaded, pivot=:partial)` | LU par blocs ; `pivot=:complete` ou `:rook` (`lu_complete`, `lu_rook`) |
| `cholesky_factor(A; blocksize, threaded)` | Cholesky, triangle inférieur |
| `ldlt_factor(A; uplo, hermitian, pivot=:bunchkaufman)`, `bunchkaufman_factor` | blocs D 1×1/2×2, `A[p,p] ≈ L·D·Lᴴ` ; `pivot=:none` possible |
| `aasen_factor` | factorisation d'Aasen |
| `solve(F, b)`, `solve_transposed` | résolution avec une factorisation |
| `least_squares(A, b)` | moindres carrés, m ≥ n, rang plein |
| `mixed_solve(A, b)` | LU en précision réduite, raffinement en `Float64` |
| `expert_solve`, `refine_solve`, `expert_cholesky`, `expert_symmetric` | équilibrage, raffinement itératif, erreur arrière |
| `native_inverse`, `cholesky_inverse`, `pseudoinverse` | inverses |
| `determinant`, `logabsdet_factor`, `condition_number`, `reciprocal_condition`, `inverse_norm_estimate` | déterminant et conditionnement |

## Orthogonalisations et rang

| Fonction | Résultat |
|---|---|
| `qr_factor(A; blocksize, threaded, pivot=false)`, `qr_pivoted` | QR compact WY ; pivot de colonnes `A[:,F.p] ≈ F.Q*F.R` |
| `lq_factor`, `ql_factor`, `rq_factor`, `complete_orthogonal` | autres décompositions orthogonales |
| `apply_q!`, `qr_q` | application ou formation de Q |
| `numerical_rank`, `null_space`, `left_null_space`, `column_space`, `row_space` | rang et sous-espaces |

## Valeurs propres et valeurs singulières

| Fonction | Résultat |
|---|---|
| `hessenberg_factor`, `hessenberg_q`, `apply_hessenberg_q!` | réduction de Hessenberg |
| `schur_factor`, `schur_multishift`, `eigenvalues`, `eigen_general` | Schur (QR à décalages), valeurs et vecteurs propres généraux |
| `schur_reorder`, `schur_eigenvectors`, `trevc_right`, `trevc_left`, `trsna_s`, `trsna_sep`, `schur_condition` | réordonnancement, vecteurs, conditionnement |
| `tridiagonal_factor`, `eigen_tridiagonal`, `eigen_tridiagonal_dc`, `eigen_tridiagonal_mrrr` | tridiagonal hermitien (QR implicite, diviser-pour-régner, MRRR) |
| `eigen_hermitian`, `eigen_hermitian_dc`, `eigen_hermitian_mrrr`, `eigen_generalized` | problèmes hermitiens, standard et généralisés |
| `qz_factor`, `eigen_pencil`, `qz_reorder`, `qz_condition`, `generalized_sylvester` | faisceaux (QZ réel ou complexe) |
| `svd_factor(A; algorithm)`, `singular_values` | SVD : Jacobi unilatéral (défaut) ou `:golub_kahan` |
| `svd_bidiagonal`, `svd_divide_conquer`, `svd_truncated`, `svd_selected`, `svd_randomized` | variantes et SVD partielles |
| `bidiagonal_factor`, `balance_matrix`, `balance_backtransform` | réductions auxiliaires |

**SVD.** Le Jacobi par défaut garantit une précision relative sur les petites
valeurs singulières, au prix de plusieurs fois le coût de `gesdd`.
`algorithm=:golub_kahan` est la voie rapide (voir le banc haut niveau,
[Performances](performance.md)).

## Fonctions matricielles

`matrix_exp`, `matrix_log`, `matrix_sqrt`, `matrix_power`, `matrix_sign`,
`matrix_sin`, `matrix_cos`, les dérivées de Fréchet (`frechet_exp`,
`frechet_log`, `frechet_sqrt`, `frechet_matrix`), `matrix_norm`, `kron_product`,
`khatri_rao`, `kron_sum` et `kronecker_action`. Les versions haut niveau
`OxiBLAS.exp`, `log`, `sqrt`, `sin` et `cos` conservent la structure quand
elle est garantie (voir [API haut niveau](highlevel.md)).

## Précision étendue

Les algorithmes génériques acceptent d'autres flottants (`BigFloat`, `Float16`,
types de précision étendue fournis par d'autres paquets). Voir
[LOT_PRECISION_ETENDUE](../dev/LOT_PRECISION_ETENDUE.md) et `gemm_extended!`,
`dot_extended`.

## Erreurs

`SingularException` et `PosDefException` sont celles de `LinearAlgebra`
(réexportées sous les mêmes noms). `ConvergenceError` signale une limite
d'itérations atteinte.
