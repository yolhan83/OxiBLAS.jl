# Tenseurs

## `einsum` et contractions

`einsum(spec, A...)` évalue une notation d'Einstein sur des tableaux denses,
des `SparseTensor` ou des matrices creuses. Les contractions par paires passent
par le GEMM natif (TTGT).

```@example tens
using OxiBLAS
A = randn(16, 16, 16); B = randn(16, 16)
C = einsum("abc,cd->abd", A, B)            # allocation
P = plan_einsum("abc,cd->abd", A, B)       # plan réutilisable
D = similar(C); ws = EinsumWorkspace(Float64)
einsum!(D, P, A, B; workspace=ws)          # en place, workspace réutilisé
D ≈ C
```

- **Plans.** `plan_einsum(spec, A...; optimize)` choisit l'ordre des
  contractions : `:greedy` par défaut, `:beam`, `:optimal`, ou `:none` pour
  la contraction directe. Le plan est immuable et sûr entre threads.
- **Évaluation en place.** `einsum!(C, plan, A...; α, β, workspace)` écrit
  dans `C`. Les temporaires viennent de l'`EinsumWorkspace`.
- **Plans découpés.** `plan_einsum_sliced` borne la mémoire en découpant le
  calcul ; `einsum_layout_report` décrit les dispositions mémoire choisies.
- **Contractions explicites.** `contract(A, B, axesA, axesB)` et `contract!`
  contractent les axes donnés (conjugaisons explicites).
- **Produits spécialisés.** `outer_product`, `mode_product(A, U, mode)`,
  `batched_matmul` et `batched_matmul!`.

## Tenseurs creux

`SparseTensor(dims, coordonnées, valeurs)` stocke les indices linéaires triés
(ordre colonne).

```@example tens
X = SparseTensor((30, 30, 30), [(rand(1:30), rand(1:30), rand(1:30)) for _ in 1:500], randn(500))
U = randn(30, 4)
Y = einsum("abc,cz->abz", X, U)            # creux × dense
size(Y)
```

- Les produits creux × creux utilisent un SpGEMM par colonnes sur les
  dépliages.
- Les opérandes matriciels (CSR, CSC, `SparseMatrixCSC`, `SparseTensor` 2-D)
  restent en CSC natif.
- `einsum!` vers une destination creuse livre directement le résultat CSC.
- Le résultat d'un calcul creux est un `SparseTensor`.

## Décompositions

| Fonction | Résultat |
|---|---|
| `hosvd(A; ranks)` | Tucker (`TuckerFactor`) |
| `tensor_train(A; maxrank, rtol)` | train de tenseurs (`TensorTrainFactor`) |
| `cp_als(A, rank)` | CP par moindres carrés alternés (`CPFactor`) |
| `reconstruct(F)` | tenseur reconstruit |

## Performances

Le banc haut niveau compare `einsum` à TensorOperations ([Performances](performance.md)).
