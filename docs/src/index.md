# OxiBLAS.jl

OxiBLAS.jl est une bibliothèque d'algèbre linéaire écrite entièrement en Julia.
C'est un portage, adapté et étendu, de la bibliothèque Rust
[OxiBLAS](https://github.com/cool-japan/oxiblas) (référence figée au commit
`8d82bb6`).

- **Calcul natif.** Aucun calcul n'est délégué à BLAS, LAPACK, Accelerate,
  SuiteSparse ou au Rust d'origine. Ces bibliothèques ne servent que de
  références dans les tests et les benchmarks. `LinearAlgebra` fournit au code
  du paquet ses types enveloppes (`Transpose`, `Adjoint`, `Hermitian`…), ses
  exceptions et l'interface `Factorization`, jamais ses calculs.
- **Micro-noyaux SIMD.** Ils utilisent des vecteurs LLVM (`llvmcall`), sans
  SIMD.jl ni LoopVectorization.jl. NEON est validé sur Apple Silicon ; les
  variantes AVX2 et AVX-512 sont implémentées mais pas validées sur du
  matériel x86.
- **Types.** Les chemins optimisés couvrent `Float32`, `Float64`, `ComplexF32`
  et `ComplexF64`. Beaucoup d'algorithmes acceptent aussi des flottants
  génériques (`BigFloat`, `Float16`, types de précision étendue), sans les
  mêmes performances.

## Installation

Le paquet n'est pas encore enregistré :

```julia
using Pkg
Pkg.add(url="https://github.com/yolhan83/OxiBLAS.jl")
```

## Premiers exemples

**Multiplication matricielle native.** `gemm!` écrit `C ← α·op(A)·op(B) + β·C`,
avec un espace de travail réutilisable facultatif :

```@example accueil
using OxiBLAS
A = randn(513, 257); B = randn(257, 389); C = zeros(513, 389)
w = GemmWorkspace(Float64, 513, 389, 257)
gemm!(C, A, B, 0.7, 0.0; workspace=w)
C ≈ 0.7 * A * B
```

**API haut niveau.** Elle reprend les noms de `LinearAlgebra`, mais ses
fonctions sont propres à OxiBLAS et ne sont pas exportées :

```@example accueil
using LinearAlgebra      # seulement pour I et la vérification
M = randn(50, 50) + 50I
F = OxiBLAS.lu(M)        # LU native (LinearAlgebra.Factorization)
b = randn(50)
x = OxiBLAS.ldiv(F, b)   # réutilise F ; F \ b donne le même résultat
M * x ≈ b
```

**Matrices creuses.** Les formats OxiBLAS (`CSRMatrix`, `CSCMatrix`,
`COOMatrix`…) prennent `\` en charge, avec un choix de factorisation selon la
structure, comme SparseArrays :

```@example accueil
using SparseArrays
S = sprand(200, 200, 0.02) + 10I
A = CSRMatrix(S)
x = A \ ones(200)
S * x ≈ ones(200)
```

**Tenseurs.** `einsum` contracte des tableaux denses ou creux :

```@example accueil
X = randn(8, 9, 10); Y = randn(10, 7)
Z = einsum("abc,cd->abd", X, Y)
size(Z)
```

## Organisation de la documentation

- Les **guides** présentent chaque famille de fonctionnalités et ses
  conventions.
- La **référence de l'API** regroupe les docstrings par famille de sources.
- La partie **Développement** archive les journaux de conception et de
  validation des versions précédentes.

## Noms partagés avec LinearAlgebra

`OxiBLAS.lu`, `OxiBLAS.qr`, `OxiBLAS.svd`, `OxiBLAS.ldiv`, `OxiBLAS.exp`, etc.
sont des fonctions distinctes de celles de `LinearAlgebra` et de `Base`
(`OxiBLAS.lu !== LinearAlgebra.lu`). Elles ne sont pas exportées : `using
LinearAlgebra, OxiBLAS` ne crée donc aucun conflit de noms. Les extensions de
fonctions de `LinearAlgebra` et de `Base` (par exemple `A \ b` sur une
`CSRMatrix`) ne portent que sur des signatures qui contiennent un type
OxiBLAS : aucune méthode n'est ajoutée pour `Matrix` ou `SparseMatrixCSC`.
