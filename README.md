# OxiBLAS.jl

[![CI](https://github.com/yolhan83/OxiBLAS.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/yolhan83/OxiBLAS.jl/actions/workflows/CI.yml)
[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://yolhan83.github.io/OxiBLAS.jl/dev/)

Algèbre linéaire dense, structurée, creuse et tensorielle, écrite entièrement
en Julia. OxiBLAS.jl est un portage adapté et étendu de la bibliothèque Rust
[OxiBLAS](https://github.com/cool-japan/oxiblas) (référence figée au commit
`8d82bb6`), sous licence Apache-2.0 (voir `LICENSE` et `NOTICE`).

- **Calcul natif.** Aucun calcul n'est délégué à BLAS, LAPACK, Accelerate,
  SuiteSparse ou au Rust d'origine. Ces bibliothèques servent uniquement de
  références dans les tests et les benchmarks.
- **Noyaux SIMD.** Ils sont écrits avec des vecteurs LLVM, sans dépendance
  SIMD.jl ni LoopVectorization.jl. NEON est validé (Apple Silicon) ; AVX2 et
  AVX-512 sont implémentés mais pas validés sur du matériel x86.
- **Couverture.** BLAS 1/2/3 et GEMM multitâche, factorisations denses (LU,
  Cholesky, LDLᵀ, QR, Schur, QZ, SVD, MRRR…), matrices compactes et bandes,
  formats creux et solveurs directs multifrontaux, solveurs itératifs et
  préconditionneurs, ordonnancements et graphes, méthodes spectrales
  itératives, `einsum` et tenseurs creux.
- **API haut niveau.** Elle reprend les conventions de `LinearAlgebra` :
  `OxiBLAS.lu`, `OxiBLAS.qr`, `OxiBLAS.svd`, `OxiBLAS.factorize`,
  `OxiBLAS.ldiv`, `A \ b` sur les formats creux OxiBLAS… Ses fonctions sont
  propres au paquet et non exportées.

## Installation

Le paquet n'est pas encore enregistré :

```julia
using Pkg
Pkg.add(url="https://github.com/yolhan83/OxiBLAS.jl")
```

Versions de Julia : 1.10 (LTS) et suivantes (`[compat] julia = "1.10"`).

## Exemple

```julia
using OxiBLAS, SparseArrays, LinearAlgebra

A = randn(512, 512); B = randn(512, 512); C = similar(A)
gemm!(C, A, B)                        # GEMM natif

F = OxiBLAS.lu(A + 512I)              # factorisation native (LinearAlgebra.Factorization)
x = OxiBLAS.ldiv(F, randn(512))       # ou F \ b

S = CSRMatrix(sprand(10_000, 10_000, 1e-3) + 10I)
y = S \ ones(10_000)                  # LU multifrontale, choisie par la structure

T = einsum("abc,cd->abd", randn(32, 32, 32), randn(32, 16))
```

## Documentation

La documentation est publiée sur <https://yolhan83.github.io/OxiBLAS.jl/dev/>. Elle se construit aussi localement :

```sh
julia --project=docs -e 'using Pkg; Pkg.instantiate()'
julia --project=docs docs/make.jl     # site dans docs/build/
```

Elle contient :

- **les guides**, une page par famille ;
- **la référence de l'API**, par famille de sources ;
- **les journaux de développement** des versions 0.1 à 0.6.

## Tests

```sh
julia --project=. -e 'using Pkg; Pkg.test()'                           # tests courants + QA
OXIBLAS_TEST_GROUP=extended julia --project=. -e 'using Pkg; Pkg.test()'  # campagnes lourdes
OXIBLAS_TEST_GROUP=sparse julia --project=. -e 'using Pkg; Pkg.test()'    # une famille
```

Les groupes possibles sont :

- **par niveau :** `core` (défaut), `extended`, `all` ;
- **par famille :** `blas`, `structured`, `lapack`, `sparse`, `graphs`,
  `spectral`, `tensors`, `generic`, `api`, `qa`.

Le détail est dans `test/groups.jl`. La QA comprend Aqua (piraterie,
ambiguïtés, paramètres non liés) et ExplicitImports. Elle comprend aussi
`Test.detect_closure_boxes`, à partir de Julia 1.14.

## Benchmarks

```sh
julia --project=benchmark -e 'using Pkg; Pkg.instantiate()'
julia --project=benchmark --threads=1 benchmark/run.jl          # banc haut niveau
julia --project=benchmark --threads=1 benchmark/run.jl gemm lu  # sélection par préfixe
```

Les campagnes historiques et les micro-benchmarks de noyaux sont dans
`benchmark/archive/` (voir `benchmark/README.md`).

## Organisation

```
src/        OxiBLAS.jl, puis utils/ blas/ structured/ lapack/ sparse/ graphs/
            spectral/ tensors/ api/ (un seul module OxiBLAS) et precompile.jl
test/       runtests.jl, groups.jl, testsetup.jl, helpers/, un dossier par famille
benchmark/  run.jl (banc haut niveau), latency/, archive/
docs/       make.jl, src/ (guides, référence générée, journaux de développement)
```
