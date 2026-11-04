# BLAS 1, 2, 3 et multiplication matricielle

Toutes les fonctions travaillent sur des tableaux indexés à partir de 1. Les
chemins optimisés couvrent `Float32`, `Float64`, `ComplexF32` et `ComplexF64`.
Les sorties doivent être modifiables et sans indices répétés.

## GEMM

```@example blas
using OxiBLAS
A = randn(Float32, 300, 200); B = randn(Float32, 200, 150); C = zeros(Float32, 300, 150)
gemm!(C, A, B)                           # C ← A·B
gemm!(C, A, B, 2f0, 1f0; transA=:N)      # C ← 2·A·B + C
G = zeros(Float32, 200, 200)
gemm!(G, A, A; transA=:C)                # G ← Aᴴ·A
kernel_info(Float32)
```

- `transA`/`transB` valent `:N`, `:T` ou `:C`. Les vues à pas non unitaire,
  `transpose` et `adjoint` sont acceptés.
- Une entrée qui recouvre la sortie est copiée avant l'écriture. `β = 0` ne lit
  pas l'ancienne sortie ; `α = 0` ou `k = 0` ne lit pas les entrées.
- `algorithm` choisit une voie : `:auto`, `:reference`, `:blocked`, `:packed`,
  `:register`, `:simd`. `kernel` vaut `:auto`, `:scalar`, `:neon`, `:avx2` ou
  `:avx512` ; une ISA indisponible provoque une erreur avant toute écriture.
- **Espace de travail.** `GemmWorkspace(T, m, n, k; config, workers)` réserve
  des panneaux réutilisables. Son partage entre appels est protégé par un
  verrou ; pour des appels simultanés indépendants, il faut un espace par tâche.
- **Parallélisme.** Il est contrôlé par `julia --threads=N`, `threaded=false`
  et `GemmConfig(parallel_threshold=...)`. Le nombre de tâches suit la taille du
  produit.
- Variantes : `gemm_cache_oblivious!`, `gemm_strassen!`, `gemm_winograd!`
  (choisies explicitement), `gemm_batched!`, `gemm_strided_batched!`,
  `gemm_extended!` (types génériques).

## `OxiBLAS.mul!` et `OxiBLAS.mul`

`OxiBLAS.mul!(C, A, B, α, β)` et `OxiBLAS.mul(A, B)` (non exportés) dispatchent
vers le bon noyau selon les types : GEMM dense, produit creux × dense, creux ×
creux, matrices structurées. `A * B` sur les formats creux OxiBLAS passe par la
même couche.

## BLAS 1

| Fonction | Convention |
|---|---|
| `axpy!(y, x, α=1)`, `scal!(x, α)` | y ← y + αx ; x ← αx |
| `copy_vector!(y, x)`, `swap!(x, y)` | copie avec instantané ; échange de vecteurs disjoints |
| `dotu(x, y)`, `dotc(x, y)` | somme sans / avec conjugaison du premier opérande |
| `dot_kahan`, `dot_pairwise` | sommations compensée et par paires |
| `dsdot`, `sdsdot` | vecteurs `Float32`, accumulation `Float64` |
| `nrm2`, `asum`, `iamax`, `iamin` | norme euclidienne à l'échelle ; métrique complexe \|Re\|+\|Im\| |
| `rotg`, `rot!`, `rotmg`, `rotmg!`, `rotm!`, `RotmParams` | rotations de Givens et modifiées |
| `axpy_parallel!`, `scal_parallel!`, `dot_parallel`, `nrm2_parallel`, `asum_parallel` | variantes multitâches explicites |

## BLAS 2

| Fonction | Convention |
|---|---|
| `gemv!(y, A, x, α=1, β=0; trans=:N)` | y ← α·op(A)·x + β·y |
| `ger!(A, x, y, α)`, `gerc!` | mise à jour de rang 1, yᵀ / yᴴ |
| `symv!`, `hemv!` | même ordre que `gemv!`, `uplo=:U` ou `:L` |
| `syr!`, `syr2!`, `her!`, `her2!` | seul le triangle choisi est modifié |
| `trmv!(x, A; uplo, trans, unit)`, `trsv!` | diagonale unité jamais lue |

## BLAS 3

| Fonction | Convention |
|---|---|
| `symm!(C, A, B, α, β; side, uplo)`, `hemm!` | A est la matrice structurée, côté `:L`/`:R` |
| `trmm!(B, A, α; ...)`, `trsm!` | B est modifiée ; `side`, `uplo`, `trans`, `unit` |
| `syrk!(C, A, α, β; ...)`, `herk!`, `syr2k!`, `her2k!` | seul le triangle de C est mis à jour |
| `gemmt!(C, A, B, α=1, β=0; uplo, transA, transB)` | triangle de α·op(A)·op(B) + β·C |
| `triangular_workspace` | espace réutilisable pour TRMM/TRSM |

Les BLAS 3 structurés réutilisent le moteur GEMM et ses options (`algorithm`,
`kernel`, `threaded`, `workspace`, `config`). SYRK, HERK, SYR2K, HER2K et
GEMMT peuvent répartir des tuiles indépendantes entre tâches.

## Pour aller plus loin

- Conventions détaillées de chaque fonction : [Interface Julia indépendante](../dev/API.md).
- Conception des BLAS 3 : [BLAS3_DESIGN](../dev/BLAS3_DESIGN.md) ;
  BLAS 1 : [LEVEL1_DESIGN](../dev/LEVEL1_DESIGN.md).
- Mesures et méthodologie : [Performances](performance.md).
