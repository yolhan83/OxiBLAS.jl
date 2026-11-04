# Performances et benchmarks

## Banc haut niveau

`benchmark/run.jl` mesure les opérations qu'un utilisateur appelle, contre la
référence appropriée :

| Famille | Référence |
|---|---|
| GEMM, `OxiBLAS.mul!`, BLAS haut niveau | OpenBLAS (`LinearAlgebra`), Accelerate sur macOS |
| LU, Cholesky, QR, SVD, valeurs propres | LAPACK d'OpenBLAS, Accelerate |
| `A \ b`, `OxiBLAS.ldiv`, `ldiv!` | même problème avec `LinearAlgebra` (denses) ou SparseArrays (creux) |
| factorisations et produits creux | SparseArrays (UMFPACK, CHOLMOD) |
| `einsum` | TensorOperations |
| matrices structurées | types structurés de `LinearAlgebra` |

```sh
julia --project=benchmark -e 'using Pkg; Pkg.instantiate()'
julia --project=benchmark --threads=1 benchmark/run.jl              # tous les cas
julia --project=benchmark --threads=1 benchmark/run.jl gemm lu       # sélection
OXIBLAS_REFERENCE=accelerate julia --project=benchmark benchmark/run.jl
```

**Méthode.**

- **Mêmes conditions.** Même problème mathématique, mêmes données et même
  nombre de threads pour OxiBLAS et la référence.
- **Vérification.** Chaque résultat est comparé à la référence avant les
  mesures.
- **Mesures à chaud.** Un premier appel compile et échauffe ; les temps
  rapportés sont des médianes de mesures à chaud.
- **Allocation séparée.** Les formes allouantes (`OxiBLAS.lu(A)`, `A \ b`) et
  non allouantes (`ldiv!`, `gemm!` avec espace de travail) sont des cas
  distincts.
- **Compilation ciblée.** Seuls les cas sélectionnés sont compilés.
- **Sorties.** Temps médian, octets alloués et rapport OxiBLAS/référence, en
  CSV et en Markdown dans `benchmark/results/`.

Les benchmarks ne sont pas lancés par la CI : les temps mesurés sur les
machines GitHub ne sont pas assez stables.

## Campagnes historiques

Les micro-benchmarks de noyaux et les campagnes complètes (BLAS 1/2/3, LAPACK,
SVD dense, creux, tenseurs, graphes) sont archivés dans `benchmark/archive/`.
Ils restent utilisables pour de futures optimisations ; leurs rapports de
campagne ne font pas partie du dépôt. La méthode historique et l'inspection du
code machine sont décrites dans [PERFORMANCE](../dev/PERFORMANCE.md).

## Latence et précompilation

Le paquet précompile, avec PrecompileTools, une petite charge de chemins
courants :
- GEMM `Float32` et `Float64` ;
- LU et Cholesky denses ;
- une factorisation creuse et sa résolution ;
- `OxiBLAS.ldiv` et `\` sur une `CSRMatrix` ;
- une contraction `einsum`.

Le premier appel de ces chemins ne paie donc plus la compilation. Les autres
types et variantes se compilent au premier usage.

Mesures sur Apple M5 Max, Julia 1.14, chaque chemin dans un processus neuf
(`benchmark/latency/run_all.jl`, résultats dans `benchmark/latency/latency_*.tsv`) :

| Premier appel | Sans charge | Avec charge |
|---|---|---|
| `gemm!` Float64 | 0,38 s | 0,09 s |
| `OxiBLAS.lu(A) \ b` | 2,85 s | 0,10 s |
| `OxiBLAS.cholesky` | 2,74 s | 0,01 s |
| `OxiBLAS.ldiv(A, b)` | 13,5 s | 0,13 s |
| `A \ b` creux (Cholesky, LU) | 7,2 s | 0,005 s |
| `einsum` | 3,2 s | 0,06 s |
| `using OxiBLAS` | 0,03 s | 0,15 s |

**Coût de la charge.** La précompilation du paquet passe d'environ 3 s à
20 s, et son cache de 10 Mo à environ 50 Mo.

**Désactiver la charge.** Pour la désactiver, ajouter `precompile_workload = false`
dans la section `[OxiBLAS]` de `LocalPreferences.toml`.
