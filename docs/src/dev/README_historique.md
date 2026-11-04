# README historique (versions 0.1 à 0.6)

> Ancien README du dépôt, conservé tel qu'il était avant la restructuration du 4 novembre 2026. Les commandes de benchmark qu'il cite sont désormais dans `benchmark/archive/`.

> Reprise en phase A supervisée : voir [protocole](PROTOCOLE_SUPERVISE.md) et [état courant](REPRISE_SUPERVISEE.md). Les indications d’arrêt précédentes sont historiques.

## OxiBLAS.jl

Le [bilan d’audit prioritaire](AUDIT_PRIORITAIRE.md) décrit les corrections
mémoire, le verrou numérique avant benchmarks et les couvertures encore inconnues.

Le lot [tenseurs : chemins et tranches](LOT_TENSEURS_PLANIFICATION.md) ajoute
la recherche beam/exacte et les contractions découpées, avec validation préparée.

La [campagne unique préparée](RUN_ALL.md) est accessible par `bash run_all.sh`
**après autorisation** : suite complète, benchmarks et quatre colonnes
Julia/Rust/OpenBLAS/Accelerate dans un seul log. Elle n’a pas été exécutée.

Portage **partiel** d’OxiBLAS en Julia natif, sous Apache-2.0.
L’état courant est **0.6.0-dev, non validé numériquement**. À la demande de
l’utilisateur, aucun test ni benchmark de cet incrément n’a été lancé.
Les résultats v0.5 restent une référence historique et ne valident pas ces
modifications. Le [bilan de développement](DEVELOPMENT_v0.6.md) distingue
la première passe. Le **[suivi courant](REPRISE_SUPERVISEE.md)** décrit
la reprise et les limites ; le [point d’arrêt antérieur](POINT_ARRET_2026-11-02.md)
reste historique. Les derniers ajouts comprennent estimations stochastiques,
plans/workspaces einsum, méthodes spectrales par blocs, SVD creuse itérative
et mises à jour SVD incrémentales par lignes/colonnes, filtres spectraux
polynomiaux, recherche par intervalle et graphes/ordonnancements multiniveau,
puis [LDL pivoté, LU multifrontale/statique et LU/Cholesky sur disque](LOT_SPARSE_DIRECT.md),
et la [complétion de la ligne Graphes](LOT_GRAPHES_COMPLETION.md),
tous écrits et non exécutés. La [correction de l’architecture numérique](LOT_PRECISION_ETENDUE.md)
supprime le scalaire maison et adapte les algorithmes aux flottants génériques,
avec fournisseurs optionnels et campagnes préparées, sans exécution.

Les calculs sont écrits en Julia, avec des vecteurs LLVM pour les micro-noyaux.
`LinearAlgebra` fournit uniquement les types `Transpose` et `Adjoint` au code du
paquet. Aucun calcul n’est délégué à BLAS, LAPACK, Accelerate ou au Rust d’origine.
Ces bibliothèques ne servent que de références dans les tests et benchmarks.
Aucune dépendance à SIMD.jl ou LoopVectorization.jl.

```julia
using Pkg
Pkg.develop(path="/chemin/vers/OxiBLAS.jl")
using OxiBLAS

A = randn(Float64, 513, 257)
B = randn(Float64, 257, 389)
C = Matrix{Float64}(undef, 513, 389)
w = GemmWorkspace(Float64, 513, 389, 257)

gemm!(C, A, B, 0.7, 0.0; workspace=w)
```

Pour une multiplication transposée, la sortie doit avoir les dimensions du
produit :

```julia
G = Matrix{Float64}(undef, 257, 257)
gemm!(G, A, A; transA=:C)   # équivalent à A' * A
println(kernel_info(Float64))

F = lu_factor(randn(65,65))
x = solve(F, randn(65))

M = randn(ComplexF64, 17, 17)
S = schur_factor(M)         # M ≈ S.Z * S.T * S.Z'
E = eigen_hermitian(M + M') # vecteurs E.vectors, valeurs réelles E.values
V = svd_factor(M)           # M ≈ V.U * Diagonal(V.S) * V.Vt
```

GEMM accepte `Float32`, `Float64`, `ComplexF32`, `ComplexF64`, les vues à strides
non unitaires, `transpose` et `adjoint`. Les entrées recouvrant la sortie sont
copiées avant mutation. β=0 ne lit pas l’ancienne sortie ; α=0 ou K=0 ne lit pas
les entrées. Les vues de sortie comportant des indices répétés sont rejetées.

Les voies `:reference`, `:blocked`, `:packed`, `:register`, `:simd` sont accessibles
avec le mot-clé `algorithm`. `kernel=:auto` sélectionne NEON, AVX2+FMA,
AVX-512+FMA ou le repli scalaire selon les capacités du compilateur Julia.
Les variantes x86 sont implémentées mais restent à valider sur du matériel x86.

Le parallélisme se contrôle avec `julia --threads=8`, `threaded=false` et
`GemmConfig(parallel_threshold=...)`. Un workspace possède un couple de panneaux
par tâche de travail. Son partage entre appels est protégé par verrou ; pour
des appels simultanés indépendants, utiliser plusieurs workspaces.

Depuis ce dossier, exécuter :

```sh
julia --startup-file=no --threads=4 --check-bounds=yes --project=. test/runtests.jl
julia --startup-file=no -C generic --threads=1 --check-bounds=yes --project=. test/runtests.jl
julia --startup-file=no --project=benchmark -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --startup-file=no --threads=8 --project=benchmark benchmark/benchmarks.jl --large
julia --startup-file=no --project=benchmark benchmark/inspect_assembly.jl
```

Les CSV, les configurations matérielles et l’assembleur inspecté sont conservés
dans `reports/` (hors dépôt). La méthode et les limites des mesures sont décrites dans
[docs/PERFORMANCE.md](PERFORMANCE.md), et les résultats finaux dans
[docs/VALIDATION.md](VALIDATION.md) pour l’incrément 0.1 et
[docs/SPECTRAL_VALIDATION.md](SPECTRAL_VALIDATION.md) pour la version 0.2.
Les mesures BLAS 3 v0.3 sont dans
[docs/BLAS3_VALIDATION.md](BLAS3_VALIDATION.md). La validation BLAS 1 v0.4
figure dans [docs/LEVEL1_VALIDATION.md](LEVEL1_VALIDATION.md). La validation
historique v0.5 est [COMPACT_PARALLEL_VALIDATION.md](COMPACT_PARALLEL_VALIDATION.md).
L’environnement exact de la campagne
Julia 1.14 est archivé dans `reports/benchmark-Project-julia1.14.toml` et
`reports/benchmark-Manifest-julia1.14.toml` ; l’environnement de benchmark courant
est résolu pour la version Julia utilisée par l’appelant.

Sur macOS, la comparaison avec Accelerate utilise un processus distinct :

```sh
OXIBLAS_REFERENCE=accelerate julia --startup-file=no --threads=1 --project=benchmark benchmark/benchmarks.jl --large
```

Les décompositions Hessenberg/Schur, EVD symétrique/hermitien et SVD Jacobi ont
un script de campagne (un thread, matrices 16–256, SVD carrées et rectangulaires),
non exécuté intégralement dans les résultats v0.2 :

```sh
julia --startup-file=no --threads=1 --project=benchmark benchmark/spectral.jl
OXIBLAS_REFERENCE=accelerate julia --startup-file=no --threads=1 --project=benchmark benchmark/spectral.jl
```

Ces algorithmes spectraux sont séquentiels. Ils signalent la non-convergence et
restent à optimiser pour les grandes dimensions. La SVD ne forme pas A'·A.

Le dépôt Rust de référence est figé à
[`8d82bb6`](https://github.com/cool-japan/oxiblas/tree/8d82bb6bf1df133aa90525aa9f478ebe41514f9b).
L’inventaire couvre ses 370 fichiers source Rust et distingue les correspondances
partielles des modules non portés : [upstream_inventory.csv](upstream_inventory.csv).
Voir aussi `NOTICE`, `LICENSE`, [API](API.md) et la
[mission d’origine](MISSION.md).

Les BLAS 3 structurés et GEMMT réutilisent le moteur GEMM natif ; les principes
et limites sont décrits dans [BLAS3_DESIGN.md](BLAS3_DESIGN.md).
Le benchmark conserve une copie exacte de `level3.jl` v0.2 pour comparer
les deux implémentations avec les mêmes noyaux GEMM :

```sh
julia --startup-file=no --threads=1 --project=benchmark benchmark/level3.jl
OXIBLAS_REFERENCE=accelerate julia --startup-file=no --threads=1 --project=benchmark benchmark/level3.jl
```

Les opérations BLAS 1 complémentaires sont natives :

```julia
F = rotmg(1.0, 1.0, 3.0, 4.0)
x = [3.0, 1.0]; y = [4.0, 2.0]
rotm!(x, y, F.params)
s = dsdot(Float32[1, 2, 3], Float32[4, 5, 6]) # Float64
r = sdsdot(0.7f0, Float32[1, 2, 3], Float32[4, 5, 6]) # Float32
```

ROTM/ROTMG concernent Float32/Float64 réels ; DSDOT/SDSDOT prennent des
vecteurs Float32 et accumulent en Float64. Leurs benchmarks et leur inspection
LLVM/assembleur s'exécutent ainsi :

```sh
julia --startup-file=no --threads=1 --project=benchmark benchmark/level1.jl
OXIBLAS_REFERENCE=accelerate julia --startup-file=no --threads=1 --project=benchmark benchmark/level1.jl
julia --startup-file=no --project=. benchmark/inspect_level1.jl
```

Depuis v0.5, les opérations packed/banded lisent directement les coefficients
utiles. Les mises à jour de rang k et GEMMT peuvent distribuer des tuiles
indépendantes, sans GEMM multithread imbriqué. Conception et limites :
[COMPACT_PARALLEL_DESIGN.md](COMPACT_PARALLEL_DESIGN.md).

```sh
julia --startup-file=no --threads=1 --project=benchmark benchmark/compact.jl
OXIBLAS_REFERENCE=accelerate julia --startup-file=no --threads=1 --project=benchmark benchmark/compact.jl
julia --startup-file=no --threads=1 --project=benchmark benchmark/tile_parallel.jl
julia --startup-file=no --threads=2 --project=benchmark benchmark/tile_parallel.jl
julia --startup-file=no --threads=4 --project=benchmark benchmark/tile_parallel.jl
julia --startup-file=no --threads=8 --project=benchmark benchmark/tile_parallel.jl
julia --startup-file=no --project=. benchmark/inspect_compact.jl
```

### Lot QZ/Schur préparé (non exécuté)

Le [port algorithmique QZ/Schur](QZ_SCHUR_PORT.md) décrit la référence Rust,
les corrections numériques démontrées et les variantes Julia natives ajoutées.
`qz_factor(A,B)` conserve maintenant une forme réelle pour les entrées réelles ;
`form=:complex` choisit explicitement la forme triangulaire complexe.
`schur_multishift` conserve aussi le réel et accepte un `SchurWorkspace` réutilisable.
Réordonnancements par blocs, TREVC/TRSNA et diagnostics de condition sont exposés.
Les tests et douze cas supplémentaires du rapport comparatif sont préparés ;
aucun résultat numérique ni gain de performance actuel n'est revendiqué.

La [SVD réellement partielle](LOT_SVD_PARTIELLE.md) est préparée :
`svd_selected(A; indices=...)` ou `interval=(low,high)` ne calcule que les rangs
demandés et leurs vecteurs. `svd_truncated(A,k)` utilise ce moteur par défaut.
Les tests et comparaisons GESVDX sont écrits, sans exécution numérique.

[Arnoldi redémarré et XTrace optimisé](LOT_ARNOLDI_XTRACE.md) sont également
préparés : `thick_restart_arnoldi` / `block_arnoldi(...; restart=true)`, soft locking
et réorthogonalisation sélective ; `trace_xtrace` optimisé et `trace_xtrace_loo`
comme référence explicite. Tous ces nouveaux lots attendent la validation supervisée.

Le [lot mémoire et triangulaires](LOT_MEMOIRE_WORKSPACES_TRIANGULAIRES.md)
prépare un défaut Golub–Kahan économe pour `sparse_svd_lanczos`,
`SparseSVDWorkspace`, `triangular_workspace` et des feuilles natives TRMM/TRSM.
La dilatation et les chemins génériques restent accessibles. Code et campagnes
écrits, sans test ni benchmark exécuté.
