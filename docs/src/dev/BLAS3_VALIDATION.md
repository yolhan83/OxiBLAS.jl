# Validation BLAS 3 v0.3 — 2 novembre 2026

P0 est terminé pour les vérifications locales : revalidation des sources v0.2, rapport spectral ajouté, workflows Linux/macOS/Windows présents et renforcés avec conservation des journaux. Les workflows distants ne sont pas exécutés ici. AVX2/AVX-512 restent non validés sur du matériel x86.

## Suite complète sur les sources finales

| Configuration locale Apple M5 Max arm64 | Tests réussis |
|---|---:|
| Julia 1.14-alpha2, CPU natif, 4 threads | 18200 |
| Julia 1.14-alpha2, CPU générique, 1 thread | 18200 |
| Julia 1.13.3, CPU natif, 4 threads | 18200 |

Bornes vérifiées dans les trois configurations. Journaux `reports/tests-v03-final-*.log`, hashes des sources `reports/baseline-v0.3/source-sha256.txt`, hashes des tests/scripts/docs/CI `reports/baseline-v0.3/artifact-sha256.txt`. Les tests couvrent les quatre types, côtés/triangles, N/T/C, vues inversées et adjointes, alias, triangles/diagonales unité interdits à la lecture, sorties sans lecture pour β=0, entrées ignorées pour α=0/K=0, singularités avant mutation, canaris et workspaces partagés entre tâches. Les références de calcul sont indépendantes du Rust (BLAS/LAPACK Julia et résidus).

Les deux processus Julia 1.14-alpha ont émis un message non fatal de cache objcache/LMDB (MDB_INVALID), conservé dans les logs ; les suites ont néanmoins terminé avec succès. La validation stable Julia 1.13.3 est consignée séparément.

## Mesures locales

492 lignes v0.2/native/OpenBLAS et 156 lignes Accelerate. Tailles 32/128/512, côtés L/R, triangles U/L et transpositions pertinentes, un thread, jusqu’à sept échantillons. Remise à zéro/restauration des sorties hors chronométrage. Les API comprennent l’allocation des panneaux. Les bibliothèques de référence sont confinées aux tests/benchmarks. Environnements et CSV dans `reports/`.

Rapport temps v0.2 / temps natif, médianes à N=512, côté gauche, triangle supérieur, sans transposition ; A ou B rectangulaire 512×384 selon l’opération. Un rapport >1 indique un gain local.

| Type | SYMM | TRMM | TRSM | SYRK | SYR2K |
|---|---:|---:|---:|---:|---:|
| Float32 | 1.07× | 20.61× | 19.49× | 1.47× | 1.38× |
| Float64 | 1.06× | 18.32× | 16.72× | 1.32× | 1.32× |
| ComplexF32 | 1.01× | 5.47× | 19.55× | 1.57× | 1.57× |
| ComplexF64 | 1.03× | 5.26× | 4.65× | 1.65× | 1.57× |

| Type complexe | HEMM | HERK | HER2K |
|---|---:|---:|---:|
| ComplexF32 | 1.02× | 1.57× | 1.55× |
| ComplexF64 | 1.02× | 1.58× | 1.55× |

Temps TRMM correspondants, millisecondes ; la comparaison BLAS externe reste indispensable car un progrès face à la v0.2 ne signifie pas la parité avec BLAS.

| Type | Julia native | OpenBLAS | Accelerate |
|---|---:|---:|---:|
| Float32 | 1.414 | 1.004 | 0.105 |
| Float64 | 2.251 | 1.765 | 0.363 |
| ComplexF32 | 7.662 | 3.860 | 0.682 |
| ComplexF64 | 9.336 | 7.737 | 2.206 |

SYR2K Float64 N=512 alloue 8,652,128 octets dans la v0.2 et 1,064,288 dans la nouvelle API (panneaux compris). Le tampon de produit est limité à une tuile diagonale de 128² éléments. Les snapshots d’alias demeurent nécessaires. TRSM échange le faible scratch de l’ancienne résolution scalaire contre des panneaux GEMM réutilisables.

Les petits cas ne gagnent pas systématiquement : notamment certaines mises à jour de rang k à N=32 et TRMM ComplexF32 gauche à N=32. Les temps natifs restent supérieurs à BLAS externe dans de nombreux cas. Les feuilles SIMD spécialisées, l’autotuning, GEMM ComplexF32 et le parallélisme entre tuiles restent des priorités. Ces mesures courtes sont locales ; aucune généralisation à d’autres machines ou grandes dimensions n’est faite.

## Contrat hermitien des références

Les mesures HERK/HER2K partent d’une diagonale réelle. Un contrôle distinct avec partie imaginaire initiale arbitraire relève 16/16 réussites OpenBLAS et 16/16 échecs de la diagonale imaginaire dans l’Accelerate chargé ici ; voir `reports/hermitian-contract-*.csv` et `benchmark/hermitian_reference_probe.jl`. Le code natif passe ce contrôle dans les deux processus, ainsi que les tests avec partie imaginaire NaN. Le contrat d’effacement de cette partie imaginaire est donné par [Netlib CHERK](https://netlib.org/blas/cherk.f). Ce constat ne porte que sur les versions/environnements enregistrés ici.

## Provenance et suite

`reports/baseline-v0.2/level3.jl` conserve la source exacte comparée. Les sources hardware/SIMD/GEMM sont inchangées depuis la campagne v0.1 ; les mesures GEMM historiques restent identifiées par leurs propres hashes. Les rapports spectraux ne prétendent pas contenir une campagne complète de temps.

Voir [BLAS3_DESIGN.md](BLAS3_DESIGN.md), [PERFORMANCE.md](PERFORMANCE.md), [SPECTRAL_VALIDATION.md](SPECTRAL_VALIDATION.md) et [STATUS.md](STATUS.md). La suite P1 comprend ROTM/ROTMG, les produits scalaires étendus, les noyaux BLAS 2 banded/packed, la validation x86 et le tuning. Le portage complet OxiBLAS reste inachevé.
