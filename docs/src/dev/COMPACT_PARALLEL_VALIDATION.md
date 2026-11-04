# Validation compact/inter-tuiles v0.5 — 2 novembre 2026

Les kernels BLAS 2 exploitent les coefficients stockés et les BLAS 3 de rang k peuvent distribuer leurs tuiles indépendantes. Le code numérique est natif Julia, avec SIMD LLVM sur les mises à jour compactes réelles contiguës.

## Suite complète

| Configuration locale Apple M5 Max arm64 | Tests réussis |
|---|---:|
| Julia 1.14-alpha2, natif, 4 threads | 32749 |
| Julia 1.14-alpha2, CPU générique, 1 thread | 32749 |
| Julia 1.13.3, natif, 4 threads | 32749 |

Bornes vérifiées, dépôts de cache primaires isolés. Les journaux sont `reports/tests-v05-final-*.log`. La suite précédente est conservée et inclut les quatre types ; les nouveaux contrôles couvrent N/T/C, U/L, vues inversées et buffers non contigus, alias des coefficients, padding et diagonales unité interdits à la lecture, α/β sans lecture, singularité avant mutation, coefficients hermitiens à partie imaginaire NaN, zéros structurels avec Inf, poids diagonaux extrêmes/subnormaux et solutions de référence BigFloat.

Un adaptateur de sortie trace les tâches ayant écrit chaque tuile. Il vérifie le nombre de tâches distinctes, les régions disjointes et les appels partageant un workspace. Des contrôles indépendants denses/BLAS complètent les 48 groupes de fixtures Rust (428 valeurs, quatre types) réellement compilés et exécutés au commit 8d82bb6. Le Rust n’est jamais utilisé par le paquet.

## Stockage compact

424 lignes v0.4/native/OpenBLAS et 140 lignes Accelerate, tailles 128/1024 et bandes 16 384 de largeur 8, N/C lorsque pertinent. Les extensions symétriques complexes sans équivalent BLAS portable sont vérifiées contre la matrice dense aux tailles 128/1024. Les grandes bandes évitent toute expansion dense et toute exécution quadratique de la baseline.

Médianes à N=1024, largeur de bande 8, sans transposition ; microsecondes. Les références hermitiennes utilisent ComplexF64, les autres lignes Float64.

| Opération | v0.4 | Native | OpenBLAS | Accelerate | Gain v0.4/native |
|---|---:|---:|---:|---:|---:|
| GBMV | 1632.79 | 4.29 | 6.00 | 3.38 | 380.43× |
| SBMV | 1800.96 | 3.54 | 11.00 | 2.79 | 508.60× |
| HBMV | 2073.29 | 7.83 | 9.42 | 8.12 | 264.69× |
| TBMV | 776.00 | 3.67 | 4.17 | 1.67 | 211.62× |
| TBSV | 600.58 | 6.25 | 5.17 | 4.25 | 96.09× |
| SPMV | 1210.29 | 67.50 | 108.75 | 105.71 | 17.93× |
| HPMV | 1643.50 | 360.08 | 351.33 | 449.54 | 4.56× |
| TPMV | 830.92 | 58.38 | 58.88 | 42.17 | 14.23× |
| TPSV | 549.83 | 68.92 | 69.12 | 45.46 | 7.98× |
| SPR | 861.71 | 66.96 | 66.92 | 67.96 | 12.87× |
| SPR2 | 818.38 | 122.33 | 119.42 | 70.83 | 6.69× |
| HPR | 1173.88 | 233.58 | 180.00 | 156.62 | 5.03× |
| HPR2 | 1361.54 | 480.83 | 361.62 | 203.25 | 2.83× |

Les 156 appels natifs mesurés (sans alias) allouent zéro octet selon BenchmarkTools. Les références mesurées allouent également zéro octet dans ces wrappers. Le dispatch SIMD à largeur typée évite les allocations par colonne ; les conversions d’options et recherches de fonctions sont hors chronométrage. Les alias imposent encore des snapshots du stockage utile. Les gains face à la v0.4 ne signifient pas que les BLAS externes sont dépassés dans toutes les tailles/types.

## Scalabilité inter-tuiles

Quatre processus distincts à 1/2/4/8 threads : 124 lignes chacun, v0.4/native/OpenBLAS, tailles 512/1024, K=3N/4. Accelerate fournit 36 lignes supplémentaires à un thread. SYMM/HEMM sont des contrôles conservant le GEMM parallèle existant ; leur algorithme n’est pas réécrit. GEMMT n’a pas de ligne BLAS externe dans cette campagne.

SYRK à N=1024, K=768, bloc 128 ; médianes en millisecondes et allocations incluant contextes/tampons diagonaux. Panneaux GEMM réutilisés.

| Threads | Float64 ms | Gain vs 1 thread | Octets Float64 | ComplexF32 ms | Gain vs 1 thread |
|---|---:|---:|---:|---:|---:|
| 1 | 20.248 | 1.00× | 157840 | 61.047 | 1.00× |
| 2 | 10.640 | 1.90× | 295168 | 30.972 | 1.97× |
| 4 | 6.586 | 3.07× | 559440 | 16.240 | 3.76× |
| 8 | 4.353 | 4.65× | 1087888 | 11.548 | 5.29× |

À huit threads, rapports v0.4/natif dans le même processus et sur les mêmes entrées à N=1024 :

| Opération | Float64 | ComplexF32 |
|---|---:|---:|
| SYRK | 2.37× | 1.26× |
| SYR2K | 2.40× | 1.43× |
| GEMMT | 2.40× | 1.30× |

Chaque tâche conserve son couple de panneaux et exécute un GEMM monothread. Le parent du workspace est verrouillé pendant le plan ; les wrappers enfants ont leurs propres verrous, sans réentrée sur le parent. La propriété suit les tâches et supporte leur migration. Le nombre de tâches est borné par les buffers, les tuiles et les threads disponibles. Les contextes et tampons diagonaux restent alloués par appel, ce qui explique leur croissance avec le nombre de tâches.


## GBMV Float64 sur entrées identiques au Rust

Largeur 8, un thread, neuf échantillons après warmup, mêmes coefficients et α/β. Les vecteurs complets Rust sont vérifiés contre un BLAS indépendant. Les formats physiques restent ceux de chaque API. Médianes en microsecondes :

| N | Rust | Julia native | OpenBLAS | Accelerate |
|---|---:|---:|---:|---:|
| 4096 | 34.50 | 17.12 | 24.04 | 11.08 |
| 16384 | 97.67 | 73.54 | 105.71 | 45.46 |
| 65536 | 322.33 | 289.67 | 399.25 | 152.88 |

Ce jeu ne concerne qu’un noyau Float64 monothread ; il ne permet pas de conclure à une supériorité générale. Scripts, résultats et vecteurs binaires sont conservés dans `reports/rust-compact-workload/` et `benchmark/`.


## Assembleur et provenance

Les deux kernels AXPY compacts réels inspectés contiennent des instructions FMLA vectorielles et aucun appel de fonction. LLVM/assembleur, environnements et bibliothèques par symbole sont conservés dans `reports/compact-*` et `reports/blas2-3-providers-*.txt`. Seul NEON a été exécuté sur le matériel local. La validation x86 et la CI distante restent à faire.

Les sources v0.4 comparées sont figées dans `reports/baseline-v0.4/`. Les sources GEMM, SIMD commun et BLAS 1 sont inchangées ; leurs campagnes historiques gardent leurs propres hashes. Les manifests courants `reports/source-sha256.txt`, `compact-source-sha256.txt`, `tiles-source-sha256.txt` et `artifact-sha256.txt` identifient les livrables.

## Limites et suite

Les campagnes restent courtes et locales, sans généralisation statistique à toutes les machines. Les kernels SIMD complexes/stridés et le parallélisme compact restent à optimiser. Les lots indépendants, la parallélisation des factorisations et des autres BLAS 1, les microkernels triangulaires, le tuning et les extensions LAPACK/sparse ne sont pas déclarés terminés.

Voir [conception](COMPACT_PARALLEL_DESIGN.md), [protocole](PERFORMANCE.md), [API](API.md), [état](STATUS.md) et [continuation actualisée](CONTINUATION_2026-11-02.md).
