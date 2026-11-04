# Validation de l’incrément 0.1 — 2 novembre 2026

Ce rapport décrit la campagne initiale GEMM/BLAS. Voir [SPECTRAL_VALIDATION.md](SPECTRAL_VALIDATION.md) pour la version 0.2.

Apple M5 Max, 18 cœurs logiques, macOS arm64. Tests sous Julia 1.14.0-alpha2 (LLVM 23.1.4) et Julia 1.13.3. Les résultats concernent le code identifié dans `reports/baseline-v0.1/source-sha256.txt`. Le portage complet reste inachevé ; voir [STATUS.md](STATUS.md).

## Correction

| Configuration | Réussis | Durée Test, hors précompilation séparée |
|---|---:|---:|
| Julia 1.14, CPU natif, 4 threads, bornes vérifiées | 7593 | 43.2 s |
| Julia 1.14, CPU générique, 1 thread, bornes vérifiées | 7593 | 44.6 s |
| Julia 1.13.3, CPU natif, 4 threads, bornes vérifiées | 7593 | 55.3 s |

La suite inclut quatre types, N/T/C, matrices rectangulaires et vides, vues inversées/non contiguës, canaris autour des sorties, alias, cas α/β sans lecture, espaces de travail partagés, résidus de factorisation, moindres carrés, stockages compactés, CSR/CG et 36 fixtures issues du Rust original. Les tests ne constituent pas une preuve de stabilité sur toutes les entrées extrêmes.

Les références Rust proviennent du commit `8d82bb6bf1df133aa90525aa9f478ebe41514f9b`, compilé en release avec `-C target-cpu=native`, LTO fin. Linux/Windows et les noyaux x86 restent à valider sur leur matériel ; les jobs CI fournis n’ont pas été lancés à distance.

## GEMM 4096 × 4096 × 4096

Débit médian en GFLOP/s. Alpha=0,7, bêta=0. Panneaux Julia réutilisés ; tous les calculs sont vérifiés avant mesure. OpenBLAS est mesuré à 1/8 threads et Accelerate en mode monothread, dans des processus successifs. Certains essais de cette taille ne comportent qu’un échantillon : ces chiffres sont des observations locales, sans garantie statistique ni supériorité générale.

| Type | Julia 1 thread | Julia 8 threads | OpenBLAS 1 thread | OpenBLAS 8 threads | Accelerate 1 thread |
|---|---:|---:|---:|---:|---:|
| Float32 | 115.0 | 720.9 | 119.7 | 803.3 | 1659.9 |
| Float64 | 63.1 | 356.5 | 60.5 | 395.9 | 439.2 |
| ComplexF32 | 62.2 | 404.8 | 108.2 | 684.3 | 1410.1 |
| ComplexF64 | 47.9 | 304.7 | 52.8 | 334.4 | 348.8 |

La campagne finale conserve 288 lignes avec OpenBLAS et 52 avec Accelerate, tailles 16–4096 et rectangles. Les CSV contiennent temps minimaux et médians, GFLOP/s, nombre d’échantillons, allocations, temps du premier appel, compilation et erreur relative. Les variantes référence/cache/packed/register sont également mesurées sur les tailles prévues par le protocole.

| Type | Erreur relative maximale observée, campagne OpenBLAS |
|---|---:|
| Float32 | 5.73e-07 |
| Float64 | 9.3e-16 |
| ComplexF32 | 6.67e-07 |
| ComplexF64 | 1.17e-15 |

Le contrôle d’un appel Float64 1024 avec workspace alloue encore 352 octets en 11 allocations à un thread. Les panneaux ne sont pas réalloués à chaque micro-noyau. Les tâches multithread ajoutent leurs propres allocations, enregistrées dans les CSV. Trafic mémoire et compteurs matériels non mesurés.

## Comparaison Rust sur entrées identiques

Float64, un thread, minima en millisecondes, cinq échantillons. « API » inclut la construction des panneaux par appel, comme l’API Rust ; « workspace » les réutilise. Les entrées déterministes et α/β sont identiques. Cette comparaison ne porte que sur GEMM Float64.

| N | Rust OxiBLAS | Julia API | Julia workspace | OpenBLAS |
|---|---:|---:|---:|---:|
| 16 | 0.0012 | 0.0010 | 0.0008 | 0.0002 |
| 64 | 0.0118 | 0.0156 | 0.0160 | 0.0190 |
| 256 | 0.5531 | 0.7427 | 0.6852 | 0.6574 |
| 512 | 4.3743 | 4.6779 | 4.5966 | 4.6960 |
| 1024 | 34.4057 | 36.6855 | 36.6977 | 36.1089 |
| 2048 | 273.5714 | 274.2425 | 274.1849 | 285.6882 |

Écart maximal entre checksums agrégés : 1.19e-09. Les produits Julia sont également comparés aux matrices de référence, et les 36 fixtures testent les valeurs élément par élément.

## Instructions machine et restriction des backends

| Micro-noyau inspecté | Instructions FMLA dans la fonction | Appels de fonction |
|---|---:|---:|
| neon-complexf32-native.s | 71 | 0 |
| neon-complexf64-native.s | 143 | 0 |
| neon-float32-native.s | 96 | 0 |
| neon-float64-native.s | 144 | 0 |

Le compte inclut boucle déroulée, reste et épilogue. Les sauvegardes ABI de registres en entrée/sortie ne sont pas des appels numériques externes. Le code LLVM et l’assembleur sont conservés dans `reports/`. L’évolution avant/après inlining est conservée séparément ; ses premières mesures sont exploratoires.

L’audit source ne relève aucun appel numérique BLAS/LAPACK/ccall dans `src/` ; seuls les types de wrappers LinearAlgebra sont importés. Voir `reports/backend-audit.txt`. Cet audit n’est pas une preuve formelle.

## Reproduction

Voir les commandes du [README](../index.md), le protocole [PERFORMANCE.md](PERFORMANCE.md), `benchmark/benchmarks.jl`, `benchmark/compare_rust.jl`, `tools/run_rust_reference.py` et les environnements enregistrés dans `reports/`. Les journaux Test sont livrés avec les résultats. Les limites fonctionnelles restent détaillées dans [STATUS.md](STATUS.md).
