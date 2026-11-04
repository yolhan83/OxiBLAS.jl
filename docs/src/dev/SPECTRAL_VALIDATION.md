# Validation spectrale v0.2 — 2 novembre 2026

Sources de référence : `reports/baseline-v0.2/source-sha256.txt`.
Les trois suites complètes passent avec vérification des bornes :

| Configuration locale, Apple M5 Max arm64 | Tests réussis | Durée |
|---|---:|---:|
| Julia 1.14.0-alpha2, CPU natif, 4 threads | 11 574 | 56,6 s |
| Julia 1.14.0-alpha2, CPU générique, 1 thread | 11 574 | 60,5 s |
| Julia 1.13.3, CPU natif, 4 threads | 11 574 | 75,4 s |

Journaux : `reports/tests-spectral-final-*.log`. Les références indépendantes
utilisent LAPACK dans les tests uniquement. Douze comparaisons supplémentaires
vérifient les spectres du Rust au commit `8d82bb6`, conservés dans
`reports/rust-spectral.csv`. Le Rust ne remplace pas les contrôles de résidu.

Hessenberg : reconstruction et unitarité, application implicite N/T/C à gauche
et à droite, compact WY, vues et alias. Schur : Francis réel, QR complexe,
blocs réels 2×2, spectre et résidus, détection de non-convergence. EVD :
tridiagonalisation symétrique/hermitienne, phases complexes, valeurs et vecteurs
triés. SVD : Jacobi unilatéral économique, valeurs seules, matrices larges,
rang déficient et complétion orthogonale. Les quatre types sont couverts,
avec matrices vides, répétitions, échelles extrêmes et valeurs subnormales.

Mesure exploratoire de formation de Q, N=256, un thread, minima en secondes :

| Type | Non bloqué | WY avant optimisation | WY après optimisation |
|---|---:|---:|---:|
| Float64 | 0,001799 | 0,002278 | 0,001316 |
| ComplexF64 | 0,006832 | 0,007320 | 0,004112 |

Cette mesure courte ne constitue pas une campagne spectrale générale.
Le script `benchmark/spectral.jl` est fourni ; sa campagne complète n'a pas été
exécutée pour la v0.2. Les performances GEMM historiques concernent les sources
GEMM inchangées depuis la v0.1 ; voir [VALIDATION.md](VALIDATION.md).

Les workflows Linux/macOS/Windows sont présents dans `.github/workflows/ci.yml`.
Ils n'ont pas été exécutés à distance ici. AVX2 et AVX-512 ne sont pas validés
sur du matériel x86. Les limites algorithmiques restent dans [STATUS.md](STATUS.md).

Revalidation P0 avant tout changement BLAS 3 : 11 574 / 11 574 tests en
55,6 s, Julia 1.14, quatre threads, bornes vérifiées ; voir
`reports/tests-p0-revalidated.log`. Les hashes v0.2 ont été contrôlés sans écart.
