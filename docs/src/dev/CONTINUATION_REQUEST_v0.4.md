# OxiBLAS.jl v0.4.0 — Bilan et suite du développement

## Objectif

Poursuivre le portage intégral d'OxiBLAS Rust en Julia natif, en conservant toutes les fonctionnalités déjà développées.

**Ne repars pas de zéro.** Le projet possède déjà un moteur BLAS fonctionnel, plusieurs algorithmes LAPACK, des kernels SIMD LLVM, des tests et des benchmarks.

L'objectif reste de reproduire les fonctionnalités pertinentes du dépôt Rust, sans déléguer les calculs numériques à des bibliothèques externes, et en exploitant toutes les possibilités de bas niveau de Julia.

## État actuel — v0.4.0

Le package dispose désormais de :

- **GEMM :** cache blocking, packing, microkernels SIMD NEON/AVX2/AVX-512, workspaces, multithreading, quatre types numériques.
- **BLAS 1 :** opérations principales, ROTM/ROTMG, DSDOT/SDSDOT, sommation compensée et optimisations SIMD.
- **BLAS 2 :** opérations denses, triangulaires, symétriques, hermitiennes, packed et banded.
- **BLAS 3 :** SYMM, HEMM, TRMM, TRSM, SYRK, SYR2K, HERK, HER2K et GEMMT avec algorithmes bloqués ou récursifs.
- **LAPACK :** LU, Cholesky, QR, Hessenberg, Schur, EVD hermitienne et SVD Jacobi.
- **Sparse :** CSR, SpMV, CG, Jacobi.
- **Batched :** GEMM sur listes et tableaux 3D.

Les journaux enregistrent **27 811 tests réussis** sur trois configurations Julia, notamment Julia 1.13.3 et 1.14-alpha2.

Le projet comporte un inventaire de 370 fichiers Rust : 137 partiellement couverts, 190 non portés et 43 fichiers FFI optionnels non repris.

La version reste un portage partiel, pas encore un remplacement complet d'OxiBLAS Rust.

## Priorité 1 — Finaliser les BLAS

### A. Optimiser BLAS 2 banded/packed

Les opérations existent, mais passent encore par des parcours génériques.

Implémenter des kernels spécialisés :

- GBMV, SBMV, HBMV, TBMV, TBSV.
- SPMV, HPMV, TPMV, TPSV.
- Opérations de rang sur stockages compacts.

Exploiter directement le stockage utile, les strides et la largeur de bande.

Ajouter des optimisations SIMD, des benchmarks et les variantes parallèles lorsque pertinentes.

### B. Compléter le parallélisme

Actuellement, GEMM est la principale opération dotée d'un parallélisme explicite.

Ajouter un véritable parallélisme natif Julia aux opérations qui le permettent :

- SYRK, HERK, SYR2K, HER2K et GEMMT.
- SYMM/HEMM.
- Certaines étapes de TRMM/TRSM.
- Factorisations bloquées.
- Opérations batched indépendantes.
- Opérations vectorielles suffisamment grandes.

Paralléliser les tuiles indépendantes avec des workspaces appropriés.

Attention aux dépendances des solveurs triangulaires et des factorisations.

Éviter le parallélisme imbriqué inefficace : lorsqu'un algorithme distribue déjà les tuiles entre plusieurs threads, ne pas lancer automatiquement un GEMM multithread dans chaque tuile.

Mesurer la scalabilité et les allocations en 1, 2, 4 et 8 threads.

### C. Continuer les optimisations CPU

- Microkernels SIMD triangulaires.
- GEMM complexe, particulièrement ComplexF32.
- Optimisation des petites matrices et matrices rectangulaires.
- Réglage MC/NC/KC/MR/NR.
- Autotuning runtime.
- Validation physique des kernels AVX2 et AVX-512 sur x86_64.
- Benchmarks comparatifs avec OpenBLAS, Accelerate et Rust.

Ne pas annoncer de validation matérielle lorsque celle-ci n'a pas été exécutée.

## Priorité 2 — Compléter LAPACK

Porter les familles avancées encore absentes :

- **LU :** pivotage complet, rook, matrices bandes, versions récursives et parallèles.
- **Cholesky :** LDLᵀ/LDLᴴ, Bunch–Kaufman, Aasen, versions récursives et parallèles.
- **QR :** QR pivoté, bloqué, récursif, LQ, QL, RQ, moindres carrés de rang déficient.
- **EVD :** valeurs et vecteurs propres généraux, balancing, divide-and-conquer, MRRR et méthodes spectrales avancées.
- **Schur :** réordonnancement, multishift, AED, QZ.
- **SVD :** bidiagonalisation, Golub–Kahan, divide-and-conquer, SVD complète et variantes tronquées.
- **Solveurs :** tridiagonaux, problèmes généralisés, raffinement mixte et solveurs experts.
- **Utilitaires :** inverse, pseudoinverse, déterminant, conditionnement, rang et espace nul.

Réutiliser systématiquement les BLAS natifs déjà implémentés pour accélérer les méthodes bloquées.

## Priorité 3 — Sparse et tenseurs

Compléter les fonctionnalités d'OxiBLAS Rust :

**Matrices creuses :**
- Formats CSC, COO, ELL, DIA, BSR, BSC, HYB et SELL.
- SpMM, conversions et factorisations creuses.
- GMRES, FGMRES, BiCGStab, MINRES, QMR, TFQMR, IDR(s).
- ILU, ILUT, Cholesky incomplète, AMG, SPAI, AINV, Schwarz.
- Lanczos, Arnoldi, IRAM, SVD creuse.
- Factorisations multifrontales et réordonnancements.

**Tenseurs :**
- Contractions N-D.
- Einstein summation.
- Batched GEMV et AXPY.
- Opérations batched parallèles.

**Précision étendue :**
- Float16.
- Précision quadruple.
- Accumulation compensée et précision mixte.

## Contraintes techniques

- Calculs 100 % Julia natif.
- `Base.llvmcall`, SIMD LLVM, intrinsics, pointeurs et assembleur bas niveau autorisés.
- `ccall` autorisé pour les services système, pas pour les calculs numériques.
- Aucune dépendance à OpenBLAS, MKL, Accelerate ou Rust pour l'exécution des algorithmes.
- Les bibliothèques externes peuvent servir de références dans les tests et benchmarks.
- Préserver la compatibilité Float32, Float64, ComplexF32 et ComplexF64.
- Garantir la sûreté des threads, des pointeurs, des alias et des accès mémoire.
- Tester numériquement les algorithmes, y compris les cas extrêmes et mal conditionnés.
- Vérifier les algorithmes de l'amont : ne pas reproduire aveuglément leurs éventuelles erreurs.
- Conserver les mesures précédentes afin d'identifier les régressions.

## Consignes de développement

Pour chaque nouvelle fonctionnalité :

1. Examiner l'implémentation Rust originale.
2. Porter réellement l'algorithme en Julia.
3. Ajouter les tests de correction.
4. Ajouter un benchmark et mesurer les allocations.
5. Optimiser les performances et le parallélisme.
6. Inspecter LLVM et l'assembleur lorsque nécessaire.
7. Mettre à jour l'inventaire de couverture et la documentation.
8. Exécuter les tests et signaler précisément les limites restantes.

Mettre également à jour `docs/CONTINUATION_2026-11-02.md`, qui décrit toujours la v0.2 et contient des tâches déjà réalisées.

## Prochaine action

**Commence par les kernels BLAS 2 banded/packed, puis le parallélisme inter-tuiles BLAS 3.**

Ne refais pas les algorithmes déjà présents. Optimise et complète leurs implémentations avant de passer aux grandes extensions LAPACK et sparse.

Ne te contente pas de proposer une architecture ou un planning : écris le code, teste, benchmarke et fournis les résultats réels.

L'objectif final reste une bibliothèque BLAS/LAPACK native Julia, complète, performante et multiplateforme.