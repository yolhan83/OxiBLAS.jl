# OxiBLAS.jl — État du projet et instructions de continuation

## Objectif général

Continuer le portage complet d'[OxiBLAS Rust](https://github.com/cool-japan/oxiblas) vers Julia natif.

**Ne repars pas de zéro.** Le projet `OxiBLAS.jl` v0.2.0 possède déjà une architecture fonctionnelle, des kernels optimisés, des tests et des benchmarks.

L'objectif est de reproduire progressivement toutes les fonctionnalités pertinentes d'OxiBLAS, avec des performances compétitives et sans dépendre d'un backend numérique externe.

## 1. État actuel

Le projet contient 2 326 lignes de Julia dans `src/` et couvre déjà :

- **GEMM :** algorithmes scalaire, bloqué, packed, register et SIMD LLVM, support des quatre types BLAS principaux, NEON Apple Silicon, implémentations AVX2/AVX-512, multithreading et workspaces.
- **BLAS 1 :** opérations vectorielles principales, produits scalaires, normes, rotations et sommations compensées.
- **BLAS 2 :** opérations matricielles denses, triangulaires, symétriques, hermitiennes, banded et packed.
- **BLAS 3 :** GEMM, SYMM, HEMM, TRMM, TRSM, SYRK, SYR2K, HERK, HER2K.
- **LAPACK :** LU, Cholesky, QR, Hessenberg, Schur, EVD symétrique/hermitienne, SVD Jacobi, solveurs élémentaires.
- **Sparse :** CSR, SpMV, gradient conjugué et Jacobi.
- **Batched :** GEMM sur listes et tableaux 3-D.

L'inventaire existant répertorie 370 fichiers Rust du commit `8d82bb6`, dont 136 partiellement portés, 191 non portés et 43 fichiers FFI optionnels non repris.

Les journaux indiquent 7 593 tests réussis et des performances GEMM encourageantes sur Apple M5 Max. Attention : les derniers changements des modules spectraux sont postérieurs aux journaux de tests ; la version finale doit être revalidée.

## 2. Priorité P0 — Stabiliser la version actuelle

Avant toute nouvelle fonctionnalité :

1. Relancer toute la suite de tests sur la dernière version du code.
2. Corriger les éventuelles régressions.
3. Ajouter le rapport `docs/SPECTRAL_VALIDATION.md` actuellement absent.
4. Ajouter les workflows CI Linux/macOS/Windows annoncés mais absents de l'archive.
5. Vérifier la conformité des rapports, hashes et benchmarks avec les sources finales.

Ne pas annoncer de validation x86 tant que les kernels n'ont pas été réellement exécutés sur cette architecture.

## 3. Priorité P1 — Compléter le moteur BLAS

C'est le prochain gros chantier.

Les opérations BLAS 3 autres que GEMM fonctionnent, mais plusieurs reposent encore sur des matrices temporaires complètes ou des traitements successifs de vecteurs.

Réimplémenter et optimiser :

- SYMM / HEMM.
- TRMM / TRSM.
- SYRK / SYR2K.
- HERK / HER2K.
- GEMMT et autres variantes pertinentes de l'amont.

Utiliser des algorithmes bloqués, des accès mémoire efficaces et les kernels GEMM existants lorsque pertinent.

Ensuite :

- Terminer les opérations BLAS 1 manquantes, notamment ROTM/ROTMG et les produits scalaires étendus.
- Optimiser BLAS 2 banded/packed.
- Valider AVX2 et AVX-512.
- Optimiser GEMM complexe, particulièrement ComplexF32.
- Ajouter l'autotuning des blocs et des kernels.
- Benchmark systématique de toutes les opérations importantes.

## 4. Priorité P2 — Compléter LAPACK

Ajouter progressivement les grandes familles manquantes :

- LU pivot complet/rook, bande, récursif et parallèle.
- LDLᵀ/LDLᴴ, Bunch–Kaufman, Aasen correctement validé.
- QR pivoté, bloqué, récursif, LQ, QL et RQ.
- Factorisations pour systèmes sous-déterminés ou de rang déficient.
- Raffinement mixte Cholesky et QR.
- Vecteurs propres généraux, balancing et réordonnancement Schur.
- EVD divide-and-conquer, MRRR, bisection et inverse iteration.
- SVD bidiagonale, Golub–Kahan et divide-and-conquer.
- Problèmes aux valeurs propres généralisés.
- Solveurs et utilitaires : déterminant, inverse, pseudoinverse, conditionnement, rang, espace nul, etc.

Réutiliser le BLAS natif du package pour accélérer les algorithmes bloqués.

## 5. Priorité P3 — Fonctionnalités supplémentaires d'OxiBLAS

Étendre progressivement :

**Sparse :** CSC, COO, ELL, DIA, BSR, BSC, HYB, SELL, conversions, GMRES, FGMRES, BiCGStab, MINRES, ILU, AMG, SPAI, Schwarz, Lanczos, Arnoldi, solveurs multifrontaux et méthodes spectrales sparse.

**Tenseurs :** contractions N-D, einsum, opérations batched supplémentaires et parallélisme adapté.

**Précision :** Float16, précision étendue et méthodes numériques compensées.

**Intégration :** interfaces Julia standard, puis éventuellement adaptateur BLAS ABI/libblastrampoline. Ce dernier est optionnel et distinct du portage numérique.

Éviter de recopier les conteneurs ou abstractions Rust lorsqu'une abstraction Julia existante fournit déjà le même service.

## 6. Contraintes techniques fondamentales

- Calculs 100 % Julia.
- Aucune délégation à BLAS/LAPACK externe.
- `Base.llvmcall`, SIMD LLVM, pointeurs et assembleur bas niveau autorisés.
- `ccall` autorisé uniquement pour les services système nécessaires.
- Aucune obligation d'utiliser SIMD.jl ou LoopVectorization.jl.
- Priorité à la correction mathématique, puis aux performances.
- Respect de l'ABI, des contraintes de GC, des alias et de la sécurité des threads.
- Tests systématiques sur les quatre types numériques lorsque pertinents.
- Respect de la licence et des attributions d'OxiBLAS Rust.

Important : le projet Rust lui-même a fait l'objet d'un audit signalant des défauts numériques. Ne considère pas systématiquement son code comme une référence correcte. Vérifie les algorithmes et compare avec des références indépendantes.

## 7. Méthode de travail

Pour chaque nouvelle famille de fonctionnalités :

1. Étudier l'implémentation Rust d'origine.
2. Identifier les fonctionnalités, variantes et optimisations à reproduire.
3. Implémenter réellement les algorithmes en Julia.
4. Ajouter les tests de correction, stabilité et cas limites.
5. Comparer les résultats aux références indépendantes.
6. Optimiser les kernels et inspecter LLVM/assembleur lorsque pertinent.
7. Exécuter les benchmarks et détecter les régressions.
8. Mettre à jour l'inventaire et la documentation.

Ne pas créer de fonctions factices, de simples wrappers ou de résultats artificiels pour prétendre avoir terminé un module.

Maintenir un état explicite des fonctionnalités : non porté, partiel, implémenté, testé, optimisé.

## 8. Prochaine tâche

**Commence par P0, puis attaque P1, principalement les kernels BLAS 3 hors GEMM.**

Nous avons déjà une base GEMM performante. Il faut maintenant construire dessus pour obtenir un moteur BLAS complet, avant de consacrer l'essentiel du développement aux algorithmes avancés LAPACK et sparse.

Ne te limite pas à proposer un plan : effectue les modifications, teste-les et documente ce qui reste à faire après chaque incrément.