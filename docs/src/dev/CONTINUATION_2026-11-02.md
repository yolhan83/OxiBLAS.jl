> État antérieur conservé. La dernière passe et l’arrêt demandé sont décrits
> dans [POINT_ARRET_2026-11-02.md](POINT_ARRET_2026-11-02.md). Les listes
> de manques ci-dessous précèdent cette passe et ne constituent plus le bilan courant.

# OxiBLAS.jl 0.6.0-dev — continuation

État courant : implémentations écrites, tests numériques **non exécutés** sur
instruction explicite de l'utilisateur. Ne pas relancer tests/benchmarks avant
l'accord demandé pour les tests. Un accord pour les tests ne vaut pas accord
pour les benchmarks/autotuning, qui restent différés.

Lire [DEVELOPMENT_v0.6.md](DEVELOPMENT_v0.6.md) et [STATUS.md](STATUS.md).
Les nouveautés et les limites y sont détaillées ; le portage complet reste
inachevé, notamment QZ, SVD Golub–Kahan/D&C et de larges variantes sparse.

- Base 0.5 conservée : `reports/baseline-v0.5/` et archive voisine
  `OxiBLAS-v0.5.jl.zip`. Les anciens résultats et manifests restent historiques.
- Source courante : Julia natif, `version = "0.6.0-dev"`. Aucune nouvelle
  dépendance, aucun backend BLAS/LAPACK/Rust et aucun type dense Tensor.
- Ajouts : contextes/scratch/jobs réutilisables ; SIMD complexe et BLAS 2
  parallèles ; lots et TRMM/TRSM indépendants ; LU complet/rook, QR bloqué/pivoté,
  LDL/Bunch–Kaufman, solve expert, mixte complexe, EVD généralisée définie,
  bidiagonalisation, Schur complexe réordonné et vecteurs propres ; COO/CSC,
  SpGEMM/SpMM, ILU0/GMRES/BiCGStab/Lanczos ; contractions N-D/einsum et plans,
  lots N-D, HOSVD/TT-SVD/CP-ALS ; autotuning strictement explicite.
- Tests préparés : quatre nouveaux fichiers inclus dans `test/runtests.jl`.
  Seule la syntaxe a été analysée sans charger ni évaluer le paquet.
- Le verrou du workspace parent couvre tout ordonnancement utilisant des
  enfants. Leurs verrous sont distincts, les buffers appartiennent aux tâches,
  et le GEMM interne reste monothread lorsqu'un plan externe est parallèle.

Après accord, exécuter les tests appropriés puis corriger les échecs sans
fabriquer de résultats. Conserver les résidus, références indépendantes,
cas d'alias/lecture interdite/GC/concurrence et quatre types pertinents.
Une fois validé, créer de nouveaux logs/manifests avec un identifiant de source
courant ; ne jamais étiqueter les anciens logs comme validation de 0.6.
