> État antérieur conservé. La dernière passe et l’arrêt demandé sont décrits

Actualisation QZ/Schur : [port algorithmique et complétion de toute la ligne](QZ_SCHUR_PORT.md), code écrit sans exécution. Les mentions antérieures de limites Schur ci-dessous sont historiques et remplacées par ce bilan.
> dans [POINT_ARRET_2026-11-02.md](POINT_ARRET_2026-11-02.md). Les listes
> de manques ci-dessous précèdent cette passe et ne constituent plus le bilan courant.

# OxiBLAS.jl 0.6.0-dev — implémentation en attente de validation

## Contrat de cette étape

La demande courante impose d'écrire les implémentations et les cas de test avant
l'autorisation d'exécuter les tests. **Aucun test numérique, smoke test,
benchmark, autotuning, contrôle d'allocation ou inspection JIT/assembleur n'a
été exécuté pour cet incrément.** L'analyse syntaxique utilise uniquement
`Meta.parseall`, sans évaluation des sources ni chargement du paquet. Une
syntaxe acceptée ne prouve ni la compilation des spécialisations ni leur justesse.

Les résultats historiques 0.5 restent dans les rapports et dans
`reports/baseline-v0.5/`. L'archive voisine `OxiBLAS-v0.5.jl.zip` est préservée.
Les rapports, manifestes de validation et SHA historiques ne doivent pas être
régénérés à partir de leurs anciens logs en les étiquetant 0.6. Les fichiers
`reports/development-v0.6.json` et `reports/development-v0.6-sha256.txt` décrivent
les sources de développement, sans attestation numérique.

## Implémentations écrites

Toutes les lignes ci-dessous sont **non validées numériquement**.

| Domaine | Changement | Contrats prévus |
|---|---|---|
| Workspaces BLAS 3 | Contextes enfants, jobs triangulaires et scratch diagonal conservés dans `GemmWorkspace` | Verrou parent durant tout le plan ; verrous enfants distincts ; propriété des buffers par tâche |
| GEMM batched | Ordonnancement externe des lots listes/3-D | Validation préalable, sorties disjointes, snapshots croisés, GEMM interne monothread, workspace partagé sérialisé |
| TRMM/TRSM | Répartition des colonnes de seconds membres à gauche, des lignes à droite | Dépendances triangulaires conservées au sein de chaque tâche ; pivots vérifiés avant mutation |
| BLAS 2 | GEMV/SYMV/HEMV, GBMV/SBMV/HBMV/SPMV/HPMV et SpMV parallèles | Propriété exclusive des sorties ; GBMV N restreint les colonnes à la bande utile ; autres symétriques lisent le stockage sans tampon dense par tâche |
| Compacts complexes | AXPY LLVM intercalé et réduction fusionnée AXPY+dot, dot transposé/adjoint | GC préservé à la frontière pointeur, strides non unitaires en boucle générique, diagonales/padding ignorés |
| Petits GEMM | Bandes de quatre lignes sans packing pour petites sorties, K jusqu'à 512 | Aucun buffer de packing ; α/β et alias conservés ; gain de performance non mesuré |
| GEMM ComplexF32 | Déroulage K de quatre itérations du microkernel SIMD complexe | Même ordre des produits, queues scalaires/vectorielles existantes ; performance non mesurée |
| Feuilles triangulaires | Accumulation réelle/complexe explicite par composants | FMA sans fast-math ; comportement numérique à vérifier |
| LU/Cholesky | Réutilisation des panneaux, TRSM parallèles, mise à jour Cholesky par HERK triangulaire | Pas de parallélisme externe autour d'un GEMM déjà parallèle |
| LU avancé | Pivot complet ou rook, permutations de lignes/colonnes, solve | Rectangulaire pour la factorisation ; carrée non singulière pour solve |
| QR | QR bloqué compact-WY, QR à pivotage de colonnes et normes recalculées | QR pivoté : solution de moindres carrés de base ; ne prétend pas donner la norme minimale en rang déficient |
| LDL | LDL symétrique/hermitien, sans pivot ou Bunch–Kaufman 1×1/2×2, solve | Triangle choisi uniquement, échanges symétriques, mises à jour indépendantes |
| Propre généralisé | Problème hermitien A, B définie positive | Cholesky + deux TRSM + EVD native ; vecteurs B-orthonormés attendus |
| Solve expert | Équilibration lignes/colonnes, LU, raffinements et erreur arrière composante par composante | Résultat et convergence explicites ; pas d'estimation de condition ni borne d'erreur avant |
| Raffinement mixte | Extension de Float64/Float32 à ComplexF64/ComplexF32 | Résidus en haute précision, statut de convergence explicite |
| Bidiagonalisation | Householder des deux côtés, stockage compact, U/V matérialisés à la demande | Tall : bidiagonale haute ; wide : basse via adjoint ; ce n'est pas encore un algorithme SVD Golub–Kahan |
| Schur/QZ avancé | QZ réel/complexe, réordonnancements 1×1/2×2, TREVC/TRSNA, conditions couplées, AED/train small-bulge bloqué | Lot écrit non validé ; stratégies et corrections Rust documentées dans QZ_SCHUR_PORT.md |
| Formats creux | COO canonique, CSC, conversions CSR/CSC/COO | Doublons COO sommés, indices validés, stockage possédé |
| Produits creux | SpMV/SpMM et produit CSR×CSR de Gustavson | Pas de développement dense ; accumulateur de ligne O(n) pour SpGEMM |
| Préconditionnement | ILU(0), substitutions | Motif sans fill ; pivot nul signalé ; pas d'ILUT/ILUK |
| Solveurs creux | GMRES redémarré à réorthogonalisation double, BiCGStab | Diagonale ou ILU(0), résidus vrais et raisons d'arrêt |
| Spectral creux | Lanczos hermitien réorthogonalisé, Ritz et résidus | Sous-espace déterminé par le vecteur initial ; pas de redémarrage implicite ni garantie d'explorer tous les sous-espaces invariants |
| Autotuning | `autotune_gemm`, sauvegarde/relecture de configuration liée à l'environnement | Appel explicite uniquement ; aucune calibration au chargement ; aucune calibration effectuée ici |

Les seuils de parallélisme sont configurables. Sans benchmarks, les choix de
seuils et de déroulage restent des hypothèses d'optimisation, pas des résultats.
Les tâches utilisent les plages et buffers qui leur sont confiés, jamais
`threadid()` comme identifiant de stockage.

## Tenseurs sans conteneur dense spécifique

- `contract` / `contract!` : contractions de rang quelconque, axes multiples,
  produit extérieur et conjugaisons explicites ; implémentation par GEMM natif.
- `plan_einsum`, `einsum`, `einsum!` : nombre quelconque d'opérandes, labels ASCII,
  sorties explicites ou implicites, permutations, diagonales répétées, traces,
  réductions unaires, ellipses avec broadcast. Une expression scalaire renvoie
  une `Array{T,0}`. Il n'y a pas de conjugaison implicite.
- Plan glouton par coût estimé puis taille d'intermédiaire ; option `:left`.
  Les contractions binaires compatibles passent par GEMM/batched ; le parcours
  général couvre les diagonales, réductions exclusives et broadcast.
- `batched_matmul` / `batched_matmul!` : dimensions 1/2 matricielles, axes 3…N
  broadcastables ; les lots indépendants partagent le plan natif de tâches.
- `mode_product`, `hosvd` / `TuckerFactor`, `tensor_train` / `TensorTrainFactor`,
  `cp_als` / `CPFactor`, `reconstruct` : décompositions natives. Les types de
  factorisation contiennent des tableaux Julia, sans introduire de dense Tensor.
  HOSVD et TT-SVD utilisent la SVD Jacobi existante ; CP-ALS rapporte erreur,
  stagnation et convergence séparément.
- Alias : `einsum!` et `contract!` terminent leur calcul avant d'écrire une
  sortie recouvrante ; α=0 et β=0 disposent des parcours sans lecture associés.

Exemples d'API, **non exécutés à cette étape** :

```julia
P = plan_einsum("...ij,...jk->...ik", A, B)
C = einsum(P, A, B)
einsum!(C, P, A, B; α=0.7, β=0.2)
F = lu_factor(M; pivot=:rook)
Q = qr_factor(M; blocksize=32, pivot=false)
D = bunchkaufman_factor(H; uplo=:L)
S = CSRMatrix(COOMatrix(m, n, rows, cols, values))
r = gmres(S, b; preconditioner=ilu0(S))
T = tensor_train(X; maxrank=32)
Xhat = reconstruct(T)
```

## Travail encore nécessaire pour le portage complet

Le portage complet d'OxiBLAS **n'est pas achevé**. Restent notamment :

- validation numérique/concurrence/GC de tous ces changements et non-régression
  v0.5 ; performances à 1/2/4/8 threads uniquement après autorisation distincte ;
- validation physique AVX2/AVX-512 ; réglage mesuré de GEMM ComplexF32/petites
  matrices, noyaux triangulaires SIMD dédiés, autres algorithmes BLAS ;
- QZ général et crayons singuliers, balancing, Schur réel réordonné,
  multishift/AED, conditionnement des sous-espaces/vecteurs ;
- itération SVD Golub–Kahan, divide-and-conquer, variantes sélectionnées,
  tronquées et randomisées ; la réduction bidiagonale ne les remplace pas ;
- Aasen, variantes bande/packed de LAPACK, LQ/QL/RQ, décomposition orthogonale
  complète, solveurs experts avec bornes avant et davantage de raffinements ;
- formats ELL/DIA/BSR/BSC/HYB/SELL, ILUT/IC/AMG/SPAI/AINV/Schwarz, solveurs
  flexibles/blocs, sparse direct multifrontal, Arnoldi/IRL/SVD creuse,
  graphes, Matrix Market et hors mémoire ;
- planification tensorielle optimale sous contrainte mémoire, intermédiaires
  réutilisables et out-of-core/distribution ; tuning des contractions génériques ;
- précisions supplémentaires et intégrations optionnelles.

## Validation préparée, non exécutée

Les nouveaux fichiers `test/continuation_parallel.jl`,
`test/advanced_factorizations.jl`, `test/sparse_extended.jl` et `test/tensors.jl`
sont inclus dans la suite complète. Ils contiennent des références indépendantes
LinearAlgebra/BLAS dans le répertoire de tests uniquement, des résidus explicites,
des cas de petites dimensions/queues, des alias, des strides, des lectures
interdites, du partage concurrent de workspace et un passage GC.

La prochaine action requiert l'accord demandé par l'utilisateur : lancer la
suite numérique avec vérification des bornes. Les benchmarks et l'autotuning
restent différés, même si les tests sont autorisés.
