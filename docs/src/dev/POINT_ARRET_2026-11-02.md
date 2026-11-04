# Point d’arrêt — 2 novembre 2026

Arrêt demandé par l’utilisateur après ce bilan. Aucune implémentation numérique
n’a été poursuivie après cette demande. État : **0.6.0-dev, portage partiel,
non validé**. Il reste du travail d’implémentation et de stabilisation.

## Ce qui est écrit

Le paquet comporte actuellement 47 fichiers source Julia, soit 9 103 lignes.
Ce volume n’est pas un pourcentage de portage et ne prouve aucune correction.

En plus du moteur et des BLAS v0.5, les sources contiennent désormais :

- SIMD complexe compact, réductions fusionnées, BLAS 2 parallèles, contextes
  BLAS 3 réutilisables, lots et seconds membres TRMM/TRSM indépendants ;
- LU complet/rook/bande, Cholesky bande/packed, LDL/Bunch–Kaufman, Aasen,
  QR bloqué/pivoté, LQ/QL/RQ et décomposition orthogonale complète ;
- SVD Golub–Kahan, Cuppen divide-and-conquer, SVD Jacobi parallèle,
  variantes sélectionnées/tronquées/randomisées, EVD divide-and-conquer/MRRR,
  QZ complexe, équilibrage et Schur avec fenêtre AED et décalages successifs ;
- utilitaires de normes/inverse/pseudoinverse/rang/noyaux, fonctions de matrices,
  dérivées de Fréchet, estimation de condition et raffinement expert ;
- COO/CSC/DIA/ELL/BSR/BSC/HYB/SELL, conversions et produits ; LU/Cholesky/QR
  sparse, LU bloqué, Cholesky multifrontal/supernodal, ILU/IC, relaxations,
  polynômes, SPAI, Schwarz et AMG ;
- GMRES/BiCGStab/FGMRES/PCG/MINRES/TFQMR/QMR/IDR(s), CG/GMRES par blocs,
  Arnoldi/IRAM/shift-invert/LOBPCG, SVD sparse randomisée, graphes et Matrix Market ;
- einsum, plans gloutons, contractions et lots N-D, HOSVD/TT-SVD/CP-ALS sur
  tableaux Julia standards ; autotuning explicite ; premiers scalaires
  double-double et calculs Float16/étendus.

Ces éléments sont des **implémentations écrites**, pas des fonctionnalités
certifiées prêtes à l’emploi. Plusieurs sont des variantes natives adaptées,
pas une reproduction exhaustive de toutes les variantes amont.

## Ce qui manque encore

### Stabilisation et validation — prioritaire avant une version utilisable

- Charger et compiler le paquet courant : cela n’a pas encore été fait pour
  les ajouts de développement, conformément à la consigne de ne pas tester.
- Écrire les tests dédiés de la dernière grande passe. Les quatre fichiers
  de test ajoutés à la première passe existent ; ils ne couvrent pas tous
  les algorithmes ajoutés ensuite.
- Vérifier numériquement les quatre types pertinents, les cas singuliers,
  les spectres serrés, les échelles extrêmes, les vues et alias, les lectures
  interdites, les durées de vie GC et le partage concurrent des workspaces.
- Corriger les erreurs de compilation, de dispatch et de calcul qui seront
  révélées. L’analyse syntaxique ne remplace aucune de ces vérifications.
- **Défaut repéré à corriger :** dans `src/sparse_direct.jl`, le choix de pivot
  de `sparse_lu` accepte `pivot_threshold=0` et peut alors conserver un pivot
  diagonal nul malgré la présence d’un pivot non nul plus bas. La correction
  n’a pas été faite après la demande d’arrêt.
- Réexaminer les conversions/limites du double-double, les pertes de rang des
  méthodes par blocs, les critères de déflation et les chemins de secours.

### Couverture algorithmique encore incomplète

- QZ : représentation complexe implémentée ; forme réelle quasi triangulaire,
  réordonnancement généralisé complet et estimations de condition de crayons
  restent à porter. Robustesse des valeurs infinies/indéterminées à vérifier.
- Schur : la nouvelle stratégie utilise AED et des décalages successifs ;
  elle n’est pas l’ensemble des noyaux bloqués multishift/small-bulge amont.
  Les permutations réelles passent par une forme complexe.
- SVD : le D&C écrit utilise une dilatation symétrique puis Cuppen ; les
  variantes bidiagonales spécialisées et leurs réglages amont ne sont pas
  toutes reproduites. Les variantes sélectionnées calculent encore un spectre
  complet avant extraction. Les algorithmes incrémentaux restent absents.
- LAPACK compact : toutes les variantes packed LDL/pivotage et les modes
  avancés de diagnostic/bornes d’erreur ne sont pas couverts. Les erreurs
  avant/conditions fournies sont des estimations, pas des bornes certifiées.
- Sparse direct : LU multifrontale complète, LDL sparse à pivots, pivotage
  statique avec diagnostics et véritables factorisations hors mémoire absents.
  Le LU sparse bloqué n’est pas un LU multifrontal.
- Préconditionneurs : AINV et certaines variantes spécialisées/avancées de
  préconditionnement, de remplissage et de coarsening restent absents.
- Spectral sparse : variantes bloc Arnoldi/Lanczos dédiées, thick restart,
  sélection par intervalle/filtrage polynomial, autres transformations
  généralisées et SVD Lanczos/incrémentale encore incomplètes ou absentes.
- Graphes : minimum-degree exact écrit ; toutes les variantes AMD/COLAMD,
  partitionnement multilevel et appariement pondéré amont ne sont pas portées.
- Estimation stochastique Hutchinson/Hutch++/XTrace non portée.
- Tenseurs : plans sous contrainte mémoire, réutilisation des intermédiaires,
  variantes distribuées/hors mémoire et toutes les optimisations amont restent
  à faire. Aucun type dense Tensor spécifique n’a été introduit.
- Précision étendue : noyau double-double et GEMM/produit scalaire seulement ;
  pas de couverture BLAS/LAPACK étendue complète ni de tous les traits amont.
- Intégrations facultatives ABI/libblastrampoline, mmap/lazy/COW et détails de
  certains conteneurs Rust ne sont pas reproduits ; les tableaux/vues Julia
  fournissent déjà une partie des fonctions pertinentes.

### Performance, portabilité et suivi

- Aucun benchmark nouveau : les changements de seuils, blocs et déroulage
  restent des hypothèses d’optimisation. Campagnes 1/2/4/8 threads à effectuer
  après autorisation, en conservant strictement les historiques v0.5.
- Autotuning écrit mais jamais lancé ; configurations persistantes à calibrer.
- Matériel AVX2/AVX-512 non disponible dans cette session ; validation physique,
  inspection assembleur et CI multiplateforme distante restent nécessaires.
- Microkernels triangulaires SIMD dédiés, réglages ComplexF32, allocations des
  nouveaux plans et parallélisme des nouveaux algorithmes restent à mesurer.
- L’inventaire CSV à 370 fichiers et son générateur ne sont **pas encore
  réconciliés avec la dernière passe**. Leurs anciens comptes ne doivent pas
  être présentés comme la couverture courante. La documentation API et les
  scripts de référence doivent également être complétés.

## Exécution et archives

**Aucun test numérique, benchmark, autotuning ni chargement du paquet courant
n’a été exécuté.** La dernière analyse syntaxique portait sur 45 sources et
23 fichiers de test ; des changements ont eu lieu ensuite, notamment les
sources `extended_precision.jl` et `expert_refinement.jl`. Il ne faut donc pas
présenter tout l’état arrêté comme syntaxiquement revalidé.

Les 195 fichiers de mesures historiques v0.5 restent inchangés. Leur validation
ne s’applique pas aux sources de développement. La v0.5 est conservée dans
l’archive voisine `OxiBLAS-v0.5.jl.zip`. Le ZIP courant est un instantané du
travail non validé. Aucun pourcentage global d’achèvement fiable n’est annoncé.

Aucune reprise automatique ni demande de lancement des tests n’est programmée.
