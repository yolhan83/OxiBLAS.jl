# Tableau d’implémentation — 2026-11-02

État après la campagne de correction du 2 novembre 2026 : **les 50 fichiers de tests passent à 1/2/4/8 threads**, bornes activées. Voir [périmètre et résultats](VALIDATION_CORRECTIONS_20261102.md). Les variantes fonctionnelles restantes ci-dessous demeurent ouvertes.
Sources, tests et benchmarks écrits ne signifient pas résultats validés.
Le suivi détaillé par famille est FUNCTIONAL_COVERAGE.csv. Les abstractions et
helpers Rust remplacés par Julia ne constituent pas des fonctionnalités manquantes.

| Domaine | État actuel | Reste significatif |
| --- | --- | --- |
| QZ et Schur | Toute la ligne écrite : QZ réel double shift, blocs 2×2, réordonnancements réel/généralisé, conditions, TREVC/TRSNA et train small-bulge/AED avec mises à jour bloquées ; [port et corrections Rust](QZ_SCHUR_PORT.md) | Tests enregistrés réussis ; performances et validation x86 à exécuter après autorisation |
| SVD et LAPACK avancé | LU partiel/complet/rook, LDL/BK, QR pivoté/bloqué, solveurs experts et variantes SVD écrits ; [sélection réellement partielle](LOT_SVD_PARTIELLE.md) par indices/intervalle ajoutée | Tests du lot partiel réussis ; kernels bidiagonaux spécialisés, packed LDL/pivotage et bornes d’erreur/diagnostics avancés |
| Sparse direct | Ligne écrite pour le périmètre amont audité : LDL pivoté, LU multifrontale/statique, diagnostics et LU/Cholesky disque | Tests numériques/concurrence/GC réussis ; performances en attente ; pas de nouveau manque fonctionnel confirmé pour cette ligne |
| Spectral et préconditionnement | Blocs Krylov, Arnoldi redémarré/Krylov–Schur avec soft locking et réorthogonalisation sélective, filtres/intervalles et préconditionneurs écrits ; [lot courant](LOT_ARNOLDI_XTRACE.md) | Tests enregistrés réussis ; transformations généralisées supplémentaires et variantes avancées de préconditionnement/coarsening à préciser |
| Graphes | Ligne écrite pour la surface pertinente auditée : appariements, ordonnancements et partitionnement multiniveau | Tests enregistrés réussis ; variantes Julia adaptées, sans prétention d’identité avec SuiteSparse/METIS ni d’optimalité globale |
| Tenseurs | Contractions/einsum, plans/workspaces, beam/exact, tranches et HOSVD/TT/CP écrits | Clôture des écarts fonctionnels de variantes ; auto-choix des tranches et coût physique à optimiser si nécessaire. Distribué/hors mémoire tensoriel : pertinence à confirmer, pas un manque amont établi |
| Précision générique | Architecture corrigée : types matériels spécialisés, fallback AbstractFloat/complexes, aucun scalaire DoubleDouble maison | BigFloat et half testés ; Double64/Float128 restent à vérifier ; étendre si pertinent les familles restant limitées aux types natifs, notamment sparse/spectral |
| Optimisations | SIMD/packing existants conservés ; [GK creux borné, workspaces SVD/triangulaires et microkernels TRMM/TRSM](LOT_MEMOIRE_WORKSPACES_TRIANGULAIRES.md) écrits | Tests mémoire/GC/concurrence réussis ; performances en attente ; réglages ComplexF32/petites matrices, autres allocations/contextes et tuning AVX2/512 sur matériel adapté |
| Stochastique — point isolé du lot Optimisations | Hutchinson/Hutch++, leave-one-out de référence et XTrace optimisé avec downdates inverse-triangulaires/rang déficient écrits ; [lot courant](LOT_ARNOLDI_XTRACE.md) | Tests des estimations/rangs/normalisation réussis ; performances en attente ; erreur empirique non certifiée |
| Validation transversale | run_all.sh et rapport unique Julia/Rust/OpenBLAS/Accelerate préparés ; tests bounds=yes, benchmarks bounds=auto ; gate numérique | Tests et corrections terminés ; benchmarks à autoriser séparément |

## Écarts complémentaires repérés, sans grossir artificiellement la liste

- **EVD dense randomisée** : l’amont propose oversampling et power iterations.
  Des Ritz ciblés via Krylov existent déjà en Julia ; l’équivalence de ce contrat
  de travail doit être précisée avant de décider d’un port dédié. Les valeurs
  propres approximatives ne sont pas une fonctionnalité entièrement absente.
- **Préconditionnement** : partitions arbitraires de Block Jacobi et variantes
  SAMG/coarsening demandent un rapprochement des paramètres avec les équivalents
  Julia existants. Ce sont des écarts de variantes, pas l’absence de Jacobi/AMG.
- **Calcul sparse générique** : les formats et certains moteurs restent bornés
  aux quatre types matériels. La généricité dense ne valide pas BigFloat en sparse.

Les solveurs tridiagonaux ont un équivalent fonctionnel via les facteurs bandés
(kl=ku=1) et leurs résolutions : pas de nouvelle ligne « solveur absent ».
Mat/MatRef/MatMut, Tensor3/NdTensor, accesseurs, itérateurs et traits sont traités
par les abstractions Julia lorsque cela couvre la fonctionnalité, sans port 1:1.

Priorité de reprise : **SVD/LAPACK avancé → spectral et écarts confirmés** ; les tests de correction QZ/Schur sont désormais exécutés. Les performances restent non mesurées.
Les optimisations spéculatives et la réconciliation des helpers ne bloquent pas
cette reprise. Aucune autorisation de test/benchmark n’est déduite de ce tableau.
