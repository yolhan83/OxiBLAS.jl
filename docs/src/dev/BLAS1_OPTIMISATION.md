# Optimisation BLAS 1

Dernière campagne (séquentiel et parallèle, 1–18 threads, avant/après) : rapport du 3 novembre 2026 (`reports/BLAS1_OPTIMISATION_20261103.md`). Campagne précédente : 2 novembre 2026 (`reports/BLAS1_OPTIMISATION_20261102.md`).

Reproduction limitée à BLAS 1 : `python3 benchmark/level1_targeted.py nom` (séquentiel) ou `OXIBLAS_L1_MODE=parallel python3 benchmark/level1_targeted.py nom` (threads `OXIBLAS_L1_THREADS`, défaut 1,2,4,6,12,18).
Sélection : `OXIBLAS_L1_SELECT`, `OXIBLAS_L1_TYPES`, `OXIBLAS_L1_SIZES`, `OXIBLAS_L1_BACKENDS` ; essais hors `reports/` : `OXIBLAS_L1_REPORT_DIR` ; budget : `OXIBLAS_L1_SAMPLES`, `OXIBLAS_L1_SECONDS` ; exécutable : `OXIBLAS_JULIA`.
Tableaux : `python3 benchmark/level1_targeted.py table APRES.log [AVANT.log]`.
Préparer les dépendances de benchmark/Project.toml au préalable ; aucun téléchargement automatique.
BLAS 2 : même script avec `OXIBLAS_TARGET=level2` (worker `benchmark/full_worker.jl`, groupe « BLAS LEVEL 2 », tailles par défaut 64,256,1024,2048) ; rapport du 3 novembre 2026 (`reports/BLAS2_OPTIMISATION_20261103.md`).
BLAS 3 : `OXIBLAS_TARGET=level3` (groupe « BLAS LEVEL 3 », mêmes variables) ; rapport du 3 novembre 2026 (`reports/BLAS3_OPTIMISATION_20261103.md`).
LAPACK dense : `OXIBLAS_TARGET=lapack` (factorisations du groupe « LAPACK » ; sélection par défaut `LAPACK_SELECT` dans le script) ; rapport du 3 novembre 2026 (`reports/LAPACK1_OPTIMISATION_20261103.md`).
