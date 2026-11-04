# Journaux de développement

Ces pages sont les notes de conception, d'audit et de validation écrites
pendant le portage (versions 0.1 à 0.6). Elles sont conservées telles quelles :
leur état (« préparé », « non exécuté », etc.) décrit le moment où elles ont
été écrites, pas l’état actuel du paquet. L’état à jour est décrit dans les
guides.

- [Restructuration du paquet (4 novembre 2026)](RESTRUCTURATION_20261104.md) — `RESTRUCTURATION_20261104.md`
- [Interface Julia indépendante](API.md) — `API.md`
- [Audit prioritaire — 2026-11-02, phase A](AUDIT_PRIORITAIRE.md) — `AUDIT_PRIORITAIRE.md`
- [Audit statique du lanceur unique — 2026-11-02](AUDIT_STATIQUE_RUN_ALL.md) — `AUDIT_STATIQUE_RUN_ALL.md`
- [Static architecture audit — 2026-11-02](AUDIT_STATIQUE_TYPES.md) — `AUDIT_STATIQUE_TYPES.md`
- [Optimisation BLAS 1](BLAS1_OPTIMISATION.md) — `BLAS1_OPTIMISATION.md`
- [BLAS 3 structuré — conception de la v0.3](BLAS3_DESIGN.md) — `BLAS3_DESIGN.md`
- [Validation BLAS 3 v0.3 — 2 novembre 2026](BLAS3_VALIDATION.md) — `BLAS3_VALIDATION.md`
- [BLAS 2 compact et tuiles BLAS 3 — v0.5](COMPACT_PARALLEL_DESIGN.md) — `COMPACT_PARALLEL_DESIGN.md`
- [Validation compact/inter-tuiles v0.5 — 2 novembre 2026](COMPACT_PARALLEL_VALIDATION.md) — `COMPACT_PARALLEL_VALIDATION.md`
- [OxiBLAS.jl 0.6.0-dev — continuation](CONTINUATION_2026-11-02.md) — `CONTINUATION_2026-11-02.md`
- [OxiBLAS.jl v0.4.0 — Bilan et suite du développement](CONTINUATION_REQUEST_v0.4.md) — `CONTINUATION_REQUEST_v0.4.md`
- [OxiBLAS.jl — État du projet et instructions de continuation](CONTINUATION_v0.2_original.md) — `CONTINUATION_v0.2_original.md`
- [OxiBLAS.jl 0.6.0-dev — implémentation en attente de validation](DEVELOPMENT_v0.6.md) — `DEVELOPMENT_v0.6.md`
- [Rotations modifiées et produits étendus — v0.4](LEVEL1_DESIGN.md) — `LEVEL1_DESIGN.md`
- [Validation BLAS 1 complémentaire v0.4 — 2 novembre 2026](LEVEL1_VALIDATION.md) — `LEVEL1_VALIDATION.md`
- [Arnoldi par blocs et XTrace optimisé — lots écrits](LOT_ARNOLDI_XTRACE.md) — `LOT_ARNOLDI_XTRACE.md`
- [Filtres spectraux et intervalles — phase A, 2 novembre 2026](LOT_FILTRES_INTERVALLES.md) — `LOT_FILTRES_INTERVALLES.md`
- [Ligne « Graphes » — complétion des livrables en phase A](LOT_GRAPHES_COMPLETION.md) — `LOT_GRAPHES_COMPLETION.md`
- [Graphes creux — continuation du 2 novembre 2026](LOT_GRAPHES_CREUX.md) — `LOT_GRAPHES_CREUX.md`
- [SVD creuse, contextes réutilisables et feuilles triangulaires](LOT_MEMOIRE_WORKSPACES_TRIANGULAIRES.md) — `LOT_MEMOIRE_WORKSPACES_TRIANGULAIRES.md`
- [Types flottants génériques — correction d’architecture](LOT_PRECISION_ETENDUE.md) — `LOT_PRECISION_ETENDUE.md`
- [Ligne « Sparse direct » — livrables écrits, non exécutés](LOT_SPARSE_DIRECT.md) — `LOT_SPARSE_DIRECT.md`
- [Spectral par blocs — 2 novembre 2026](LOT_SPECTRAL_BLOCS.md) — `LOT_SPECTRAL_BLOCS.md`
- [Continuation du 2 novembre 2026 — écrit, non exécuté](LOT_STOCHASTIQUE_TENSEURS.md) — `LOT_STOCHASTIQUE_TENSEURS.md`
- [SVD incrémentale — continuation du 2 novembre 2026](LOT_SVD_INCREMENTALE.md) — `LOT_SVD_INCREMENTALE.md`
- [SVD réellement partielle — code préparé](LOT_SVD_PARTIELLE.md) — `LOT_SVD_PARTIELLE.md`
- [Tenseurs — chemins avancés et exécution par tranches](LOT_TENSEURS_PLANIFICATION.md) — `LOT_TENSEURS_PLANIFICATION.md`
- [Mission: Reimplement OxiBLAS in Native Julia — Full Low-Level Performance](MISSION.md) — `MISSION.md`
- [Mesures et inspection du code machine](PERFORMANCE.md) — `PERFORMANCE.md`
- [Point d’arrêt — 2 novembre 2026](POINT_ARRET_2026-11-02.md) — `POINT_ARRET_2026-11-02.md`
- [PROTOCOLE_SUPERVISE.md](PROTOCOLE_SUPERVISE.md) — `PROTOCOLE_SUPERVISE.md`
- [QZ / Schur — port algorithmique et complétion](QZ_SCHUR_PORT.md) — `QZ_SCHUR_PORT.md`
- [Reprise — correction numérique terminée](REPRISE_SUPERVISEE.md) — `REPRISE_SUPERVISEE.md`
- [Campagne unique — préparation uniquement](RUN_ALL.md) — `RUN_ALL.md`
- [Validation spectrale v0.2 — 2 novembre 2026](SPECTRAL_VALIDATION.md) — `SPECTRAL_VALIDATION.md`
- [État courant — 0.6.0-dev, tests de correction](STATUS.md) — `STATUS.md`
- [Tableau d’implémentation — 2026-11-02](TABLEAU_IMPLEMENTATION.md) — `TABLEAU_IMPLEMENTATION.md`
- [Validation de l’incrément 0.1 — 2 novembre 2026](VALIDATION.md) — `VALIDATION.md`
- [Correction numérique — 2 novembre 2026](VALIDATION_CORRECTIONS_20261102.md) — `VALIDATION_CORRECTIONS_20261102.md`

Données tabulaires du même dossier (fichiers du dépôt, non rendus) :
`docs/src/dev/BENCHMARK_API_AUDIT.csv`, `docs/src/dev/COVERAGE_SUPERVISED.csv`, `docs/src/dev/FUNCTIONAL_COVERAGE.csv`, `docs/src/dev/LOW_LEVEL_SITES.csv`, `docs/src/dev/MEMORY_ACCESS_AUDIT.csv`, `docs/src/dev/UPSTREAM_FUNCTION_AUDIT.csv`, `docs/src/dev/upstream_inventory.csv`.
