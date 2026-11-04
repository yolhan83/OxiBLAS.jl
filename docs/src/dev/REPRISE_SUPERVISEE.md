# Reprise — correction numérique terminée

> Mise à jour BLAS 1 : optimisation ciblée et tableau complet (`reports/BLAS1_OPTIMISATION_20261102.md`). Julia 1.14.0-alpha2, un thread, mesures séquentielles OpenBLAS/Accelerate ; 26 173 assertions BLAS 1 réussies. La campagne globale ci-dessous est antérieure aux optimisations. Performances x86 non validées.

La demande utilisateur « ok on commence par les test resous tte les erreurs »
autorise la campagne de correction, les chargements/précompilations nécessaires,
les correctifs et leurs relances. Elle remplace les anciennes restrictions
interdisant les tests ou exigeant un accord pour chaque correctif.

Les 50 fichiers de tests passent à 1/2/4/8 threads sous Julia 1.13.3 macOS arm64,
avec bornes activées : 362 040 assertions réussies sur les quatre configurations,
zéro échec et zéro erreur. Voir [état courant](STATUS.md),
[rapport des corrections](VALIDATION_CORRECTIONS_20261102.md) et
sorties complètes (`reports/correction_tests_20261102.log`).

L’autorisation ne comprend pas les benchmarks, l’autotuning, la compilation Rust
ou la génération LLVM/assembleur. Aucun de ces travaux n’a été exécuté.
DoubleFloats/Quadmath et x86 restent non validés. La réussite de la suite ne
signifie pas que toutes les variantes Rust restantes sont implémentées.

Pour continuer : [tableau fonctionnel](TABLEAU_IMPLEMENTATION.md) et
[couverture](FUNCTIONAL_COVERAGE.csv). Les sources Rust et les historiques v0.5
sont inchangés. Les prochaines mesures de performance requièrent un accord.
