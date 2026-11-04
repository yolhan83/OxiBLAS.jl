# État courant — 0.6.0-dev, tests de correction

> Mise à jour LAPACK, première partie (3 novembre 2026) : LU, Cholesky, QR et variantes, séquentiel et parallèle (`reports/LAPACK1_OPTIMISATION_20261103.md`). `test/lapack_dense.jl` et les tests de factorisation existants réussis (suite complète du paquet non relancée).
>
> Mise à jour BLAS 3 (3 novembre 2026) : optimisation séquentielle et parallèle, tableaux avant/après (`reports/BLAS3_OPTIMISATION_20261103.md`). Tests ciblés BLAS 3 réussis à 4 threads (suite complète du paquet non relancée). Performances x86 non validées.
>
> Mise à jour BLAS 2 (3 novembre 2026) : optimisation séquentielle et parallèle, tableaux avant/après (`reports/BLAS2_OPTIMISATION_20261103.md`). Tests ciblés BLAS 2 et BLAS 1 réussis à 4 threads (suite complète du paquet non relancée). Performances x86 non validées.
>
> Mise à jour BLAS 1 (3 novembre 2026) : optimisation séquentielle et parallèle, tableaux avant/après (`reports/BLAS1_OPTIMISATION_20261103.md`). Julia 1.14.0-alpha2, M5 Max, OpenBLAS/Accelerate à 1–18 threads ; 32 380 assertions BLAS 1 réussies (suite complète du paquet non relancée). Performances x86 non validées.

La campagne autorisée le 2 novembre 2026 est terminée : **362 040 assertions
réussies à 1/2/4/8 threads, zéro échec et zéro erreur**. Les 50 fichiers enregistrés
ont été exécutés dans chaque configuration, avec `--check-bounds=yes`.
Voir [corrections, résultats et limites](VALIDATION_CORRECTIONS_20261102.md)
et le rapport intégral (`reports/correction_tests_20261102.log`).

Julia 1.13.3, macOS arm64 ; références OpenBLAS ILP64 0.3.30 limitées à un thread.
Les tests applicables couvrent les quatre types natifs, Float16/ComplexF16 et
BigFloat/Complex{BigFloat}. DoubleFloats/Quadmath restent non validés car absents.
Aucune performance 0.6 n’est attestée ; les benchmarks restent en attente d’accord.

## Modifications confirmées par les tests

- Lanceur compatible avec l’API publique de Test, dépendances des relances ciblées,
  métadonnées d’environnement et codes de retour fiables.
- Validation des types GEMM et des dimensions de Sylvester avant les opérations.
- Redémarrage filtré : chaque direction retenue progresse et une marge d’expansion
  est conservée, sans relâcher les tests de résidus.
- CP-ALS : linéarisation correcte des indices cartésiens, y compris quatre modes,
  rang deux, vues et conservation des entrées ; `einsum optimize=:none` direct.
- EVD sans vecteurs : construction correcte du résultat DC/MRRR.
- Raffinement QR mixte : gradient en précision de travail avant conversion basse
  précision ; normalisation binaire exacte des corrections.
- Oracles Arnoldi/appariement et fixtures XTrace/NonFMAFloat corrigés.

QZ/Schur, SVD partielle, Arnoldi/XTrace, SVD creuse à mémoire bornée et
microkernels triangulaires sont inclus dans la campagne complète. Les fallback
génériques sont conservés ; aucun calcul n’est délégué à un backend externe.
Rust reste inchangé au commit `8d82bb6bf1df133aa90525aa9f478ebe41514f9b` et
l’archive v0.5 conserve son SHA256 historique.

## Suite du projet

Le portage global demeure partiel. Le [tableau d’implémentation](TABLEAU_IMPLEMENTATION.md)
et [FUNCTIONAL_COVERAGE.csv](FUNCTIONAL_COVERAGE.csv) distinguent les fonctionnalités
présentes des variantes restantes : LAPACK compact/diagnostics, variantes de
préconditionnement/spectral, tenseurs et généricité sparse. Les helpers remplacés
par les abstractions Julia ne sont pas des fonctionnalités manquantes.

Performances, allocations en campagne dédiée, réglages ComplexF32/petites matrices,
validation x86/AVX2/AVX-512 et fournisseurs optionnels restent à traiter.
Le [lanceur complet](RUN_ALL.md) contient aussi des benchmarks et ne doit pas être
lancé sous la seule autorisation de tests. Les anciens bilans « non exécuté »
décrivent la préparation antérieure ; ce document et le rapport courant font foi
pour les tests effectivement exécutés.
