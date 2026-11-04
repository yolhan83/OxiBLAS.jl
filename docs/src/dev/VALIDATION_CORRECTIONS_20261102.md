# Correction numérique — 2 novembre 2026

Campagne de tests autorisée par « ok on commence par les test resous tte les erreurs ».
Les corrections et relances sont couvertes par cette demande. Benchmarks,
autotuning et génération LLVM/assembleur ne sont pas autorisés et n’ont pas été lancés.

Rapport stdout/stderr conservé : `reports/correction_tests_20261102.log`.
Julia stable 1.13.3, macOS arm64, `--check-bounds=yes`, BLAS de référence limité à un thread.
Chargement initial également réussi avec Julia 1.14.0-alpha2 ; ceci ne valide pas
la suite sur cette version alpha.

## Corrections

| Défaut constaté | Cause et correction | Confirmation ciblée |
| --- | --- | --- |
| Lanceur incompatible avec Julia 1.13 | Suppression de l’usage des fonctions internes Test.push_testset/pop_testset ; utilisation de @testset, comptage récursif conservé, sortie non nulle en cas d’échec | Rapports ciblés réussis et échoués, avec compteurs et codes de retour cohérents |
| GEMM sur un type élément union | one(T)/zero(T) évalués avant le contrôle de type ; coefficients par défaut entiers, conversion après validation ; même garde dans le fallback générique | audit_priority : 597 assertions réussies |
| Sylvester généralisé : erreur de validation | Vérification des dimensions avant la forme de Schur | qz_schur_complete : 1 128 assertions réussies |
| Filtrage spectral redémarré stagnant | Réserver de l’espace pour l’expansion et appliquer le filtre à chaque direction retenue avant le redémarrage ; contrôles de résidus inchangés | interval_spectral + weighted_matching : 3 277 assertions réussies |
| Oracle Arnoldi sur un seuil coupant une paire conjuguée | Vérifier l’appartenance au spectre sans réutilisation et les modules extrémaux ; les résidus restent contrôlés | schur + restarted_arnoldi_xtrace : 1 943 assertions réussies |
| Test XTrace : construction de configuration | Point-virgule manquant avant les arguments nommés | Même campagne de 1 943 assertions, toutes les branches XTrace exécutées |
| Test d’appariement : égalité flottante exacte | Résultat -3.0000000000000004 correct en précision de travail ; comparaison approchée, permutation optimale toujours exacte | Inclus dans les 3 277 assertions ci-dessus |
| CP-ALS : indices cartésiens utilisés comme lignes matricielles | enumerate(CartesianIndices(...)) fournit les indices linéaires de K sans changer l’ordre column-major | Tenseurs + génériques : 4 369 assertions réussies, dont CP-ALS quatre modes/rang deux |
| EVD divide-and-conquer/MRRR sans vecteurs | Constructeur EigenFactor explicite pour Nothing, sans paramètre de type non inférable | extended_spectral : 4 536 assertions réussies |
| Raffinement QR mixte sur un système inconsistant | Former le gradient en précision de travail avant conversion basse précision ; préconditionner avec les deux résolutions triangulaires du facteur QR, sans former AᴴA | extended_lapack : 3 832 assertions réussies, seuils conservés |
| Fallback BigFloat mal conditionné | Normalisation binaire exacte du RHS au lieu d’un diviseur arbitraire injectant un arrondi | Même suite LAPACK étendue réussie |
| Fixture NonFMAFloat ambiguë sur Julia 1.13 | Constructeur identité explicite du type de test ; aucune méthode ajoutée aux types de tiers | generic_contract : 15 assertions réussies |
| einsum optimize=:none refusé | Activation explicite de la contraction directe existante, sans intermédiaires de paires | Même relance tensorielle : 4 369 assertions réussies |
| Tests ciblés dépendants d’autres fichiers | Résolution explicite des dépendances de helpers de tests, sans désactiver d’assertion | Campagnes ciblées et campagne complète |

Les 2 124 assertions du nouveau lot mémoire SVD/workspaces/triangulaires sont
également passées lors d’une campagne ciblée. Cette réussite partielle ne remplace
pas la validation globale.

## Traçabilité des essais non concluants

Le premier passage complet a été interrompu au test disque par un signal de
diagnostic SIGUSR1 non pris en charge sur ce processus Julia/macOS (code 158).
Il reste dans le rapport, explicitement incomplet. Le diagnostic suivant utilise
un échantillon de pile sans signal. Celui-ci montre une compilation LLVM active, pas un verrou bloqué. Le test disque isolé termine normalement : 2 782 assertions réussies en 2 min 06 s. Un essai de séparation des cas a été retiré après n’avoir pas réduit le coût ; une référence de variable locale oubliée durant cet essai reste tracée dans le rapport, puis corrigée par ce retrait. Aucun résultat global réussi n’en est déduit.
Un essai diagnostique --compile=min échoue car les intrinsics llvmcall requièrent
la compilation ; ce mode ne sert pas de campagne de validation des kernels.

Les fournisseurs optionnels DoubleFloats et Quadmath sont absents du dépôt Julia
local. Les tests compatibles exécutent Float16, Float32, Float64, leurs complexes,
BigFloat et Complex{BigFloat}. Les fournisseurs absents ne sont pas déclarés validés.
Les performances, x86/AVX2/AVX-512 et les comparaisons de backends restent non mesurés.

## Résultat global

Les 50 fichiers de tests enregistrés passent avec contrôle des bornes activé
sur les mêmes sources à 1, 2, 4 et 8 threads. Total : **362 040 assertions réussies**
sur les quatre configurations, zéro échec et zéro erreur. Les assertions répétées
entre configurations ne sont pas des tests uniques supplémentaires.

| Threads Julia | Assertions réussies | Échecs | Erreurs | Ignorés Test.Broken | Code retour |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 90 510 | 0 | 0 | 0 | 0 |
| 2 | 90 510 | 0 | 0 | 0 | 0 |
| 4 | 90 510 | 0 | 0 | 0 | 0 |
| 8 | 90 510 | 0 | 0 | 0 | 0 |

Versions/environnement : Julia 1.13.3, macOS 27.2 arm64, cible CPU Julia apple-m1,
OpenBLAS ILP64 0.3.30 pour les références, limité à un thread. La cible CPU Julia
n’est pas une identification indépendante du modèle physique de processeur.
Les configurations ont tourné dans des processus indépendants simultanément ;
leurs durées de test sont tracées dans le rapport et ne sont pas des benchmarks.

Les huit types annoncés dans le rapport sont couverts par les suites qui leur
sont applicables : Float16, Float32, Float64, ComplexF16/F32/F64, BigFloat et
Complex{BigFloat}. Cela n’étend pas les API sparse/spectrales restreintes aux
quatre types matériels. DoubleFloats et Quadmath, absents, ne sont pas validés ;
leur absence n’est pas comptée comme une assertion Test.Broken.

Les quatre tests Python du contrat de rapport passent également. Aucun benchmark,
autotuning, compilation Rust ou fichier LLVM/assembleur n’a été lancé/généré.
La réussite concerne les cas de test existants, pas une certification universelle
de stabilité, d’équivalence Rust ou de performance.

