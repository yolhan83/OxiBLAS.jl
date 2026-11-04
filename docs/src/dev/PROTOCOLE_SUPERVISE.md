OxiBLAS.jl 0.6.0-dev — Mission de complétion et protocole de validation supervisée
Contexte exact (archive OxiBLAS 3.jl.zip, 2 novembre 2026)
Tu reprends le projet existant, sans le réinitialiser. Le ZIP correspond à OxiBLAS.jl 0.6.0-dev : 47 fichiers Julia dans src/, 9 103 lignes de sources, avec 23 fichiers de tests environ (1 977 lignes) et des scripts de benchmark principalement hérités de la v0.5. Les nouveautés v0.6 n'ont été ni chargées, ni testées numériquement, ni benchmarkées. Les résultats v0.5 (32 749 tests rapportés) ne certifient pas le code v0.6.

Lis d'abord, dans l'ordre : docs/POINT_ARRET_2026-11-02.md, docs/STATUS.md, docs/DEVELOPMENT_v0.6.md, docs/MISSION.md, docs/upstream_inventory.csv, src/OxiBLAS.jl, test/runtests.jl et l'inventaire des scripts de benchmark/.

L'amont de référence pour la comparaison est le commit Rust 8d82bb6bf1df133aa90525aa9f478ebe41514f9b. N'invente pas un pourcentage d'avancement à partir du nombre de fichiers. La dernière grille de 370 fichiers est obsolète : rapproche-la des fonctions, variantes et comportements réellement portés.
But — d'abord la complétion du code, ensuite les campagnes autorisées
Ne change pas cet ordre :
1. Achever les fonctionnalités pertinentes de l'OxiBLAS Rust de référence, y compris BLAS, LAPACK, sparse, solveurs, décompositions spectrales, tenseurs, einsum, et précision étendue ; conserver une liste explicite des limitations. Ne pas reproduire les couches Rust sans intérêt sous Julia (types de conteneurs génériques, no_std, FFI Rust) si les AbstractArray natifs couvrent déjà le besoin.
2. Écrire une suite de tests complète pour chaque API nouvelle ou modifiée, et non seulement un « smoke test » global.
3. Écrire les scripts et protocoles de benchmarks complets de toute la bibliothèque, y compris les outils d'inspection LLVM/assembleur, sans les exécuter.
4. Une fois ces trois livrables préparés, t'arrêter et me demander mon autorisation explicite avant toute exécution.
Il ne s'agit pas de prétendre que tout fonctionne avant les tests. Distingue systématiquement écrit, couvert par un test non exécuté, testé, benchmarké, validé sur une architecture et absent. Ne marque jamais « terminé et validé » un algorithme qui n'a pas été exécuté.
A. Première phase — travail autorisé, SANS exécutions numériques
Pendant cette phase, tu peux lire, comparer et modifier les sources, écrire les tests, les benchmarks, les scripts de diagnostics et la documentation ; produire des diffs, tableaux de couverture, listes de commandes. Ne lance pas using OxiBLAS, Pkg.test, test/runtests.jl, tests ciblés, benchmarks, autotuning, précompilation, @code_llvm, @code_native ni scripts de génération. Ne confonds pas analyse statique et validation numérique. Ne lance pas de commande qui exécute du code du projet sans accord ; le contrôle de sources purement statique est permis.
Fonctionnalités restant à traiter — minimum à auditer
Utilise docs/POINT_ARRET_2026-11-02.md comme liste de départ et confronte-la à chaque sous-système amont :
- BLAS/CPU : compléter les variantes non couvertes, kernels petits/complexes/triangulaires SIMD, stratégies de parallélisme, plans réutilisables ; éviter le parallélisme imbriqué ; préparation de l'autotuning sans l'exécuter.
- LAPACK/Spectral : QZ réel et réordonnancement généralisé, gestion des crayons singuliers ; Schur réel, multishift/small-bulge ; SVD bidiagonale spécialisée, variantes sélectionnées et incrémentales ; packed LDL/pivotages, diagnostics condition/erreurs et raffinements réellement fondés ; solveurs avancés.
- Sparse : LU multifrontal et LDL avec pivotage lorsque pertinent ; AINV, préconditionneurs et variantes manquantes ; bloc Arnoldi/Lanczos, thick restart, SVD itérative ; graphes AMD/COLAMD/partitionnement, appariement ; estimations Hutchinson/Hutch++/XTrace ; conversions et formats. Les types sparse spécialisés sont autorisés.
- Tenseurs : porter toute la partie tensorielle pertinente d'OxiBLAS : einsum, contractions N-D, planification sous contrainte mémoire, réutilisation des intermédiaires, batched, décompositions et variantes avancées. Ne crée aucun type Tensor dense : utiliser AbstractArray{T,N}, les vues et les wrappers de Julia. Des types pour les factorisations ou tenseurs creux sont acceptables.
- Précision étendue : compléter les opérations et types réellement utiles et documenter les limites numériques du double-double et de Float16 ; ne pas annoncer un BLAS/LAPACK étendu complet si seuls quelques kernels existent.
- Défaut déjà repéré : sparse_lu dans src/sparse_direct.jl, pivot_threshold=0 pouvant conserver un pivot diagonal nul malgré une ligne non nulle plus bas. Corrige la logique et écris un test de régression, sans prétendre l'avoir passé.
Si des modules ne sont pas portables, dépassent la portée pertinente de Julia ou restent incomplets, justifie fonction par fonction et indique une alternative native, sans marquer la fonction couverte.

Conserve le calcul numérique 100 % Julia : pointeurs, Base.llvmcall, NEON, AVX2, AVX-512, assembleur LLVM bas niveau autorisés. Pas de délégation à OpenBLAS/Accelerate/Rust ni de type piracy. Pas de dépendance SIMD.jl ou LoopVectorization.jl ajoutée par défaut. Vérifie statiquement les chemins @inbounds, la sûreté GC, les strides, l'aliasing, la stabilité numérique et la concurrence.
Tests à ÉCRIRE (sans les lancer)
Associe chaque fonction exportée, chaque type et chaque variante à des fichiers/cas de test : dimensions 0/1/petites/irrégulières, Float32/Float64/ComplexF32/ComplexF64 quand applicable, Float16/étendue selon API, transposition/adjoint, strides négatifs si acceptés, vues, diagonale unité ignorée, β=0 et α=0 sans lecture, NaN/Inf, singularités, pivotage, rang déficient, spectres proches, matrices mal conditionnées, échelles extrêmes, résidus et invariants mathématiques, collisions d'alias, GC.@preserve et collectes, partages concurrents, threads 1/2/4/8. Tests indépendants avec LinearAlgebra/LAPACK/Rust dans le dossier test uniquement et références analytiques pour les cas sensibles.

Prévois les tests complets de décompositions, sparse et tenseurs, pas seulement ceux des anciennes v0.5. Intègre-les explicitement à test/runtests.jl ou à un lanceur documenté. Prépare un rapport de couverture API/fonction → tests, en marquant « non exécuté » partout.
Benchmarks à ÉCRIRE (sans les lancer)
Prépare une campagne reproductible pour toutes les grandes familles, et pas uniquement GEMM :
- BLAS 1, 2 dense/banded/packed, BLAS 3, GEMM (petites à très grandes, rectangulaires, transposées et complexes), batched et scaling 1/2/4/8 threads ;
- LU/Cholesky/QR/LDL, solveurs, SVD/EVD/Schur/QZ, variantes spectrales et précision mixte ;
- sparse (formats/conversions, SpMV/SpMM/SpGEMM, solveurs et préconditionneurs, direct, spectral), densités, motifs et tailles contrôlés ;
- tenseurs (einsum 2+ opérandes, contractions variées, plans, coûts d'intermédiaires, batch, CP/HOSVD/TT) ;
- autotuning, performance d'appel, petites tailles, compilation/TtFX, allocations, mémoire temporaire, scalabilité, absence de sursouscription BLAS.
Ajouter les références OpenBLAS, Accelerate sur Mac, et OxiBLAS Rust lorsque disponibles, avec données/threads/précisions équivalents et processus distincts quand nécessaire. Vérifier les sorties avant de chronométrer. Séparer temps de compilation, temps chaud, temps minimal/médian, allocations et workspaces ; CSV + configuration Julia/CPU/BLAS/threading/commit + logs + hashes. Garder intactes les données v0.5 : ne jamais écraser les historiques ni attribuer leurs résultats à la v0.6. Prévoir durées/budgets raisonnables et possibilité de reprendre une campagne interrompue.
Livrables obligatoires avant de demander à lancer quoi que ce soit
- Code source complet de la passe avec liste de modifications par fichier.
- Tableau fonction upstream → API Julia → statut d'implémentation → tests → benchmark → limite à jour ; ne plus utiliser les anciens comptes non réconciliés.
- Suite de tests et scripts de benchmark complets, non exécutés.
- Scripts d'inspection LLVM/assembleur préparés, non exécutés, avec signatures réelles prévues des kernels.
- Liste des risques et points encore non implémentés ; indiquer explicitement qu'aucune performance ou conformité v0.6 n'est attestée.
- Proposition d'une première campagne d'exécution : commandes exactes, ordre, coût/risque et fichiers de sortie attendus.
ARRÊT OBLIGATOIRE : demande-moi alors : « Autorises-tu cette campagne d'exécution précise ? » Ne la démarre pas sans réponse affirmative.
B. Après mon accord — première campagne et analyse exhaustive
Exécute uniquement les commandes/campagnes approuvées. Commence par la capacité à charger/précompiler le package, puis les tests de correction et bornes (--check-bounds=yes) en 1 puis plusieurs threads, et enfin les benchmarks si la correction est suffisante et si ces campagnes ont été autorisées. N'exécute pas de benchmarks interprétables comme valides après échecs numériques non résolus ; demande un nouvel accord si le plan doit changer.

Pour chaque commande, sauvegarde les sorties stdout/stderr complètes, code de retour, versions, environnement, timestamp, artefacts et durée observée. Lis les logs intégralement ou via des analyses qui conservent toutes les erreurs ; ne te contente pas des dernières lignes ni d'un « tests passés » global. Classe les défauts : chargement/dispatch/ambiguïtés, valeurs fausses, stabilité, alias/pointeurs/GC, race, allocations, performance, régression, portabilité. Sépare erreurs de référence et erreurs du code natif.

Produis ensuite un rapport priorisé : symptôme → preuve (log et test) → cause probable → fonction et fichier → correction proposée → conséquence attendue → test/benchmark de confirmation. Donne les comparaisons chiffrées à la v0.5 et aux bibliothèques externes quand elles existent, sans extrapoler les chiffres absents.
C. Avant chaque correction — mon approbation
Sur la base des résultats, propose les modifications précises, puis demande ma validation. Si l'IR LLVM ou l'assembleur sont utiles, indique exactement quelles fonctions spécialisées, quels types/tailles/flags, et quel problème l'inspection doit départager (vectorisation, FMA, allocations, dispatch, appels parasites, branches, code size). Prépare les commandes @code_llvm, @code_native, code_llvm/`code_native` ou scripts benchmark/inspect_*.jl adaptés, et nomme les artefacts .ll/`.s` prévus. Ne génère rien avant accord, et n'inspecte pas systématiquement tout le package.

Après approbation, effectue uniquement les diagnostics et modifications approuvés. Si les diagnostics invalident l'hypothèse ou impliquent des changements supplémentaires importants, présente un nouveau plan et demande accord avant de les appliquer. Mets à jour les tests, benchmarks et docs correspondants, sans effacer les logs précédents.

Puis ARRÊT OBLIGATOIRE à nouveau : présente la liste des patches appliqués et la prochaine liste exacte de tests/benchmarks/générations à exécuter. Demande explicitement mon accord avant de lancer ce nouveau cycle. Ne déduis jamais l'autorisation des campagnes suivantes de l'accord précédent.
D. Boucle de travail définitive
Écrire les fonctionnalités, tests et benchmarks (sans exécuter) → demander autorisation → exécuter la campagne autorisée → lire et interpréter toutes les sorties → proposer corrections et éventuels LLVM/assembleur → demander validation → effectuer les travaux validés → demander autorisation pour la campagne suivante → recommencer.

Il peut y avoir plusieurs cycles. Aucune modification hors du périmètre approuvé après la première campagne ; aucune campagne non autorisée ; aucune déclaration de tests « verts » non exécutés.
Format attendu des comptes rendus
À chaque point d'arrêt, présente dans cet ordre :
1. État exact de la version et distinction « écrit / non exécuté / validé ».
2. Réalisé depuis le dernier accord (fichiers, APIs, tests, scripts, patchs).
3. Résultats vérifiés : synthèse tests, échecs, tableaux perf, allocations, scaling, provenance de chaque mesure.
4. Diagnostic et modifications proposées par priorité ; besoin éventuel de sorties LLVM/assembleur.
5. Étape nécessitant mon accord, avec commandes et artefacts précis.
Tu ne dois pas me faire répéter une autorisation déjà clairement donnée pour le lot courant, mais tu ne dois jamais l'étendre au lot suivant.