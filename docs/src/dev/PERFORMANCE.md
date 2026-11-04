# Mesures et inspection du code machine

Les CSV de `reports/` contiennent les mesures brutes, et les fichiers
`*-environment.txt` les versions et paramètres. Les temps sont des observations
locales sur Apple M5 Max ; ils ne prédisent pas les performances d’autres CPU.
Les noyaux x86 n’ont pas été chronométrés ni exécutés sur du matériel x86.

`benchmark/benchmarks.jl` utilise BenchmarkTools, un échauffement, des sorties
préallouées et des workspaces réutilisables. Chaque ligne vérifie le résultat
contre le backend de référence avant de chronométrer. Les tailles carrées vont de
16 à 4096 avec `--large`, pour les quatre types, avec des cas rectangulaires.
Les six familles GEMM sont comparées sur les petites tailles ; les boucles
de référence coûteuses sont limitées à 128. Pour les grandes tailles, les voies
SIMD et BLAS de référence sont mesurées. Le contrôle `:auto` sélectionne la voie
bloquée jusqu’à 8, puis les panneaux SIMD ou le repli registre ; cette limite a
été abaissée après les premières mesures des tailles 16 et 32.

Les colonnes `threads` et `active_workers` distinguent les threads configurés
des tâches Julia effectivement lancées ; pour BLAS, la seconde valeur est le
nombre de threads rapporté par le backend, sans mesure de l’occupation des cœurs.
Le seuil de parallélisme évite de lancer
des tâches pour les petits produits. Les mesures OpenBLAS incluent des lignes
à un et huit threads pour permettre des comparaisons équivalentes. La
scalabilité Julia se calcule entre ses propres lignes à un et huit threads.
Accelerate est explicitement placé en mode monothread, via AppleAccelerate.jl,
dans un processus distinct. Cette dépendance sert uniquement au benchmark.

Le temps minimal et la médiane sont tous deux conservés, avec au plus 15
échantillons et un budget indicatif de 0,15 s par essai. Une opération longue
peut dépasser ce budget ; les lignes à peu d’échantillons doivent être lues avec
prudence. Les FLOPs sont calculés indépendamment : 2mnk en réel, 8mnk en complexe,
conventions dominantes ne comptant pas le coût inférieur d’α/β ou de l’empaquetage.
Les allocations incluent le contrôle de l’appel, sans la construction préalable
du workspace. Le premier appel et le temps de compilation rapporté par `@timed`
sont consignés séparément. Trafic mémoire et compteurs matériels non mesurés.

L’amont Rust est compilé séparément en release, `target-cpu=native`, LTO fin.
Ses 36 fixtures couvrent les quatre types et N/T/C sur A et B, avec α/β non
triviaux. `test/rust_conformance.jl` les vérifie sans exécuter Rust dans le paquet.
Pour les timings Rust, `benchmark/compare_rust.jl` reproduit exactement les mêmes
matrices déterministes Float64, α=0,7, β=0 et cinq échantillons. `julia_api`
inclut l’allocation des panneaux à chaque appel comme l’API Rust ;
`julia_workspace` mesure aussi le cas réutilisable. Comparer les minima entre
`rust-gemm.csv` et `julia-rust-workload.csv`, en tenant compte des runtimes
distincts. Ces résultats ne sont pas une comparaison complète de toutes les
fonctionnalités des deux projets.

L’inspection `@code_native` / `@code_llvm` est reproduite par
`benchmark/inspect_assembly.jl`. Les fichiers `neon-*-native.s` montrent les FMA
vectorielles réelles et complexes. Le premier noyau réel contenait des appels
de wrappers LLVM et des spills d’accumulateurs : l’ajout d’`alwaysinline` dans
la fonction LLVM et d’annotations Julia a éliminé ces appels dans le micro-noyau.
L’observation avant/après est enregistrée dans `kernel-evolution.csv`, et l’ancien
assembleur conservé à titre d’audit. Le code final n’emploie pas de fast-math.

Les fichiers `exploratory-*` proviennent de la première campagne de développement,
pendant laquelle d’autres travaux pouvaient encore compiler ou s’exécuter. Les
fichiers sans ce préfixe sont issus de la passe finale, dont les commandes
numériques ont été lancées successivement. Ils constituent la base du bilan.

Le bilan chiffré final et les journaux de vérification sont dans
[VALIDATION.md](VALIDATION.md). Les performances complexes et les tailles réduites
peuvent rester inférieures aux références ; aucune supériorité générale n’est
revendiquée. Les choix d’alignement, de blocage et de déroulage restent des
paramètres à réévaluer sur chaque famille de processeurs.

## Campagne BLAS 3 v0.3

`benchmark/level3.jl` compare l'implémentation native aux fonctions v0.2 figées
dans `reports/baseline-v0.2/level3.jl` et aux BLAS externes, dans le seul dossier
benchmark. Tailles 32/128/512, quatre types, côtés L/R, triangles U/L et
transpositions pertinentes. Les opérations hermitiennes réelles équivalentes
aux symétriques ne sont pas doublonnées. GEMMT est mesuré en natif, sans
équivalent portable dans l'interface BLAS de Julia ; il est validé par un
produit indépendant et par les contrôles de triangle.

Un seul thread est utilisé pour les comparaisons de temps. La sortie est remise
à l'état initial avant chaque évaluation, hors de la zone chronométrée.
Les API ordinaires incluent les allocations des panneaux ; leur réutilisation
est disponible dans l'API et testée séparément. Le premier appel et son coût
de compilation, minimum/médiane, allocations et erreur relative sont conservés.
Sept échantillons au plus et 0,12 s par essai : les gains sont des observations
locales de courte durée, et non une garantie pour toutes les matrices.

Les campagnes OpenBLAS et Accelerate s'exécutent dans des processus
successifs, sans test ou autre benchmark lancé simultanément par cet agent.
La comparaison v0.2/native constitue la mesure avant/après conservée.

La diagonale initiale de C est réelle pour HERK/HER2K dans la campagne de temps.
Une vérification séparée (`benchmark/hermitian_reference_probe.jl`) teste les
parties imaginaires initiales arbitraires. Le contrat de référence impose de
les ignorer et de produire une diagonale réelle :
[Netlib CHERK](https://netlib.org/blas/cherk.f).
Dans l'environnement Accelerate local, les seize cas HERK/HER2K N/C U/L des
deux types complexes conservent une partie imaginaire non nulle ; les seize
cas OpenBLAS passent. Le code natif est vérifié dans les deux processus et
ses tests couvrent aussi une partie imaginaire NaN. Ces observations concernent
les bibliothèques chargées ici, sans conclusion sur d'autres versions d'Accelerate.
Les CSV `hermitian-contract-*.csv` conservent tous les résultats, échecs compris.

`benchmark/inspect_level3.jl` conserve LLVM et assembleur des feuilles
TRMM/TRSM pour les quatre types ; ces feuilles restent scalaires ou auto-vectorisées
selon le compilateur. Les micro-noyaux GEMM explicites inchangés disposent
de leur inspection séparée. Aucun résultat assembleur x86 n'est fourni.

## Campagne BLAS 1 v0.4

`benchmark/level1.jl` mesure ROTM réel Float32/Float64, DSDOT/SDSDOT sur entrées
Float32, puis ROTMG réel. Tailles vectorielles 32/4096/1 000 000 et strides
1/3/-2, flags ROTM -1/0/1, un thread. Les références passent par les symboles
BLAS explicites absents de l'interface Julia standard ; ils sont confinés au
fichier `test/level1_references.jl`. La bibliothèque réellement responsable de
chaque symbole est enregistrée via libblastrampoline dans les environnements.
Les campagnes OpenBLAS et Accelerate s'exécutent séparément.

Les temps proviennent de onze échantillons au maximum, evals=1, 0,15 s par
jeu. Le premier appel et sa compilation sont conservés séparément ; sorties
ROTM restaurées hors mesure. Une cellule de résultat rend les calculs lus
observables. ROTMG mesure des lots de 256 entrées variées et conserve chaque
sortie dans un tableau préalloué : les durées CSV sont celles du lot, à diviser
par 256 pour un coût moyen par transformation. Cela évite les constantes
repliées par le compilateur et les durées sous la résolution de l'horloge.
Les entrées des deux variantes et des deux backends sont identiques par jeu.
Les éventuelles références ROTMG invalides sont exclues de la mesure et
signalées dans les journaux de campagne.

Les contrôles hors chronométrage utilisent produits matriciels indépendants,
sommes BigFloat à 256 bits et invariants de génération à haute précision.
Les deux opérandes Float32 sont convertis avant multiplication pour DSDOT/SDSDOT.
Le champ type des lignes de produits étendus désigne le type de résultat.
Les allocations incluent le wrapper de référence retournant les paramètres ;
les buffers des lots sont préalloués pour les deux méthodes.

`reports/baseline-level1-generic/` garde la passe avant paquets ROTM explicites,
sa source exacte reconstruite et vérifiée contre le hash mesuré, et le script
utilisé. Le reste de l'implémentation est inchangé entre les deux passes ; les
vues à strides 3/-2 gardent la même voie générique. Les mesures sont locales et
courtes, sans garantie de supériorité sur d'autres machines.

## Campagne compact/inter-tuiles v0.5

`benchmark/compact.jl` compare la source v0.4 exacte à la nouvelle implémentation
et aux BLAS externes pour les combinaisons de types standard. Tailles 128/1024,
plus 16 384 pour les bandes : largeur 8, N/C quand pertinent, un thread.
Les extensions symétriques complexes du paquet n'ont pas de référence BLAS
portable ; elles restent comparées à une matrice dense indépendante aux tailles
128/1024. À 16 384, la référence externe évite de construire une matrice dense
et la baseline quadratique v0.4 n'est pas exécutée. Les sorties et stockages
mis à jour sont restaurés hors chronométrage ; les conversions d'options sont
également préparées hors mesure. Sept échantillons au plus, 0,12 s, evals=1.

`benchmark/tile_parallel.jl` s'exécute dans des processus successifs avec
`--threads=1`, `2`, `4`, `8`. Il compare v0.4, natif et OpenBLAS aux tailles
512/1024, K=3N/4, bloc triangulaire 128. SYMM/HEMM sont conservés comme points
de contrôle utilisant le GEMM inchangé. GEMMT n'a pas de ligne BLAS externe
portable dans cette campagne. Accelerate est mesuré séparément à un thread.
Les panneaux sont préalloués dans les deux variantes Julia, les contextes et
tampons diagonaux par tâche sont inclus dans les temps/allocations. Cinq
échantillons au plus, 0,15 s, evals=1 ; contrôle numérique avant chronométrage.

Les processus ne se chevauchent pas et ne s'exécutent pas en même temps que les
tests/compilations lancés par cet agent. Environnements, temps minima/médians,
allocations, premier appel, compilation et erreurs sont conservés en CSV.
Les durées restent des observations locales courtes. Les scripts d'inspection
conservent LLVM/assembleur des AXPY compacts réels. Les sources Rust ont été
exécutées pour 48 groupes de fixtures compactes ; elles sont confrontées aux
contrôles indépendants, et ne servent pas de backend au paquet.

Une comparaison Rust supplémentaire mesure GBMV Float64 N, largeur 8,
ordres 4096/16 384/65 536 sur des entrées déterministes identiques, α=0,7,
β=-0,2. Le Rust est compilé en release, LTO thin et target-cpu=native.
Neuf échantillons sont pris après warmup, restauration de y hors mesure ;
le côté Julia utilise aussi au plus neuf échantillons. Les vecteurs complets
Rust sont exportés et vérifiés contre un BLAS indépendant avant comparaison.
Les formats physiques restent ceux de chaque API (Mat Rust et tableau Julia).
Cette comparaison ne concerne qu'un noyau Float64 monothread, sans extrapolation
au reste du portage. Voir `benchmark/rust_compact_benchmark.rs`,
`benchmark/compare_compact_rust.jl`, `tools/run_rust_compact_benchmark.py`
et `reports/rust-compact-workload/`.
