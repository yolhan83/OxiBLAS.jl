# Ligne « Graphes » — complétion des livrables en phase A

**0.6.0-dev : sources, tests, benchmarks et diagnostics préparés, non exécutés.**
La ligne utilisateur « variantes AMD/COLAMD, partitionnement multilevel et
appariement pondéré » est écrite pour les fonctionnalités pertinentes de
l'amont Rust `8d82bb6bf1df133aa90525aa9f478ebe41514f9b`. Ce constat porte sur
les livrables ; ni le chargement, ni la correction, ni les performances ne sont
attestés. Aucun code Julia/Rust du projet, test, benchmark, compilateur,
précompilation, autotuning ou générateur LLVM n'a été exécuté pendant ce lot.

Le [premier lot graphes](LOT_GRAPHES_CREUX.md) reste la description des bases.
Ce complément ferme son audit pour cette ligne et remplace ses mentions
« raffinement seulement glouton », « variantes encore à auditer » et
« adaptateur Rust absent ». La mission globale OxiBLAS reste partielle.

## Changements de cette passe

| Fichier | Changement |
| --- | --- |
| src/graph_refinement.jl | refine_partition ; passes FM pondérées avec verrouillage, gains incrémentaux, deux tas, exploration puis rollback au meilleur préfixe |
| src/graph_matching.jl | Hopcroft–Karp avec DFS itérative pour l'incidence rectangulaire ; bipartite_graph_matching pour le graphe carré non orienté |
| src/graph_structure.jl | Mesures de largeur/profil selon la convention graphe Rust ; dissection par niveaux avec composantes déconnectées, feuilles minimum-degree exactes, profondeur et arbre inspectable |
| src/multilevel_partition.jl | Option refinement=:fm dans partition et dissection, défaut :greedy conservé ; maxdepth et profondeur des nœuds ; rejet du sous-débordement des poids d'arêtes normalisés |
| src/sparse_operations.jl | API bipartite_matching conservée, moteur Hopcroft–Karp ; nested_dissection raccordée au moteur complet ; postordre symbolique itératif et option uplo explicite |
| src/sparse_ordering_advanced.jl | Mesures graph_bandwidth/graph_profile/graph_average_bandwidth ajoutées sans changer les champs historiques de convention matrice |
| src/OxiBLAS.jl ; test/runtests.jl | Exports/includes des trois sources et trois tests |
| test/graph_matching.jl | Références exhaustives rectangulaires/graphes, chemin alternant de 4096 sommets, formats, entrées intactes, GC et concurrence |
| test/graph_refinement.jl | Oracle dense sans tas ni gains incrémentaux ; rollback explicite, poids, fractions, erreurs et rattachement aux deux algorithmes multiniveaux |
| test/graph_structure.jl | Distances et composantes indépendantes, conventions de profils, séparateurs/depth, triangles ignorés, chaîne de postordre de 4096 sommets |
| test/run_graphs.jl | Lanceur des six fichiers de tests graphes, dont les trois existants |
| benchmark/sparse_graphs.jl | Mesures natives étendues, gardes avant timing, hashes des sources/données/résultats et empreintes CSR communes au Rust |
| benchmark/rust_graph_reference/ | Adaptateur Cargo séparé vers le checkout amont local, données Float64 identiques pour les cas communs, gardes puis temps chauds |
| benchmark/run_graph_campaign.sh | Phases séparées correction, performance 1/2/4/8 threads, référence Rust hors ligne et inspection ; logs/statuts/timestamps/hashes |
| benchmark/compare_graph_inputs.py | Comparaison préparée des empreintes d'entrée, sans prétendre certifier résultats ou performances |
| benchmark/inspect_graphs.jl | Signatures réelles prévues pour Hopcroft–Karp, passes FM et éléments-colonnes ; sorties .ll/.s |
| README.md ; docs/STATUS.md ; docs/REPRISE_SUPERVISEE.md ; docs/COVERAGE_SUPERVISED.csv | État et correspondance actualisés |

## Audit complet de la surface fonctionnelle pertinente

Chemins relatifs à `crates/oxiblas-sparse/src/`. Les types de résultats/configuration
Rust sont remplacés par les mots-clés, tableaux et NamedTuple Julia appropriés.
Les indices Julia sont 1-based ; 0 indique une absence de partenaire/parent.
Les variantes numériques réelles/complexes restent entièrement Julia dans src/.

| Amont | Julia | Statut et adaptation |
| --- | --- | --- |
| graph/functions.rs::connected_components ; connected_components_csc | connected_components sur NativeSparse | Écrit existant ; labels/components/count, tailles par length.(components) ; motif symétrisé |
| build_symmetric_adjacency ; build_symmetric_adjacency_csc | _sparse_graph | Adaptation interne commune, hors diagonale et zéros numériques |
| bandwidth_profile ; bandwidth_profile_csc | graph_bandwidth_profile | Écrit dans cette passe ; demi-largeur max, profil de portée de ligne et moyenne ; convention historique bandwidth_profile conservée séparément |
| level_sets | level_sets | Écrit existant ; retourne directement les listes par niveau ; distances dérivables de leurs indices, sommets inaccessibles absents |
| pseudo_peripheral_vertex | pseudo_peripheral_vertex | Écrit existant ; recherche dans la composante initiale ; vide → 0 |
| is_structurally_symmetric ; degree_sequence | mêmes APIs | Écrit existant ; symétrie des arcs présents et degrés non orientés |
| is_bipartite | bipartite_coloring | Écrit existant ; couleurs 1/2, gauche/droite ; self-loops ignorées comme l'adjacence amont |
| bipartite_matching ; hopcroft_karp ; dfs_augment | bipartite_graph_matching ; _hopcroft_karp | Écrit ; parcours augmentants par couches et pile explicite, sans récursion proportionnelle au chemin |
| Besoin Julia incidence lignes/colonnes | bipartite_matching | API existante conservée, moteur remplacé par Hopcroft–Karp ; rectangularité conservée ; appariement maximum, partenaires en cas d'égalité potentiellement différents |
| weighted_bipartite_matching | weighted_graph_matching | Écrit précédent ; coloration puis affectation à chemins augmentants/coûts réduits, poids max des directions transformées |
| Besoin Julia affectation rectangulaire | weighted_bipartite_matching | Écrit précédent ; objectifs cardinalité puis poids, ou poids seul ; transformations value/abs/logabs ; négatifs, complexes et débordement du total documentés |
| partition_graph_kway ; partition_graph_bisect | multilevel_partition(A,k) ; multilevel_partition(A,2) | Adaptation native multiniveau plus générale ; pas de reproduction des labels heuristiques Rust |
| graph/types.rs : résultats de composantes, niveaux, bipartitions, matching, poids et partitions | NamedTuple retournés et tableaux natifs | Pas de duplication des classes Rust ; couverts par les invariants des tests et conventions ci-dessous |
| linalg/ordering/functions.rs::approximate_minimum_degree | minimum_degree_ordering | Écrit existant ; degré exact et remplissage clique, conforme à la nature documentée de l'amont, pas au véritable AMD SuiteSparse |
| multiple_minimum_degree | multiple_minimum_degree | Écrit précédent ; lots de degrés minimaux, candidats indépendants d'abord ; remplissage exact |
| colamd ; colamd_aggressive | column_minimum_degree(;dense_threshold,absorb,tie) | Écrit précédent ; degrés exacts sur éléments-cliques de lignes, filtre dense/absorption ; l'amont n'est pas un vrai COLAMD non plus |
| reverse_cuthill_mckee | reverse_cuthill_mckee | Écrit existant ; composantes et BFS trié par degrés |
| nested_dissection ; nested_dissection_recursive ; local_amd | nested_dissection(;leafsize,maxdepth,balance_tolerance,return_tree) | Complété : feuilles en minimum-degree exact, séparateurs par niveaux, composants séparés et profondeur plafonnée |
| find_vertex_separator ; find_pseudo_peripheral ; compute_level_sets ; widen_separator | _level_nested_dissection ; _graph_levels | Adaptation : choix de niveau évaluant déséquilibre/taille, plutôt qu'élargissement forcé du séparateur ; arbre et déséquilibre inspectables, aucune garantie d'optimum |
| symmetric_permutation ; inverse_permutation | copy(p) ; invperm(p) natifs Julia | Pas de wrappers publics redondants ; sparse_permute valide les permutations avant application |
| estimate_fill_in ; fill_in_ratio | ordering_metrics : factor_nnz, fill, fill_ratio | Remplissage symbolique explicite ; définition du ratio native conservée et distinguée ci-dessous |
| compute_ordering ; OrderingAlgorithm::name | sparse_ordering(A,symbol) ; symboles Julia | Toutes les familles pertinentes routées, et permutations explicites vérifiées ; pas d'enum Rust artificielle |
| compare_orderings ; select_best_ordering ; OrderingComparison | compare_orderings ; select_sparse_ordering | Comparaison des facteurs symboliques, sélection par nnz puis largeur/profil ; pas de mesure de temps implicite |
| bandwidth_with_ordering ; profile_with_ordering | ordering_metrics : graph_bandwidth et profile | Demi-largeur graphe ; profil inférieur adapté (différence du +1 amont indiquée ci-dessous) ; graph_profile relève du module graph |
| elimination_tree ; EliminationTree::new,parent,children,len,is_empty,roots,depth | symbolic_cholesky : parent/children/permutation et size(A), findall(iszero,parent), chaîne des parents | Forêt accessible sans type dédié ; racine parent=0 ; ordre des parents croissant après permutation |
| postorder_tree | symbolic_cholesky.postorder | Parcours itératif complété, sans profondeur de pile proportionnelle à la forêt |
| column_counts ; compute_l_pattern ; SymbolicCholesky::new,n,nnz,l_col_ptrs,l_row_indices,parent,post_order | symbolic_cholesky : columns, nnz, parent, postorder ; length.(columns) et préfixes pour le stockage CSC | Variante exacte par remplissage explicite, pas l'algorithme GNP near-linear Rust ; uplo=:L/:U disponible, défaut full-pattern historique conservé |
| NestedDissectionConfig ; son Default | leafsize, maxdepth, balance_tolerance | Mots-clés validés ; valeurs par défaut Julia propres, pas identité des heuristiques |
| linalg/ordering/multilevel.rs::PartitionConfig ; MultilevelPartitioner::new,partition | multilevel_partition(;coarse_size,refine_passes,imbalance,...) | API fonctionnelle, poids de sommets/arêtes pris en charge ; vide accepté avec k=1 et k>n rejeté explicitement |
| recursive_bisect ; bisect_recursive ; bisect | récursion de multilevel_partition ; _multilevel_bisect | Nombre de sommets suffisant pour chaque partie finale, bisections proportionnelles au nombre de parties |
| heavy_edge_matching ; coarsen ; CoarseGraph ; Matching | _coarsen_partition ; _PartitionGraph | Paires lourdes sous contrainte de poids, contraction et somme des arêtes parallèles ; tables de projection natives |
| initial_partition | _initial_bisection ; _partition_bfs | Deux départs BFS, choix selon charge/coupe, composantes déconnectées |
| refine_partition ; uncoarsen_and_refine | raffinement :greedy existant ; option :fm et refine_partition ajouté | FM à verrouillage/meilleur préfixe dépasse la seule recherche de gain positif amont ; aucun label METIS-équivalent |
| multilevel_nested_dissection ; récursion et sous-graphes privés | multilevel_nested_dissection(;refinement,maxdepth,return_tree) ; _partition_induced ; _separator_cover | Contraction/bisection, couverture gloutonne des arêtes coupées, récursion avec profondeur et minimum-degree local |
| OrderingError, conversions/Display d'erreurs | ArgumentError, DimensionMismatch, BoundsError natifs | Erreurs explicites, pas de copie de la hiérarchie Rust |

## Contrats et limites restant explicites

Les noms AMD et COLAMD de l'amont ne désignent pas les algorithmes SuiteSparse
correspondants. Le port ne remplace pas les implémentations exactes déjà écrites
par des alias trompeurs. Les variantes pertinentes de l'amont sont rapprochées
ci-dessus ; développer le vrai quotient-graph AMD/SuiteSparse COLAMD avec leurs
heuristiques optimisées constitue un travail supplémentaire, pas une fonction
Rust pertinente encore cachée derrière ces noms. Aucune performance équivalente
à SuiteSparse/METIS n'est annoncée.

Les algorithmes d'ordonnancement/graphes restent séquentiels. Les appels sont
indépendants et ne partagent pas de workspace mutable ; les tests préparent GC
et appels simultanés. Les campagnes 1/2/4/8 threads servent aussi à contrôler
l'environnement et les appels concurrents ; aucun speedup intrinsèque de ces
algorithmes séquentiels n'est revendiqué. La recherche par degrés exacts et le
remplissage symbolique peuvent être coûteux en temps/mémoire. La construction
d'un grand graphe symbolique n'est pas un algorithme streaming/distribué.
Aucune API graphes distribués n'est présente dans les modules audités.

L'appariement carré de graphe et l'incidence rectangulaire ont des sens distincts.
Pour un graphe, saturated_smaller_side indique la saturation de la plus petite
couleur et covers_all_vertices indique si tous les sommets sont couverts. Cela
évite de reproduire le champ amont is_perfect, dont l'implémentation teste la
saturation de la plus petite couleur malgré une documentation « tous sommets ».
Les objectifs et limites de normalisation Float64 de l'appariement pondéré sont
inchangés et décrits dans LOT_GRAPHES_CREUX.md ; pas de certificat exact des poids.

Le raffinement :greedy reste le défaut des APIs multiniveaux, préservant leur
politique antérieure. :fm est optionnel ; refine_partition utilise :fm par défaut.
Chaque passe verrouille un sommet après son déplacement, actualise les gains
voisins et conserve seulement le meilleur préfixe selon violation d'équilibre
puis coupe. Une exploration temporaire peut dépasser l'intervalle de charge d'un
sommet maximal ; les tailles minimales des deux parties restent imposées. La
coupe et la charge sont recomputées après rollback, avec retour à l'état initial
si l'amélioration n'est pas retrouvée en précision de travail. Ce n'est ni une
recherche globale ni une implémentation de toutes les variantes KL/FM. Les tas
utilisent des versions paresseuses ; la recherche d'un déplacement admissible
peut rescanner des candidats bloqués par les poids, donc le pire cas n'est pas
annoncé linéaire. Le scratch des candidats bloqués est réutilisé dans la passe.

Les séparateurs par niveaux n'ont pas d'arêtes entre leurs deux côtés. Si le
sous-graphe est déconnecté, ses composants sont séparés sans supprimer de sommet.
Les feuilles utilisent un minimum-degree local exact. Les séparateurs multiniveaux
restent des couvertures gloutonnes, pas nécessairement minimales. maxdepth borne
la récursion ; les feuilles éventuellement grandes restent traitées explicitement.
Les poids indivisibles, tolérances impossibles ou heuristiques peuvent empêcher
l'équilibre : les résultats exposent le critère obtenu sans prétendre le garantir.
Les poids d'arêtes qui deviennent nuls par normalisation sont désormais rejetés
au lieu d'être silencieusement perdus ; la même politique existait pour les sommets.

Les conventions de mesures sont désormais explicites : bandwidth_profile garde
sa largeur matricielle lower+upper+1 et son profil inférieur ;
graph_bandwidth_profile fournit max(lower,upper), somme des portées de lignes et
moyenne, comme le module graph Rust. ordering_metrics expose les deux familles.
Son fill_ratio = factor_nnz / pattern_nnz, où pattern_nnz inclut la diagonale et
le triangle du motif symétrisé. L'amont fill_in_ratio soustrait le nnz stocké de
l'entrée avant division : cette convention dépend du stockage et peut différer.
Elle n'est pas présentée comme le même chiffre. Le profil du module ordering
Rust ajoute aussi 1 par ligne occupée ; son code soustrait des indices non signés
sans borner les lignes strictement supérieures. Le profil inférieur Julia garde
la diagonale implicite et évite cette soustraction non définie pour certains
motifs. Il n'est donc pas annoncé bit-à-bit identique au profil ordering Rust. Les zéros numériques sont ignorés
par le port ; l'amont compte parfois des zéros explicitement stockés comme arcs.

## Tests et campagnes préparés

Les six fichiers du lanceur ciblé couvrent les API de cette ligne, avec
références exhaustives/denses et invariants indépendants : quatre types natifs,
vides/rectangles/isolés, triangles uniques, signes/phases/poids logarithmiques,
appels simultanés, GC, permutations, remplissage, coupes/charges/séparateurs,
BFS/composantes, rollback des passes FM, échelles et options invalides. Les
longs chemins HK et postordres ont 4096 sommets. Les erreurs et cas d'équilibre
impossible sont préparés. Tous ces tests sont intégrés, **non exécutés**.

Le benchmark natif mesure chaque famille de la ligne avec gardes avant timing,
premier appel/compilation, min/médiane chauds, allocations et stockage conservé.
Le Rust préparé mesure en processus séparé un sous-ensemble Float64 pertinent
au même checkout : matching, degrés/colonnes/dissection, mesures et partitionnement.
Les fichiers `cases-fnv64.csv` et la colonne `csr_fnv64` Rust utilisent exactement
le même encodage des CSR (m/n, indices zéro, bits Float64 little-endian). Le script
de comparaison refuse une entrée Rust absente du manifeste Julia. FNV sert à
la correspondance des cas, pas à une garantie cryptographique ; les sources,
matrices natives, logs et résultats disposent aussi d'empreintes SHA-256.
Les heuristiques peuvent produire des permutations/coupes différentes : les
qualités sont rapportées, jamais supposées identiques au Rust. Les premiers
appels Julia incluent la compilation, les appels Rust compilés ne l'incluent pas.
Les benchmarks d'ordonnancement Rust/natif partagent des entrées, sans prétendre
avoir le même algorithme optimisé ou le même temps. Pas de référence BLAS/LAPACK
pour ces algorithmes combinatoires, et pas de provider SuiteSparse/METIS ajouté.

Commandes préparées depuis le paquet, **à ne pas exécuter maintenant** :

```sh
bash benchmark/run_graph_campaign.sh /tmp/oxiblas-graphs-correction-NEW correctness
bash benchmark/run_graph_campaign.sh /tmp/oxiblas-graphs-performance-NEW performance
bash benchmark/run_graph_campaign.sh /tmp/oxiblas-graphs-rust-NEW rust
python3 benchmark/compare_graph_inputs.py /tmp/oxiblas-graphs-performance-NEW/graphs-1/cases-fnv64.csv /tmp/oxiblas-graphs-rust-NEW/rust.csv
bash benchmark/run_graph_campaign.sh /tmp/oxiblas-graphs-ir-NEW inspect
```

Les phases sont séparées et ne s'enchaînent pas automatiquement. Les dossiers
doivent être neufs ; un échec conserve les logs complets, le code de retour,
les timestamps et hashes puis arrête la phase. Rust est construit --offline,
sans téléchargement implicite ; une dépendance absente produira un échec explicite.
Le coût exact reste inconnu ; les 7 échantillons / 0,15 s cible par cas ne bornent
pas une compilation ou un appel individuel. La reprise automatique reste du
ressort du lanceur global à terminer ; les résultats partiels restent intacts.

Ce n'est pas une demande anticipée d'exécution. Selon le protocole, la première
autorisation sera demandée lorsque les livrables globaux seront prêts. Les
archives/mesures v0.5 et manifests historiques ne sont pas remplacés par ce lot.

## Contrôles statiques effectués

Contrôle textuel des 62 includes source, 38 includes du lanceur de tests général
et 6 includes du lanceur graphes : fichiers présents, sans doublons. Registre
partiel à 58 entrées : sept colonnes, chemins de tests/scripts présents et
statut not executed conservé. Vérification des liens Markdown, du chemin de
dépendance Rust et de la syntaxe Python du comparateur via AST, sans exécuter
son corps. Syntaxe shell contrôlée par bash -n. Recherche textuelle de délégation
numérique externe dans les trois nouvelles sources : aucun appel relevé.
Ces contrôles ne remplacent pas un parseur Julia, un chargement ou les tests.

L'archive historique v0.5 conserve son SHA-256 :
`8978f5348aad68fd19fe86f7da68b9a6329e32ebca67825f05ae98fc48d44df6`.
Aucun résultat de cette archive n'est attribué à la ligne Graphes actuelle.
