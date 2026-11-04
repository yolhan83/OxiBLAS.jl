# Graphes creux — continuation du 2 novembre 2026

> Suite de cette passe : [complétion de la ligne Graphes](LOT_GRAPHES_COMPLETION.md).
> Elle actualise les limites FM/dissection/audit et les scripts de référence Rust
> mentionnés ci-dessous ; tout reste écrit et non exécuté.

**0.6.0-dev : implémentations, tests et benchmark écrits, non exécutés.**
Aucun chargement, test, chronométrage ou diagnostic LLVM/assembleur n’a été
lancé. Aucune performance, qualité d’ordonnancement ou validation numérique
de ce lot n’est attestée. La phase A supervisée continue.

## Fichiers modifiés

| Fichier | Contenu |
| --- | --- |
| src/weighted_matching.jl | Appariement pondéré sparse, coloration, adaptation graphe, degrés, symétrie structurelle, sommet pseudo-périphérique |
| src/sparse_ordering_advanced.jl | Multiple minimum-degree ; minimum-degree des colonnes par éléments ; sélection, métriques et comparaison |
| src/multilevel_partition.jl | Contraction heavy-edge, bisection, raffinement, partitions k-way et dissection avec séparateurs inspectables |
| src/sparse_operations.jl | Raccordement des nouveaux ordonnancements/permutations explicites à symbolic_cholesky ; somme vide explicite pour nnz |
| src/OxiBLAS.jl ; test/runtests.jl | Exports/includes et intégration des tests |
| test/weighted_matching.jl ; test/sparse_ordering_advanced.jl ; test/multilevel_partition.jl | Références exhaustives/structurelles et invariants |
| benchmark/sparse_graphs.jl | Campagne ciblée préparée avec références analytiques/invariants |
| README.md ; docs/STATUS.md ; docs/REPRISE_SUPERVISEE.md ; docs/COVERAGE_SUPERVISED.csv | Suivi et limites |

## Correspondance fonctionnelle avec Rust

Commit amont `8d82bb6bf1df133aa90525aa9f478ebe41514f9b`, sous oxiblas-sparse/src.

| Fonction amont | API Julia | Adaptation |
| --- | --- | --- |
| graph/functions.rs::weighted_bipartite_matching | weighted_graph_matching | Graphe carré, coloration puis affectation sparse ; retourne nothing si non biparti |
| Besoin affectation lignes/colonnes | weighted_bipartite_matching | Matrice rectangulaire, interprétation distincte du graphe carré |
| graph/functions.rs::is_bipartite | bipartite_coloring | Couleurs 1/2 ; diagonale ignorée ; graphe non orienté |
| degree_sequence / is_structurally_symmetric / pseudo_peripheral_vertex | mêmes noms Julia | Réutilisation de la structure native ; composante du sommet initial pour le périphérique |
| ordering/functions.rs::approximate_minimum_degree | minimum_degree_ordering existant | L’amont indique que sa fonction est du degré exact, pas le vrai AMD |
| ordering/functions.rs::multiple_minimum_degree | multiple_minimum_degree | Lot de degrés minimaux égaux, indépendants d’abord, remplissage exact |
| ordering/functions.rs::colamd / colamd_aggressive | column_minimum_degree | Degrés exacts sur éléments de lignes, filtrage des lignes denses ; pas de prétention au vrai COLAMD |
| compute_ordering / compare_orderings / select_best_ordering / estimate_fill_in / fill_in_ratio | sparse_ordering ; compare_orderings ; select_sparse_ordering ; ordering_metrics | Symboles Julia et remplissage symbolique calculé explicitement |
| ordering/multilevel.rs::MultilevelPartitioner::partition ; graph/functions.rs::partition_graph_kway/bisect | multilevel_partition | Contraction/bisection/uncoarsening ; méthode adaptée, pas reproduction exacte des partitions amont |
| ordering/multilevel.rs::multilevel_nested_dissection | multilevel_nested_dissection | Séparateurs par couverture gloutonne des arêtes coupées ; feuilles minimum-degree |

## Appariement pondéré

Le solveur lignes/colonnes emploie des chemins augmentants à coûts réduits,
potentiels et tas Dijkstra. Les arcs résiduels restent sparse : pas de matrice
dense m×n de coûts. Les coûts sont normalisés et calculés en Float64 pour les
quatre types natifs. Les poids sont signés réels, modules ou log-modules.
Les zéros numériques du problème d’origine ne sont pas des arêtes ; un poids
transformé nul (log(abs)=0) est conservé comme arête.

objective=:cardinality maximise d’abord le nombre de paires puis le poids.
objective=:weight permet de laisser des sommets non appariés, notamment pour
les poids négatifs. Les indices 0 indiquent l’absence de partenaire.
L’optimalité est annoncée en précision de travail, sans certificat exact.
Des écarts de poids extrêmes peuvent être perdus par normalisation/arrondi.
Un total non représentable est signalé par weight_overflow, sans prétendre
que ce total flottant est exploitable.

L’adaptateur graphe colore le motif symétrisé hors diagonale puis prend le
maximum des deux poids directionnels transformés pour une arête non orientée.
Cette politique prend en charge les triangles uniques et motifs asymétriques,
mais n’est pas une équivalence bit à bit au choix directionnel Rust. Les
diagonales n’interviennent pas dans cette interprétation graphe.

## Ordonnancements : terminologie et mémoire

L’amont documente lui-même que ses fonctions AMD/COLAMD ne sont pas les
algorithmes SuiteSparse correspondants. Le port emploie donc des noms décrivant
les opérations réellement présentes. Le minimum-degree exact existant reste
conservé ; il n’est pas réimplémenté sous un faux nom AMD.

L’ordonnancement des colonnes représente chaque ligne comme un élément-clique,
calcule les unions de voisins actifs et remplace les éléments absorbés par
l’élément de remplissage. Il ne forme ni A'*A numérique, ni le graphe initial
d’intersection de toutes les paires de colonnes. L’absorption d’éléments inclus
est optionnelle ; les lignes trop denses peuvent être ignorées. Les degrés
sont néanmoins recalculés exactement : coût et remplissage restent potentiellement
superlinéaires. Pas de supercolonnes, degrés approchés paresseux ou équivalence
à l’aggressive absorption du vrai COLAMD.

Les permutations satisfont p[new]=old. symbolic_cholesky accepte les nouveaux
symboles ainsi qu’une permutation explicite vérifiée ; Cholesky multifrontale
en bénéficie par son analyse symbolique existante. ordering_metrics calcule
vraiment le remplissage, ce qui peut coûter cher. La sélection compare nnz du
facteur puis largeur de bande/profil ; elle ne garantit pas un optimum global.

## Partitionnement et séparateurs

La structure non orientée est symétrisée, sans diagonale ni zéros numériques.
Les arêtes sont unitaires par défaut ou pondérées par maximum des modules
directionnels. Des poids de sommets positifs sont acceptés. Les poids sont
normalisés pour la contraction puis remis à l’échelle dans les diagnostics.

Chaque bisection contracte des paires liées par des arêtes fortes, effectue une
bisection BFS sur le graphe grossier puis projette et raffine les labels. Le
raffinement utilise des déplacements gloutons et échanges de paires de frontière,
avec priorité à la réduction de déséquilibre. Il ne reproduit pas un KL/FM
complet à meilleurs préfixes et n’est pas annoncé METIS-équivalent.

La récursion k-way impose des partitions non vides si n>0 et k<=n. Pour le
graphe vide, seul k=1 est accepté et donne une partie vide. Les indivisibilités
de poids peuvent rendre l’équilibre irréalisable. balanced et achieved_imbalance
rapportent le critère global obtenu ; requested_imbalance n’est pas une garantie.
Pas de recherche multistart ni de parallélisme des récursions à ce stade.

La dissection couvre toutes les arêtes entre les deux côtés par un ensemble
de sommets séparateurs, puis ordonne les deux côtés avant ces séparateurs.
L’arbre retourné optionnellement permet de vérifier l’absence d’arêtes gauche/droite.
La couverture est gloutonne, pas nécessairement minimale. Si la séparation
dégénère, le sous-graphe passe en feuille de minimum-degree.

## Tests et benchmark préparés

Les trois fichiers de tests sont intégrés, **non exécutés** : appariements
comparés à une énumération exhaustive indépendante, objectifs/poids négatifs,
rectangles/vides, phases complexes, log-poids nuls, débordement du total,
coloration et invariants ; ordonnancement des colonnes comparé à un graphe
dense de référence ; remplissage symbolique comparé à Cholesky indépendante ;
partitions, charges et coupes recomputées, séparateurs vérifiés, erreurs,
répétabilité, absence de mutation, GC et concurrence.

Le benchmark exige une destination neuve et garde les historiques. Il vérifie
avant timing les affectations à optimum analytique, coupes, permutations et
remplissage d’étoiles. Il distingue premier appel/compilation, minimum/médiane
chauds, allocations et stockage conservé. Les algorithmes de graphes actuels
sont séquentiels ; aucune scalabilité 1/2/4/8 threads n’est attestée. Les
comparaisons temporelles externes Rust/SuiteSparse/METIS, grands motifs, hashes
et lanceur reprenable restent à compléter dans la campagne globale.

Le registre reste partiel. Vrais AMD/COLAMD optimisés, KL/FM avancé, graphes
distribués et les autres familles du portage restent à auditer ; les archives
et anciens manifests ne représentent toujours pas les lots supervisés.
