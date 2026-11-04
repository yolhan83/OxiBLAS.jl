# Continuation du 2 novembre 2026 — écrit, non exécuté

Actualisation : [XTrace optimisé et oracle LOO conservé](LOT_ARNOLDI_XTRACE.md), code écrit non exécuté. Les limites XTrace ci-dessous sont historiques.

Version 0.6.0-dev. Le lot est une continuation de la phase A de
`PROTOCOLE_SUPERVISE.md`. Aucun code Julia du projet, test, benchmark,
autotuning ou diagnostic LLVM/assembleur n’a été exécuté pour ces changements.
La relecture des sources n’atteste ni compilation, ni correction numérique,
ni performance, ni portabilité. La mission complète reste en cours.

## Estimations stochastiques

Le rapprochement utilise `oxiblas-sparse/src/stochastic.rs` au commit amont
`8d82bb6bf1df133aa90525aa9f478ebe41514f9b`.

| Amont | API Julia écrite | Limite ou adaptation |
| --- | --- | --- |
| StochasticConfig / ProbeType | StochasticConfig | Symboles Julia ; confiance informative |
| trace_hutchinson | trace_hutchinson | Réel/complexe, erreur scalaire quadratique complexe |
| trace_hutch_plus_plus | trace_hutch_plus_plus | Sketch et correction indépendants ; budget de produits A*x |
| trace_xtrace | trace_xtrace | Contrôle leave-one-out reconstruit ; pas de formule inverse triangulaire optimisée |
| diagonal | stochastic_diagonal | Conjugaison des sondes complexes |
| frobenius_norm | stochastic_frobenius | Carré non biaisé, racine généralement biaisée ; matrices rectangulaires |
| log_det | stochastic_logdet | Lanczos réorthogonalisé, solveur tridiagonal natif ; SPD requise |

Les sondes sont réelles ±1 ou complexes à quatre phases, gaussiennes réelles
ou complexes normalisées, ou sphériques de covariance identité. Le générateur
SplitMix64 est local à l’appel ; aucun RNG global et aucun buffer mutable partagé.
Les vecteurs temporaires sont réutilisés dans les boucles. CSC/COO est converti
en CSR une seule fois par appel ; les SpMV sparse internes sont séquentiels.
Les estimations ne sont pas des algorithmes parallèles par sondes à ce stade.

`std_error` porte uniquement sur l’échantillonnage indépendant : Inf pour une
seule sonde, NaN pour les échantillons leave-one-out corrélés. Aucun intervalle
de confiance certifié n’est déduit de `confidence`. Pour Hutch++,
`n_probes_used` compte les échantillons de correction ; `matvecs` compte aussi
le sketch et les produits de la base. Pour la norme, l’erreur est fournie sur
le carré (`squared_std_error`). Pour le log-déterminant, l’erreur exclut le biais
de troncature de Lanczos ; des Ritz positifs ne certifient pas la positivité
globale. Le contrôle exact de symétrie hermitienne peut être désactivé uniquement
sur assertion de l’appelant. Pas de densification du grand opérateur sparse.

La fonction Rust appelée XTrace forme une base de Ω et combine Q'*(AΩ).
La variante Julia utilise plutôt une base de A*Ω privée d’une colonne et un
contrôle de trace indépendant de cette colonne. Aucune équivalence bit à bit
ou de budget d’opérations n’est annoncée entre ces deux variantes. L’optimisation
XTrace publiée reste une lacune explicite ; le coût de reconstruction peut être
élevé. Les tests analytiques écrits visent les invariants mathématiques.

## Plans et workspaces einsum

Les APIs de contraction pertinentes de `oxiblas-blas/src/tensor.rs` et
`ndtensor.rs` continuent d’utiliser des tableaux/vues Julia. Les conteneurs
NdTensor/Tensor3 ne sont pas reproduits. Les ajouts de ce lot concernent la
planification et l’exécution répétée, sans prétendre couvrir toutes les
variantes tensorielles de l’amont.

`plan_einsum(...; memory_limit=bytes)` conserve le choix glouton ou gauche,
filtre les paires selon leur pic de données intermédiaires et vérifie que le
résultat tient. Le pic compte les intermédiaires vivants plus deux sorties de
paire pour anticiper une permutation. Si aucun chemin retenu ne tient, le plan
contracte directement tous les opérandes, avec un seul résultat temporaire.
Cette solution peut augmenter fortement le travail. Ni recherche globalement
optimale, ni découpage en tranches, ni exécution hors mémoire/distribuée.

La limite exclut les entrées/sorties fournies par l’appelant, les métadonnées,
les indices des tâches, les buffers de packing BLAS et la capacité inactive
conservée par un workspace. Le résultat calculé est inclus. Il s’agit d’une
borne de données tensorielles vivantes prévues, **pas d’une limite de RSS**.
Le type scalaire est fixé pour un plan contraint afin de préserver les octets
prévus. Les produits de dimensions en octets utilisent une arithmétique vérifiée.

`EinsumWorkspace(T)` conserve un pool de vecteurs Julia, remodélisés en vues
N-D. Les données simultanément nécessaires ont des leases distinctes ; les
buffers consommés sont libérés après chaque paire. Les permutations des
intermédiaires en pool sont des vues. `einsum!` peut aussi réutiliser son résultat
temporaire, calculé avant toute écriture susceptible d’aliaser les entrées.
`einsum` renvoie un résultat indépendant du pool. `empty!` libère les références
conservées entre campagnes. L’exécution reste susceptible d’allouer métadonnées,
packing BLAS et résultat hors place : aucun appel zéro allocation n’est promis.

Le verrou couvre l’exécution et la copie finale ; deux appelants partagent le
contexte en série, tandis que les tâches internes peuvent traiter des sorties
indépendantes. L’usage récursif est rejeté ; le bloc finally libère les leases.
Une entrée ou sortie qui expose directement le stockage du pool est rejetée.
La collecte effective des buffers retirés reste à la charge du GC Julia.

## Tests et mesures préparés

`test/stochastic.jl` et `test/tensor_memory.jl` sont intégrés à `test/runtests.jl`.
Les cas des quatre types natifs comprennent références analytiques/denses
indépendantes, vues, alias, GC, concurrence et erreurs. **Non exécutés**.

`benchmark/stochastic.jl` compare aux formules d’une matrice Toeplitz connue ;
`benchmark/tensor_plans.jl` vérifie les contractions contre des produits denses
de référence. Les scripts séparent premier appel, compilation lorsque disponible,
minimum/médiane chauds, allocations et données de workspace. La référence BLAS
est réglée à un thread ; le processus Julia choisit 1/2/4/8 threads dans une future
campagne. Ces scripts exigent un répertoire inexistant pour préserver l’historique.

Ils ne constituent pas encore la campagne complète demandée : lanceur reprenable,
logs/hashes/configuration globale, références temporelles Rust et fournisseurs
BLAS distincts, autres familles et inspections ciblées restent à préparer.
Aucune mesure nouvelle n’est disponible ; les résultats v0.5 restent historiques.

## Suite de la mission

La grille `COVERAGE_SUPERVISED.csv` s’étend aux fonctions examinées ici mais
n’est toujours pas l’inventaire global réconcilié. QZ réel/réordonnancement,
small-bulge bloqué, variantes SVD, LDL sparse pivoté/LU multifrontal, méthodes
spectrales par blocs, graphes avancés, précision étendue et couverture complète
des tests/benchmarks restent notamment à traiter. Les archives et anciens manifests
ne sont pas actualisés par ce lot. Aucune autorisation d’exécution n’est sollicitée
avant que les livrables globaux exigés par le protocole soient prêts.
