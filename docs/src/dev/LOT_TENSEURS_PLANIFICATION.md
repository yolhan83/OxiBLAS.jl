# Tenseurs — chemins avancés et exécution par tranches

**Écrit, tests et benchmarks préparés, non exécutés.** Ce lot commence la ligne
Tenseurs du tableau ; il ne déclare pas terminées les décompositions restantes,
la recherche automatique des meilleurs axes de découpe ou le traitement distribué.

## Correction après audit physique

Le [bilan prioritaire](AUDIT_PRIORITAIRE.md) complète le modèle : strides d’entrée,
packing/gather et permutations d’intermédiaires sont estimés sans lecture numérique.
Le défaut `cost_model=:work` minimise toujours le travail, puis le trafic estimé.
`cost_model=:layout, layout_weight=...` choisit explicitement un score pondéré ;
il n’est pas calibré en temps réel. `einsum_layout_report` expose les hypothèses.
L’optimalité ne vaut que pour l’objectif choisi et les dispositions estimées.

## Recherche de chemins

`plan_einsum`, `einsum` et `einsum!` acceptent maintenant `optimize=:beam`
ou `:optimal`, en plus de `:greedy` et `:left` conservés.

- Beam conserve au maximum `beam_width` états à chaque profondeur, classés
  par somme des travaux de contraction puis pic mémoire estimé. Défaut : 32.
- Optimal explore les séquences de paires avec élagage par coût déjà accumulé.
  Le minimum porte sur le modèle de produits de dimensions des labels uniques,
  avec le pic de payload pour départager. Ce n’est pas un optimum de temps réel.
- Les coûts sont des BigInt de planification : pas de dépassement Float64 pour
  classer les grandes dimensions. L’arithmétique des éléments reste celle de T.
- `search_limit`, défaut 100000, borne les états/candidats explorés. Une exhaustion
  est une erreur explicite, jamais un résultat approximatif étiqueté optimal.
  La recherche exacte peut être exponentielle ; la mémoire des métadonnées de
  recherche n’est pas incluse dans les limites de payload numérique.
- Les candidats qui dépassent `memory_limit` sont exclus. Si aucun chemin de
  paires ne tient, le calcul direct antérieur reste disponible si la sortie tient.
  Une beam étroite peut manquer un chemin réalisable ; elle ne certifie rien.

Le parseur, les diagonales répétées, les hyperarêtes, ellipses, strides et
kernels de contraction existants sont réutilisés. Les choix sont déterministes
à métadonnées et options identiques. Une recherche ne lit aucun élément numérique.

## Planification par tranches

Nouvelle API, sur les tableaux Julia ordinaires :

```julia
plan = plan_einsum_sliced("ik,kj->ij", A, B;
    chunks=(i=16, j=16, k=32), scratch_limit=64*1024, optimize=:beam)
workspace = EinsumWorkspace(eltype(A))
einsum!(C, plan, A, B; workspace, α=1, β=0)
```

`SlicedEinsumPlan` contient des métadonnées, aucun stockage dense spécifique.
Les clés chunks sont des labels ASCII nommés ; les largeurs sont positives.
La découpe d’un label de sortie définit des tuiles ; celle d’un label réduit
additionne les résultats partiels. Plusieurs labels et les dernières tranches
incomplètes sont pris en charge. Les ellipses continuent à broadcaster ; leurs
axes implicites ne sont pas sélectionnables comme clés chunks dans ce lot.

Le chemin est choisi une fois sur les dimensions maximales d’une tuile. Les
tranches utilisent des vues et réutilisent les mêmes étapes ; leurs dimensions
et formes de travail sont propres à l’appel, sans modifier le plan partagé.
Les plans de bord ne sont pas reconstruits par une nouvelle recherche. Des
métadonnées et wrappers de vues peuvent encore allouer.

Les tranches sont exécutées en série, avec ordre canonique des labels et des
indices ; une réduction par tranches ne crée pas de conflits d’écriture. Les
kernels internes peuvent employer le parallélisme natif habituel. Les types
à contexte arithmétique inconnu restent dans la tâche appelante selon le trait
existant. La réassociation des sommes par tranches change potentiellement les
derniers bits : aucune identité bit à bit avec une contraction entière n’est
promise. Il n’y a aucun backend de calcul externe.

### Mémoire, alias et contexte

`scratch_limit` borne le modèle de payload des tuiles/intermédiaires, y compris
leur résultat temporaire et une marge de permutation. Il **exclut** la sortie
complète possédée par l’appelant, les entrées, leurs snapshots d’alias, le packing
BLAS, les métadonnées et la capacité inactive du pool. Ce n’est pas une borne
RSS, ni un algorithme qui stocke le résultat sur disque. `einsum(plan,...)`
alloue toujours une sortie complète ; `einsum!` reçoit celle de l’appelant.

Le modèle ne prétend pas mesurer la mémoire détenue par un fournisseur de
scalaires non-isbits : une limite en octets est rejetée pour ces types, tandis
que les tranches sans limite sont disponibles pour BigFloat et types compatibles.
Les champs peak_intermediate_bytes et slice_count exposent les estimations.

Toutes les entrées aliasées à la sortie sont copiées avant sa modification.
Alpha nul évite ces copies et toute lecture numérique des opérandes ; beta nul
ne lit pas l’ancienne sortie. Les collisions avec les buffers possédés par le
workspace restent interdites, y compris pour alpha nul. Un contexte partagé
sérialise les appels. Les leases sont libérées dans finally, même après erreur.
Une erreur numérique en milieu d’appel peut laisser une sortie partielle.
Les plans exposent des collections internes qui doivent rester non modifiées
par l’utilisateur pendant leur utilisation, comme les plans einsum existants.

## Préparation de validation et mesures

`test/tensor_planning.jl` est intégré à la suite générale (46 fichiers). Il prépare :
coûts optimaux contre les cinq parenthésages indépendants d’une chaîne,
reconstructions, plafonds mémoire, budgets épuisés, traces et hyperarêtes,
tranches de sortie/réduction, bords, scalaire/vide, broadcasting, vues inversées,
alpha/beta, alias, GC, partage de workspace et récupération après erreur.
Les précisions génériques et fournisseurs optionnels utilisent les suites déjà
préparées ; aucune compatibilité supplémentaire n’est déclarée validée.

`benchmark/tensor_plans.jl` conserve les cas historiques et ajoute les modes
beam/optimal et l’exécution découpée. Le catalogue du rapport unique ajoute
sept opérations de planification/exécution : **209 entrées par famille,
201 noms distincts**. Les contractions numériques sont comparées au même produit
matriciel, via OpenBLAS et Accelerate vérifiés dans des processus séparés ;
les colonnes de références sans planificateur équivalent restent N/A.
Tous les nouveaux cas seront inclus par `run_all.sh`, sans lancement actuel.

## Correspondance et reste de la ligne

L’audit local des fichiers amont `oxiblas-blas/src/tensor.rs` et `ndtensor.rs`
retrouve contractions, einsum, batches et opérations de tableaux. Le port Julia
réutilise Array/AbstractArray/vues pour les opérations élémentaires, sans recopier
les conteneurs Tensor3/NdTensor. Les deux fichiers ne fournissent pas un moteur
équivalent de recherche de chemins ou de tranches ; ces plans sont des
adaptations Julia. Le module hors mémoire identifié dans l’amont audité concerne
les factorisations sparse et n’est pas déclaré comme moteur tensoriel.

Restent à réconcilier globalement les variantes de décomposition pertinentes,
les besoins éventuels de streaming/distribution, l’auto-sélection des axes et
les compromis de chemins tenant compte de copies/packing réels. Ce lot ne les
marque pas couverts. La phase A supervisée et l’interdiction d’exécution restent
actives ; les historiques v0.5 sont conservés.
