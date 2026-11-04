# Audit prioritaire — 2026-11-02, phase A

**Audit statique et correctifs écrits ; aucune validation numérique exécutée.**
Pas de chargement Julia, précompilation, test (y compris les nouveaux tests
Python), benchmark, autotuning, inspection LLVM/assembleur ou analyse
SnoopCompile. Les seuls contrôles effectués lisent du texte, analysent la syntaxe
Python/Bash et structurent les inventaires. Le checkout Rust a été identifié
par lecture Git : `8d82bb6bf1df133aa90525aa9f478ebe41514f9b`.

## Correctifs certains et organisation mémoire

L’inventaire détaillé des parcours est [MEMORY_ACCESS_AUDIT.csv](MEMORY_ACCESS_AUDIT.csv).
Il sépare les chemins conservés, les corrections/améliorations et les décisions
qui demandent des mesures. « Conservé » signifie architecture examinée, pas
validation d’exécution ni certification générale de sûreté.

- **GEMM** conserve packing, blocage, formes SIMD et ordonnanceur. La capacité
  physique des vecteurs, les offsets, l’absence de buffers entrelacés entre
  workers et les collisions avec les opérandes sont maintenant contrôlés avant
  l’accès brut. Une capacité déclarée ne suffit plus. Les types natifs doivent
  être concrets : un tableau à élément Union n’est pas un buffer BLAS homogène.
  Les algorithmes sans packing honorent également le verrou et la propriété
  d’un workspace explicitement fourni. Les contrôles de tuiles triangulaires
  s’effectuent sous le verrou du parent, sans inspection concurrente d’un pool
  en cours de redimensionnement. Les batches vérifient tous leurs opérandes
  avant de traiter le premier batch.
- **Leading dimensions et vues** : le chemin par pointeur de C exige un stride
  de ligne unitaire et un stride de colonne positif suffisant. Les autres vues
  passent par les stores indexés. Les vues à indices répétés et les wrappers
  reshape/permutation sont contrôlés ; les sorties Transpose/Adjoint qui
  enveloppent un vecteur sont prises en charge. Les alias d’entrées ordinaires
  restent copiés avant modification de C.
- **SYMV/HEMV** : parcours du triangle par colonnes quand les strides le
  justifient, update et dot miroir fusionnés en séquentiel. En parallèle, chaque
  tâche possède ses lignes de sortie et les dots correspondants ; aucune
  accumulation partagée. Les autres dispositions gardent le fallback par lignes.
  Le snapshot d’alias ne lit plus le triangle inutilisé.
- **GEMV transposé/adjoint** : les dots suivent les colonnes physiques du parent
  lorsqu’il s’agit d’une matrice dense adaptée. Le chemin normal reste par
  colonnes avec sorties réparties par lignes.
- **TRMV** : parcours par colonnes in-place dans un ordre qui conserve x[j]
  original jusqu’à sa consommation, sans copie complète du vecteur sur ce chemin.
  Les autres dispositions conservent la copie. Les snapshots de TRMV/TRSV
  respectent le triangle choisi et la diagonale unité. L’ordre de substitution
  de TRSV reste intact.
- **BLAS 1/updates** : contrôle d’injectivité des sorties standard avant mises à
  jour ; les updates de rang alpha nul ne copient plus les vecteurs aliasés.
- **Cholesky** : le buffer initial ne copie que le triangle inférieur documenté,
  sans lire les coefficients supérieurs non définis d’un wrapper ou d’un type
  à références. La sortie supérieure est toujours mise à zéro en fin d’algorithme.
- **LU/LDL solve** : suppression de la copie redondante après `b[F.p]` ou
  `b[F.p,:]`, qui crée déjà un résultat possédé pour les tableaux Julia usuels.
  Les permutations, pivots et opérations triangulaires restent identiques.
- **CSC** : produit normal séquentiel et produit transposé/adjoint directement
  dans les colonnes compressées, sans construire un CSR à chaque appel.
  Aucun ordre trié des indices de ligne n’est supposé. Le produit normal
  parallèle garde sa conversion CSR et son ownership par lignes ; le supprimer
  sans mesure ou répartir les colonnes en scatter partagé serait injustifié.
  COO conserve son adaptateur sparse ; aucune conversion dense n’est ajoutée.

Les parcours stridés des permutations de pivots, des petites feuilles
triangulaires et des similarités spectrales ne sont pas réécrits au prétexte
qu’ils visitent des lignes. Les dépendances et le blocage GEMM existants priment.
Les seuils de création de tâches et tailles de blocs n’ont pas été retunés.

## Tenseurs : modèle physique explicite

Les plans inspectent les strides disponibles sans lire les éléments ni
matérialiser les opérandes. Ils suivent les dispositions estimées des
intermédiaires, l’ordre des axes, le packing/gather et une éventuelle permutation.
Un layout inconnu reste inconnu, avec charge conservatrice. Les chemins avec
workspace peuvent garder des permutations en vues ; le modèle ne prétend pas
prouver qu’un reshape copie ou qu’un certain accès gagne en temps réel.

`cost_model=:work` reste le défaut : travail arithmétique exact en BigInt,
puis trafic estimé et pic mémoire pour départager. `cost_model=:layout` permet
explicitement de minimiser `work + layout_weight*traffic`, avec poids entier
non négatif. C’est une heuristique non calibrée, pas un autotuning lancé.
`optimize=:optimal` ne certifie que l’objectif symbolique choisi, dans son budget.
Un changement de strides lors de la réutilisation du plan ne change pas la
correction du chemin, mais peut invalider son intérêt estimé.

`einsum_layout_report` expose ces hypothèses pour les plans entiers et découpés.
Le constructeur de métadonnées antérieur est conservé avec layouts inconnus.
Les limites de scratch excluent toujours sortie possédée, alias snapshots,
packing, métadonnées et capacité inactive ; aucun RSS ni payload de scalaires
non-isbits n’est certifié. Aucun Tensor dense spécifique n’est créé.

## Lanceur et équivalence des comparaisons

`Campaign.julia(mode="test")` émet `--check-bounds=yes` ; le mode benchmark,
par défaut, émet `--check-bounds=auto`. Les benchmarks historiques enveloppés
par `full_legacy.jl` et les wrappers Bash ciblés suivent cette séparation.
Les comparateurs numériques/probes tournent en mode test.

Les quatre processus de tests doivent se terminer avec zéro Fail/Error,
compteurs disponibles et assertions réellement exécutées avant toute mesure.
Un échec ferme le **numerical validation gate** : les autres configurations de
**tests** continuent, mais performances Julia/Rust/historiques restent bloquées.
Les fournisseurs optionnels des benchmarks sont limités aux types présents dans
les tests réussis de toutes les configurations. Cela ne prouve pas une opération
sur tous les cas : chaque mesure conserve aussi son contrôle de résultat.

Rust doit être disponible, sans modification suivie du working tree, et sa révision doit correspondre au pin attendu
(`OXIBLAS_RUST_REVISION` permet un changement explicite, tracé). Les builds
restent offline/release et isolés du dépôt dans un répertoire temporaire.
Les empreintes des données communes incluent aussi la matrice H, pas seulement A/B/x.
Les guards du GEMM Rust, de son dot et du GBMV Rust précèdent leurs essais
chronométrés ; la consommation observable du résultat GEMM est explicite.
Le probe QuadFloat doit passer son oracle indépendant avant ses timings.

Les quatre colonnes Julia/Rust/OpenBLAS/Accelerate restent systématiques.
Les symboles requis, le fournisseur réel, les demandes de threads et les
informations effectivement renvoyées restent séparés. Accelerate n’est jamais
supposé utiliser exactement N workers. Les fonctions matricielles et le produit
extérieur utilisent désormais leur référence stdlib directe : aucun eigensolver
ou GEMM de substitution n’est inventé uniquement pour remplir Accelerate.
Sans attribution de symbole vérifiée, la référence est marquée LinearAlgebra
et Accelerate vaut N/A. Les variantes sélectionnées/randomisées qui demandent
le rang complet l’indiquent : aucune efficacité de spectre partiel n’est inférée.

Le budget mémoire vérifie aussi le payload n⁴ des sommes de Kronecker.
Les erreurs, stderr, codes de sortie et allocations restent dans le rapport
unique ; les historiques ne sont pas écrasés. Les lignes legacy non appariées
ne produisent pas de speedups artificiels. Les dépendances manquantes restent
des échecs explicites, sans installation implicite.

## Dispatch, intrinsics et parallélisme

Les quatre types matériels gardent leurs méthodes prioritaires, intrinsics et
packing. Les chemins génériques gardent leur type de travail, l’addition/produit
sans FMA obligatoire, et la politique de contexte `generic_threadsafe` prudente.
Les conversions Float64 restantes dans les moteurs denses génériques examinés
concernent des tailles/coûts de scheduling, pas les données. Les coefficients
spectraux calculés en Float64 dans les modules limités aux types natifs et les
RNG à grille de 53 bits sont des restrictions de ces algorithmes, pas une
compatibilité BigFloat annoncée. Sparse reste explicitement limité à ses types
natifs, sans fausse généralisation de signatures.

Les fonctions de packing et réduction par pointeur sont restreintes aux types
natifs ; les microkernels complexes aux deux types complexes matériels. Les
vecteurs LLVM déclarent l’alignement du scalaire réel, pas un alignement excessif
sur les tableaux utilisateur. Le padding des panneaux vise 128 octets, sans que
la correction dépende de cette optimisation. Les invariants des indices sparse sont établis par les constructeurs ; une
mutation directe des patterns doit les préserver. Aucun appelant ne doit
redimensionner les buffers ou changer leurs indices pendant un calcul. Les
checks de capacités ne prétendent pas rendre sûre une mutation externe concurrente.
Les pointeurs de C ne sont utilisés
que sur tuiles complètes ; les panneaux sont paddés et les queues traitées sans
chargement hors de leur capacité. `GC.@preserve buf C` couvre les racines des
panneaux et parents de C ; les frontières compactes préservent leurs vecteurs.
Ces arguments statiques restent soumis aux tests GC/alias/bornes préparés.

Les macros @inbounds existantes ne sont pas dupliquées dans leurs sous-blocs.
Aucune propagation à travers un appel non annoté n’est présumée pour conclure
à l’absence de contrôles ; le mode test force au contraire les contrôles.
Les générateurs dépendent de types/tailles de microtuiles et largeurs ISA connues,
pas des dimensions générales des matrices. Leur coût de compilation reste à
observer. Les tâches gardent leurs propres buffers, sans dépendre de threadid().
Les seuils existants et les solveurs intrinsèquement séquentiels sont conservés.

Les extensions Base repérées dans src concernent des types OxiBLAS possédés,
pas les scalaires tiers. Aucune méthode LinearAlgebra remplacée, aucune extension
DoubleFloats/Quadmath ajoutée, aucune nouvelle dépendance SIMD/LoopVectorization.
Les listes mécaniques de sites restent des pistes d’inspection, pas une preuve
complète de dispatch, d’absence d’invalidation ou de type piracy.

## Inventaire fonctionnel prioritaire

La vue de référence est désormais [FUNCTIONAL_COVERAGE.csv](FUNCTIONAL_COVERAGE.csv),
centrée sur les API publiques et familles numériques. Elle distingue équivalents
Julia, adaptations/algorithmes alternatifs, variantes réellement manquantes et
helpers remplacés par les abstractions Julia. Les milliers de déclarations brutes
ne servent pas de liste de fonctionnalités à porter. La réconciliation ligne à
ligne des accesseurs et helpers internes n’est pas un prérequis à la reprise.

Les correctifs prioritaires sont écrits et les manques significatifs connus.
Conformément au recadrage utilisateur, l’audit s’arrête ici ; il ne sera pas
prolongé par des optimisations spéculatives. La suite fonctionnelle reprend
QZ/Schur, puis variantes LAPACK/SVD et les autres restrictions de cette vue.

## Annexes techniques et limites de l’inventaire brut

- [BENCHMARK_API_AUDIT.csv](BENCHMARK_API_AUDIT.csv) : chaque export public,
  occurrences Julia/tests, références de benchmarks candidates ou exclusion
  explicitement motivée. Les types, IO, autotuning et auxiliaires non isolés ne
  deviennent pas des faux benchmarks. Une occurrence textuelle ne prouve pas
  qu’une méthode particulière est chronométrée : cette limite reste indiquée.
- [UPSTREAM_FUNCTION_AUDIT.csv](UPSTREAM_FUNCTION_AUDIT.csv) : déclarations
  Rust `pub fn` et méthodes de traits publics, fichier/ligne/hash/révision,
  correspondance candidate, statut et restrictions. Les macros/cfg et résolutions
  de types ne sont pas développées. Les déclarations **Non réconcilié** ne sont
  ni qualifiées d’absentes, ni comptées comme portées.
- [LOW_LEVEL_SITES.csv](LOW_LEVEL_SITES.csv) : localisation mécanique des
  pointeurs, preserve, SIMD/generated, tâches, conversions et inbounds.
  Les mentions ne sont pas toutes des défauts : elles orientent les inspections.

Le vocabulaire sépare Porté fidèlement, Adapté à Julia, Algorithme alternatif,
Partiellement implémenté, Absent et Implémenté mais non validé. L’audit automatique
ne décerne pas « Porté fidèlement » ou « Absent » par simple ressemblance de nom.
Les correspondances candidates demeurent à réconcilier manuellement ; l’ancien
inventaire au fichier `upstream_inventory.csv` reste historique. **Il n’y a pas
encore de couverture globale réconciliée ni de pourcentage de portage fiable.**
QZ réel, réordonnancement Schur/QZ, small-bulge, SVD spécialisée et divergences
sparse/graphes conservent leurs restrictions documentées. Le présent audit ne
les déclare pas complétés.

## Fichiers modifiés

Sources : gemm/simd/compact_blas, level1/level2/modified_rotations, level3/batched,
factorizations/advanced_factorizations, sparse/sparse_extended,
tensors/tensor_planning. Les architectures SIMD, blocs et seuils existants restent.
Tests : audit_priority.jl, runtests.jl, run_all_supervised.jl et
 test_full_report_contract.py. Benchmarks : full_report/full_worker/full_cases/
full_extra_cases, adaptateurs Rust GEMM/GBMV, tensor_plans et wrappers de campagnes.
Documentation : présent bilan, inventaires CSV, RUN_ALL/STATUS/REPRISE et lots
concernés. `tools/audit_static.py` ne lit que les sources pour reconstruire les
inventaires ; il n’exécute pas les algorithmes du projet.

## Livrables de validation préparés et inspections futures

`test/audit_priority.jl` prépare leading dimensions, strides négatifs/non unitaires,
Transpose/Adjoint, triangles/diagonales ignorés, sorties/buffers qui se chevauchent,
capacités corrompues, GC, précision BigFloat, layout tensoriel et CSC non trié.
`test/test_full_report_contract.py` prépare, avec mocks, modes bounds, propagation
d’erreurs, quatre colonnes et blocage des timings après échec des tests.
Ils n’ont pas été exécutés. Le lanceur général inclut le nouveau fichier Julia.
Les benchmarks existants plus les cas CSC et les modèles tensoriels sont prêts.

Après accord explicite seulement, les commandes candidates sont :

```sh
python3 -m unittest discover -s test -p test_full_report_contract.py
bash run_all.sh
```

La campagne complète peut durer plusieurs heures ou davantage. L’accord de
lancement reste à obtenir ; la mission fonctionnelle n’est pas déclarée achevée.
Après une campagne convenue, lire le log intégral et proposer les corrections
avant tout nouveau cycle. Inspections **à proposer ensuite**, jamais lancées ici :
GEMM natif et stores stridés, variantes de largeur LLVM et queues compactes,
absence de pointeurs/FMA imposées dans BigFloat, allocations des contextes,
propagation des checks et coût des nouveaux guards. AVX2/AVX-512 restent
**non validés sur matériel x86**. SnoopCompile est une piste distincte, non lancée.

## Contrôle de clôture statique

Syntaxe Python analysée par AST sans importer les scripts, délimiteurs Julia
contrôlés lexicalement (pas de parseur/compilateur), cibles des 66 includes source
et 46 includes de tests présentes, CSV structurés, registre à 209 entrées /
201 noms sans doublons. Les tableaux fonctionnels remplacent les comptes
de helpers comme suivi prioritaire. Archive v0.5 inchangée. Aucun test ni
benchmark exécuté ; fonctionnement et performances restent à valider.
