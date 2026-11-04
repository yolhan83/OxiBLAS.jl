# Spectral par blocs — 2 novembre 2026

Complément : [Arnoldi par blocs redémarré/Krylov–Schur](LOT_ARNOLDI_XTRACE.md), soft locking, réorthogonalisation sélective et images conservées, code non exécuté.

État : **0.6.0-dev, sources/tests/scripts écrits, non exécutés**. Aucune
compilation, validation numérique, mesure, inspection LLVM ou validation
d’architecture n’a été effectuée. La phase A du protocole continue ; le portage
complet reste inachevé. Les anciens ZIP/manifests ne décrivent pas ce lot.

## Fichiers de la continuation

| Fichier | Modifications écrites |
| --- | --- |
| src/block_spectral.jl | BlockKrylovFactor ; projections et Ritz Arnoldi/Lanczos par blocs ; redémarrage épais |
| src/sparse_svd_iterative.jl | Dilatation hermitienne implicite ; sparse_svd_lanczos |
| src/OxiBLAS.jl | Exports et includes |
| test/block_spectral.jl | Relations de projection, structures par blocs, Ritz, rang, cibles, redémarrages, erreurs, GC/concurrence |
| test/sparse_svd_iterative.jl | Deux équations singulières, phases, dilatation/alias, rang, vides, échelles, GC/concurrence |
| test/runtests.jl | Inclusion des deux tests |
| benchmark/block_spectral.jl | Projections et solveurs séparés ; références indépendantes avant timing |
| benchmark/inspect_block_spectral.jl | Diagnostics SpMM Float64/ComplexF32 préparés |
| README.md ; docs/STATUS.md ; docs/REPRISE_SUPERVISEE.md ; docs/COVERAGE_SUPERVISED.csv | Suivi courant et limites |

## Rapprochement des fonctions amont

Référence : commit Rust `8d82bb6bf1df133aa90525aa9f478ebe41514f9b`, sous
`oxiblas-sparse/src/linalg/`.

| Fonction amont | API Julia | Adaptation et variantes absentes |
| --- | --- | --- |
| eigenvalue/block.rs::BlockArnoldi::compute | block_arnoldi_factor ; block_arnoldi | Base commune déflatée ; Ritz complexes ; pas de redémarrage Arnoldi par blocs intégré |
| eigenvalue/block.rs::BlockLanczos::compute | block_lanczos_factor ; block_lanczos | Blocs de largeur variable ; réorthogonalisation complète ; variantes partielles/sélectives absentes |
| Config/Result des méthodes par blocs | Mots-clés ; BlockKrylovFactor ; NamedTuple | Wrappers Rust remplacés par des conventions Julia ; mode eigenvalues-only optimisé absent |
| eigenvalue/thick_restart.rs::ThickRestartLanczos::compute | thick_restart_lanczos | Plusieurs Ritz et directions résiduelles conservés ; projection complète ; pas de verrouillage dur |
| svd/types.rs::TruncatedSVD::compute | sparse_svd_lanczos | Dilatation hermitienne plutôt que les équations normales Rust ; pas de complétion des vecteurs singuliers nuls |

Ces lignes décrivent les variantes écrites ; elles ne certifient pas les
variantes manquantes. Les extensions complexes sont également non validées.
Les SVD incrémentales restent absentes.

## Algorithmes et contrats

Les bases et petites projections sont denses, le grand opérateur sparse ne
l’est pas. A*V utilise GEMM/SpMM natifs ; CSC/COO est converti une fois en CSR.
Le facteur réserve dix blocs par défaut ; les solveurs dimensionnent leur
budget par rapport à nev. Une base complète doit être demandée explicitement
sur les grandes matrices. Les tailles restent plafonnées par subspace et n.
Deux passes projettent contre la base, puis les colonnes du bloc sont pivotées
par norme et réorthogonalisées. La déflation compare le résidu à son échelle
avant projection, pour éviter de normaliser du seul bruit d’arrondi. Elle réduit
la largeur du bloc sans lancer de solveurs scalaires indépendants.
Le seuil de déflation par défaut est plafonné sous la tolérance relative de
convergence par défaut, notamment en Float32 sur les grands n. Une tolérance
de convergence explicitement plus serrée peut nécessiter d’abaisser aussi
deflation_tol ; aucune calibration de ces seuils n’a été effectuée.

Le facteur expose V, AV, H, une base résiduelle W et son couplage B, avec la
relation visée AV ≈ VH+WB. La projection d’Arnoldi est Hessenberg par blocs ;
Lanczos sans redémarrage est tridiagonal par blocs à l’arrondi près. Les
coefficients de la petite projection ne sont pas artificiellement annulés.
Les buffers pleins sont transférés sans copie finale ; les bases arrêtées tôt
sont réduites. Stocker AV et réorthogonaliser intégralement coûte de la mémoire
et du calcul ; aucun gain de performance n’est attesté. Les contextes de
factorisation et redémarrage ne sont pas encore réutilisables entre cycles.

Les résidus de Ritz sont recalculés avec A et comparés à
`max(atol,rtol*abs(lambda))`. Le décompte distingue construction et vérification ;
une application réelle à des vecteurs complexes compte deux produits réels.
Le motif d’arrêt expose les budgets et le sous-espace invariant. Un résidu
convergé ne certifie pas que les extrêmes globaux ou toutes les multiplicités
ont été rencontrés. Les limites du petit Schur natif pour les matrices
défectives/non normales restent applicables.

Les cibles sont algébriques, par module, alternées aux deux bouts hermitiens
ou proches d’une valeur. La proximité ne remplace pas un filtrage/shift-invert.
Pas de métrique généralisée, filtre par intervalle/polynôme ou comptage certifié.

Le redémarrage conserve plusieurs Ritz et ajoute des directions résiduelles
avant d’étendre la base. Il résout le problème projeté sur tout l’espace retenu
et étendu, avec historique des dimensions, largeurs et résidus. Ce n’est pas
un simple nouveau départ scalaire. Les images des directions conservées sont
recalculées ; pas de récurrence courte optimisée ni de verrouillage dur.

La SVD applique `[0 A; A' 0]` sans construire A'*A ou AA'. La base a m+n lignes,
soit plus de stockage qu’un Golub–Kahan spécialisé. Les Ritz positifs donnent
U/V normalisés séparément ; A*v=σ*u et A'*u=σ*v sont ensuite vérifiées.
Le solveur interne emploie des tolérances divisées par deux ; seul le contrôle
final des deux équations détermine la convergence de la SVD. Les produits A/A'
de construction et de vérification sont comptés séparément.

Les valeurs retenues dépassent `singular_tol*maximum(Ritz positifs)`. Ce seuil
ne certifie pas le rang global et peut écarter de faibles valeurs réelles.
Moins de triplets demandés donnent `insufficient_positive_ritz_values`, sans
compléter les vecteurs nuls. Le rang demandé zéro produit des facteurs vides,
y compris pour 0×n ou m×0. Les spectres serrés, échelles extrêmes et fortes
multiplicités demandent encore une validation et des méthodes spécialisées.

## Préparation de la validation

Les deux fichiers de tests sont intégrés et **non exécutés** : quatre types,
tailles irrégulières, références analytiques/LinearAlgebra, rang, multiplicités,
couples complexes, budgets, vues inversées, alias de l’opérateur implicite,
résidus vrais, échelles 2^-60/2^60, erreurs, GC et tâches concurrentes.

Le benchmark exige une destination neuve, fixe la référence BLAS à un thread
et vérifie les sorties avant chronométrage. Il distingue projection seule et
résolution, premier appel/compilation, temps chaud minimal/médian, allocations,
taille du résultat et nombre de produits opérateurs. Les processus 1/2/4/8
threads et les références temporelles Rust/OpenBLAS/Accelerate séparées restent
à intégrer à la campagne globale reprenable/loggée/hachée. Aucune mesure obtenue.

L’inspection préparée vise `inspect_block_apply!` pour
`Tuple{Matrix{T},CSRMatrix{T,Int},Matrix{T}}`, T=Float64 et ComplexF32,
`threaded=false`. Question : dispatch résiduel, appels dynamiques inattendus,
abaissement des opérations complexes SpMM. Sorties futures :
`block-spmm-Float64.ll/.s`, `block-spmm-ComplexF32.ll/.s`, avec
`code_llvm(optimize=true,debuginfo=:none)` et `code_native(debuginfo=:none)`.
**Aucun artefact généré.** Pas de validation AVX2/AVX-512 annoncée.

## Suite

Le registre de couverture reste partiel. QZ réel/réordonnancement, small-bulge
bloqué, SVD spécialisées/incrémentales, LDL sparse à pivots/LU multifrontal,
graphes avancés, filtres spectraux/généralisés, tenseurs avancés, précision
étendue et tests/benchmarks de toutes les familles restent notamment à traiter.
Aucune autorisation d’exécution n’est sollicitée avant les livrables globaux.
