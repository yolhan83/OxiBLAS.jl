# SVD incrémentale — continuation du 2 novembre 2026

Version **0.6.0-dev**. Sources, tests et script de benchmark **écrits, non
exécutés**. Aucune compilation, validation numérique, mesure de performance,
inspection LLVM/assembleur ou validation d’architecture n’est attestée.
La phase A supervisée continue ; aucune autorisation d’exécution n’est demandée.

## Fichiers modifiés ou ajoutés

| Fichier | Contenu de cette passe |
| --- | --- |
| src/incremental_svd.jl | État incrémental, initialisation/réinitialisation, ajouts de lignes/colonnes, reconditionnement, snapshots/rang/copie |
| src/OxiBLAS.jl | Exports et include après les SVD et helpers utilisés |
| test/incremental_svd.jl | Références indépendantes, in-span, rang, alias, erreurs, concurrence, échelles, initialisation sparse |
| test/runtests.jl | Inclusion explicite du nouveau test |
| benchmark/incremental_svd.jl | Initialisations denses/sparse, ajouts avec/sans nouvelles directions, réinitialisation/reconditionnement |
| docs/REPRISE_SUPERVISEE.md ; docs/STATUS.md ; README.md ; docs/COVERAGE_SUPERVISED.csv | État courant et couverture limitée au lot audité |

## Correspondance avec l’amont

Référence : `oxiblas-sparse/src/linalg/svd/types.rs` au commit
`8d82bb6bf1df133aa90525aa9f478ebe41514f9b`.

| Fonction amont | API Julia écrite | Adaptation |
| --- | --- | --- |
| IncrementalSVDConfig / IncrementalSVD::new | IncrementalSVDFactor ; incremental_svd | Mots-clés Julia, état typé réel/complexe et cap de rang |
| IncrementalSVD::initialize | incremental_svd ; initialize_svd! | Initialisation dense Jacobi ou randomisée ; sparse randomisée ou Lanczos ; remise à zéro transactionnelle |
| IncrementalSVD::add_rows | svd_add_rows! | Bloc p×n, résidu de rang révélé et petite SVD de couplage systématique |
| IncrementalSVD::add_columns | svd_add_columns! | Bloc m×q, symétrique de l’ajout de lignes, avec adjoints complexes |
| IncrementalSVD::get_svd | svd_snapshot | Copies cohérentes sous verrou plutôt que références empruntées au stockage mutable |
| IncrementalSVD::rank / dimensions | svd_rank ; size | Lecture protégée, rang sans copie des facteurs |
| Configuration de réorthogonalisation | reorthogonalize_svd! ; reorthogonalize | QR des deux bases et SVD du petit cœur, optionnel avant les mises à jour |

Il ne s’agit pas d’une équivalence bit à bit avec Rust. Contrairement aux
branches amont traitant un résidu nul par simple extension de vecteurs, la
variante Julia recalcule aussi la SVD de couplage dans ce cas : les valeurs
singulières peuvent changer même si le sous-espace reste identique. Les tests
analytiques 3/4/5 écrits visent spécifiquement cette propriété.

## Algorithme et choix de stockage

Pour un ajout de lignes B, la base de résidu Q complète la base droite V :
`[A; B] ≈ blockdiag(U,I) * [Sigma 0; B*V B*Q] * [V Q]'`.
Pour des colonnes C, le cœur est `[Sigma U'*C; 0 Q'*C]`. La décomposition du
cœur utilise la SVD Jacobi native ; les facteurs sont relevés par GEMM natif.
Les grandes matrices blockdiag ne sont pas matérialisées. Le résidu du bloc
rejeté est mesuré explicitement ; les directions dépendantes sont déflatées.

Les bases sont éventuellement reconditionnées avant la mise à jour. Les QR
produisent seulement les colonnes minces utiles ; seuls les petits triangles R
sont extraits. Une petite SVD de `Ru*Sigma*Rv'` préserve la matrice représentée,
contrairement à une normalisation indépendante laissant Sigma inchangée.
`reorthogonalize=false` évite cette étape, avec le risque d’accumuler une dérive
d’orthogonalité ; l’appel explicite de reconditionnement reste disponible.

L’état ne conserve pas l’historique ni une copie de la matrice initiale. Une
mise à jour agit sur **l’approximation stockée**, augmentée du nouveau bloc.
Des directions déjà tronquées ne peuvent pas être reconstruites à partir des
seuls facteurs. `maxrank` fixe le cap ; sa valeur par défaut est le rang demandé.
Pour permettre une croissance de rang, fournir un cap supérieur dès la création.
Le rang retenu peut être plus faible à cause de `ranktol` ou du rang des données.

L’initialisation dense utilise Jacobi par défaut et accepte une projection
randomisée. En sparse, le défaut est randomisé et le rang par défaut est borné
à 16 ; une initialisation Lanczos utilise la dilatation native existante.
Aucune densification globale du grand opérateur sparse dans ces deux chemins.
Les facteurs et les nouveaux blocs de données sont denses. Un bloc sparse de
mise à jour est copié en dense comme le bloc ajouté, pas comme l’ancienne matrice.
Le rang initial zéro ignore les données et représente une approximation nulle,
y compris pour des dimensions vides. L’état peut ensuite acquérir du rang.

`maxsweeps` règle le Jacobi initial dense et les petits cœurs ; les initialisations
randomisées/Lanczos gardent les budgets de leurs implémentations natives.
Seed, oversampling et power_iterations sont configurables. Les produits et le
QR peuvent utiliser le parallélisme natif, sans boucle externe parallèle ajoutée.

## Concurrence, erreurs et diagnostics

Un verrou couvre chaque mise à jour, réinitialisation ou snapshot. Les nouvelles
valeurs sont calculées dans des tableaux séparés puis affectées seulement après
succès. Une exception numérique/dimensionnelle laisse les facteurs et les
compteurs précédents intacts. Les vues des facteurs fournies comme bloc sont
copiées sous le verrou avant toute modification. Les snapshots et `copy`
possèdent leur stockage ; deux clones ne partagent pas leur verrou.

Les lectures directes des champs U/S/Vt pendant une écriture concurrente ne
sont pas une interface cohérente : utiliser svd_snapshot. Les appels récursifs
sur le même état sont rejetés. Plusieurs ajouts de lignes concurrents sont
sérialisés dans l’ordre d’acquisition du verrou, donc l’ordre des lignes ajouté
n’est pas déterministe. La mutation externe des champs n’est pas synchronisée.

`last_truncation_norm` est la norme des singularités abandonnées dans le dernier
cœur, `last_projection_residual` celle du résidu local non représenté par les
bases. Ce ne sont pas des bornes certifiées cumulées pour le flux d’origine.
Le premier diagnostic est NaN à l’initialisation randomisée/Lanczos ou pour une
approximation initiale nulle non vide, car les directions initiales omises ne
sont pas évaluées complètement. Pas de certification globale de rang.

Allocations de blocs/cœurs, packing GEMM et reconditionnement restent à mesurer.
Il n’existe pas encore de workspace persistant d’update ni de mise à jour
distribuée/hors mémoire, de suppression de données ou de correction générale
à rang un dans cette API. Aucun gain de performance n’est revendiqué.

## Tests et benchmarks préparés

Le test intégré couvre les quatre types natifs : reconstruction, singularités
comparées à LinearAlgebra, orthogonalité, ajouts successifs, résidus dépendants
et nuls, croissance/troncature de rang, états vides, vues à strides négatifs,
alias des facteurs, snapshots/copies indépendants, réinitialisation, refus de
NaN/Inf et d’options/dimensions incorrectes, échec Jacobi après préparation de
l’update sans mutation de l’état, GC et appels concurrents. Échelles 2^-60/2^60
et initialisations denses/sparse sont incluses. **Tous ces tests sont non exécutés.**

Le benchmark vérifie les facteurs contre la SVD indépendante de l’approximation
stockée augmentée. Chaque échantillon utilise `evals=1` et une copie fraîche de
l’état dans le setup non chronométré ; le premier appel rapporté inclut cette
copie et le signale. Le temps des initialisations est mesuré séparément.
Il prépare minimum/médiane, compilation lorsque disponible, allocations,
stockage conservé, diagnostics locaux et options de reconditionnement.
La référence BLAS est à un thread, les processus Julia 1/2/4/8 threads restant
à intégrer à la campagne supervisée. Lecture de rang/snapshot seuls, références
temporelles externes et lanceur global reprenable/loggé/haché restent à compléter.
Aucun chronométrage ni diagnostic LLVM n’a été lancé pour ce lot.

Les absences mentionnées dans les bilans antérieurs sont remplacées uniquement
pour ces fonctions et variantes précises. QZ réel/réordonnancement, Schur bloqué,
autres variantes SVD, sparse direct/graphes/filtres avancés, tenseurs, précision
étendue et couverture globale des tests/benchmarks restent notamment en cours.
