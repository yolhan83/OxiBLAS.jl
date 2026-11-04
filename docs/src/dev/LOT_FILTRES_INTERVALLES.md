# Filtres spectraux et intervalles — phase A, 2 novembre 2026

> Mise à jour du 2 novembre 2026 : les tests enregistrés de ce lot passent à 1/2/4/8 threads, bornes activées. Voir [résultats et limites](VALIDATION_CORRECTIONS_20261102.md). Les déclarations « non exécuté » ci-dessous décrivent la préparation antérieure ; benchmarks et fournisseurs optionnels restent non validés.

**0.6.0-dev : écrit, non exécuté.** Aucun chargement, test, benchmark,
autotuning ou diagnostic LLVM/assembleur n’a été lancé. La relecture statique
ne valide ni les algorithmes, ni leurs performances, ni une architecture.

## Fichiers et correspondance amont

Référence Rust : `8d82bb6bf1df133aa90525aa9f478ebe41514f9b`, fichiers
`oxiblas-sparse/src/linalg/eigenvalue/special/polynomial_filtered.rs` et
`interval.rs`.

| Amont / besoin | API Julia écrite | Source ; tests ; benchmark |
| --- | --- | --- |
| Bornes et coefficients Chebyshev | hermitian_spectral_bounds ; polynomial_filter ; ChebyshevFilter ; filter_response | src/spectral_filter.jl ; test/spectral_filter.jl ; benchmark/interval_spectral.jl |
| Application du filtre | apply_filter! ; FilterWorkspace | mêmes fichiers ; buffers réutilisables, vecteurs/blocs, alias |
| PolynomialFilteredLanczos::compute / polynomial_filtered_eigenvalues | polynomial_filtered_eigenvalues | src/interval_spectral.jl ; test/interval_spectral.jl ; benchmark/interval_spectral.jl |
| IntervalEigen::compute / eigenvalues_in_interval | eigenvalues_in_interval | mêmes fichiers ; diagonal/tridiagonal ou projection par blocs |
| count_eigenvalues_in_interval | count_eigenvalues_in_interval | mêmes fichiers ; réponse avec provenance, pas un entier sans qualification |

Les sources sont intégrées à `src/OxiBLAS.jl`, les deux tests à `test/runtests.jl`.
Le suivi README/STATUS/REPRISE et le registre de couverture sont mis à jour.
Les types Config/Result Rust sont remplacés par mots-clés, plans et NamedTuple.
Ces correspondances ne couvrent pas les variantes absentes décrites ci-dessous.

## Filtre et mémoire

Le filtre approche l’indicatrice d’une fenêtre par une série de Chebyshev,
avec amortissement Jackson par défaut ou sans amortissement. La récurrence
travaille sur une copie CSR mise à l’échelle, pour éviter de former d’abord
un grand A*X susceptible de déborder alors que le résultat normalisé tient.
Le plan possède une copie de l’opérateur original et une copie normalisée.

Les bornes par défaut sont celles de Gershgorin avec marge en précision de
travail. Elles ne constituent pas un calcul d’intervalles à arrondis dirigés.
Des bornes fournies par l’appelant sont une assertion d’encadrement, pas une
certification indépendante. Des bornes trop étroites peuvent faire diverger
la récurrence ; les valeurs non finies sont rejetées avant écriture de sortie.
Une fenêtre constante/disjointe ou une plage spectrale dégénérée devient un
polynôme constant ; le filtre ne décide pas à lui seul de l’appartenance aux
bornes d’un intervalle fermé. Une fenêtre de largeur nulle est refusée pour le
filtre, mais acceptée pour la sélection structurelle par intervalle.

FilterWorkspace conserve quatre matrices n×capacité et croît selon les blocs.
Les appels partageant le contexte sont sérialisés et l’usage récursif est
rejeté. Les alias de stockage du plan/workspace sont refusés ; entrée/sortie
peuvent se recouvrir, car le résultat est complet avant toute écriture.
alpha=0 ne lit pas les données d’entrée ; beta=0 ne lit pas l’ancienne sortie.
Des vues et métadonnées ainsi que le packing SpMM peuvent encore allouer.
Pas de promesse zéro allocation ni de borne RSS.

## Recherche filtrée et multiplicité

La base de Krylov est construite sur p(A), mais Rayleigh–Ritz et les résidus
sont calculés sur **A**, pour restituer les valeurs originales même lorsque
le filtre rapproche leurs réponses. Les multiplicités sont préservées par les
directions du sous-espace, sans déduplication par valeur numérique.

Les redémarrages retiennent des Ritz proches de la fenêtre, appliquent à nouveau
le filtre à toutes ces directions puis les réorthogonalisent. Les résidus du
problème original enrichissent cette base, en réservant au moins une place pour
une nouvelle expansion filtrée. Cela évite un sous-espace plein qui ne progresse
plus. Une exploration locale déterministe est ajoutée si ces résidus ne
fournissent pas de direction.
Un filtre constant fait développer la base avec A directement. La dimension
reste plafonnée par subspace ; l’historique distingue candidats et paires
vérifiées. Les comptes séparent applications polynomiales, projection et
vérification/enrichissement de l’opérateur original.

La convergence de nev paires n’est pas la preuve que toutes les valeurs de
l’intervalle ont été trouvées. Pas de verrouillage dur, réorthogonalisation
partielle/sélective, métrique généralisée ou certification de multiplicité.
Des spectres serrés, une trop petite base ou un degré inadapté peuvent limiter
la convergence. Aucun réglage de degré/largeur n’a été calibré.

## Sélection et décompte : contrats distincts

L’intervalle est fermé [low,high], à bornes finies. boundary_tol permet une
extension numérique explicite, avec indicateurs boundary_uncertain. Les
bornes sont converties dans le type réel des données ; si cette conversion
change une borne, même le chemin diagonal ne certifie pas le compte par
rapport aux bornes originales de l’appelant. Les
paires rapportées satisfont un résidu calculé sur A, comparé à
`max(atol,rtol*abs(lambda))` ; les candidats non convergés ne sont pas comptés.

- Diagonal : comparaison exacte des entrées réelles, y compris les doublons
  et les bornes. certified_count/certified est vrai si boundary_tol=0 et si
  la conversion des bornes est exacte.
- Tridiagonal hermitien : phases complexes réduites à des bandes réelles ;
  l’extraction utilise le MRRR natif existant, puis vérifie les résidus sur A.
  Ce chemin calcule encore le spectre et les vecteurs complets, même si les
  vecteurs ne sont pas renvoyés. L’extraction réellement sélectionnée reste
  à optimiser ; elle peut consommer une base n×n.
- Décompte tridiagonal seul : Sturm sur bandes normalisées, sans former les
  vecteurs ; les pivots nuls prennent des signes distincts pour les limites
  stricte/inclusive aux bornes. `complete=true` indique le traitement de
  toutes les bandes ; `certified=false` rappelle les limites d’arrondi.
- Général : projection par blocs bornée ; les paires convergées sont une
  découverte partielle. Même si tous les candidats de cette projection
  convergent, complete reste faux tant que la base ne couvre pas n directions.

complete décrit la couverture numérique de la méthode et des résultats
rapportés ; ce n’est pas un certificat formel aux frontières. Hors diagonale,
certified_count/certified reste faux. Une recherche ne trouvant rien dans une
petite base ne certifie pas que l’intervalle est vide. Les valeurs aux bornes
peuvent rester ambiguës en précision flottante, notamment pour Sturm.
Le comptage sparse général par inertie LDL pivotée n’est pas ajouté ici.

## Tests et mesures préparés

Les tests couvrent les quatre types : formules analytiques du filtre de degré
un, moments par intégration indépendante, calcul fonctionnel spectral par
cosinus, zéros/identité, bornes dégénérées, échelles extrêmes, vues/alias,
alpha/beta sans lecture, croissance/réutilisation des tampons, GC/concurrence,
intervalles fermés, multiplicités, phases tridiagonales complexes, recherche
partielle explicitement non exhaustive, budgets et enrichissement/restart.
**Tests écrits, non exécutés ; aucune conformité attestée.**

Le benchmark préparé utilise un spectre Toeplitz et une base de sinus connus,
vérifie les sorties avant chronométrage, exige un répertoire neuf et fixe la
référence BLAS à un thread. Il sépare premier appel/compilation, minimum/médiane
chauds, allocations et comptes de produits. Les processus 1/2/4/8 threads,
références temporelles externes et protocole global reprenable/loggé/haché
restent à intégrer. Aucun résultat nouveau ; archives/manifests antérieurs
non réactualisés. Aucun nouvel artefact LLVM/assembleur n’a été généré.

Le portage complet reste ouvert : QZ réel/réordonnancement, Schur bloqué,
variantes SVD/LDL compact, sparse direct/graphes, autres transformations
généralisées, tenseurs avancés, précision étendue et validation globale.
