# Arnoldi par blocs et XTrace optimisé — lots écrits

> Mise à jour du 2 novembre 2026 : les tests enregistrés de ce lot passent à 1/2/4/8 threads, bornes activées. Voir [résultats et limites](VALIDATION_CORRECTIONS_20261102.md). Les déclarations « non exécuté » ci-dessous décrivent la préparation antérieure ; benchmarks et fournisseurs optionnels restent non validés.

2 novembre 2026. Code, tests et benchmarks préparés, **sans exécution numérique**.
Rust inchangé au commit `8d82bb6bf1df133aa90525aa9f478ebe41514f9b`.

## Arnoldi par blocs

Les corps de `linalg/eigenvalue/block.rs::BlockArnoldi::compute`, projection par
blocs, orthonormalisation, QR locale, résolution Hessenberg et reconstruction ont
été lus. La révision Rust construit un seul sous-espace ; le champ
`max_iterations` n'entraîne pas de redémarrage. Le port de base dans
`block_spectral.jl` était déjà présent : il n'est pas réimplémenté.

`thick_restart_arnoldi`, aussi accessible par `block_arnoldi(...; restart=true)`,
complète ce moteur avec un redémarrage Krylov–Schur par blocs :

- sélection des blocs de Schur du problème projeté selon la cible ;
- réordonnancement réel conservant les paires conjuguées, sans conversion en
  vecteurs complexes pour un opérateur réel ;
- conservation des vecteurs de Schur et de leurs **images déjà calculées** ;
  le redémarrage ne rappelle pas A pour ces colonnes ;
- conservation de la frontière résiduelle, expansion avec blocs de largeur
  variable après déflation de rang, sous-espace borné par `subspace` ;
- `locking=true` : soft locking, priorité aux blocs contenant les valeurs de
  Ritz demandées déjà convergées. Ces blocs sont conservés en tête, puis les
  blocs encore recherchés. Chaque cycle revérifie les résidus et peut déverrouiller
  une paire dont le résidu se dégrade. Pas de projection avec une base de
  vecteurs propres non orthogonaux d'une matrice non normale ;
- réorthogonalisation `:full` par défaut, ou `:selective` avec contrôle de perte
  de norme et d'orthogonalité déclenchant la deuxième projection. Le cycle
  initial utilise la construction de base existante entièrement réorthogonalisée ;
- vrais résidus A·v−λv pour décider la convergence ; historique séparant les
  nouveaux produits opérateur, les vérifications et les soft locks.

Les échanges d'un crayon projeté quasi inséparable restent susceptibles d'être
refusés par les contrôles de stabilité Schur. Un espace trop petit pour conserver
une paire réelle entière et une frontière d'expansion est refusé explicitement.
Un résultat `:maxrestarts` ou `:invariant_subspace` non convergé n'est pas présenté
comme une solution. Les résidus ne certifient pas que toutes les valeurs
extrémales mondiales ont été trouvées. Il s'agit de soft locking, pas d'un
verrouillage dur des vecteurs propres individuels.

Corrections de divergences numériques de l'amont, uniquement côté Julia :
la soustraction Rust lit `h_ij[l][m]` alors que les produits sont stockés avec
l'ancien vecteur en ligne et le nouveau en colonne. Pour V=[e₁,e₂] et
W=[e₁+2e₂,3e₁+4e₂], la matrice de projection est [1 3;2 4] ; la lecture transposée
ne laisse pas un résidu nul, contrairement à VᵀW. Le port utilise les produits
matriciels correctement orientés. La QR Rust remplace aussi des colonnes
résiduelles nulles par une base canonique non nécessairement orthogonale : Julia
réduit le rang du bloc. Enfin, les colonnes de Z d'une itération QR sur une
matrice non normale ne sont généralement pas ses vecteurs propres ; la résolution
projetée Julia utilise Schur puis TREVC par blocs et vérifie A·v−λv.

Le calcul reste natif Julia pour dense et formats sparse natifs, réels et
complexes matériels. Stockage O(n·subspace), hors petit problème projeté, sans
conversion dense de l'opérateur creux. Aucun solveur scalaire n'est lancé pour
simuler un bloc. Les applications de blocs et projections réemploient nos moteurs
GEMM/SpMM ; aucun parallélisme de solveurs imbriqués n'est ajouté.

## XTrace optimisé

Le corps Rust `stochastic.rs::trace_xtrace`, son générateur sphérique et sa QR
ont été lus. Cette routine forme une QR de Ω, pas de AΩ, et son estimateur
`(n/p) sum(qᵢᵀAωᵢ)` n'est pas la formule XTrace publiée. Pour A=I, p=2,
ω₁=e₁, ω₂=(e₁+e₂)/√2, il vaut n·(1+1/√2)/2, inférieur à n.
Cela démontre que le facteur n/p ne corrige pas la modification du sketch.
Cette erreur est corrigée dans le port Julia, sans toucher au Rust.

Les références numériques étudiées sont le
[code des auteurs](https://github.com/eepperly/XTrace), notamment
[xtrace.m](https://raw.githubusercontent.com/eepperly/XTrace/main/code/xtrace.m),
[xtrace_helper.m](https://raw.githubusercontent.com/eepperly/XTrace/main/code/xtrace_helper.m)
et la normalisation des sondes. L'implémentation Julia repose sur les identités
de projecteurs, sans dépendance ou appel à ce code.

`trace_xtrace` remplace le chemin coûteux par :

1. génération locale reproductible de Ω et calcul en bloc Y=AΩ ;
2. une QR fine de Y, sans Q n×n ;
3. colonnes normalisées de R⁻ᴴ pour les downdates leave-one-out de projecteurs ;
4. calcul unique en bloc de A·Q ;
5. calcul des projecteurs des résidus et de leurs images à partir des données
   précédentes, avec deux GEMM natifs ; aucun nouveau produit A·résidu par sonde ;
6. évaluation des contrôles de trace et des formes quadratiques sur les résidus.

Le coût en produits opérateur est **p+rank(Y) ≤ 2p**, au lieu de reconstruire une
base ambiante et ses images pour chacune des p exclusions. `num_probes` conserve
son sens historique de p sondes d'entrée ; ce n'est pas le budget total de
produits des scripts des auteurs. Coût arithmétique O(n·p²+p³) et stockage
O(n·p+p²) sur le chemin de rang plein.

Un sketch déficient ou un inverse triangulaire trop mal conditionné prend un
chemin explicite de rang déficient : une seule base ambiante commune, puis QR
uniquement des **petits** sketches rank(Y)×(p−1). Ce chemin évite aussi les
produits opérateur supplémentaires et fonctionne pour p>n, A=0 et rang faible.
Il peut coûter davantage en petites opérations que le chemin inverse-triangulaire,
mais ne reconstruit jamais p bases n×p. Le rang numérique est contrôlé par
`ranktol` ; une garantie d'absence de biais exacte au-delà de cette décision de
rang en précision finie n'est pas revendiquée.

`normalization=:none` reproduit le contrôle leave-one-out existant.
`:auto` active le resphering pour les sondes sphériques ; `:resphere` accepte
également les gaussiennes. Le facteur est la dimension du complément divisée
par la norme carrée du résidu, calculée sur les résidus eux-mêmes pour éviter
une soustraction de normes presque égales. Cette normalisation est refusée pour
Rademacher : la transposer sans hypothèse d'invariance sphérique serait biaisée.

La version naïve reste accessible sous `trace_xtrace_loo` comme oracle/benchmark
séparé. `std_error` reste NaN et le contrat de corrélation est préservé.
`empirical_error` donne en plus l'estimation empirique échangeable du type utilisé
par les auteurs ; elle est explicitement **heuristique, non certifiée**.
Les champs `path`, `sketch_rank`, `matvecs` et `normalization` rendent le travail
observable. `return_samples=true` permet la comparaison sonde par sonde.

## Validation préparée

`test/restarted_arnoldi_xtrace.jl` prépare :

- Arnoldi dense/CSR/CSC/COO, réel/complexe, vrai problème non symétrique,
  comparaison LAPACK, résidus indépendants, relation de projection et orthogonalité ;
- redémarrages avec images conservées, historique des nouveaux produits,
  modes full/selective et soft locking, paires réelles avec nev impair,
  bloc initial dépendant, budgets et dimensions invalides ;
- XTrace optimisé contre leave-one-out explicite avec mêmes sondes, quatre
  types matériels et trois distributions ; rang plein/déficient, p>n, zéro et rang un ;
- resphering sur identité et oracle indépendant QR complet dans le test,
  comparaisons sonde par sonde, formats sparse, reproductibilité, compte ≤2p,
  erreurs de paramètres et distinction d'erreur empirique/certifiée.

`benchmark/full_arnoldi_xtrace_cases.jl` ajoute sept cas au rapport unique,
avec le même gate numérique et les demandes 1/2/4/8 threads. Les cas XTrace lent
et optimisé utilisent les mêmes sondes et le même opérateur préconstruit.
Les colonnes Rust/OpenBLAS/Accelerate restent N/A sans adaptateur **équivalent** :
une diagonalisation dense ou un autre estimateur n'est pas présenté comme le
même calcul. Les matrices/probes, allocations et compteurs sont conservés.
Le benchmark historique stochastique garde le cas LOO et ajoute le cas optimisé.

Les trois demandes — SVD partielle, Arnoldi par blocs redémarré et XTrace optimisé —
sont maintenant écrites, avec tests/mesures préparés. Aucune validation numérique
ou performance de ces nouveaux lots n'est revendiquée avant lancement autorisé.
