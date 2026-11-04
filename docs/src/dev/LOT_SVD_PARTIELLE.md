# SVD réellement partielle — code préparé

> Mise à jour du 2 novembre 2026 : les tests enregistrés de ce lot passent à 1/2/4/8 threads, bornes activées. Voir [résultats et limites](VALIDATION_CORRECTIONS_20261102.md). Les déclarations « non exécuté » ci-dessous décrivent la préparation antérieure ; benchmarks et fournisseurs optionnels restent non validés.

Écrit le 2 novembre 2026, sans chargement Julia, tests, benchmarks ou compilation.
Le code n'est pas encore validé numériquement. Rust reste inchangé au commit
`8d82bb6bf1df133aa90525aa9f478ebe41514f9b`.

## Algorithmes portés et corrections

Lecture des corps numériques de `svd/selective.rs` : bidiagonalisation,
construction du tridiagonal BᵀB, comptage de Sturm, sélection par rang/intervalle,
bissection, itération inverse, résolution tridiagonale et rétrotransformation.
Lecture également du dispatch et des branches de `svd/truncated.rs`.

Le port conserve cette stratégie de sélection par Sturm/bissection/itération
inverse, avec deux représentations explicites :

- `representation=:gram` porte le tridiagonal BᵀB du Rust, sans former de matrice
  dense de Gram. Les valeurs jugées non fiables près de la résolution absolue
  de ce Gram provoquent `ConvergenceError(:selected_gram_accuracy,...)`, y compris
  en mode valeurs seules. Il ne fabrique pas de triplets dans ces cas.
- `representation=:dilation`, par défaut, applique la même famille d'algorithmes
  au tridiagonal symétrique intercalé de diagonale nulle et d'arêtes d₁,e₁,d₂,… .
  Ses valeurs propres positives sont les singularités ; cela évite le carré
  du conditionnement de BᵀB. Ce choix suit la linéarisation TGK exposée dans
  [Netlib DBDSVDX](https://www.netlib.org/lapack/explore-html/dc/d73/group__bdsvdx_ga981379d866b6ed3cb60fb363e35e3e14.html).
  L'algorithme est implémenté ici en Julia, sans appel à cette bibliothèque.

Divergences motivées et corrigées uniquement côté Julia :

1. `compute_singular_values_by_index` Rust, lignes 753–776, calcule toutes les
   valeurs avant de découper. Julia bisecte chaque **rang distinct demandé**,
   sans calculer les racines non sélectionnées.
2. `compute_singular_values_in_range`, lignes 664–750, utilise bien deux
   comptages, mais recalcule ensuite tout le spectre pour trouver l'offset.
   Julia déduit les deux bornes de rang directement des comptages et ne fait
   aucune passe spectrale complète pour l'offset.
3. Former BᵀB peut supprimer une singularité représentable : pour
   B=[1 1; 0 δ], δ=2⁻³⁰ en Float64, `1+δ²` s'arrondit à 1 alors que la petite
   singularité est proche de δ/√2. La linéarisation par défaut évite cette
   perte démontrable ; la variante Gram explicite diagnostique ce cas au lieu
   de retourner un zéro artificiel ou une base inventée.
4. L'itération inverse Rust utilise une élimination Thomas régularisée et
   orthogonalise seulement après convergence. Julia utilise le pivotage
   adjacent des tridiagonales (avec seconde surdiagonale), redimensionne les
   seconds membres avant les divisions risquant un dépassement, réorthogonalise
   deux fois à chaque itération au sein des clusters sélectionnés et contrôle
   les résidus. Échec explicite si le budget est épuisé ; aucun solveur complet
   n'est appelé en secours.
5. Les zéro-vecteurs de l'amont peuvent être remplacés par des colonnes de la
   base canonique, qui ne sont généralement pas des vecteurs du noyau.
   Julia découpe le chemin biparti de la dilation aux arêtes exactement nulles.
   Chaque composante impaire fournit un zéro-vecteur sur U ou V. Les deux
   familles sont appariées pour les singularités nulles demandées, avec
   récurrence logarithmique échelonnée, sans calculer une base n×n entière.
6. Les vecteurs U/V complets produits pendant la bidiagonalisation Rust ne
   sont pas nécessaires à une sortie de rang q. Julia garde les réflecteurs
   compacts de `BidiagonalFactor` et les applique directement aux q colonnes.
7. La branche large de `SelectiveSvd::compute` copie Vᵀ dans un U de mauvaise
   orientation/dimension. Le port travaille sur l'adjoint large et échange
   correctement les deux facteurs fins, complexes compris.

## API et coût réel

`svd_selected(A; indices=..., interval=..., vectors=true, ...)` renvoie le même
`SVDFactor` que les autres SVD. Les indices sont **1-based et descendants**,
comme auparavant. Leur ordre et leurs doublons sont conservés en sortie ; les
rangs distincts sont calculés une seule fois. Sans sélecteur, tous les rangs
sont demandés explicitement. L'intervalle Julia reste **(low,high]**, avec des
bornes réelles éventuellement infinies ; une borne basse négative permet de
sélectionner les zéros. Un intervalle vide renvoie un facteur fin vide.

`svd_truncated(A,k)` choisit désormais la sélection réellement partielle par
défaut, même pour k proche de min(m,n) : pas de bascule cachée vers une SVD
complète. Les algorithmes complets `:golub_kahan` / `:divide_conquer` restent
accessibles lorsqu'ils sont explicitement demandés via ce wrapper.
`full=true` est refusé par la sélection, qui ne calcule que les colonnes demandées.

Les limites d'itérations `maxiter` et `maxinverse`, `rtol` et `atol` sont
explicites. Les singularités non représentables après rescaling provoquent une
erreur de dépassement. La variante Gram a un diagnostic conservateur pour les
petites racines : elle peut refuser un problème que la dilation peut traiter.
Les valeurs nulles structurelles ne sont pas remplacées par un seuil arbitraire.
Une matrice numériquement presque singulière reste soumise à l'erreur arrière
normwise de sa bidiagonalisation ; aucune précision relative universelle n'est
revendiquée pour une matrice dense arbitrairement mal conditionnée.

`diagnostics=Ref{Any}()` reçoit : représentation, ordre du tridiagonal,
nombre demandé/distinct, cibles de bissection et d'itération inverse,
comptages de Sturm réellement évalués et itérations. Ce sont des compteurs de
travail, pas des mesures de performance.

Pour r=min(m,n), la réduction dense complète reste O(m·n·r), avec stockage
compact O(m·n). La partie spectrale sélectionnée a un stockage O(r·q) avec
vecteurs, O(r) sans vecteurs, hors sorties et métadonnées ; les facteurs fins
ont un coût O((m+n)·q). Il n'y a ni Q de taille m², ni V de taille n², ni
spectre complet masqué derrière un filtre. Les bissections indépendantes peuvent
se paralléliser ; le traitement des clusters reste ordonné. La politique
`generic_threadsafe` conserve les types à contexte de précision en séquentiel.
Float16 est élargi une fois en Float32 ; le chemin générique reste AbstractFloat
et Complex{<:AbstractFloat}, sans type scalaire maison.

## Validation et comparaisons préparées

`test/svd_selected.jl` est inclus par la suite supervisée :

- matrices hautes/larges, réelles/complexes, indices extrêmes et centraux,
  sélections désordonnées et répétées ; oracles indépendants LAPACK complets ;
- les deux résidus A·V−U·Σ et Aᴴ·U−V·Σ, orthogonalité et entrée inchangée ;
- intervalles demi-ouverts, valeurs répétées, intervalles vides, zéros structurels
  dont les noyaux ne sont pas canoniques ;
- contre-exemple du Gram, petites singularités et échelles extrêmes ;
- rétrotransformation compacte contre les bases matérialisées **dans le test**,
  résolution tridiagonale pivotée contre un système indépendant ;
- budgets nuls, dimensions vides, arguments invalides, vues à strides négatifs,
  GC et appels concurrents ;
- BigFloat et fournisseurs optionnels, élargissement Half, deux représentations.

`benchmark/full_selected_svd_cases.jl` remplace le précédent cas sélectionné
`indices=1:n` par une véritable sélection. Il ajoute valeurs seules, rangs
centraux/petits, intervalle, haut/large, fractions 1/4 et 1/2, Gram et tronquée.
Le garde numérique compare aux singularités de l'oracle complet, mais cet oracle
est exclu du chronométrage. Les sorties partielles sont vérifiées par leurs
résidus ; leur reconstruction n'est pas faussement comparée à la matrice entière.

Les références sont des appels directs GESVDX confinés à
`benchmark/selected_svd_reference.jl`, pour les quatre types matériels.
ABI et espaces de travail sont préparés d'après
[DGESVDX](https://www.netlib.org/lapack/explore-html/dc/d4a/group__gesvdx_ga004948b9a80afcbb01d95119db268cbc.html)
et [ZGESVDX](https://netlib.org/lapack/explore-html/dc/d4a/group__gesvdx_ga6bb41a877ef35c9caf8d1be4f29db67d.html).
Les requêtes de workspace, copies, allocations et résultats font partie des
deux contrats chronométrés. Le symbole GESVDX actif est vérifié **avant** tout
appel externe dans le worker. OpenBLAS et Accelerate utilisent les processus
isolés déjà préparés. Si le symbole n'est pas disponible, la colonne porte N/A,
sans substitution d'une SVD complète présentée comme partielle.

Le rapport unique conserve Julia/Rust/OpenBLAS/Accelerate, allocations et demandes
1/2/4/8 threads. Le Rust est inchangé : faute d'adaptateur chronométré équivalent,
sa colonne reste N/A pour ces nouveaux cas. La campagne ne prétend pas qu'Accelerate
respecte exactement le nombre de threads demandé. Aucun lancement effectué.

Les compléments Arnoldi par blocs et XTrace optimisé sont suivis séparément
dans LOT_ARNOLDI_XTRACE.md ; ce document décrit uniquement la SVD partielle.
