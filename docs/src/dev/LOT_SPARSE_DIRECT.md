# Ligne « Sparse direct » — livrables écrits, non exécutés

La ligne du tableau utilisateur est traitée pour les fonctionnalités pertinentes
identifiées au commit Rust `8d82bb6bf1df133aa90525aa9f478ebe41514f9b` : LU
multifrontale, LDL à pivotage, LU à pivotage statique et diagnostics, ainsi que
LU et Cholesky hors mémoire. **Cela désigne les sources, tests et scripts prêts ;
aucun chargement Julia, test, benchmark ou diagnostic LLVM n'a été exécuté.**
Aucune correction numérique, performance ou conformité de cette ligne n'est
encore attestée. Le reste de la mission OxiBLAS reste incomplet.

## Inventaire des changements

| Fichier | Livrable |
| --- | --- |
| src/sparse_indefinite.jl | LDL sparse Bunch–Kaufman et sans pivot, blocs 1×1/2×2, inertie ; LU statique avec perturbations ; résolution et diagnostics |
| src/multifrontal_lu.jl | Arbre d'assemblage par éléments de lignes, fronts, pivotage partiel, contributions de Schur, parallélisme ; déterminants/log-déterminants des nouveaux facteurs en mémoire |
| src/out_of_core.jl | LU/Cholesky par tuiles disque, cache privé, budget des tampons numériques, résolutions N/T/C, contrôles de fichiers et nettoyage |
| src/OxiBLAS.jl | Exports et includes, après les helpers sparse/ordonnancements existants |
| test/sparse_indefinite.jl | Reconstruction, résidus et références indépendantes, perturbations/inertie, erreurs, extrêmes, alias/GC/concurrence |
| test/multifrontal_lu.jl | Arbres/forêts, pivots hors diagonale, panels, permutations, reconstruction, résolutions N/T/C, déterminants, partage concurrent |
| test/out_of_core.jl | Résidus, pivots entre tuiles et anciens L, budgets, fichiers corrompus, fermeture, concurrence, triangles ignorés |
| test/runtests.jl ; test/run_sparse_direct.jl | Intégration générale et lanceur ciblé incluant la régression seuil zéro existante |
| benchmark/sparse_direct_complete.jl | Facteurs/résolutions/diagnostics en mémoire et LU/Cholesky disque de bout en bout ; références denses séparées ; configuration et hashes |
| benchmark/run_sparse_direct_campaign.sh | Commandes préparées pour 1/2/4/8 threads, bornes forcées, logs complets, codes de retour et empreintes |
| benchmark/inspect_sparse_direct.jl | Signatures Float64/ComplexF64 pour les blocs LDL, solve multifrontal et balayage disque ; fichiers .ll/.s préparés |
| README.md ; docs/STATUS.md ; docs/REPRISE_SUPERVISEE.md ; docs/COVERAGE_SUPERVISED.csv | État, correspondance et limites actualisés |

## Audit fonctionnel amont → Julia

Les chemins ci-dessous sont relatifs à `crates/oxiblas-sparse/src/linalg/`.
Il s'agit d'équivalences fonctionnelles adaptées à Julia ; aucune identité des
pivots, de l'arbre, des arrondis ou du format disque Rust n'est annoncée.

| Module / fonction Rust | API ou chemin Julia | Décision et tests préparés |
| --- | --- | --- |
| pivoting.rs / SparseLuThreshold::new, with_threshold ; compute_with_threshold | sparse_lu(;pivot_threshold) existant, patch seuil zéro conservé | Pas de réimplémentation ; test/sparse_pivot_regression.jl ; timing threshold_lu |
| SparseLuThreshold::l, u, perm, threshold, solve | F.L, F.U, F.p ; seuil passé à sparse_lu ; solve | Facteur existant ne conserve pas une copie du seuil ; aucun accesseur fictif ajouté |
| SparseLuStaticPivot::new ; compute_static_pivoting | sparse_lu_static | Ordre fixe prescrit, remplissage sparse complet, perturbation diagonale à phase conservée ; test/sparse_indefinite.jl |
| SparseLuStaticPivot::l, u, epsilon, num_perturbed, solve | F.L, F.U, F.epsilon ; count(!iszero,F.perturbations) ; solve | Quantité/amplitude des perturbations inspectables ; résolution du système modifié |
| SparseLdlt::new ; bk_alpha | sparse_ldlt(;pivot=:bunchkaufman) ; constante interne | Blocs 1/2, permutation symétrique, triangle choisi ; dictionnaires sparse plutôt que workspace dense de l'amont |
| SparseLdlt::pivots, perm, n, num_2x2_pivots | F.blocks, F.diagonal, F.offdiagonal, F.D ; F.p ; size(F) ; count(==(2),F.blocks) | Champs Julia inspectables, sans reproduire PivotBlock enum |
| SparseLdlt::solve | solve(F,b;trans,threaded) | Réel/complexe, symétrique/Hermitien, vecteurs et multi-RHS ; test/sparse_indefinite.jl |
| SparseLuFactors, LuSpa, caches privés du module pivoting | Dictionnaires/incidences de sparse_lu_static et stockage existant de sparse_lu | Organisation interne adaptée ; pas de type public artificiel à porter |
| multifrontal_lu.rs / LuFrontalMatrix::new, nrows, ncols, get, get_mut | Matrices Julia F propres à chaque front ; size, indexation native | Pas de conteneur dense dédié ; tests des reconstructions multi-fronts |
| LuFrontalMatrix::assemble_original, assemble_update | factor_front : assemblage des lignes originales, extend-add des enfants | Chaque entrée d'origine introduite une fois ; contribution propagée une fois |
| LuFrontalMatrix::factor_and_schur | Panel pivoté, TRSM natif puis GEMM natif ; _LUFrontUpdate | Schur local réel, pas d'appel à sparse_lu_blocked ni à un backend externe |
| MultifrontalLU::new | multifrontal_lu | Arbre d'éléments de lignes adapté au motif non symétrique ; fronts indépendants par niveaux |
| symmetrize_pattern, permute_columns, build_col_etree, compute_postorder_lu, symbolic_lu | _row_element_lu_tree ; q/iq ; parcours par hauteurs | Analyse adaptée par éléments de lignes ; pas de matrice A'A numérique ni de reproduction du motif symétrisé Rust |
| permutation_sign | _direct_permutation_sign | Cycles des permutations p/q ; tests de déterminant |
| MultifrontalLU::solve | solve(F,b;trans) | Permutations distinctes p/q ; N/T/C ; test/multifrontal_lu.jl |
| MultifrontalLU::n, row_perm, col_perm, nnz | size(F), F.p, F.q ; length(F.L.values)+length(F.U.values) | Les deux diagonales stockées sont comptées dans ce nnz, comme dans les facteurs séparés |
| MultifrontalLU::determinant | determinant ; logabsdet_factor | Parités des deux permutations ; tests LA.det/logabsdet |
| out_of_core.rs / OutOfCoreConfig::default, new | Mots-clés blocksize, memory_budget (octets), directory, threaded | Constructeur Julia factorise immédiatement ; pas d'état public non factorisé |
| OutOfCoreLu::factorize_csr ; OutOfCoreSolver::factorize | out_of_core_lu(A) | LU disque par panneaux, recherche du pivot dans toutes les lignes restantes du panneau |
| OutOfCoreCholesky::factorize_csr ; trait factorize | out_of_core_cholesky(A;uplo,ordering) | Cholesky Hermitienne disque ; les deux algorithmes sont pertinents et inclus |
| OutOfCoreLu::solve ; OutOfCoreCholesky::solve ; trait solve | solve(F,b;trans) | Résultat alloué, multi-RHS/vues ; pas d'API buffer sol Rust redondante requise |
| OutOfCoreSolver::memory_usage_bytes | out_of_core_memory_usage | Payload chiffré séparé des métadonnées retenues ; pas de promesse sur le RSS |
| write_dense_block, read_dense_block, TempBlockFile::new/drop | _ooc_write!, _ooc_read!, mktempdir, close, finalizer | Format local natif contrôlé ; cycle de vie Julia ; tests de corruption/nettoyage |
| dense_lu_inplace, dense_cholesky_inplace ; forward_sub, backward_sub, chol_forward_sub, chol_backward_sub ; dgemm_sub | Panneaux natifs _ooc_lu_factorize!/_ooc_cholesky_factorize!, _ooc_triangular!, gemm! | Factorisation et substitutions adaptées aux tuiles et au pivotage global de panneau |
| unique_prefix, tile_index, read_block, write_block, extract_dense_block | mktempdir ; _ooc_path/_ooc_range ; _ooc_read!/_ooc_write! ; _ooc_initialize! | Métadonnées Julia, extraction CSR par bandes de lignes ; tests disque |
| OutOfCoreError::fmt/source/from | Exceptions Julia ArgumentError, SingularException, PosDefException et IO natives | Pas de hiérarchie Rust à reproduire ; échecs propagés |
| Blocs temporaires, substitution et mises à jour privés | _ooc_lu_factorize!, _ooc_cholesky_factorize!, _ooc_triangular! | Boucles, TRSM et GEMM natifs uniquement ; test/out_of_core.jl |
| Diagnostics supplémentaires Julia | sparse_direct_diagnostics ; ldl_inertia | Résidu/erreur arrière sur A original ; inertie des blocs D en précision de travail |

Les autres familles de solveurs, préconditionneurs et spectraux creux relèvent
d'autres lignes. Le packed LDL dense reste dans « SVD et LAPACK avancé » ; il
n'est pas comptabilisé comme implémenté par le présent LDL sparse.

## Invariants et limites des algorithmes en mémoire

- LDL : `A[p,p] = L*D*(hermitian ? L' : transpose(L))`. Seul `uplo` est lu ;
  la partie imaginaire d'une diagonale Hermitienne est ignorée. Les zéros exacts
  singuliers sont signalés ; aucune régularisation implicite. Les inverses
  2×2 sont mis à l'échelle. Pas de certificat d'inertie exact/arbitraire ; la
  tolérance de `ldl_inertia` concerne les blocs D, pas les valeurs propres de A.
  Pas de pivotage rook ou LDL multifrontal annoncé par cette API.
- LU statique : `A[p,q]+Diagonal(perturbations)=L*U`, avec p=q prescrit.
  `relative=true` multiplie epsilon par le plus grand module d'entrée (si non
  nul). La perturbation conserve la phase d'un pivot non nul. Les diagnostics
  réévaluent le système original et signalent l'existence des perturbations ;
  ils ne certifient pas que ce système a été résolu. Pas de borne d'erreur avant
  ou de condition certifiée. Le raffinement générique existant n'est pas
  remplacé ni présenté comme un solveur sparse expert optimisé.
- LU multifrontale : `A[p,q]=L*U`. Une ligne d'origine entre à sa première
  colonne ordonnée ; ses lignes survivantes appartiennent à une seule
  contribution. Les fronts candidats à une colonne se réunissent dans son
  arbre. Les permutations ultérieures conservent les multiplicateurs L
  associés à l'identité originale des lignes. Les chaînes simples peuvent être
  regroupées jusqu'à maxwidth. Aucun dense Schur global n'est formé, mais un
  front dense et le remplissage total peuvent être très grands. Ce n'est pas
  une implémentation de pivotage retardé distribué ni une garantie de faible
  remplissage. Les fronts indépendants sont parallèles ; les noyaux internes
  désactivent leurs threads lorsque plusieurs fronts sont actifs.
- Les facteurs en mémoire sont lus par solve et peuvent servir des appels
  simultanés ; ils possèdent leur stockage. Modifier leurs champs tableaux
  pendant une résolution n'est pas pris en charge. Les entrées ne sont pas
  modifiées. Un débordement/pivot non représentable est signalé ; aucun solveur
  ne promet la stabilité de tout problème arbitrairement mal conditionné.

## Hors mémoire : portée précise

Les deux variantes Rust sont pertinentes : elles factorisent une entrée CSR
avec des blocs temporaires sur disque. Le port Julia fournit ces deux calculs
pour les quatre types natifs, avec des tuiles denses sur disque. **Ce n'est pas
une compression sparse des facteurs disque.** Le disque peut occuper O(n²).
L'entrée CSR et ses conversions restent résidentes ; ce n'est pas un lecteur
de matrice source distribuée ou en streaming.

Le budget vérifié à la construction couvre le panneau n×b, quatre tampons b²
et les panneaux de packing GEMM alignés de chaque worker. Les TRSM utilisent une feuille de la largeur de tuile, sans copies
numériques récursives dues aux alias de sous-vues ; GEMM réutilise le workspace. La taille b est réduite si nécessaire ; un budget insuffisant
pour b=1 est rejeté. Les solutions retournées, les représentations CSR d'entrée,
les indices/métadonnées, les objets de planification, l'IO Julia, l'allocateur
et les caches du système ne sont pas inclus : **ce budget n'est pas une borne
RSS**, ni une promesse d'absence d'allocations temporaires non numériques.
`out_of_core_memory_usage` rapporte séparément le payload et les métadonnées
retenues estimées à l'instant de l'appel.

Le facteur possède un sous-dossier unique créé sous directory. close est
idempotent et ne supprime que ce sous-dossier ; un finalizer nettoie au mieux,
sans garantie de délai. Une construction qui échoue tente le même nettoyage.
Chaque fichier porte type, dimensions, ordre des octets et checksum FNV-1a ;
les tailles erronées, fichiers manquants ou payloads modifiés sont rejetés.
Ce contrôle vise les corruptions accidentelles, pas une authentification.
Les fichiers sont temporaires et natifs à cette machine : pas de compatibilité
binaire Rust, de checkpoints persistants, de reprise après crash ou de protocole
distribué annoncé. Ces fonctionnalités ne sont pas exposées par les variantes
locales auditées. solve/close partagent un verrou ; deux résolutions du même
facteur disque sont sérialisées, deux facteurs distincts ont des caches propres.

## Couverture préparée et campagne future

Les trois nouveaux fichiers sont inclus dans test/runtests.jl. Le lanceur ciblé
ajoute la régression existante de seuil nul. Les tests préparent des références
LinearAlgebra indépendantes et des identités analytiques : dimensions 0/1 et
irrégulières, quatre précisions, blocs 2×2, permutations avant/après stockage de L,
triangles ignorés contenant NaN, N/T/C, multi-RHS et vues à strides négatifs,
absence de mutation, cas singuliers/non finis, échelles extrêmes, GC et appels
concurrents. Les tests disque ajoutent budget insuffisant/réduit, fermeture,
nettoyage après échec, isolation et altération de chaque élément du format.

Le benchmark distingue premier appel/compilation, temps chauds min/médian,
allocations et threads, garde les résidus avant mesure et conserve des hashes
sources/données/résultats. Motifs bande/flèche/blocs, quatre types, tailles
32/128/384. La LU statique est chronométrée sur le motif diagonalement dominant
approprié et son propre hash, pas sur une entrée retournée exigeant de grosses
perturbations. Les références LAPACK denses sont explicitement identifiées et
ne représentent pas une référence sparse multifrontale. OpenBLAS/Accelerate
sont sélectionnés dans des processus distincts ; l'identité réelle du backend
est enregistrée. Une comparaison temporelle au Rust nécessite encore son
adaptateur dans la campagne globale ; aucune équivalence de vitesse n'est
annoncée. Les temps disque incluent factorisation/résolution/IO/fermeture et
utilisent un cache OS non contrôlé ; pas de mesure « disque froid » affirmée.

Commandes **préparées, à ne pas lancer maintenant** depuis le dossier du paquet :

```sh
bash benchmark/run_sparse_direct_campaign.sh /tmp/oxiblas-direct-correctness-NEW correctness
bash benchmark/run_sparse_direct_campaign.sh /tmp/oxiblas-direct-performance-NEW performance
julia --project=benchmark --threads=1 benchmark/inspect_sparse_direct.jl /tmp/oxiblas-direct-ir-NEW
```

Le lanceur sépare correction et performance ; un échec arrête la phase et
conserve logs/code de retour/timestamps/hash. Les dossiers doivent être neufs.
La première phase produit load.log puis tests-1/2/4/8.log avec bornes forcées.
La seconde produit les CSV/configurations/hashes native et reference pour
1/2/4/8 threads ; l'inspection écrit des .ll/.s. Les durées restent inconnues :
compilation et IO peuvent dépasser les petits budgets par échantillon. Aucun
script ne lance Pkg.instantiate ni n'installe de dépendance implicitement.
La reprise automatique de campagne interrompue et l'adaptateur Rust font partie
du lanceur global encore à compléter ; les sorties partielles restent intactes.

Ces commandes ne constituent pas une demande d'accord anticipée. Selon le
protocole, l'autorisation d'exécution sera demandée lorsque les livrables de la
mission globale seront prêts. L'archive v0.5 et ses mesures sont conservées ;
les ZIP/manifests v0.6 antérieurs restent historiques et n'incluent pas ce lot.

## Contrôles statiques effectués

Sans exécuter Julia : vérification textuelle des includes (59 sources, 35 tests
au lanceur général et 4 au lanceur ciblé), existence des chemins référencés,
absence de doublons dans ces listes, cohérence des 50 entrées du registre CSV
partiel et de leur statut « not executed », liens Markdown locaux et recherche
de délégation numérique externe dans les trois nouvelles sources. La syntaxe
shell du lanceur a été contrôlée par `bash -n`. Ces contrôles ne constituent
ni une analyse syntaxique Julia ni une validation de chargement ou numérique.
