# 0.6.0-dev — Restructuration du paquet (4 novembre 2026)

- **Sources.** Elles sont rangées par famille sous `src/` : `utils/`, `blas/`,
  `structured/`, `lapack/`, `sparse/`, `graphs/`, `spectral/`, `tensors/`,
  `api/`. Il n'y a toujours qu'un seul module `OxiBLAS`, et l'ordre des
  `include` n'a pas changé. Les 3 672 signatures de méthodes et les exports
  sont identiques avant et après.
- **Précompilation.** `src/precompile.jl` (PrecompileTools) couvre GEMM
  Float32/Float64, LU et Cholesky denses, `\` et `OxiBLAS.ldiv` sur les
  formats creux, et `einsum`. Le premier `OxiBLAS.ldiv(A, b)` passe de
  13,5 s à 0,13 s. La charge se désactive par la préférence
  `precompile_workload = false`.
- **Tests.**
  - `test/<famille>/`, prélude commun `testsetup.jl` et helpers partagés dans
    `test/helpers/` : chaque fichier tourne seul.
  - Groupes `core` (CI) et `extended`, et une famille à la demande
    (`OXIBLAS_TEST_GROUP`), avec une ligne de progression par fichier
    (`groups.jl`).
  - Les tests de la campagne SVD dense sont branchés.
  - Tous les tests et fixtures liés à la référence Rust sont retirés du paquet.
- **Benchmarks.**
  - Nouveau banc haut niveau `benchmark/run.jl` (Chairmarks), contre
    OpenBLAS/Accelerate, SparseArrays et TensorOperations, avec vérification
    préalable de chaque résultat.
  - Les campagnes historiques sont archivées dans `benchmark/archive/`.
  - Les programmes et scripts Rust sont sortis du paquet.
- **Documentation.**
  - Documenter (`docs/make.jl`), avec une page d'accueil et huit guides.
  - Une référence générée par famille de sources.
  - Les rapports, et les journaux de développement conservés dans
    `docs/src/dev/`.
- **CI.** Trois workflows GitHub Actions :
  - `CI.yml` : tests courants et QA, Linux et macOS, Julia 1.10 et 1, 1 et
    4 threads, cible CPU générique ;
  - `Extended.yml` : manuel et hebdomadaire ;
  - `Documentation.yml` : construction sur PR, déploiement GitHub Pages sur
    `main`.

# 0.6.0-dev — Solveurs sparse bas niveau (4 novembre 2026)

- Analyse symbolique : AMD natif (`amd_ordering`, CSparse `cs_amd`), arbre d'élimination, postordre (pondéré par les comptes de colonnes), comptes de colonnes en O(nnz(A)·α).
- `multifrontal_cholesky` supernodal : supernœuds relâchés de CHOLMOD, largeur non plafonnée, pile de blocs de contribution, kernel de front bloqué en place jusqu'à f = 192 puis POTRF/TRSM/HERK ; bascule simpliciale (up-looking) quand flops/nnz(L) < 40 (`supernodal=:auto`). Résolution par supernœuds. `PosDefException` donne la colonne de A.
- Parallélisme entre sous-arbres du Cholesky supernodal (tâches `@spawn`, tampons par tâche, colonnes de L disjointes), puis haut de l'arbre avec les kernels denses parallèles.
- `sparse_qr(A; pivot=false, ordering)` : QR multifrontal (supernœuds relâchés de l'arbre des colonnes, lignes en escalier, front entier factorisé avec contribution trapézoïdale, panneaux WY), ordres `:natural`, `:amd` (AMD de AᵀA) ou vecteur ; `pivot=true` inchangé. `sparse_apply_q!` et `solve` plus rapides.
- `sparse_cholesky` exact par ce code en ordre naturel (zéros des supernœuds relâchés retirés) ; IC0/ICT inchangés.
- `multifrontal_lu` : `ordering=:auto` par défaut. Stratégie symétrique de UMFPACK (motif symétrique, diagonale pleine) : supernœuds de A+Aᵀ, fronts carrés, pivotage partiel à seuil parmi les lignes complètement sommées, pivots reportés au parent. Motif non symétrique : singletons puis Gilbert–Peierls avec AMD(AᵀA). Ordres explicites : moteur à éléments-lignes inchangé. Résolution par lignes CSR.
- `sparse_lu` exact : argument `ordering` (`:auto` par défaut, `:natural`, `:amd_ata`, vecteur), champ `q` dans `SparseLUFactor` (A[p,q] = L·U) ; Gilbert–Peierls left-looking en ordre donné ; ILUT inchangé (ordre naturel). `multifrontal_lu` : seuil de pivot par défaut 0,1 ; cœurs non symétriques par un LU multifrontal sur les fronts de colonnes (pivotage à seuil préférant la ligne la plus creuse, lignes restantes acheminées au front de leur première colonne non nulle).
- `colamd_ordering` (portage de COLAMD de SuiteSparse, BSD-3, permutations identiques) et `sparse_ordering(:colamd)` ; ordre du cœur du LU non symétrique, option `:colamd` de `sparse_lu` et `sparse_qr`. Pivots du LU sur fronts de colonnes : longueurs de lignes mises à jour à chaque pivot, colonne pivot la plus creuse du bloc ; largeur des fronts plafonnée à 256 colonnes.
- `minimum_degree_ordering` et `multiple_minimum_degree` : graphe quotient avec supervariables et degré externe exact (MMD : élimination multiple).
- `sparse_ldlt` : multifrontal supernodal, Bunch–Kaufman (1×1, 2×2) parmi les variables complètement sommées, seuil 0,01 (MA57) quand le maximum de la colonne est hors des candidats, pivots reportés sinon ; API inchangée (`hermitian=false`, `pivot=:none`, inertie). Résolution par colonnes directes.
- `sparse_triangular_solve` : kernels CSR/CSC natifs sans conversion.
- BLAS3 : pas de mise à l'échelle pour β = 1 dans le produit triangulaire packé (HERK/SYRK).
- Rapports : `reports/SPARSE_DIRECT_DIAGNOSTIC_20261103.md`, `reports/SPARSE_CHOLESKY_ETAPE3_20261104.md`, `reports/SPARSE_LU_LDLT_ETAPES4_5_20261104.md`, `reports/SPARSE_QR_ETAPE6_20261104.md`, `reports/SPARSE_DIRECT_PARALLELE_20261104.md`.

# 0.6.0-dev — SparseArrays, tenseurs sparse, `OxiBLAS.mul!` (3 novembre 2026)

- SparseArrays devient une dépendance directe. Conversions dans les deux sens entre `SparseMatrixCSC`/`SparseVector` et les formats OxiBLAS sans passer par le dense ; `CSCMatrix(S)` et `CSRMatrix(transpose(S))` partagent le stockage de `S`. Les routines sparse d'OxiBLAS (`sparse_mul!`, `sparse_matmul`, solveurs directs et itératifs, `ilu0`) acceptent les matrices SparseArrays.
- Constructeurs des formats sparse OxiBLAS étendus aux types `GenericFloat` (BigFloat compris), sans changer les chemins Float32/Float64.
- `SparseTensor{T,N}` (COO, indices linéaires triés) ; `einsum`, `einsum!`, `contract`, `contract!` et les plans acceptent des opérandes sparse (mélanges dense/sparse, traces, diagonales, produits extérieurs, ellipses) avec sortie `:auto`, `:dense` ou `:sparse` et destinations sparse.
- `OxiBLAS.mul!(C,A,B[,α,β])` et `OxiBLAS.mul(A,B)`, non exportés et distincts de `LinearAlgebra.mul!` (ni étendu ni aliasé) : dense, types structurés de LinearAlgebra (y compris sur contenu sparse, sans densification), SparseArrays, formats OxiBLAS, destinations sparse ; conversion de type qui garde le format ; copie des opérandes dont le stockage chevauche `C`.
- Types sparse OxiBLAS dans la hiérarchie SparseArrays : CSR, CSC, COO, DIA, ELL, BSR, BSC, HYB, SELL `<: AbstractSparseMatrix`, `SparseTensor{T,N} <: AbstractSparseArray{T,Int,N}` (l'ordre 1 est un `AbstractSparseVector`). `CSCMatrix` n'hérite pas d'`AbstractSparseMatrixCSC`, afin que les opérations sur nos types finissent dans les kernels OxiBLAS (`reports/ARCHITECTURE_CSC_A_vs_B.md`). Indices de ligne de `CSCMatrix` triés par colonne ; `copy`/`similar`/`zero` gardent le format (CSC, CSR, COO).
- `Base.*` défini pour les types sparse OxiBLAS uniquement (et leurs transposées et wrappers structurés), via `OxiBLAS.mul` : 16 méthodes, aucune ambiguïté de dispatch (test dédié). `CSC * CSC` reste CSC, les autres produits OxiBLAS donnent du CSR.
- Opérations Base/LinearAlgebra sur les types sparse OxiBLAS, toutes dans des kernels OxiBLAS (aucun kernel SparseArrays ou LinearAlgebra, aucun BLAS externe) : `LinearAlgebra.mul!` (vers `OxiBLAS.mul!`), `+`/`-` (`sparse_add`), transpositions et `permutedims` (`sparse_transpose`), `SparseArrays.permute` (`sparse_permute`), `norm`/`opnorm` 1 et Inf (`sparse_norm`) ; indexation et `setindex!`, broadcast et `map` (style `OxiSparseStyle`), réductions (`sum`, `maximum`, `count`, `any`, `all`, par dimension), `==`, `findall`, `findnz`, `dropzeros`/`droptol!`, `triu`/`tril`, `diag`, `tr`, `issymmetric`/`ishermitian`, `dot`, `kron`, concaténation, `reverse`, `circshift`, `vec`/`reshape`, conversion dense, `zero`/`similar` (`src/sparse_base.jl`, `src/sparse_base_ops.jl`). Coût en O(nnz) pour les `SparseTensor` de tout ordre. Les solveurs (`\`, `lu`, `cholesky`, `qr`) ne sont pas encore branchés.
- Tests : `test/sparse_interop.jl`, `test/mul_layer.jl`, `test/sparse_supertypes.jl`, `test/sparse_base_ops.jl`. Banc (non lancé) : `benchmark/sparse_worker.jl`, cible `OXIBLAS_TARGET=sparse`.

# 0.6.0-dev — LAPACK, première partie (3 novembre 2026)

- `lu_factor` (getrf) : panneau récursif (getrf2), échanges de lignes groupés par panneau (laswp), anticipation parallèle (le panneau suivant est factorisé pendant la mise à jour de la matrice restante), blocs de 128, feuilles de 32 (réel) ou 16 (complexe) colonnes.
- `cholesky_factor` (potrf) : récursif (potrf2) via TRSM/HERK ; feuilles de 64 (réel) ou 32 (complexe) colonnes.
- QR (geqrf) : forme WY compacte, T calculé par GEMM, tampons réutilisés, panneau en BLAS 1 vectorisé, taille de panneau selon la forme. `apply_q!` (ormqr/unmqr) et `qr_q`/`F.Q` (orgqr/ungqr) bloqués, en parallèle par découpe des seconds membres.
- `qr_pivoted` (geqp3) : mise à jour des normes de colonnes avec le critère de recalcul de LAPACK.
- `ldlt_factor` (sytrf) : Bunch–Kaufman bloqué (lasyf), mêmes choix de pivots.
- Solves sur vecteur via TRSV ; `native_inverse` (getri) via TRTRI ; nouveau `cholesky_inverse` (potri : TRTRI puis LAUUM récursif, HERK unique jusqu'à N = 64).
- BLAS : `trsv!` complexe transposé ou adjoint via les produits scalaires BLAS 1 ; test d'alias de `gemm!` à travers vues et transpositions ; le pool de panneaux garde les plus grands.
- Tests : `test/lapack_dense.jl`. Harness : cible `OXIBLAS_TARGET=lapack`, cas LQ, QR rectangulaire, `apply_q`, `potrs` et `potri` équivalents. Voir `reports/LAPACK1_OPTIMISATION_20261103.md`.

# 0.6.0-dev — BLAS 3 (3 novembre 2026)

- Micro-noyau GEMM complexe à deux accumulateurs (`a·Re(b)`, `a·Im(b)`), recombinés à l'écriture ; formes de registres ComplexF64 NEON 1×4×3 et AVX2 2×2×3.
- Tuiles de bord GEMM calculées par le noyau SIMD dans une tuile temporaire, au lieu du noyau scalaire.
- `syrk`/`herk`/`syr2k`/`her2k`/`gemmt` : GEMM packée restreinte au triangle (tuiles hors triangle sautées, tuiles diagonales masquées) ; l'ancien ordonnancement par tuiles reste le repli.
- Feuilles `trmm`/`trsm` réelles d'ordre ≤ 8, à gauche et à droite : noyaux générés, déroulés et vectorisés.
- Parallèle : workers selon le travail (≈ 100 µs séquentielles par worker), thread appelant actif, grille 2D pour GEMM ; panneaux internes réutilisés depuis un pool (au plus 4 par type) au lieu d'une allocation par appel.
- Harness : cible `OXIBLAS_TARGET=level3`, échantillonnage direct (sans BenchmarkTools ni échauffement supplémentaire pour les appels longs), `full_case` non spécialisé par type. Voir `reports/BLAS3_OPTIMISATION_20261103.md`.

# 0.6.0-dev — BLAS 2 (3 novembre 2026)

- `trsv`/`trmv`/`gemv`/`symv`/`hemv` : quatre colonnes fusionnées par passage (bloc diagonal 4×4 à part) ; produits complexes symétriques/hermitiens en quatre sommes réelles.
- `ger`/`gerc`/`syr`/`her`/`syr2`/`her2` : coefficient par colonne puis boucle SIMD fusionnée.
- `gbmv` réel et bandes étroites par accès direct à la bande ; `tbmv`/`tbsv` étroits par lignes, parties réelle/imaginaire séparées en ComplexF32 ; réductions compactes en `@simd` simple ; rang 2 compact en un passage.
- Parallèle BLAS 2 : workers selon le trafic estimé (règle BLAS 1, appelant actif) ; `symv`/`hemv`/`spmv`/`hpmv`/`sbmv`/`hbmv` par colonnes à travail égal avec vecteurs partiels (triangle lu une fois) ; `parallel_threshold` vaut par défaut `nothing` (automatique), une valeur explicite garde l'ancienne règle. Corrige l'erreur `first` du `symv` parallèle.
- Harness : cible `OXIBLAS_TARGET=level2` dans `benchmark/level1_targeted.py`, filtres `OXIBLAS_FULL_*` dans `full_worker.jl`, références `syr2`/`her2`. Voir `reports/BLAS2_OPTIMISATION_20261103.md`.

# 0.6.0-dev — BLAS 1 (3 novembre 2026)

- `dotu`/`dotc` complexes en quatre sommes réelles SIMD ; `dotu`/`dotc` réels en `@simd` simple.
- `dsdot`/`sdsdot` : somme Float64 réassociée (comme OpenBLAS), repli ordonné si non fini.
- `iamax` en un passage par blocs avec clés entières ; `rotg` complexe sans `hypot` dans la plage sûre ; `nrm2` Float64 avec max vectorisé et échelle en puissance de deux.
- Variantes `*_parallel` : noyaux SIMD par morceau, nombre de workers selon le trafic estimé (seuil 4 Mio), thread appelant actif ; `parallel_threshold` vaut par défaut `nothing` (automatique).
- Harness BLAS 1 : mode parallèle réparé, références imprécises marquées, tableaux générés. Voir `reports/BLAS1_OPTIMISATION_20261103.md`.

# 0.6.0-dev — implémentation, validation en attente

Aucun test numérique ni benchmark exécuté pour cet incrément, conformément à
la demande de l'utilisateur. Le portage complet reste inachevé. Voir
`docs/DEVELOPMENT_v0.6.md` pour les API, garanties prévues et limites exactes.

- Contextes, jobs et scratch BLAS 3 réutilisables ; lots parallèles sans
  parallélisme GEMM imbriqué ; TRMM/TRSM sur seconds membres indépendants.
- SIMD complexe compact et réductions fusionnées ; BLAS 2 denses/compacts
  parallèles ; voie GEMM petite sortie sans packing et déroulage ComplexF32.
- Panneaux LU/Cholesky réutilisés, HERK Cholesky ; LU complet/rook,
  QR bloqué/pivoté, LDL/Bunch–Kaufman, solve expert et mixte complexe.
- EVD généralisée hermitienne définie, Schur complexe réordonné et vecteurs
  propres droits, réduction bidiagonale compacte.
- COO/CSC, conversions, SpMM/SpGEMM, ILU0, GMRES/BiCGStab et Lanczos.
- Contractions N-D, einsum et plans, lots N-D, HOSVD, TT-SVD, CP-ALS ;
  tableaux Julia standards, sans type dense Tensor.
- Autotuning strictement explicite, persistance contrôlée par environnement.
  Aucun calibrage ni résultat de performance nouveau annoncé.
- Nouveaux cas de test écrits ; base et archive v0.5 préservées.

# 0.5.0 — stockage compact direct et parallélisme inter-tuiles

- GBMV/SBMV/HBMV/TBMV/TBSV bornés à la bande, sans parcours logique dense.
- SPMV/HPMV/TPMV/TPSV et SPR/SPR2/HPR/HPR2 par colonnes packed stockées.
- Multiplications triangulaires sur place, pivots vérifiés avant substitution,
  snapshots compacts en cas d'alias, paddings et diagonales unité ignorés.
- AXPY SIMD LLVM réel contigu, largeurs dans le domaine des types pour éviter
  les allocations par colonne, racines GC protégées et restes explicites.
- SYRK/HERK/SYR2K/HER2K/GEMMT : tâches sur tuiles disjointes, panneaux propres
  à chaque tâche et protection globale du workspace partagé ; GEMM interne
  monothread lorsqu'un plan extérieur distribue déjà les tuiles.
- Références denses/BLAS/Rust, vues, alias, lectures interdites, données non
  finies structurelles et échelles extrêmes ; mesures 1/2/4/8 threads.
- État de continuation actualisé ; sources et mesures v0.4 conservées.

Le parallélisme compact, les lots/factorisations parallèles, le tuning complexe
et la validation physique x86 restent à poursuivre.

# 0.4.0 — rotations modifiées et produits scalaires étendus

- ROTM Float32/Float64, paquets SIMD LLVM sur les sorties contiguës,
  paramètres immuables ou vecteur BLAS, vues/strides,
  paramètres ignorés non lus, recouvrement des paramètres et sorties contrôlé.
- ROTMG pur et mise à jour d'état, mantisses/exposants séparés pour les produits
  extrêmes, coefficients préservés pendant les redimensionnements répétés.
- DSDOT/SDSDOT : produits élargis et accumulation ordonnée en Float64 ;
  conversion finale Float32 pour SDSDOT, sans fast-math ni réassociation.
- Tests par invariants en haute précision et BLAS indépendant ; défaut ROTMG
  du Rust figé reproduit, fixtures et outil de reproduction conservés.
- Benchmark des vecteurs contigus/inversés/non unitaires, génération par lots
  variés, inspection LLVM/assembleur et archivage des hashes historiques.

Le code numérique reste autonome. Les plages de résultats non représentables,
la portée des références et les tâches restantes sont documentées.

# 0.3.0 — BLAS 3 structuré bloqué et stabilisation

- Revalidation P0 de la v0.2 (11 574 tests), rapport spectral manquant ajouté,
  archives des sources et journaux, CI Linux/macOS/Windows avec logs conservés.
- SYMM/HEMM : empaquetage des triangles directement vers les panneaux GEMM,
  sans expansion symétrique/hermitienne dense.
- TRMM/TRSM : décomposition récursive, mises à jour GEMM, feuilles sur place,
  panneaux dimensionnés pour les sous-problèmes ; résolution droite contiguë.
- SYRK/SYR2K/HERK/HER2K : calcul des seules tuiles du triangle sélectionné,
  tampon diagonal réutilisé et borné, diagonale hermitienne exactement réelle.
- GEMMT : N/T/C indépendants, matrices rectangulaires, triangle opposé préservé.
- Tests des quatre types, triangles ignorés, diagonale unité non lue, α/β/K sans
  lecture, alias, vues, singularités avant mutation et workspaces partagés.
- Benchmarks reproductibles contre la source v0.2 figée et BLAS externe ;
  rapports, inventaire et hashes actualisés. Aucun backend numérique ajouté.

Les kernels x86 et la CI distante restent non exécutés localement. Les autres
travaux P1/P2/P3 sont détaillés dans `docs/STATUS.md`.

# 0.2.0 — décompositions spectrales natives

- Hessenberg compact réel/complexe, application implicite N/T/C de Q à gauche
  et à droite, génération de Q par compact WY et GEMM natif.
- Réduction des multiplications WY aux régions utiles et réutilisation du
  workspace de panneaux pendant tout l’appel.
- Schur réel par Francis double décalage et Schur complexe par QR implicite,
  avec déflation relative, décalages exceptionnels et signalement de non-convergence.
- Tridiagonalisation symétrique/hermitienne rang 2, correction explicite des
  phases et valeurs/vecteurs propres par QR implicite ; voie valeurs seules.
- SVD Jacobi unilatérale économique pour les quatre types, matrices larges,
  rang déficient et valeurs seules ; normes/corrélations mises à l’échelle.
- Tests d’invariants, références Rust/LAPACK, entrées très grandes/petites,
  sous-normales, vues et non-convergence ; script de campagne spectrale fourni
  (campagne complète non exécutée pour la v0.2).

Les variantes spectrales bloquées/avancées, les vecteurs propres généraux,
le balancing et les autres fonctions énumérées dans `docs/STATUS.md` restent
inachevés. Aucun backend numérique externe n’a été ajouté au paquet.

# 0.1.0 — incrément initial

- Paquet Julia autonome et interface explicite pour les quatre types BLAS usuels.
- GEMM de référence, cache, panneaux, registres et vecteurs LLVM, avec sélection
  de l’architecture et répartition native Julia en tâches de travail.
- Buffers alignés réutilisables, garanties α/β, vues et gestion du recouvrement.
- SIMD réel et complexe NEON validé sur Apple Silicon ; formes x86 implémentées
  et en attente de validation matérielle.
- Opérations BLAS denses et compactes, LU/Cholesky bloqués, QR Householder,
  résolutions, raffinement LU mixte, lots GEMM, CSR/SpMV et CG/Jacobi.
- Tests de conformité, références Rust figées, benchmarks, inspection LLVM et
  assembleur, documentation des écarts et inventaire exhaustif de l’amont.

Le portage de l’ensemble d’OxiBLAS reste incomplet. En particulier, Schur/EVD/SVD,
les solveurs et formats creux avancés, les variantes tensorielles, la précision
étendue, l’autotuning et l’ABI sont encore à réaliser. Voir `docs/STATUS.md`.
