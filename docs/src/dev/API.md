# Interface Julia indépendante

Toutes les fonctions numériques travaillent avec des tableaux à indices commençant
à 1. Les entrées/sorties flottantes utilisent un même type parmi Float32, Float64,
ComplexF32 et ComplexF64. Les opérations vectorielles génériques acceptent également
certains autres nombres, sans promesse de performance ou de couverture de test.
Les sorties doivent offrir des entrées logiques distinctes et être modifiables.

| Fonction | Convention |
|---|---|
| `gemm!(C,A,B,α=1,β=0; transA=:N,transB=:N,...)` | C ← α op(A) op(B) + β C |
| `gemv!(y,A,x,α=1,β=0; trans=:N)` | y ← α op(A) x + β y |
| `axpy!(y,x,α=1)`, `scal!(x,α)` | y ← y+αx ; x ← αx |
| `copy_vector!(y,x)`, `swap!(x,y)` | copie avec snapshot ; échange de vecteurs disjoints |
| `dotu(x,y)`, `dotc(x,y)` | somme sans / avec conjugaison du premier opérande |
| `dot_kahan`, `dot_pairwise` | mot-clé `conjugate=false` ; pas de fast-math |
| `asum`, `iamax`, `iamin` | métrique complexe abs(Re)+abs(Im) ; indices Julia à partir de 1, 0 si vide |
| `nrm2(x)` | norme euclidienne avec somme de carrés mise à l’échelle |
| `rotg(a,b)`, `rot!(x,y,c,s)` | `(c,s,r)`, puis x′=cx+sy, y′=cy−conj(s)x |
| `ger!(A,x,y,α)`, `gerc!` | mise à jour rang 1, yᵀ / yᴴ |
| `symv!`, `hemv!` | même ordre que gemv ; `uplo=:U` ou `:L` |
| `syr!`, `syr2!`, `her!`, `her2!` | A en premier, puis x / x,y, puis α ; seul le triangle sélectionné est modifié |
| `trmv!(x,A; ...)`, `trsv!(x,A; ...)` | `uplo=:U`, `trans=:N`, `unit=false` ; diagonale unité non lue |
| `symm!(C,A,B,α,β;side=:L,uplo=:U)`, `hemm!` | A est la matrice structurée, côté L/R |
| `trmm!(B,A,α;...)`, `trsm!` | B est modifiée ; `side=:L/:R`, `uplo`, `trans`, `unit` |
| `syrk!(C,A,α,β;...)`, `herk!` | triangle de C mis à jour ; N/T pour symétrique, N/C pour hermitien |
| `syr2k!(C,A,B,α,β;...)`, `her2k!` | deux produits de rang k ; HER2K conjugue α dans le second produit |
| `gemm_batched!(Cs,As,Bs,α,β;...)` | listes de matrices ; sorties disjointes, snapshots entre lots |
| `gemm_strided_batched!(C,A,B,α,β;...)` | tableaux 3-D ; dimension 3 = lot |
| `lu_factor(A;blocksize=32,check=true,threaded=true)` | `F.L*F.U ≈ A[F.p,:]` ; `info` si check=false |
| `cholesky_factor(A;blocksize=32,threaded=true)` | triangle inférieur ; `F.L*F.L' ≈ A` |
| `qr_factor(A)` | `F.Q*F.R ≈ A`, Q carrée ; facteurs compacts stockés séparément |
| `solve(F,b)` | LU/Cholesky carrés ou moindres carrés QR |
| `least_squares(A,b)` ou `(F::QRFactor,b)` | m≥n, rang plein ; défaut de rang signalé |
| `mixed_solve(A::Matrix{Float64},b;maxiter=10,rtol=1e-13)` | LU basse précision, résidu Float64 ; résultat nommé x/converged/iterations/residual |

`algorithm` sélectionne explicitement une famille GEMM. `kernel` peut valoir
`:auto`, `:scalar`, `:neon`, `:avx2`, `:avx512` ; une ISA indisponible provoque une
erreur avant toute écriture. Sur les anciennes versions Julia sans interrogation
fiable du CPU cible, x86 utilise le repli scalaire. `kernel_info(T)` rapporte
la voie choisie ; pour les complexes, `lanes` compte les éléments complexes,
le vecteur LLVM contenant deux fois plus de composantes réelles.

Le constructeur `GemmWorkspace(T,m,n,k;config,workers)` réserve des panneaux
réutilisables de taille bornée par les blocs. La configuration passée à l’appel
doit tenir dans ces capacités ; un workspace trop petit provoque une erreur.
Les champs internes ne doivent pas être redimensionnés ou modifiés par l’appelant.
Le noyau lui-même n’alloue pas de panneau. Les appels possèdent encore un petit
coût d’allocation de contrôle, consigné dans les benchmarks.

`PackedMatrix(data,n;uplo,kind,unit)` emploie l’ordre compact BLAS colonne par
colonne. `pack_matrix(A;...)` crée ce stockage. `kind` vaut `:symmetric`,
`:hermitian` ou `:triangular`. Les opérations dédiées sont `spmv!`, `hpmv!`,
`tpmv!`, `tpsv!`, `spr!`, `spr2!`, `hpr!`, `hpr2!`.

`BandedMatrix(data,m;kl,ku,kind=:general,uplo=:U,unit=false)` interprète le
stockage BLAS `data[ku+1+i-j,j]`. Les matrices structurées sont carrées et
utilisent une seule bande : diagonale à la ligne kd+1 en supérieur, 1 en
inférieur. `band_matrix(A;...)` copie seulement la bande. Les fonctions
`gbmv!`, `sbmv!`, `hbmv!`, `tbmv!`, `tbsv!` consomment cet adaptateur.

`CSRMatrix(m,n,rowptr,colind,values)` copie et valide les buffers, en indices
Julia à partir de 1. Les doublons dans une ligne sont rejetés. Les colonnes
peuvent être non triées. Ne pas modifier ensuite les invariants structurels.
`sparse_mul!(y,A,x,α,β;trans=:N)` parcourt les non-zéros en Julia.
`cg(A,b;rtol,atol,maxiter,x0,diagonal=nothing)` suppose A hermitienne définie
positive, et retourne x/converged/iterations/residual/reason. La convergence est
confirmée sur le résidu réellement recalculé. `diagonal` fournit éventuellement
une diagonale de Jacobi réelle positive ; elle ne construit pas un préconditionneur
avancé. Les sorties ne peuvent pas recouvrir le stockage CSR.

Les BLAS 3 structurés réutilisent les options GEMM `algorithm`, `kernel`,
`threaded`, `workspace`, `config`. SYMM/HEMM lisent le triangle sélectionné au
moment d'empaqueter les panneaux. TRMM/TRSM utilisent une décomposition récursive
avec `blocksize=8` pour les feuilles ; SYRK/SYR2K/HERK/HER2K et GEMMT parcourent
les tuiles du seul triangle demandé, avec `blocksize=128`. Une tuile diagonale
est temporaire ; l'autre triangle de C reste exactement inchangé. Les cas
aliasés prennent les snapshots nécessaires avant d'écrire.

`gemmt!(C,A,B,α=1,β=0;uplo=:U,transA=:N,transB=:N,...)` calcule
`C[triangle] ← α op(A) op(B) + β C[triangle]` pour C carrée. A et B peuvent
être rectangulaires ; N/T/C sont indépendants. β=0 ne lit aucune ancienne
entrée de C ; α=0 ou K=0 ne lit pas A/B. Un workspace pour les mises à jour de
rang k ou GEMMT doit couvrir `(min(n,blocksize),min(n,blocksize),k)` ; pour
TRMM/TRSM, `(size(B,1),size(B,2),size(A,1))` convient à tous les sous-problèmes.

Les décompositions spectrales ajoutent les interfaces suivantes :

| Fonction | Résultat |
|---|---|
| `hessenberg_factor(A)` | facteurs compacts ; propriétés H, Q ; A ≈ Q H Qᴴ |
| `hessenberg_q(F;algorithm=:auto,blocksize=32)` | formation de Q non bloquée ou compact WY |
| `apply_hessenberg_q!(B,F;side=:L,trans=:N,algorithm=:auto,blocksize=32)` | application implicite de Q, Qᵀ ou Qᴴ, à gauche/droite |
| `schur_factor(A;maxiter=100max(1,n))` | T, Z, values, iterations ; A ≈ Z T Zᴴ |
| `eigenvalues(A;...)` | valeurs propres générales complexes via Schur |
| `tridiagonal_factor(A;uplo=:U)` | réduction hermitienne ; propriétés T, Q |
| `tridiagonal_q(F;algorithm=:auto,blocksize=32)` | formation de Q avec correction des phases |
| `eigen_tridiagonal(d,e;vectors=true,maxiter=100max(1,n))` | valeurs réelles triées et vecteurs optionnels |
| `eigen_hermitian(A;uplo=:U,vectors=true,maxiter=100max(1,n))` | values, vectors (ou nothing), iterations |
| `svd_factor(A;vectors=true,maxsweeps=100,rtol=...)` | SVD économique U, S décroissantes, Vt, sweeps |
| `singular_values(A;...)` | valeurs singulières seules |

`ConvergenceError` signale une limite d'itérations atteinte. Les colonnes de Z
sont des vecteurs de Schur ; les vecteurs propres généraux restent à porter.
Les sorties SVD sans vecteurs mettent U/Vt à `nothing`.


ROTM/ROTMG et produits scalaires étendus, ajoutés en v0.4 :

| Fonction | Convention |
|---|---|
| `RotmParams(T=Float64;flag=-2,h11=0,h21=0,h12=0,h22=0)` | paramètres immuables, flag -2/-1/0/1 |
| `RotmParams(p::AbstractVector)` | cinq entrées en ordre BLAS flag,h11,h21,h12,h22 ; seuls les champs utiles sont lus |
| `rotm!(x,y,p;kernel=:auto)` | transforme chaque paire par H ; p struct ou vecteur BLAS ; sortie `(x,y)` |
| `rotmg(d1,d2,x1,y1)` | résultat nommé d1/d2/x1/params ; mêmes types Float32/Float64 |
| `rotmg!(state,y1)` | modifie `[d1,d2,x1]` après génération réussie et retourne les paramètres |
| `dsdot(x::AbstractVector{Float32},y)` | produits et accumulation en Float64 (réassociée, comme OpenBLAS), résultat Float64 |
| `sdsdot(alpha::Float32,x::AbstractVector{Float32},y)` | alpha initialement élargi, accumulation en Float64, conversion finale en Float32 |

ROTM accepte les vues inversées et à strides non unitaires. Les sorties
contiguës utilisent des paquets SIMD LLVM (NEON localement, formes x86 non
validées matériellement) ; `kernel=:scalar` choisit les boucles génériques. Les deux sorties
d'une transformation non identité doivent être disjointes, sans indices répétés.
L'identité ne lit/écrit pas les vecteurs. Les paramètres peuvent recouvrir une
sortie : les champs nécessaires sont copiés dans la valeur immuable avant mutation.
Flags 0 et 1 ignorent les coefficients implicites ; flag -2 ignore les quatre
coefficients. Un flag non valide provoque une erreur.

ROTMG exige des entrées finies et protège ses produits/rapports avec des
mantisses et exposants séparés. Le poids négatif d1 ou une forme indéfinie
inadmissible produit la transformation zéro BLAS. Le dépassement de la plage
d'exposants d'une sortie, y compris le sous-dépassement d'un coefficient non nul,
provoque `OverflowError` avant modification de `state`. Les quatre coefficients
explicités sont préservés pendant chaque redimensionnement.

DSDOT/SDSDOT réassocient la somme Float64 (sommes partielles SIMD, comme OpenBLAS), sans fast-math ; un résultat non fini est recalculé dans l'ordre.
Les produits Float32 sont élargis avant multiplication ; leur précision étendue
ne constitue pas une sommation exacte ou compensée. Les valeurs non finies
suivent l'arithmétique IEEE ; les entrées ne sont jamais modifiées.

En v0.5, les opérations packed/banded exploitent les colonnes stockées, sans
parcourir les zéros structurels. `gbmv!`, `tpmv!`, `tpsv!`, `tbmv!`, `tbsv!`
et les mises à jour `spr!`/`spr2!`/`hpr!`/`hpr2!` acceptent `kernel=:auto`
ou les demandes ISA déjà décrites pour GEMM ; `:scalar` choisit la boucle
générique. Le SIMD explicite des mises à jour contiguës concerne les deux types
réels. Les quatre types restent acceptés par les opérations compactes.

Les multiplications triangulaires se font en place dans l'ordre de dépendance,
sans copie de x lorsqu'il ne recouvre pas les coefficients. Les résolutions
vérifient tous les pivots non unité avant modification. Si les coefficients
recouvrent la sortie, une copie du stockage utile protège les lectures. Les
paddings et diagonales unité sont ignorés, y compris au packing.

Pour SYRK/HERK/SYR2K/HER2K/GEMMT, `threaded=true` peut maintenant répartir
les tuiles indépendantes. `GemmConfig.parallel_threshold` contrôle le seuil ;
`workspace.buffers` borne aussi le nombre de tâches. Chaque tâche possède un
couple de panneaux et appelle GEMM en monothread ; le parent du workspace reste
verrouillé pendant tout le plan. Les tampons diagonaux et contextes restent
alloués par appel. `threaded=false` conserve l'exécution séquentielle ; un seul
buffer peut aussi limiter le plan à une tâche. SYMM/HEMM conservent leur
parallélisme GEMM, et les dépendances de TRMM/TRSM ne sont pas modifiées.


## Ajouts 0.6.0-dev — non validés numériquement

Ces signatures sont documentées sans affirmation de réussite des tests.
Leurs limites figurent dans [DEVELOPMENT_v0.6.md](DEVELOPMENT_v0.6.md).

| API | Convention |
|---|---|
| `lu_factor(A; pivot=:partial/:complete/:rook)` | pivots complets/rook : `A[F.p,F.q] ≈ F.L*F.U` |
| `qr_factor(A; blocksize=32, threaded=true, pivot=false)`, `qr_pivoted(A)` | compact-WY sans pivot ; pivot de colonnes avec `A[:,F.p] ≈ F.Q*F.R` |
| `ldlt_factor(A; uplo=:L, hermitian=true, pivot=:bunchkaufman)` | blocs D 1×1/2×2, `A[p,p] ≈ L*D*L'` ; `pivot=:none` possible |
| `expert_solve(A,b; equilibrate=true, pivot=:partial, ...)` | x, convergence, itérations, erreur arrière composante par composante |
| `eigen_generalized(A,B; uplo=:U, vectors=true, ...)` | A hermitienne, B définie positive ; A*V=B*V*Diagonal(values) |
| `bidiagonal_factor(A)` | propriétés U, B, V ; stockage compact et U/V matérialisés à la demande |
| `schur_reorder(F,select)` | Schur complexe uniquement, sélection en tête |
| `eigen_general(A)` | valeurs/vecteurs propres droits via Schur complexe natif |
| `CSRMatrix`, `CSCMatrix`, `COOMatrix` | stockage sparse spécialisé possédé, conversions natives |
| `sparse_matmul(A,B)`, `sparse_mul!(Y,A,X)` | produit creux×creux ; sparse×matrice dense |
| `ilu0(A)`, `gmres(A,b;preconditioner=...)`, `bicgstab(...)` | préconditionneur rien/diagonale/ILU0, résidus vrais |
| `lanczos(A;nev,maxiter,which,start,rtol)` | Ritz hermitien, résidus et statut explicites |
| `plan_einsum(spec,A...)`, `einsum(plan,A...)`, `einsum!(C,plan,A...;α,β)` | plan réutilisable ; `optimize=:none` pour la contraction directe sans planification par paires ; chaînes acceptées directement ; scalaires en Array{T,0} |
| `contract(A,B,axesA,axesB)`, `contract!(C,A,B,axesA,axesB;α,β)` | axes multiples ; conjugaisons explicites ; sorties libres de A puis de B |
| `outer_product`, `mode_product(A,U,mode)` | produits extérieur et modal |
| `batched_matmul(A,B)`, `batched_matmul!(C,A,B,α,β)` | dimensions 1/2 matricielles ; broadcast des axes 3…N |
| `hosvd(A;ranks)`, `tensor_train(A;maxrank,rtol)`, `cp_als(A,rank;...)` | facteurs Tucker/TT/CP et `reconstruct(F)` |
| `autotune_gemm(T,m,n,k;...)` | exécute une calibration uniquement sur appel explicite ; non appelé pendant cet incrément |
| `save_gemm_tuning`, `load_gemm_tuning(T,path)` | cache de configuration contrôlé par CPU/version/threads/type |

`threaded=false` désactive les plans de tâches concernés. Les produits BLAS 2
acceptent `parallel_threshold`, les lots `batch_parallel_threshold`. Les lots
parallèles utilisent toujours des GEMM internes monothread. Les contextes
réutilisés par `GemmWorkspace` restent protégés par le verrou parent.
