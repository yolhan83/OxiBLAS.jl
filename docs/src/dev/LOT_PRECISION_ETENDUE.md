# Types flottants génériques — correction d’architecture

> Mise à jour du 2 novembre 2026 : les tests enregistrés de ce lot passent à 1/2/4/8 threads, bornes activées. Voir [résultats et limites](VALIDATION_CORRECTIONS_20261102.md). Les déclarations « non exécuté » ci-dessous décrivent la préparation antérieure ; benchmarks et fournisseurs optionnels restent non validés.

État : **sources et campagnes écrites, aucune exécution**. La correction
utilisateur « Correction de l’architecture des types numériques » remplace
le projet de scalaire double-double interne. Ce document remplace le bilan
provisoire correspondant. Les résultats v0.5 et anciens ZIP restent historiques.

## Deux chemins numériques

`NativeFloat` reste exactement `Float32`, `Float64`, `ComplexF32`, `ComplexF64`.
Leurs méthodes GEMM, contextes de packing, microkernels LLVM/SIMD, tailles de
blocs et dispatch spécialisés sont conservés. Un test prépare la vérification
de la méthode sélectionnée et des allocations après échauffement.

`GenericFloat` est l’union de `AbstractFloat` et `Complex{<:AbstractFloat}`.
Les algorithmes utilisent les opérations du fournisseur du scalaire, sans
importer DoubleFloats ou Quadmath et sans conversion de données en Float64.
Les conversions Float64 de dimensions présentes dans le planificateur servent
seulement à estimer le travail. `src/extended_scalar.jl` et
`src/extended_math.jl` sont supprimés, ainsi que leurs exports, alias et méthodes
arithmétiques. Aucun type de précision étendue n’est possédé par OxiBLAS.

Les tableaux BigFloat ne sont pas traités comme des buffers machine. Les
pointeurs SIMD compacts sont réservés aux types matériels. L’accumulation
scalaire générique utilise multiplication puis addition ; `fma`/`muladd` ne
font pas partie de son interface obligatoire. Un type de test à un seul mot,
sans précision supplémentaire, fait échouer explicitement ces deux opérations.
Ses méthodes Base appartiennent au type de test ; aucun type tiers n’est piraté.

### Interface requise

| Algorithmes | Opérations du fournisseur requises |
| --- | --- |
| BLAS et contractions | type concret ; zero/one préservant T ; conversion des coefficients/entiers ; addition, soustraction, multiplication, conjugaison pour les complexes |
| Triangles/LU/solveurs | division, abs/comparaison, détection des zéros ; prédicats de finitude selon l’API |
| Normes/Cholesky/QR/SVD/EVD | sqrt/hypot/abs2, real/imag, eps et bornes flottantes, comparaisons ; fonctions supplémentaires de l’algorithme choisi |
| Raffinement équilibré | frexp/ldexp et conversions vers un lowtype explicitement compatible |
| Fonctions matricielles | fonctions scalaires correspondantes ; exp générique utilise aussi log2/ceil/ldexp ; log/sqrt utilisent Schur complexe |
| CP sans facteurs initiaux | sin/cos pour l’initialisation déterministe |

Une borne AbstractFloat ne garantit pas que tout fournisseur implémente toutes
ces opérations. Une opération absente reste signalée par le fournisseur ou le
dispatch ; aucune compatibilité avec Double64/Float128 n’est déclarée validée.
Les packages externes restent propriétaires de leur arithmétique, y compris
leur éventuelle bibliothèque scalaire. Les calculs matriciels restent Julia.

## Algorithmes conservés et adaptations

| Surface | Mise en œuvre et tests préparés |
| --- | --- |
| BLAS 1/2 denses et compactes | parcours génériques réutilisés ; gardes natives devant les pointeurs ; contreparties scalaires des entrées BLAS 1 parallèles ; test/extended_blas.jl |
| BLAS 3 | GEMM par tuiles ; SYMM/HEMM via opérande symétrique logique ; TRMM/TRSM, rank-k, GEMMT ; test/extended_blas.jl |
| Batches | contextes enfants avec buffers distincts, GEMM interne sans tâches imbriquées ; test/extended_blas.jl |
| LU/Cholesky/LDL/QR | algorithmes Julia existants paramétriques, pivotage complet/rook, QR pivoté, LQ/QL/RQ/COD, bandes/packed/Aasen ; test/extended_lapack.jl |
| Spectral | Hessenberg/tridiagonalisation, EVD QR/DC/MRRR, Schur/QZ disponibles, SVD Jacobi/parallèle/Golub–Kahan/DC et variantes présentes ; test/extended_spectral.jl |
| Utilitaires et fonctions matricielles | inverse/pseudoinverse, espaces/rang/condition, exp/log/sqrt/power/sign et Fréchet ; test/extended_lapack.jl et extended_spectral.jl |
| Raffinement | LU/Cholesky/LDL/QR, lowtype explicite, provenance du fallback et diagnostics ; test/extended_lapack.jl |
| Tenseurs | einsum/einsum!, contract/contract!, batched_matmul, HOSVD, tensor_train, CP-ALS et reconstructions sur AbstractArray ; test/generic_tensors.jl |

L’exponentielle générique utilise une série de Taylor mise à l’échelle, arrêtée
à la précision active, plutôt qu’une approximation Padé calibrée Float64.
Les chemins matériels gardent leur Padé spécialisé. Les budgets d’itération
DC/MRRR/bidiagonalisation sont adaptés à la précision du fournisseur ; la
déflation Jacobi traite les colonnes numériquement dépendantes sans parcourir
le domaine immense des exposants BigFloat. Ce ne sont pas des garanties de
convergence universelle ou d’arrondi correct.

`dot_extended` conserve son API avec somme compensée de Neumaier et
`accumulator_type` facultatif. Float32 est élargi en Float64 ; Float64 reste
Float64 par défaut. Les autres fournisseurs gardent T. Les produits sont
arrondis dans le type d’accumulation choisi : ce n’est plus un produit double-double
implicite. `gemm_extended!` est une compatibilité vers le GEMM paramétrique.

### Précision, contextes et concurrence

`GemmWorkspace(T,m,n,k)` garde le contexte matériel pour les quatre types
natifs et retourne `GenericWorkspace` pour les autres. Une tuile par worker,
un verrou par contexte partagé et des enfants pour les batches protègent les
buffers. Les alias entrée/sortie sont copiés ; les collisions avec le stockage
possédé sont rejetées. alpha=0/k=0 évite les lectures des opérandes et beta=0
celle de l’ancienne sortie. Snapshots et planification peuvent encore allouer.
TRMM/TRSM/GEMMT génériques et BLAS 1 parallèle générique ont un parcours scalaire.

`generic_threadsafe(T)` est vrai pour les types matériels et demi-précision,
faux par défaut pour les fournisseurs inconnus. BigFloat reste dans la tâche
appelante pour conserver le contexte de précision. Un fournisseur peut étendre
ce trait OxiBLAS après avoir établi sa sûreté. Le paquet ne change pas le
contexte global BigFloat et n’invente pas un parallélisme générique mesuré.

Float16/ComplexHalf utilisent Float32/ComplexF32 pour GEMM/rank-k/TRMM/TRSM ;
les variantes LAPACK élargissent une fois et renvoient les facteurs/résultats
à la précision de travail. BLAS 1/2 peuvent arrondir à chaque opération half.
Les résidus natifs Float32 utilisent Float64 ; les fournisseurs génériques
gardent T. `mixed_solve` ne choisit pas implicitement Float64 pour un BigFloat :
le type bas générique par défaut reste T. Un lowtype explicite active le calcul
mixte. Le retour expose convergence, métrique et provenance d’un fallback vers
le facteur de travail. QR utilise le résidu normalisé des équations normales,
sans promettre un résidu original nul ni une borne d’erreur avant.

Le plan einsum distingue `:inline` et `:reference_slots`. Pour un type non-isbits,
le pic indiqué ne représente que les emplacements de référence ; une limite
mémoire finie est rejetée car le payload du fournisseur est inconnu. Ce n’est
pas une limite RSS. Les plans sans limite restent disponibles.

## Compatibilités et campagnes préparées

Les six fichiers du lanceur ciblé sont intégrés à la suite générale : contrats
scalaires, BLAS, LAPACK, spectral, dispatch/allocations, tenseurs. Ils préparent
Float16/32/64 et leurs complexes, BigFloat/ComplexBigFloat à 512 bits, puis
Double64/Float128 et leurs complexes sur activation explicite. Les contrôles
incluent des perturbations plus petites que l’ulp Float64, références MPFR,
reconstructions, alias, strides, GC, contextes partagés et ambiguïtés.

Aucune dépendance numérique obligatoire n’a été ajoutée au Project du paquet.
`test/optional/Project.toml` isole les fournisseurs et BenchmarkTools. Leurs
interfaces sont documentées par les sources primaires de
[DoubleFloats](https://juliamath.github.io/DoubleFloats.jl/stable/)
et [Quadmath](https://github.com/JuliaMath/Quadmath.jl).
Le mode `OXIBLAS_OPTIONAL_FLOATS=DoubleFloats,Quadmath` exige leur disponibilité ;
`available` journalise les absences ou échecs d’import propres à la plateforme.
Le mode par défaut n’importe aucun fournisseur optionnel. Une absence n’est
jamais comptée comme un test réussi.

La préparation future de l’environnement optionnel nécessite Pkg.develop du
paquet local puis Pkg.instantiate dans test/optional : ces actions ne sont pas
exécutées dans la phase actuelle. Les scripts n’installent rien implicitement.

`benchmark/precision_complete.jl` compare natif et référence avec les mêmes
types de stockage/travail, mêmes entrées et même précision active. BigFloat
utilise 256 bits par défaut (`OXIBLAS_GENERIC_BITS`), l’oracle au moins 512 bits
et le double de la précision de travail. Les opérations de référence absentes
sont notées unsupported, sans temps ni speedup. Half est explicitement élargi
à Float32 des deux côtés. Aucun résultat Float64 n’est présenté comme une
référence haute précision. Les benchmarks natifs existants sont conservés.

Le lanceur `benchmark/run_precision_campaign.sh` prépare les phases séparées
correctness, performance, rust et inspect, avec 1/2/4/8 threads pour les deux
premières, logs, codes de sortie et hashes dans des dossiers neufs. Le nombre
de threads du processus ne force pas la création de tâches pour un fournisseur
au trait faux. `OXIBLAS_CORRECTNESS_PROJECT=test/optional` et
`OXIBLAS_PRECISION_PROJECT=test/optional` permettent une future campagne opt-in.
La référence Rust QuadFloat reste un programme indépendant du checkout amont,
avec probe comparé à MPFR avant ses timings ; elle n’est pas un backend du paquet.

Premier appel/compilation, temps chauds, allocations et taille estimée des
contextes sont prévus. Les inspections LLVM/assembleur et les rapports de
méthodes ambiguës sont seulement écrits. Trois échantillons et 0,1 s cible
ne bornent pas le coût d’un appel ou d’une compilation. L’autorisation sera
demandée avec la campagne globale quand tous les livrables seront prêts.

## Limites de portée

Ce lot conserve les algorithmes existants ; élargir un type ne complète pas
QZ réel, les variantes SVD/LDL spécialisées ou les autres lignes encore ouvertes.
Les graphes et formats sparse spécialisés ne sont pas annoncés arbitrairement
génériques. Aucun support fournisseur, absence d’ambiguïté, seuil d’allocation,
exactitude ni performance n’est attesté avant exécution autorisée. Le code
courant n’est pas remplacé par un ancien ZIP ou validé par les chiffres v0.5.
