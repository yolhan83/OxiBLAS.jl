# Campagne unique — préparation uniquement

Le lanceur est **écrit et non exécuté**. Le [correctif d’audit prioritaire](AUDIT_PRIORITAIRE.md)
sépare `--check-bounds=yes` pour les tests et `--check-bounds=auto` pour les benchmarks,
et bloque les performances si les tests numériques échouent. Aucun chargement Julia, test, benchmark,
compilation Rust, installation de dépendance ni inspection LLVM n’a été lancé
pour le préparer. La vérification est statique, pas une validation d’exécution.
La mission de portage reste partielle ; ce lanceur ne rend pas présents les
algorithmes encore manquants.

Après autorisation explicite, depuis la racine du paquet :

```sh
bash run_all.sh
```

Un seul nouveau fichier de rapport persistant est créé :
`reports/full_run_YYYYMMDD_HHMMSS_PID.log`. stdout et stderr passent par `tee`,
avec propagation des codes de sortie, y compris un échec de tee. Les sources,
CSV, traces, données de contexte et erreurs utiles des étapes sont insérés dans
ce journal. Les intermédiaires et binaires sont placés dans un dossier temporaire,
puis supprimés ; les fichiers binaires sont identifiés par taille et SHA256.
Les rapports historiques et archives v0.5 ne sont pas écrasés. Les caches Julia
et Cargo ordinaires peuvent être alimentés par les compilations autorisées.

## Ce qui sera exécuté

1. Informations OS/CPU/architecture, versions Julia/Rust/Cargo, configuration,
   empreintes des sources et inventaire des scripts.
2. Les **46 fichiers** actuellement inclus par `test/runtests.jl`, dans quatre
   processus Julia à 1/2/4/8 threads, avec vérification des bornes. Le lanceur de
   tests relit cet inventaire ; une erreur dans un fichier n’empêche pas les
   fichiers suivants de s’exécuter. Les fichiers qui dépendent d’un précédent
   peuvent eux-mêmes échouer : ces erreurs restent visibles. Les compteurs
   Pass/Fail/Error/Broken sont extraits des résultats Test, jamais inventés depuis
   un code de sortie. Un processus interrompu garde des compteurs inconnus.
3. Compilation **offline, release** de huit adaptateurs Rust dans un projet Cargo
   temporaire, contre le checkout OxiBLAS fourni. Chaque binaire est construit
   séparément afin de poursuivre les adaptateurs indépendants après un échec.
4. Catalogue comparatif : **209 entrées par famille, 201 noms d’opération distincts**,
   sur les quatre types matériels et les stockages génériques applicables.
   BLAS 1/2/3, GEMM/batches/triangles, facteurs/solveurs/utilitaires LAPACK,
   SVD/EVD/Schur/QZ, sparse/préconditionneurs/itératifs, tenseurs et précision
   étendue figurent dans le catalogue. Les formes, variantes et scénarios
   spécialisés supplémentaires restent dans les scripts existants.
Les étapes 3 à 5 sont bloquées si les quatre configurations de tests ne passent
pas ou si leurs compteurs/types ne sont pas disponibles. Les fournisseurs de
précision optionnels sont limités aux types réellement présents dans ces tests.
Le pin Rust est contrôlé avant construction. Une correspondance de noms dans
l’inventaire n’est jamais assimilée à une validation.

5. Les **16 scripts de timing existants**, leurs variantes natives/références,
   les probes de fournisseurs et de contrat hermitien, les références Rust
   existantes et leurs comparateurs de données/MPFR. Ils tournent dans des
   processus séparés ; leurs CSV et métadonnées sont insérés dans le même log.
6. Tableaux par famille, scaling, allocations et bilan final. Les étapes
   obligatoires échouées ou les mesures en erreur donnent un code final non nul.
   Les étapes indépendantes continuent. N/A et SKIP ne sont pas des validations.

Les références artificielles par reconstruction spectrale des fonctions
matricielles ont été retirées : la référence stdlib directe est utilisée,
sans attribution Accelerate lorsque son routage interne n’est pas établi.

Les inspecteurs LLVM/assembleur sont volontairement **exclus** de `run_all.sh`.
Après la campagne autorisée, le rapport devra être lu en entier avant de
proposer des corrections ou une nouvelle inspection/campagne.

## Quatre colonnes et attribution réelle

Chaque ligne mesurée ou conservée en annexe a les colonnes :

```text
Operation                    Julia (ms)       Rust (ms)   OpenBLAS (ms) Accelerate (ms)
-----------------------------------------------------------------------------------------------
```

Julia signifie OxiBLAS.jl natif. Rust signifie OxiBLAS Rust release. Les workers
OpenBLAS effacent la table de forwarding héritée et chargent explicitement le
OpenBLAS fourni avec Julia. Les workers Accelerate chargent AppleAccelerate dans
un autre processus. Tous démarrent avec `--startup-file=no`.

Pour une comparaison BLAS/LAPACK du catalogue, le fournisseur des symboles
requis est interrogé via libblastrampoline avant le timing. Un symbole absent,
un fournisseur inconnu ou un fallback OpenBLAS dans un worker Accelerate donne
N/A avec sa raison ; il n’est pas crédité à Accelerate. Cette précaution est
nécessaire car AppleAccelerate peut conserver OpenBLAS comme fallback, notamment
pour GEMMT. Voir la [documentation AppleAccelerate](https://julialinearalgebra.github.io/AppleAccelerate.jl/stable/blas/)
et le [code de provenance LBT](https://raw.githubusercontent.com/JuliaLang/LinearAlgebra.jl/master/src/lbt.jl).

Une référence générique sans symbole BLAS vérifié est libellée explicitement
`LinearAlgebra (internal routing unaudited; not credited to OpenBLAS)`
ou, pour les scalaires génériques, `LinearAlgebra (generic scalar path; no native BLAS attribution)` dans la troisième colonne et ses
métadonnées. La quatrième reste N/A. Les opérations de base éventuellement
employées par un adaptateur sont indiquées dans sa note. Un import Accelerate
ne suffit jamais à attribuer un algorithme générique à ce framework. Sur Linux
et Windows, la colonne Accelerate reste N/A. Sur macOS, son import défaillant
est une étape obligatoire en échec, pas un succès silencieux.

Les nombres 1/2/4/8 désignent les **threads demandés**. Le rapport conserve aussi
les nombres renvoyés par BLAS et la configuration réelle. Accelerate peut choisir
ses propres workers : une demande supérieure à un peut seulement activer son
mode multithread géré, suivant sa version. Une API de contrôle indisponible est
signalée ; le nombre de workers actifs par appel reste non mesuré. Les petits
kernels Julia et certaines entrées Rust peuvent également rester séquentiels.
Les références Rust séquentielles du nouvel adaptateur ne sont comparées qu’à
la configuration un thread ; GEMM utilise le pool Rayon demandé.

## Équivalence et limites des comparaisons

Les cas du catalogue utilisent le même générateur déterministe, les mêmes types,
formes, coefficients, stockage logique et précision active. Les fingerprints
sont comparés avant de publier les ratios. Les adaptateurs Rust appariés couvrent
GEMM réel à 1/2/4/8 threads et dot/LU/Cholesky/solve réels à un thread ; les autres
cellules Rust restent N/A lorsqu’aucun adaptateur équivalent fiable n’est préparé.
Les références Rust spécialisées existantes restent présentes en annexe.

Les vérifications de résultat, préparations et remises à zéro sont hors timing.
Le catalogue mesure un premier appel séparé puis cinq échantillons, evals=1,
avec médiane/minimum et allocations Julia. L’algorithme exact, les allocations
retournées et l’utilisation d’un contexte sont indiqués. GEMM Julia réutilise son
workspace ; l’API Rust gère son packing, ce qui reste explicite dans la note.
Les GFLOP/s ne sont donnés que lorsque le décompte conventionnel est défini.
Une mesure d’allocations Julia ne couvre pas les allocations internes des
bibliothèques étrangères. Les ratios sont référence/médiane Julia ; aucune
régression v0.5 n’est déduite d’un contrat différent.

Float16 est explicitement distingué du travail Float32. Ses références GEMM,
contraction, LU/Cholesky/QR et solve convertissent au type de travail adapté,
avec conversions et stockage requis dans les appels mesurés. Aucun BLAS matériel
Float16 n’est prétendu. BigFloat utilise 256 bits par défaut dans les benchmarks ;
les tests préparent notamment 512 bits. Double64/Float128 et leurs complexes
s’ajoutent lorsqu’ils sont disponibles ou explicitement demandés. Le paquet ne
reçoit aucune dépendance numérique obligatoire.

Les scripts historiques ont des contrats propres : minimum seul dans certains
cas, anciennes implémentations, scénarios, resets et politiques de threads.
**Leurs lignes restent toutes dans les tableaux et CSV du log**, mais elles ne
sont pas fusionnées dans les ratios du catalogue sans adaptateur d’équivalence.
Leurs colonnes de références non vérifiées restent N/A ; leurs temps bruts sont
conservés avec leur provenance. Un LU dense n’est pas présenté comme un LU
multifrontal creux équivalent. Les cellules N/A explicitent une absence d’API,
de fournisseur ou d’adaptateur ; elles ne signifient pas que l’opération est
mathématiquement impossible dans l’autre bibliothèque.

## Tailles, durée et prérequis

Le catalogue propose 16/32/64/128/256/512/1024/2048/4096. GEMM conserve toute la
plage selon le budget mémoire ; les variantes BLAS 3 et LAPACK sont plafonnées à
512, BLAS 2 à 1024, spectral coûteux à 128, sparse à 256, décompositions tenseur
et Kronecker volumineux à 32. Les contractions montent à 128, les précisions
étendues utilisent 8/24. Les scripts existants gardent leurs tailles propres.
Le budget indicatif par cas est 1536 MiB ; un dépassement estimé produit SKIP.
Ce budget de scratch n’est pas une limite RSS et exclut notamment le compilateur.

La campagne complète est volumineuse et peut durer plusieurs heures ou davantage.
Il n’y a pas de timeout global qui ferait passer un résultat tronqué pour complet.
Les outils requis sont Bash, Python 3, Julia >=1.10, Rust/Cargo et les environnements
Julia déjà préparés. Aucune installation/résolution réseau n’est déclenchée
implicitement. Un prérequis manquant est enregistré comme échec. Le checkout Rust
par défaut est relatif au workspace `../../work/oxiblas-upstream` ; il peut être
fourni par `OXIBLAS_RUST_ROOT`. Le cache Cargo local du workspace est réutilisé
s’il existe, sauf si CARGO_HOME est déjà fourni.

Les dépendances de `benchmark/Project.toml` doivent être instanciées avec OxiBLAS
local développé. Sous Julia 1.10, ne pas compter sur le seul champ `[sources]` :
la préparation future nécessite Pkg.develop du chemin local. Les fournisseurs
optionnels disposent de `test/optional/Project.toml`. Cette préparation elle-même
nécessite l’autorisation de chargement/installation et n’a pas été exécutée.

Réglages facultatifs, toujours imprimés dans le rapport :

| Variable | Usage |
| --- | --- |
| OXIBLAS_BENCH_PROJECT | Environnement benchmarks ; défaut benchmark |
| OXIBLAS_TEST_PROJECT | Environnement tests ; défaut . |
| OXIBLAS_RUST_ROOT | Checkout Rust explicite |
| OXIBLAS_RUST_REVISION | Pin attendu ; défaut 8d82bb6bf1df133aa90525aa9f478ebe41514f9b |
| OXIBLAS_FULL_SIZES | Tailles du nouveau catalogue ; ne change pas celles des anciens scripts |
| OXIBLAS_FULL_SAMPLES | Échantillons du catalogue apparié ; défaut 5 |
| OXIBLAS_FULL_MEMORY_MIB | Budget indicatif du nouveau catalogue ; défaut 1536 |
| OXIBLAS_GENERIC_BITS | Précision BigFloat des benchmarks ; défaut 256 |
| OXIBLAS_OPTIONAL_FLOATS | Défaut available : absences/échecs d’import journalisés ; DoubleFloats,Quadmath les exige |

Les environnements et réglages effectifs sont enregistrés. Le mode available ne
compte jamais un fournisseur absent comme un test réussi. Les chiffres de temps,
les compteurs de tests et la compatibilité d’exécution sont encore inconnus.
