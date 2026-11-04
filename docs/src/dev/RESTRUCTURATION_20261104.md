# Restructuration du paquet (4 novembre 2026)

## Arborescence

```
OxiBLAS.jl/
├── Project.toml, README.md, LICENSE, NOTICE, CHANGELOG.md, .gitignore
├── src/
│   ├── OxiBLAS.jl                module unique, exports, include dans l'ordre de dépendance
│   ├── precompile.jl             charge PrecompileTools
│   ├── utils/        (5)  types numériques, matériel, ordonnancement, autotuning, précision étendue
│   ├── blas/        (11)  SIMD, GEMM, BLAS 1/2/3, batched, variantes, BLAS étendu
│   ├── structured/   (3)  stockages compacts et bandes
│   ├── lapack/      (24)  factorisations denses, Schur, QZ, SVD, MRRR, fonctions matricielles
│   ├── sparse/      (25)  formats, solveurs directs et itératifs, préconditionneurs
│   ├── graphs/       (6)  ordonnancements, partitionnement, couplages
│   ├── spectral/    (10)  Krylov par blocs, SVD creuse et incrémentale, filtres, stochastique
│   ├── tensors/      (3)  einsum, planification, tenseurs creux
│   └── api/          (5)  mul, API haut niveau, résolution
├── test/
│   ├── runtests.jl, groups.jl, testsetup.jl, README.md
│   ├── helpers/      références et sondes partagées
│   ├── blas/ structured/ lapack/ sparse/ graphs/ spectral/ tensors/ generic/ api/ qa/
│   ├── legacy/       anciens lanceurs (délèguent à runtests.jl ; originaux en *_v0.6.jl)
│   └── optional/     environnement des fournisseurs de précision étendue
├── benchmark/
│   ├── run.jl, README.md, Project.toml   banc haut niveau (Chairmarks)
│   ├── latency/                          latence de premier appel
│   └── archive/  (60)                    campagnes et micro-benchmarks historiques
├── docs/
│   ├── Project.toml, make.jl
│   └── src/  index.md, guide/ (8 pages), reference/ (générée), dev/ (journaux)
└── .github/workflows/  CI.yml, Extended.yml, Documentation.yml
```

`reports/` (rapports de campagne) reste local et est ignoré par git.

## Sources

**Déplacements.** Les 92 fichiers sont rangés par famille, comme le montre
l'arborescence ci-dessus.

**`OxiBLAS.jl`.**

- Seuls les chemins des `include` changent : l'ordre est strictement
  identique, vérifié par `diff`.
- L'ordre de dépendance entrelace les familles : types, constantes et `@eval`
  évalués au chargement. Des commentaires de section le rendent lisible.
- `include("precompile.jl")` est ajouté en dernier.

**Contrôle.** Les 3 672 signatures de méthodes OxiBLAS, sur Base, Core,
LinearAlgebra, SparseArrays et OxiBLAS, et la liste des exports sont
identiques avant et après le déplacement.

**Correctifs de compatibilité.** Ils ne visent que le dispatch, sans aucun
noyau touché et sans coût à l'exécution.

- **`hcat`/`vcat` sur les formats OxiBLAS.** Les éléments sont contraints à
  `<:Number`. Ces méthodes deviennent ainsi plus spécifiques que celles de
  SparseArrays jusqu'à 1.13, qui étaient ambiguës.
- **`LinearAlgebra.mul!`.** Quatre méthodes de désambiguïsation, à
  destination matricielle, couvrent les signatures `Diagonal`/bande et
  triangulaires de LinearAlgebra 1.10.
- **`\` sur un `AbstractCompressedVector`.** Une méthode par enveloppe
  triangulaire, sous `@static isdefined(SparseArrays, :AbstractCompressedVector)`.
- **`nonzeroinds`** est lu par le champ `nzind`, valable sur toutes les
  versions (ExplicitImports).

## Tests

**Prélude commun.** `testsetup.jl` contient `TYPES`, `tol` et `op`, et inclut
`helpers/` : références BLAS, fournisseurs optionnels, sondes (`UnreadableMatrix`,
`TensorNoRead`, `AuditTriangle`), vérifications (`check_triangle`,
`spectrum_error`, `check_partition`) et références de précision étendue.
Ces helpers étaient auparavant définis dans des fichiers de test et utilisés
par d'autres : chaque fichier tourne maintenant seul.

**Niveaux** (`groups.jl`).

- `core` (CI, 55 fichiers) et `extended` : `level3_blocked` et les six
  fichiers de types génériques et de précision étendue, `qz_schur_complete`,
  `sparse_memory_triangular`.
- `all`, une famille, ou des fichiers explicites (`OXIBLAS_TEST_FILES`).
- Une ligne de progression par fichier. La QA s'exécute dans le testset
  racine, même après un échec.

**Ajouts et retraits.**

- Les tests de la campagne SVD dense (`dense_svd_*`) sont branchés. Le fichier
  de campagne a perdu sa redéfinition locale de `tol`, inutilisée, qui aurait
  remplacé celle de la suite.
- Tous les tests liés à la référence Rust et leurs fixtures sont retirés du
  paquet : `rust_conformance`, `compact_rust`, `spectral_rust_conformance` et
  la comparaison ROTMG de `level1_extended`.

**Robustesse.** Deux tests sont rendus indépendants de la version et du tirage
aléatoire :

- valeurs propres de `Diagonal` comparées à une référence triée explicitement ;
- `logabsdet` triangulaire avec une tolérance absolue ;
- triangulaire unitaire bien conditionnée dans `continuation_parallel`.

**QA** (`qa/qa.jl`).

| Contrôle | Statut |
|---|---|
| Aqua : piraterie, ambiguïtés, paramètres non liés | bloquant |
| ExplicitImports | non bloquant |
| `Test.detect_closure_boxes` | non bloquant, 1.14 seulement |

## Benchmarks

**`benchmark/run.jl`.** Il compte 26 cas répartis en neuf familles :

- GEMM, multiplication creuse, BLAS 3 ;
- factorisations denses, SVD et spectres ;
- résolution dense et creuse, factorisations creuses ;
- tenseurs, matrices structurées.

**Méthode.**

- Les références sont OpenBLAS ou Accelerate (via LinearAlgebra),
  SparseArrays (UMFPACK, CHOLMOD) et TensorOperations.
- Chaque résultat est vérifié avant la mesure. Les formes allouante et en place
  sont distinguées.
- Les mesures sont des médianes à chaud (Chairmarks). Seuls les cas
  sélectionnés sont compilés, et les résultats sont écrits en CSV et en
  Markdown.

**Archives.** Les anciens scripts sont dans `archive/`, avec leurs chemins
corrigés. Les programmes Rust et les scripts qui les pilotent ont quitté le
paquet.

## Précompilation

`src/precompile.jl` utilise `@setup_workload` et `@compile_workload` sur de
petites matrices déterministes et bien conditionnées. La charge couvre :

- GEMM Float32 et Float64 ;
- LU et Cholesky denses ;
- `OxiBLAS.lu`, `OxiBLAS.cholesky`, `OxiBLAS.ldiv` et `ldiv!` ;
- `\` sur une `CSRMatrix` (Cholesky et LU) ;
- `OxiBLAS.mul` creux × vecteur ;
- deux `einsum`.

À la fin, le pool d'espaces de travail GEMM, seul état global mutable, est
vidé.

**Latence** (`benchmark/latency/`, `1.14+opt`, un processus neuf par chemin).

| Chemin, premier appel | Sans charge | Avec charge |
|---|---|---|
| `OxiBLAS.ldiv(A, b)` | 13,5 s | 0,13 s |
| `\` creux | 7,2 s | 0,005 s |
| `einsum` | 3,2 s | 0,06 s |
| LU dense | 2,85 s | 0,10 s |
| `using OxiBLAS` | 0,03 s | 0,15 s |

**Coût.** La précompilation passe d'environ 3 à 20 s, et le cache d'environ
10 à 50 Mo.

**Désactivation.** Avec `precompile_workload = false` dans
`LocalPreferences.toml`, le paquet se charge et fonctionne normalement.

## Tests exécutés (niveau `core`, 4 threads)

| Julia | Tests | Durée |
|---|---|---|
| 1.10.13 (LTS) | 101 263 / 101 263, QA comprise | 14 min 31 |
| 1.12.7 | 101 241 / 101 241, QA comprise | 21 min 23 |
| 1.13.3 | 101 273 / 101 273, QA comprise | 15 min 11 |
| 1.14.0-alpha2 (`+opt`) | 101 269 / 101 269 ; QA 686/686 après les derniers correctifs | 29 min |

**Documentation.** Documenter construit toutes les pages : les exemples
`@example` sont exécutés, les docstrings des exports sont toutes listées, et
aucun lien interne n'est brisé.

## Limitations et étapes restantes

- **Configuration GitHub.** Une fois le dépôt créé :
  - activer GitHub Pages sur la branche `gh-pages` ;
  - vérifier les liens de sources de la documentation, générés seulement dans
    un dépôt git.
- **Banc haut niveau à refaire.** Il n'a été exécuté qu'en vérification, sur
  une machine chargée. Une mesure de référence doit être faite au repos, sur
  une version stable.
- **Julia 1.14-alpha.** La compilation y est nettement plus lente (par
  exemple `out_of_core.jl` : 26 s en 1.10, 706 s en `1.14+opt`). C'est à
  revoir à la sortie de 1.14.
- **Sondes de `level3_blocked`.** Leur combinatoire coûte en compilation. À
  alléger dans le niveau courant ; les noyaux natifs se vérifieraient par
  empoisonnement `NaN`.
- **Docstrings.** 183 noms exportés n'en ont pas encore ; ils sont listés par
  famille dans la référence.
