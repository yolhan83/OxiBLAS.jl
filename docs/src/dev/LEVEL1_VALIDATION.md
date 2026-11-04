# Validation BLAS 1 complémentaire v0.4 — 2 novembre 2026

ROTM/ROTMG réels et produits scalaires étendus sont implémentés en Julia. Les formes SIMD réutilisent les primitives LLVM du paquet ; aucune délégation numérique BLAS/LAPACK/Rust n’est ajoutée dans src/.

## Tests sur les sources finales

| Configuration locale Apple M5 Max arm64 | Réussis |
|---|---:|
| Julia 1.14-alpha2, natif, 4 threads | 27811 |
| Julia 1.14-alpha2, CPU générique, 1 thread | 27811 |
| Julia 1.13.3, natif, 4 threads | 27811 |

Bornes vérifiées. Chaque processus dispose d’un dépôt de cache primaire isolé ; les dépôts existants servent de repli en lecture. Journaux `reports/tests-v04-final-*.log`, manifeste source `reports/baseline-v0.4/source-sha256.txt` et manifeste d’artefacts `reports/baseline-v0.4/artifact-sha256.txt`. Les tests v0.3 GEMM/BLAS/LAPACK/sparse sont conservés dans la suite complète.

Les nouveaux tests vérifient les deux précisions réelles pertinentes, les formes de H, les paramètres implicites non lus, les vues inversées, les canaris, les alias de paramètres, le refus d’alias entre sorties, les restes SIMD et la protection des tableaux sous GC. DSDOT/SDSDOT sont comparés à BLAS, à une accumulation indépendante ordonnée et à BigFloat. ROTMG est contrôlé par annulation et congruence de la forme quadratique à 512 bits, sur poids extrêmes, subnormaux, redimensionnements multiples et formes indéfinies admissibles. Les dépassements de plage sont explicites et laissent l’état mutable intact.

## Mesures de temps et allocations

152 lignes avec OpenBLAS et 76 avec Accelerate. Tailles vectorielles 32/4096/1 000 000, strides 1/3/-2, trois formes ROTM, un thread, jusqu’à onze échantillons. Les symboles de référence sont associés à leurs vrais fournisseurs dans les environnements enregistrés ; ils proviennent bien d’OpenBLAS puis d’Accelerate. Les sorties sont restaurées hors mesure. Les résultats lus sont rendus observables au compilateur.

ROTM contigu, flag 0, N=1 000 000, médianes en millisecondes :

| Type | Boucle avant SIMD | Native SIMD | OpenBLAS | Accelerate | Gain local avant/après |
|---|---:|---:|---:|---:|---:|
| Float32 | 0.265 | 0.107 | 0.100 | 0.038 | 2.47× |
| Float64 | 0.292 | 0.223 | 0.207 | 0.103 | 1.31× |

Produits étendus contigus, N=1 000 000, médianes en millisecondes :

| Opération | Native | OpenBLAS | Accelerate |
|---|---:|---:|---:|
| DSDOT | 0.715 | 0.151 | 0.215 |
| SDSDOT | 0.718 | 0.152 | 0.226 |

Les appels natifs de cette campagne allouent zéro octet selon BenchmarkTools. Les buffers d’entrée/sortie sont préparés hors mesure. La voie ordonnée des produits étendus reste plus lente que les références BLAS mesurées ; la précision étendue ne garantit pas une sommation exacte. Les vues non contiguës utilisent les boucles génériques. Les nombres sont des observations locales de courte durée, sans extrapolation à d’autres architectures.

ROTMG mesure des lots de 256 entrées variées ; les CSV indiquent le temps du lot, et non celui d’un seul scalaire. Les sorties de chaque entrée sont conservées afin d’éviter le repli de constantes et les durées sous la résolution de l’horloge. La génération par exposants protégés coûte davantage que la référence sur les cas ordinaires ; son objectif est de préserver les cas extrêmes et de signaler les résultats non représentables.

## Contrôle du Rust et de l’assembleur

Le programme Rust figé a été compilé et exécuté : `reports/rust-level1.csv`, `reports/rust-level1-reference.log`, `benchmark/rust_level1_reference.rs`. Il concorde sur les cas ordinaires. Les deux contre-exemples de redimensionnement violent l’annulation avec le Rust et la respectent avec Julia. Les tests enregistrent explicitement cette divergence ; aucune valeur incorrecte du Rust n’est adoptée comme cible.

Les six noyaux paquets Float32/Float64 et flags -1/0/1 inspectés contiennent chacun 14 instructions vectorielles FMLA et aucun appel de fonction. LLVM, assembleur et comptages sont conservés dans `reports/level1-assembly-*` et `reports/rotm-packet-*`. Le comptage inclut les boucles déroulées, les restes et les épilogues. Les racines restent protégées en dehors du noyau.

## Provenance et limites

La source et la campagne avant SIMD sont archivées dans `reports/baseline-level1-generic/` et vérifiées contre le hash mesuré. Les hashes et résultats v0.3 restent dans `reports/baseline-v0.3/`. Les sources GEMM et BLAS 3 sont inchangées : les campagnes historiques ne sont pas présentées comme de nouvelles mesures de v0.4.

Les entrées ROTMG doivent être finies. Si une sortie ou un coefficient non nul déborde ou sous-déborde de la plage représentable, une erreur est retournée avant mutation. La validation physique x86 et la CI distante restent à faire. Les variantes parallèles BLAS 1, les noyaux banded/packed BLAS 2 et le tuning des produits étendus restent à poursuivre.

Voir [LEVEL1_DESIGN.md](LEVEL1_DESIGN.md), [API.md](API.md), [PERFORMANCE.md](PERFORMANCE.md) et [STATUS.md](STATUS.md). Les conventions primaires et les écarts avec l’amont sont référencés dans le document de conception.
