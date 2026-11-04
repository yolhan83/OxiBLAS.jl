# Rotations modifiées et produits étendus — v0.4

L'amont étudié est `oxiblas-blas/src/level1/rot.rs` et `dot_extended.rs`, au
commit `8d82bb6`. Les conventions sont confrontées aux sources primaires
[ROTM](https://raw.githubusercontent.com/Reference-LAPACK/lapack/master/BLAS/SRC/drotm.f),
[ROTMG](https://raw.githubusercontent.com/Reference-LAPACK/lapack/master/BLAS/SRC/drotmg.f),
[DSDOT](https://netlib.org/blas/dsdot.f) et [SDSDOT](https://netlib.org/blas/sdsdot.f).

ROTM utilise les quatre formes BLAS de H. Les paramètres immuables sont stockés
comme `(flag,h11,h21,h12,h22)`. Le constructeur à partir d'un vecteur ne lit que
les coefficients définis par le flag. Les paramètres sont capturés avant
mutation ; ils peuvent donc recouvrir x ou y. Les deux sorties d'une opération
non identité doivent rester disjointes, avec des entrées logiques distinctes.

Les vecteurs à stride unité emploient les vecteurs LLVM du GEMM existant :
quatre paquets indépendants par tour, chargement des deux entrées avant écriture,
FMA vectorielles, puis paquets et éléments de reste. L'alignement requis n'est
que celui du scalaire. Les racines Julia des tableaux/vues restent protégées par
`GC.@preserve`. La largeur vient du dispatch ISA déjà contrôlé par le paquet.
NEON est exécuté localement ; aucune validation physique AVX2/AVX-512 n'est
annoncée. Les autres layouts et `kernel=:scalar` suivent les boucles génériques.

ROTMG conserve des paires mantisse/exposant pour d*x, d*x², les rapports et les
coefficients. Cela étend la plage des exposants intermédiaires sans augmenter la
précision des mantisses et sans utiliser BigFloat/MPFR dans le paquet.
Les rapports de q utilisés pour calculer u sont bornés ; les coefficients
potentiellement extrêmes ne sont convertis qu'après normalisation des poids.
Le facteur GAM=4096 donne des poids non nuls entre 2^-24 et 2^24. Les
coefficients de la matrice complète sont explicités une seule fois et restent
préservés à tous les redimensionnements.

Pour les cas admissibles non dégénérés, les tests contrôlent à haute précision :
H*[x1;y1] ≈ [x1_nouveau;0], et Hᵀ*diag(d1_nouveau,d2_nouveau)*H ≈ diag(d1,d2).
Le cas identité avec d2=0 n'annule pas nécessairement la coordonnée y1 non
pondérée, dont le poids est déjà nul. Les cas standard de remise à zéro ne
prétendent pas préserver la forme quadratique indéfinie inadmissible.

Le Rust figé utilise un `else` qui réinitialise des coefficients même après le
passage au flag -1. Les contre-exemples `reports/rust-level1.csv` reproduisent
l'erreur avec plusieurs redimensionnements. Le script Rust est livré ; les
tests vérifient son accord dans les cas ordinaires et sa violation de
l'annulation dans les contre-exemples, puis vérifient le résultat Julia.
La source primaire ROTMG emploie bien `else if flag==1`, sans écraser le cas -1.

Des entrées non finies sont rejetées. Si une sortie ou un coefficient non nul
sort de la plage représentable, une erreur explicite précède toute mise à jour
de l'état mutable. Cela comprend le sous-dépassement d'un coefficient ; la
fonction ne promet pas de résultat utilisable pour tous les quadruplets finis.
Cette limite évite de rendre silencieusement une transformation invalide ou de
boucler sur une échelle infinie.

DSDOT/SDSDOT élargissent chaque Float32 avant multiplication. Les significandes
des produits finis tiennent exactement en Float64. Une accumulation Float64
réassociée (sommes partielles SIMD, comme OpenBLAS) reste sujette à l'arrondi
de la somme, borné en O(n eps(Float64)) Σ|x||y|. SDSDOT ajoute alpha puis
convertit une fois en Float32. Il n'y a ni fast-math ni promesse de sommation
exacte ; un résultat non fini est recalculé dans l'ordre séquentiel.

BigFloat, les références BLAS et l'exécutable Rust sont confinés aux tests et
benchmarks. Le code numérique du paquet ne les appelle jamais.
