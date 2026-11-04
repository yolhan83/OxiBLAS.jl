# BLAS 2 compact et tuiles BLAS 3 — v0.5

Les algorithmes sont examinés dans les treize modules correspondants de
`oxiblas-blas/src/level2/` et les variantes bloquées de `level3/`, au commit
`8d82bb6`. Les conventions de stockage sont également confrontées à
[GBMV](https://netlib.org/blas/dgbmv.f),
[HBMV](https://netlib.org/blas/zhbmv.f) et
[TBSV](https://netlib.org/blas/dtbsv.f).

## Colonnes stockées

Les adaptateurs PackedMatrix/BandedMatrix sont conservés. Leurs kernels dédiés
utilisent directement `data`, des bases de colonnes et des bornes utiles. Ils
ne passent plus par une matrice logique dense. GBMV général coûte O(m+n+nnz)
avec les sorties à mettre à l'échelle ; les opérations structurées en bande
coûtent O(n*(kd+1)). Les opérations packed parcourent O(n*(n+1)/2) coefficients.
Les paddings ne sont pas lus, même si la largeur déclarée dépasse l'ordre.

GBMV N met à jour des plages contiguës par AXPY ; T/C effectuent un produit
scalaire par colonne stockée. Les produits symétriques/hermitiens emploient
chaque coefficient stocké pour ses deux contributions, avec conjugaison du
terme réfléchi en hermitien. La diagonale hermitienne est réelle, y compris
lorsque la partie imaginaire stockée vaut NaN. Les zéros structurels ne sont
pas multipliés ; une bande diagonale ne pollue pas les autres sorties par
0*Inf provenant d'une entrée hors bande.

Les multiplications triangulaires modifient x sur place dans l'ordre qui
préserve les entrées encore nécessaires. N utilise des mises à jour de
colonnes ; T/C utilisent les produits de colonnes stockées. La résolution
suit l'ordre de substitution inverse. Aucun vecteur complet n'est copié sans
alias de stockage. Les pivots non unité sont tous vérifiés avant mutation,
comme dans l'API précédente, contrairement à TBSV BLAS qui ne promet pas ce
contrôle. Les diagonales unité ne sont pas lues, y compris lors du packing.

SPR/SPR2/HPR/HPR2 mettent à jour directement les plages packed. Les facteurs
de colonne sont calculés une fois, les entrées qui recouvrent le stockage sont
capturées avant écriture et la diagonale hermitienne est normalisée. α=0 évite
les lectures des vecteurs ; β=0 initialise les produits sans lire l'ancienne
sortie. Le produit vide conserve la convention du paquet de mise à l'échelle
par β, déjà utilisée en v0.4.

Les AXPY réels à stride unité réutilisent les vecteurs LLVM, quatre paquets
par tour et une gestion explicite des restes. Le dispatch garde les largeurs
sous forme de `Val` littéraux ; construire `Val` depuis un entier dynamique
allouait des boîtes de pointeurs/scalaires à chaque colonne. Les racines des
tableaux/vues sont protégées par GC.@preserve aux frontières des pointeurs.
Les chemins complexes et non contigus emploient les boucles natives. NEON est
exécuté localement ; les formes x86 ne sont pas déclarées validées physiquement.

Un alias entre les coefficients et la sortie impose une copie compacte des
seuls coefficients utiles ; aucune expansion dense n'est requise. Une copie
ignore padding et diagonales implicites. Les vues à indices répétés sont
refusées pour les sorties. Les lectures d'entrées répétées restent permises.

## Tuiles indépendantes

SYRK/HERK/SYR2K/HER2K/GEMMT conservent les formules et le parcours triangulaire
de v0.4. Leur nouveau plan distribue les tuiles disjointes entre tâches lorsque
le volume estimé atteint `config.parallel_threshold`. Chaque tâche garde son
propre couple de panneaux et son tampon diagonal. Son GEMM est monothread :
il n'y a pas de distribution inter-tuiles et de GEMM multithread imbriqués.

Le nombre de tâches est borné par les threads Julia, le nombre de tuiles et
les buffers du workspace fourni. Le workspace parent reste verrouillé pendant
tout le plan ; chaque enfant emploie un wrapper à un buffer et son propre
verrou. Un appel concurrent utilisant le même parent ne peut donc pas écrire
les panneaux en cours d'emploi. La propriété des buffers suit la tâche et ne
dépend pas de threadid(), ce qui autorise leur migration.

Les snapshots d'alias et les validations ont lieu avant la distribution. Les
tuiles diagonales ne recopient que le triangle demandé ; le triangle opposé
reste inchangé. Si aucun plan extérieur parallèle n'est créé, un GEMM assez
grand peut encore utiliser son parallélisme existant. SYMM/HEMM continuent
à bénéficier du partitionnement GEMM déjà présent. Les dépendances des
factorisations et de TRMM/TRSM ne sont pas réordonnées dans cet incrément.

Les tampons diagonaux, contextes de tâches et wrappers de workspace sont encore
créés par appel. Les mesures d'allocations en 1/2/4/8 threads les incluent ; les
panneaux GEMM du benchmark sont préalloués. Le parallélisme compact, les lots
indépendants, les factorisations parallèles et les microkernels complexes
spécialisés restent à poursuivre.
