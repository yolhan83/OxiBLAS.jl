# BLAS 3 structuré — conception de la v0.3

L'amont examiné est le commit `8d82bb6` :
`oxiblas-blas/src/level3/{symm,hemm,trmm,syrk,syr2k,herk,her2k,gemmt}.rs`
et `trsm/generic.rs`. Ses grandes voies SYMM/TRMM/rank-k/GEMMT utilisent des
expansions ou résultats denses. TRSM possède des variantes bloquées/récursives.
Le port Julia conserve les contrats mathématiques et emploie son propre GEMM.
Les résultats Rust ne sont pas utilisés comme preuve de correction ; les tests
comparent les produits, résidus et solutions à Julia BLAS/LAPACK.

SYMM/HEMM présentent un opérande indexable structuré à l'empaquetage GEMM.
Les entrées symétriques réfléchies ou hermitiennes conjuguées sont produites
pendant la copie vers les panneaux. Il n'y a pas d'expansion carrée préalable.
La diagonale hermitienne fournit seulement sa partie réelle.

TRMM partage A et B en deux blocs, puis respecte l'ordre des dépendances
pour modifier B sur place. Par exemple, à gauche avec A supérieure : multiplier
B₁ par A₁₁, ajouter A₁₂ B₂ avec GEMM, puis multiplier B₂ par A₂₂.
Le bloc B₂ d'origine reste disponible pour la mise à jour. Les feuilles
parcourent les lignes/colonnes dans le même ordre de dépendance. Aucune copie
complète de B n'est nécessaire. Les feuilles par défaut sont limitées à huit
lignes/colonnes après comparaison locale des seuils 8/16/32/64 ; le réglage
reste manuel et modifiable par `blocksize`.

TRSM résout un bloc diagonal, met à jour l'autre bloc avec GEMM, puis le résout.
Aucune inversion explicite n'est formée. La diagonale non unité est contrôlée
avant toute écriture. À droite, les feuilles mettent à jour des colonnes
contiguës plutôt que de résoudre successivement des vecteurs à grands strides.

SYRK/SYR2K/HERK/HER2K et GEMMT parcourent les tuiles du seul triangle demandé.
Les tuiles strictement hors diagonale sont écrites directement par GEMM.
Une tuile diagonale temporaire évite de toucher l'autre triangle. Le tampon est
réutilisé, sa taille est au plus `blocksize²` éléments (128² par défaut).
Les mises à jour hermitiennes ignorent l'ancienne partie imaginaire diagonale
et rendent une diagonale exactement réelle. Les petites sorties n≤16
emploient directement une boucle triangulaire, sans tampon de produit.

Chaque appel crée ou réutilise un workspace GEMM pour tous ses sous-problèmes.
Les panneaux suivent les tâches et restent protégés par le verrou GEMM ; les
sous-opérations s'exécutent dans l'ordre des dépendances. Les mises à jour assez
grandes peuvent répartir GEMM entre threads. Les alias d'entrée/sortie imposent
des snapshots avant mutation ; ce coût quadratique éventuel est nécessaire à
la sémantique d'alias, et ne constitue pas le chemin sans alias.

Les triangles non sélectionnés et diagonales unité ne sont jamais lus dans
les opérations triangulaires. α=0 et K=0 évitent les lectures d'opérandes ;
β=0 évite les anciennes sorties. Les tests utilisent des matrices qui lèvent
une erreur sur ces lectures, des canaris, des vues inversées, des données
complexes avec N/T/C distincts, des alias et des workspaces partagés.

Les blocs restent réglables manuellement. La parallélisation des tuiles entre
elles, l'autotuning, les noyaux triangulaires SIMD spécialisés et la validation
physique x86 restent à réaliser. Les mesures locales et leurs limites figurent
dans [BLAS3_VALIDATION.md](BLAS3_VALIDATION.md).
