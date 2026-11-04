# SVD creuse, contextes réutilisables et feuilles triangulaires

> Mise à jour du 2 novembre 2026 : les tests enregistrés de ce lot passent à 1/2/4/8 threads, bornes activées. Voir [résultats et limites](VALIDATION_CORRECTIONS_20261102.md). Les déclarations « non exécuté » ci-dessous décrivent la préparation antérieure ; benchmarks et fournisseurs optionnels restent non validés.

Lot écrit le 2 novembre 2026, **sans chargement Julia, test, benchmark,
compilation ou inspection LLVM**. Aucun gain mesuré ni correction numérique
validée n'est revendiqué. Rust reste en lecture seule au commit
`8d82bb6bf1df133aa90525aa9f478ebe41514f9b`.

## SVD creuse à mémoire bornée

L'ancien moteur utilisait la dilatation hermitienne [0 A; Aᴴ 0] avec une base
et toutes ses images, chacune de taille (m+n)×s. Le résultat retenait aussi le
facteur Krylov complet. La variante demeure explicitement accessible par
`sparse_svd_lanczos(A,k; algorithm=:dilation)`, avec son ancien contrat.
Le benchmark historique `:sparse_svd_dilation` la demande maintenant explicitement.

Le nouveau défaut `algorithm=:golub_kahan`, dans `src/sparse_svd_golub_kahan.jl`,
emploie des bases séparées U et V :

- U a au plus s colonnes et V au plus min(n,s+max(b,k)), pour une largeur de bloc b et un rang k ;
- les seuls produits opérateur sont A·bloc et Aᴴ·bloc, sans AᴴA/AAᴴ explicite
  ou implicite et sans vecteurs de longueur m+n ;
- pivotage et réorthogonalisation double des blocs, avec réduction de rang ;
- petit problème projeté UᴴAV, traité par notre SVD bidiagonale native. Après
  redémarrage la projection est complète : elle n'est pas faussement supposée
  strictement bidiagonale ;
- redémarrage épais depuis les directions de Ritz droites conservées. La marge droite max(b,k) préserve assez de place après un redémarrage épais. La base
  droite est comprimée par tranches de lignes en scratch borné. La récurrence
  gauche est reconstruite, avec produits recomptés dans les compteurs, plutôt
  que conserver A·V et Aᴴ·U ou deux grandes copies de redémarrage ;
- démarrages supplémentaires orthogonalisés lorsque l'espace devient invariant
  avant d'avoir suffisamment de directions, notamment pour les multiplicités ;
- vérification des deux vraies équations A·v=σu et Aᴴ·u=σv, par blocs de scratch ;
- les deux bases et leurs images ne sont jamais conservées simultanément.

Le paramètre `subspace` est une capacité gauche pour Golub–Kahan, plafonnée à
min(m,n) ; pour la dilatation c'est l'ancienne dimension d'espace hermitien.
Les dimensions effectives `left_basis_dimension` / `right_basis_dimension`
sont exposées. Les deux conventions ne doivent pas être comparées comme des
capacités identiques sans regarder les octets réellement réservés.

La lecture des corps Rust `linalg/svd/types.rs::TruncatedSVD::compute_via_ata` et
`compute_via_aat` confirme la construction de matrices normales creuses dans
cette révision. Ce n'est pas adapté à la demande de mémoire courante : le fill
peut devenir important et le conditionnement est mis au carré. Le chemin Julia
conserve le contrat de triplets itératifs, mais adopte explicitement la réduction
bilatérale pour éviter ces coûts. Aucune modification Rust, aucun wrapper externe.

Le produit CSR normal est réparti par propriétaires de lignes, même pour un ou
deux seconds membres. Le produit adjoint conserve le chemin sparse existant,
avec propriétaires de colonnes de sortie : pas de scatter concurrent, ni plan
transposé persistant O(nnz) ajouté au workspace.

Les valeurs numériquement nulles non résolues ne sont pas complétées artificiellement.
Le résultat peut contenir moins de k triplets avec `converged=false`. Un résidu
petit ne certifie pas que des directions extrémales absentes du bloc de départ
ont toutes été trouvées. Le budget de redémarrages reste explicite.

## Workspaces et propriété des résultats

`SparseSVDWorkspace(A,k; subspace,blocksize,rowblock)` réserve une fois :

- bases U/V, projection et scratch de bloc gauche/droit ;
- coefficients, QR de bloc et permutation ;
- buffers de Ritz et de résidus ;
- un petit buffer de compression de lignes et une paire de panneaux GEMM.

`workspace=w` réutilise ces stockages pour le même type/forme, avec capacités
suffisantes. Les valeurs de A peuvent changer entre appels : aucun Gram ni image
opérateur périmé n'est caché. CSR est conservé ; CSC/COO sont canonisés une fois
par appel, pas à chaque produit. Le contexte ne met pas en cache la structure
ou une transposée susceptible de devenir périmée.

Type, capacités réelles, alias entre buffers, alias des valeurs sparse et du
bloc initial sont contrôlés avant utilisation. Un contexte partagé est sérialisé,
une réentrance est refusée, des contextes distincts restent indépendants.
Les résultats U/S/Vt sont des copies possédées : un appel ultérieur ne les modifie
pas. Par défaut `krylov=nothing` ; `return_basis=true` demande explicitement des
copies des bases et de la projection, avec leur coût mémoire supplémentaire.

`sparse_svd_workspace_bytes(w)` rend les octets des buffers numériques possédés,
panneaux compris. Ce n'est **ni le RSS maximal ni la totalité des allocations** :
le sparse d'entrée, les sorties, les headers/verrous/tâches et les petites
factorisations projetées en sont exclus. La SVD projetée et les métadonnées de
cycle allouent encore ; aucune promesse « zéro allocation » n'est faite.

`triangular_workspace(T,m,n; side,config,workers)` construit un contexte natif
réutilisable TRMM/TRSM adapté aux demi-blocs récursifs. C'est un `GemmWorkspace`
compatible avec l'API existante. Les appels avec ce contexte sont protégés de la
vérification d'alias jusqu'à la fin de l'opération, feuilles comprises ; les
workers utilisent les sous-contextes possédés existants. L'alpha nul et les
petites tailles ne contournent plus la vérification d'alias des buffers fournis.
Employer la même configuration lors de la construction et de l'opération.

## Microkernels triangulaires pertinents

La lecture de `oxiblas-blas/src/level3/trsm/generic.rs`, des feuilles complexes
et du dispatch TRMM montre des substitutions scalaires dans les feuilles,
avec mises à jour GEMM des grands blocs. Julia conserve cette stratégie et ses
mises à jour récursives, sans développer A en une matrice triangulaire dense.

`src/triangular_microkernels.jl` ajoute :

- côté gauche, traitement de quatre seconds membres en registres : chaque
  coefficient Aᵢₚ est chargé une fois pour les quatre accumulateurs ; queues
  traitées par la feuille générique ;
- côté droit, FMA LLVM explicites sur les lignes contiguës de B pour les mises
  à jour de substitution TRSM et les accumulations TRMM, réels et complexes ;
- arithmétique complexe entrelacée 4M, sans reconstruction par trois grands
  produits réels ;
- division diagonale Scalaire stable, sans remplacer Aᵢᵢ par un inverse qui
  pourrait déborder ou sous-déborder ;
- sélection d'ISA selon la cible Julia existante ; pas d'AVX forcé sur un CPU
  ou une cible de compilation qui ne le supporte pas.

Les pointeurs sont strictement limités aux matrices stridées reconnues par le
guard `_can_pointer`, avec lignes unitaires et leading dimension positif adapté.
Les pointeurs sont dérivés dans `GC.@preserve B`. Les queues ne lisent aucun
padding et les loads n'exigent que l'alignement naturel des composantes.
Les vues négatives/non compatibles prennent la feuille générique appropriée.

Le triangle inutilisé et la diagonale implicite ne sont pas lus. La vérification
d'une diagonale singulière TRSM reste antérieure à la mutation de B. Un A aliasé
est copié logiquement comme auparavant ; B n'est pas recopié intégralement.
`algorithm=:reference` sélectionne explicitement les anciennes feuilles.
Le chemin AbstractFloat/Complex générique, BigFloat et fournisseurs optionnels
est conservé ; Float16 garde son élargissement Float32. Aucun pointeur ou SIMD
n'est appliqué à un scalaire générique. Aucune dépendance numérique externe et
aucun type piracy ne sont introduits.

## Tests et mesures préparés

`test/sparse_memory_triangular.jl`, ajouté à la suite supervisée, prépare :

- SVD creuse haute/large réelle/complexe, blocs 1/2/3, références LAPACK,
  projection, deux résidus, budgets et comparaison au fallback dilatation ;
- réutilisation avec valeurs sparse modifiées, identité des buffers, sorties
  stables après réutilisation, appels concurrents, GC, capacité/type/alias invalides ;
- croissance linéaire du payload à capacités constantes et absence des deux
  grandes bases d'images de la dilatation ;
- TRMM/TRSM gauche/droite, U/L, N/T/C, diagonales explicites/implicites,
  feuilles à quatre RHS, SIMD et queues, vues denses/paddées/stridées/négatives ;
- accès interdits au triangle et à la diagonale implicite via une fixture qui
  lève une erreur, alias A/B, contexte partagé, buffers aliasés et singularité ;
- fallback générique et précisions optionnelles contre références indépendantes.

`benchmark/full_memory_triangular_cases.jl` ajoute neuf cas au rapport unique :
GK frais/réutilisé, dilatation, feuilles TRMM/TRSM gauche/droite et contextes
triangulaires réutilisés. Pour les feuilles, les symboles TRMM/TRSM du backend
réel sont vérifiés, puis OpenBLAS/Accelerate sont mesurés en processus séparés.
Les SVD creuses sans adaptateur équivalent ne sont pas comparées à une SVD dense
présentée comme le même calcul. Les quatre colonnes demeurent visibles, N/A inclus.

`benchmark/sparse_svd_memory.jl` prépare des opérateurs diagonaux creux de 256 à
16 384 directions, rectangles 4:1 et 1:4, sans fixture dense. Le même spectre
analytique vérifie les trois variantes. Il relève allocations Julia, mémoire
retenue du résultat, payload du workspace, produits opérateur et résidus, avec
colonnes Julia/Rust/OpenBLAS/Accelerate (références non équivalentes N/A).
La campagne 1/2/4/8 threads est intégrée à `run_all.sh`, avec ses artefacts
temporaires rassemblés dans le rapport unique et le gate numérique inchangé.
Le RSS maximal et les performances matérielles ne sont pas déduits de ces
formules ou des compteurs : il faudra les mesures autorisées.
