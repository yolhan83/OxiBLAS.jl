# QZ / Schur — port algorithmique et complétion

> Mise à jour du 2 novembre 2026 : les tests enregistrés de ce lot passent à 1/2/4/8 threads, bornes activées. Voir [résultats et limites](VALIDATION_CORRECTIONS_20261102.md). Les déclarations « non exécuté » ci-dessous décrivent la préparation antérieure ; benchmarks et fournisseurs optionnels restent non validés.

Lot écrit le 2 novembre 2026. **Aucun chargement Julia, test, benchmark,
compilation Rust ni inspection LLVM exécuté.** Les propriétés décrites ci-dessous
sont les contrats du code et des tests préparés, pas des résultats mesurés.
Le dépôt Rust de référence reste en lecture seule, HEAD
`8d82bb6bf1df133aa90525aa9f478ebe41514f9b`, sans modification suivie.

## Sources réellement étudiées

Les corps numériques de `crates/oxiblas-lapack/src/evd/schur.rs`,
`complex_schur.rs` et `qz.rs` ont été lus : réductions, itérations, rotations,
réflecteurs, finitions 2×2, extraction, TREVC droite/gauche, TRSNA-S/SEP.
Il ne s'agit pas d'un rapprochement des seules signatures publiques.

| Rust au commit de référence | Julia | Stratégie portée / complétée |
| --- | --- | --- |
| `schur.rs::compute_hessenberg_qr`, `francis_qr_step`, `householder_3` | `src/schur.jl` | Hessenberg orthogonal, polynôme de Francis à deux shifts, réflecteurs de longueur trois puis deux, chasse de bulle, déflation 1×1/2×2 et budget explicite |
| `schur.rs::compute_2x2`, `extract_eigenvalues` | `src/schur.jl` | Vecteur propre local, complétion orthogonale, blocs conjugués réels conservés et canonisés ; extraction depuis la forme produite |
| `complex_schur.rs::compute_wilkinson_shift`, `complex_qr_step`, `compute_2x2` | `src/schur.jl` | Shift de Wilkinson, Givens complexes unitaires, chasse implicite, shift exceptionnel après stagnation, finition analytique 2×2 |
| `qz.rs::qr_factorize`, `hessenberg_qz`, `qz_iterations`, `qz_step`, `extract_eigenvalues` | `src/qz.jl`, `src/qz_real.jl` | QR initial de B, réduction Hessenberg/triangulaire, équivalences gauche/droite distinctes, QZ réel double shift, restauration RQ de B, blocs réels et paires homogènes ; corrections détaillées ci-dessous |
| `schur.rs::trevc_right`, `trevc_left` | `src/schur_blocks.jl` | Substitutions arrière/avant par vrais blocs 1×1/2×2, perturbation des petits pivots, normalisation conjointe des colonnes d'une paire |
| `schur.rs::trsna_s`, `trsna_sep` | `src/schur_blocks.jl` | Produits gauche/droite normalisés ; estimation de distance spectrale conservée sous son contrat réel, sans la présenter comme une séparation par valeurs singulières |
| Pas d'implémentation correspondante dans cette révision Rust | `src/schur_multishift.jl` | Train simultané de petites bulles réel/complexe, AED par blocs, accumulation des transformations dans des panneaux et GEMM natif pour les mises à jour éloignées |
| Pas d'implémentation correspondante dans cette révision Rust | `src/schur_blocks.jl` | Réordonnancement Schur réel et QZ généralisé, échanges 1/1, 1/2, 2/1, 2/2, Sylvester couplé et conditions des crayons/sous-espaces |

Les variantes avancées absentes de cette révision Rust ont donc été ajoutées
côté Julia ; elles ne sont pas attribuées fictivement à l'amont.

## Corrections démontrables, uniquement côté Julia

1. **Réduction QZ amont, `hessenberg_qz`, lignes 421–524.** Le réflecteur P
   est appliqué à H à gauche et à droite, à Q et Z, mais à R seulement à droite.
   Partant de B=I dans cette étape, cela donne Q=P, R=P et Z=P, donc
   Q·R·Zᵀ=P, généralement différent de I. R n'est pas non plus maintenu
   triangulaire. Julia applique chaque transformation gauche aux deux matrices,
   puis une transformation droite indépendante pour annuler le nouvel élément
   sous-diagonal de B. Le test de réduction vérifie les deux reconstructions.
2. **`qz_step`, lignes 591–681.** Le commentaire annonce un double shift,
   mais le code utilise seulement H22, sans B, puis la même rotation de chaque
   côté. Le crayon triangulaire n'est pas préservé. Julia utilise un vrai
   polynôme homogène du bloc terminal du crayon et restaure B par des rotations
   RQ locales après chaque réflecteur gauche de longueur trois/deux.
   Aucune matrice inverse de B n'est formée, et le problème n'est pas remplacé
   par une diagonalisation de B⁻¹A.
3. **`extract_eigenvalues`, lignes 710–761.** Pour
   S=[0 −1; 1 0], T=diag(2,3), le polynôme est 1+6λ².
   Les racines sont ±i/√6 ; l'extraction Rust donne i/2 et −i/3.
   Julia utilise det(S−λT), y compris T12. Le cas
   S=[1 2; −3 4], T=[2 5; 0 3] vérifie aussi le coefficient −26λ
   de 6λ²−26λ+10, absent du discriminant de S seul.
4. **TREVC amont.** Lorsqu'une substitution traverse un autre bloc réel 2×2,
   les récurrences scalaires ne résolvent pas son système couplé. Julia utilise
   systématiquement les blocs, même entre deux paires conjuguées. Le vecteur
   local est construit par un noyau nul de la ligne la plus longue, évitant
   aussi l'erreur de signe de `trevc_right` dans la branche qui utilise A12
   pour la partie imaginaire. La substitution gauche résout le crayon adjoint.
5. **`trsna_sep`.** Le corps Rust estime une distance entre valeurs propres,
   malgré une description évoquant σmin. Il ignore les distances nulles et
   peut donc annoncer une grande séparation pour deux valeurs répétées.
   Julia garde cette famille d'estimateur, retourne zéro dans ce cas et Inf
   lorsqu'aucun bloc complémentaire n'existe. La vraie condition de sous-espace
   est exposée séparément par le solveur de Sylvester couplé et son estimateur.
6. Les tests absolus `epsilon` / `epsilon*(diag+1)` et sommes de carrés non
   échelonnées de l'amont sont remplacés par des normes échelonnées, des shifts
   équivalents calculés sans annulation inutile, et une déflation relative à
   deux critères. Ces changements visent l'invariance d'échelle, sans changer
   les familles Francis/Wilkinson/QZ/TREVC. Les valeurs propres sont extraites
   après les transformations ; elles ne sont pas conservées dans un ordre
   analytique susceptible de diverger de celui de la forme triangulaire.

Ces démonstrations sont algébriques. Aucun résultat de test ou performance
n'est inventé à partir d'elles, et aucun fichier Rust n'a été corrigé.

## Contrats et variantes livrés

- `qz_factor(A,B)` renvoie une forme **réelle** pour les entrées réelles :
  S quasi triangulaire et T triangulaire. Les entrées complexes, ou
  `form=:complex`, donnent deux matrices triangulaires complexes.
  `QZFactor{T,C}` sépare le type des matrices et celui des paires alpha/beta.
  Un beta nul représente l'infini ; alpha=beta=0 représente l'indétermination.
  Les petits beta non nuls ne sont pas remplacés arbitrairement par zéro.
- `qz_reorder` et `schur_reorder` conservent les blocs réels : sélectionner un
  membre sélectionne toute sa paire. Chaque échange résout un petit Sylvester,
  construit des bases orthogonales/unitaires et contrôle les erreurs faibles
  et fortes avant d'appliquer la transformation à la copie du facteur.
  Un échange instable/inséparable est signalé. Les entrées restent intactes.
- `generalized_sylvester(A,D,B,E,C,F)` résout
  A·R−L·B=scale·C et D·R−L·E=scale·F.
  Il parcourt les blocs triangulaires et factorise au plus un système 8×8 par
  paire de blocs, avec redimensionnement du second membre avant division par
  les petits pivots pour éviter les dépassements. Le `scale` rendu fait partie du contrat numérique.
- `qz_condition(F; split)` fournit les conditions des valeurs propres homogènes,
  normalisées par le champ `normalization`, les conditions de projection gauche
  et droite et une estimation de séparation par norme 1 de l'inverse couplé.
  Le système adjoint est traité par substitution de blocs, sans matrice de
  Kronecker globale O(n⁴). Un `split` à l'intérieur d'une paire réelle est refusé.
  `schur_condition` expose aussi TRSNA-S et la distance spectrale.
- `trevc_right/left` conservent le stockage réel/imaginaire apparié du Rust
  pour une entrée réelle ; `packed=false` renvoie une colonne complexe par
  valeur propre. `schur_eigenvectors` rétrotransforme par Z.
  `eigen_general` emploie maintenant la forme réelle et TREVC par blocs pour
  le réel ; `eigen_pencil(...; left=true)` fournit aussi les vecteurs gauches.
- `schur_multishift` conserve désormais le type réel de l'entrée. Les shifts
  réels sont groupés sans couper de paire conjuguée. Les bulles avancent
  simultanément, avec espacement trois pour les doubles shifts réels et deux
  pour les shifts complexes. Les réflecteurs sont accumulés par panneau ;
  H et Z éloignés sont mis à jour par notre GEMM. Ce n'est plus une succession
  de QR complets mono-shift appelée « multishift ».
- AED teste la norme du spike sur tout le bloc réel et tente les échanges pour
  déflater des candidats supplémentaires. Un échange refusé laisse une forme
  valide et interrompt seulement cette tentative de déplacement.
- `SchurWorkspace(T,n; blocksize,nshifts)` réutilise les buffers des panneaux et
  des mises à jour éloignées. Capacités, types et alias sont contrôlés. Une
  utilisation concurrente du même contexte est sérialisée ; la réentrance est
  refusée. Les contextes distincts restent indépendants. Les GEMM éloignés
  peuvent se paralléliser ; la chasse dépendante est séquentielle, sans tâches
  imbriquées. Les types non déclarés thread-safe restent séquentiels.
- Float16 est élargi une fois en Float32 pour ces calculs, complexes compris.
  BigFloat et les fournisseurs AbstractFloat passent par les mêmes algorithmes
  paramétriques. Le noyau matériel GEMM existant conserve son dispatch natif.

Les allocations de réduction Hessenberg, de fenêtre AED et de petits systèmes
Sylvester subsistent : `SchurWorkspace` ne promet pas une factorisation entière
sans allocation. Aucun optimum de blocage ni gain de vitesse n'est revendiqué.
Les diagnostics sont des estimations, pas des bornes certifiées ; des valeurs
multiples/défectives peuvent exiger les perturbations usuelles de TREVC.
Si un coefficient du déterminant 2×2 sous-déborde malgré la normalisation,
le code signale le besoin d'un type plus large au lieu d'inventer un infini.

## Validation préparée et intégration au rapport unique

`test/qz_schur_complete.jl`, inclus par `test/runtests.jl`, prépare :

- reconstructions des deux matrices, orthogonalité/unitarité, structure et
  résidus propres homogènes droits/gauches ; références LAPACK indépendantes ;
- contre-exemples algébriques ci-dessus, infinis/indéterminations, budgets nuls,
  erreurs de dimensions et données non finies ;
- échanges de tous les formats 1/1, 1/2, 2/1, 2/2, réalité conservée,
  sélection d'un seul membre d'une paire, entrées intactes ;
- TREVC traversant plusieurs blocs, normalisation commune et répétitions ;
- Sylvester et opérateur adjoint contre un petit Kronecker construit **dans le
  test seulement**, conditions et séparation, refus d'une coupure de paire ;
- train de bulles isolé avec panneaux 1/5/13 et 1/2/6 shifts, puis solveur complet
  avec petite fenêtre AED pour forcer réellement l'utilisation du train ;
- vues à strides négatifs, échelles extrêmes, débordement évité dans Sylvester, BigFloat 192/256 bits,
  fournisseurs optionnels et élargissement Half, GC et appels
  concurrents partageant le contexte. La campagne fournit 1/2/4/8 threads.

`benchmark/full_qz_schur_cases.jl` ajoute douze opérations au catalogue existant :
réordonnancements Schur/QZ, TREVC droite/gauche, TRSNA-S/SEP, conditions Schur/QZ,
Sylvester, multishift avec workspace, train isolé et QZ complexe explicite.
Les références TREVC/ORDSCHUR utilisent les **mêmes facteurs préconstruits** ;
les symboles TREVC/TRSEN/TGSEN/GEES/GGES sont vérifiés dans chaque processus.
Les trois références restent dans le tableau standard à quatre colonnes, avec
N/A lorsque l'adaptateur équivalent ou le symbole n'est pas disponible.
Les adaptateurs Rust ne sont pas modifiés ; les nouvelles opérations sans
adaptateur chronométré Rust portent N/A, jamais « fonctionnalité Rust absente ».

`run_all.sh` garde l'unique sortie, le gate numérique avant chronométrage, les
allocations Julia, les rapports aux trois références, OpenBLAS/Accelerate isolés
et les demandes 1/2/4/8 threads. Le nombre effectif de threads Accelerate reste
explicitement non garanti. Tout lancement reste soumis à autorisation.

Contrôles statiques effectués : délimiteurs textuels des fichiers modifiés,
68 inclusions sources et 47 inclusions de tests résolues, exports et catalogue
de benchmarks présents, AST Python du rapport et largeurs CSV cohérents,
`bash -n run_all.sh`, empreinte de l'archive v0.5 préservée et dépôt Rust propre.
Ce ne sont ni un parsing Julia ni une preuve de correction numérique.
