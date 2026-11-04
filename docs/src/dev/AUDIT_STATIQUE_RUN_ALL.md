# Audit statique du lanceur unique — 2026-11-02

Préparation uniquement. Vérifications réalisées sur les textes des fichiers :

- Syntaxe Bash vérifiée avec bash -n run_all.sh.
- Syntaxe Python vérifiée avec ast.parse, sans importer/exécuter l’orchestrateur.
- Délimiteurs lexicaux équilibrés dans les sept nouveaux fichiers Julia/Rust.
  Ce contrôle n’est ni un parseur Julia/Rust ni une compilation.
- 44 cibles du lanceur de tests existantes ; inventaire relu à l’exécution future.
- 16 scripts de timing et huit sources Rust classifiés ; chemins présents.
- 199 entrées (191 opérations distinctes), sans doublon de clé famille/opération.
- Rapports historiques non redirigés vers des chemins d’écriture fixes lors du
  lancement global ; copies temporaires distinctes pour chaque comparaison compacte.
- Archive v0.5 : SHA256 inchangé
  8978f5348aad68fd19fe86f7da68b9a6329e32ebca67825f05ae98fc48d44df6.

Aucun script du projet, chargement Julia, installation, parser Julia,
précompilation, build Rust, test, benchmark ou inspecteur LLVM n’a été exécuté.
Les API, symboles disponibles, compteurs, temps et compatibilités restent à
vérifier dans une campagne autorisée. Le portage global reste partiel.

## Lot tenseurs suivant

Après intégration des chemins/tranches : 45 includes de tests existants,
66 includes source existants, 206 entrées par famille / 198 noms distincts.
Délimiteurs lexicaux de six fichiers Julia contrôlés, syntaxe Python vérifiée
par AST ; aucune compilation ni exécution. Archive v0.5 inchangée.
