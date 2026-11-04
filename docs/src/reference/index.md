# Référence de l'API

Une page par famille de sources (`src/<famille>/`). Chaque page liste :

- les docstrings des noms exportés de la famille ;
- pour l'API haut niveau, les fonctions non exportées (`OxiBLAS.lu`,
  `OxiBLAS.ldiv`…), qui s'appellent avec le préfixe `OxiBLAS.` ;
- les noms exportés qui n'ont pas encore de docstring : leur convention est
  décrite dans les guides.

Ces pages sont générées par `docs/make.jl` à partir des docstrings et du
fichier source de chaque méthode.
