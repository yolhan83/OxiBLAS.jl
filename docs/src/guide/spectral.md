# Algorithmes spectraux et itératifs

Les décompositions spectrales denses (Schur, hermitien, SVD, QZ) sont décrites
dans [Factorisations denses](factorizations.md). Cette page couvre les méthodes
itératives et partielles, utilisables sur des matrices denses ou creuses.

```@example spec
using OxiBLAS, SparseArrays, LinearAlgebra
n = 200
L = spdiagm(-1 => -ones(n-1), 0 => 2ones(n), 1 => -ones(n-1))   # laplacien 1D
r = lobpcg(CSRMatrix(L); nev=3, which=:smallest)
(r.converged, r.values)
```

## Valeurs propres

| Fonction | Méthode |
|---|---|
| `lanczos(A; nev, which)`, `thick_restart_lanczos` | Lanczos hermitien, redémarrage épais |
| `arnoldi`, `iram`, `thick_restart_arnoldi` | Arnoldi, redémarrage implicite ou épais |
| `block_lanczos`, `block_arnoldi` (et `*_factor`) | variantes par blocs |
| `lobpcg(A; nev, B, which)` | LOBPCG, problème généralisé possible |
| `shift_invert` | transformation spectrale |
| `polynomial_filter`, `apply_filter!`, `filter_response` | filtres de Chebyshev (`ChebyshevFilter`, `FilterWorkspace`) |
| `eigenvalues_in_interval`, `count_eigenvalues_in_interval`, `polynomial_filtered_eigenvalues`, `hermitian_spectral_bounds` | recherche par intervalle |

## Valeurs singulières

| Fonction | Méthode |
|---|---|
| `sparse_svd_lanczos`, `sparse_svd` | SVD partielle creuse (Golub–Kahan par défaut), `SparseSVDWorkspace` |
| `svd_selected(A; indices, interval)`, `svd_truncated`, `svd_randomized` | SVD partielles denses |
| `incremental_svd`, `svd_add_rows!`, `svd_add_columns!`, `svd_snapshot`, `svd_rank` | SVD mise à jour par lignes ou colonnes (`IncrementalSVDFactor`) |

## Estimations stochastiques

`trace_hutchinson`, `trace_hutch_plus_plus`, `trace_xtrace` et
`trace_xtrace_loo` estiment la trace. `stochastic_diagonal`,
`stochastic_frobenius` et `stochastic_logdet` estiment respectivement la
diagonale, la norme de Frobenius et le log-déterminant. Tous sont configurés
par `StochasticConfig(; num_probes, seed, ...)`, avec une graine explicite et
sans état global.

Conception historique : [LOT_SPECTRAL_BLOCS](../dev/LOT_SPECTRAL_BLOCS.md),
[LOT_ARNOLDI_XTRACE](../dev/LOT_ARNOLDI_XTRACE.md),
[LOT_FILTRES_INTERVALLES](../dev/LOT_FILTRES_INTERVALLES.md),
[LOT_SVD_PARTIELLE](../dev/LOT_SVD_PARTIELLE.md),
[LOT_SVD_INCREMENTALE](../dev/LOT_SVD_INCREMENTALE.md) et
[LOT_STOCHASTIQUE_TENSEURS](../dev/LOT_STOCHASTIQUE_TENSEURS.md).
