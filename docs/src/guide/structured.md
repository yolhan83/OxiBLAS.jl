# Matrices structurées

OxiBLAS fournit deux stockages compacts et prend en charge les types
structurés de `LinearAlgebra` par son API haut niveau.

## Stockages compacts et bandes

- **`PackedMatrix(data, n; uplo, kind, unit)`.** Stockage compact BLAS,
  colonne par colonne. `pack_matrix(A; ...)` le construit à partir d'une
  matrice ; `kind` vaut `:symmetric`, `:hermitian` ou `:triangular`.
- **`BandedMatrix(data, m; kl, ku, kind=:general, uplo, unit)`.** Stockage
  bande BLAS, `data[ku+1+i-j, j]` ; `band_matrix(A; ...)` copie seulement la
  bande.

```@example st
using OxiBLAS, LinearAlgebra
A = randn(6, 6); H = A * A' + 6I
P = pack_matrix(H; uplo=:U, kind=:symmetric)
x = randn(6); y = zeros(6)
spmv!(y, P, x)                 # y ← P·x
y ≈ H * x
```

| Opérations | Fonctions |
|---|---|
| produits compacts | `spmv!`, `hpmv!`, `tpmv!`, `tpsv!`, `spr!`, `spr2!`, `hpr!`, `hpr2!` |
| produits bande | `gbmv!`, `sbmv!`, `hbmv!`, `tbmv!`, `tbsv!` |
| factorisations | `band_lu`, `band_cholesky`, `packed_cholesky` (`BandLUFactor`, `BandCholeskyFactor`, `PackedCholeskyFactor`) |

Ces opérations ne parcourent que les coefficients stockés. Elles acceptent
`kernel=:auto` ou les demandes ISA de GEMM ; `:scalar` choisit la boucle
générique.

## Types structurés de LinearAlgebra

`OxiBLAS.lu`, `cholesky`, `ldlt`, `bunchkaufman`, `factorize`, `det`, `inv`,
`eigen`, `svd`… acceptent `Diagonal`, `Bidiagonal`, `Tridiagonal`,
`SymTridiagonal`, les matrices triangulaires et leurs enveloppes
`Hermitian`/`Symmetric`. Le moteur choisi exploite la structure.

| Type | Moteur |
|---|---|
| `Tridiagonal` | `band_lu`, O(n) |
| `SymTridiagonal` réel | `band_cholesky` (`cholesky`), `eigen_tridiagonal` (`eigen`) |
| `Bidiagonal` | substitution (`factorize`), noyau QR bidiagonal (`svd`) |
| `Diagonal` | opérations directes, O(n) |
| triangulaires | `TriangularEngine` : substitution, sans élimination |
| `BandedMatrix` | `band_lu`, `band_cholesky`, multifrontal pour LDLᵀ |
| `PackedMatrix` | `packed_cholesky`, `tpsv!` |

**Les fonctions de LinearAlgebra restent intactes.** Les types structurés de
Julia ne reçoivent que des méthodes `OxiBLAS.*` : `LinearAlgebra.lu(::Tridiagonal)`
reste celle de Julia. Seuls `BandedMatrix` et `PackedMatrix`, types d'OxiBLAS,
étendent les fonctions de `LinearAlgebra`.

## Limitations

- **Aucun moteur structuré** pour QR, LQ, Schur, Hessenberg et les spectres
  non symétriques des types bande et tridiagonaux. Ces cas renvoient une
  `MethodError`, sans conversion dense cachée.
- **`PackedMatrix` indéfinie :** pas de Bunch–Kaufman compact. `factorize`
  lève une `ArgumentError`.
- **Pas de `lu!` sur les stockages bande :** il faudrait un tampon pour le
  remplissage du pivotage.

Conception historique :
[COMPACT_PARALLEL_DESIGN](../dev/COMPACT_PARALLEL_DESIGN.md).
