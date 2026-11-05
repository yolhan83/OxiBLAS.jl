# Structured matrices

OxiBLAS provides two compact storages and supports the structured types of
`LinearAlgebra` through its high-level API.

## Packed and banded storage

- **`PackedMatrix(data, n; uplo, kind, unit)`.** BLAS packed storage, column
  by column. `pack_matrix(A; ...)` builds it from a matrix; `kind` is
  `:symmetric`, `:hermitian` or `:triangular`.
- **`BandedMatrix(data, m; kl, ku, kind=:general, uplo, unit)`.** BLAS band
  storage, `data[ku+1+i-j, j]`; `band_matrix(A; ...)` copies only the band.

```@example st
using OxiBLAS, LinearAlgebra
A = randn(6, 6); H = A * A' + 6I
P = pack_matrix(H; uplo=:U, kind=:symmetric)
x = randn(6); y = zeros(6)
spmv!(y, P, x)                 # y ← P·x
y ≈ H * x
```

| Operations | Functions |
|---|---|
| packed products | `spmv!`, `hpmv!`, `tpmv!`, `tpsv!`, `spr!`, `spr2!`, `hpr!`, `hpr2!` |
| banded products | `gbmv!`, `sbmv!`, `hbmv!`, `tbmv!`, `tbsv!` |
| factorizations | `band_lu`, `band_cholesky`, `packed_cholesky` (`BandLUFactor`, `BandCholeskyFactor`, `PackedCholeskyFactor`) |

These operations only visit the stored coefficients. They accept
`kernel=:auto` or the ISA requests of GEMM; `:scalar` selects the generic
loop.

## Structured types of LinearAlgebra

`OxiBLAS.lu`, `cholesky`, `ldlt`, `bunchkaufman`, `factorize`, `det`, `inv`,
`eigen`, `svd`… accept `Diagonal`, `Bidiagonal`, `Tridiagonal`,
`SymTridiagonal`, the triangular matrices and their `Hermitian`/`Symmetric`
wrappers. The chosen engine exploits the structure.

| Type | Engine |
|---|---|
| `Tridiagonal` | `band_lu`, O(n) |
| real `SymTridiagonal` | `band_cholesky` (`cholesky`), `eigen_tridiagonal` (`eigen`) |
| `Bidiagonal` | substitution (`factorize`), bidiagonal QR kernel (`svd`) |
| `Diagonal` | direct operations, O(n) |
| triangular | `TriangularEngine`: substitution, no elimination |
| `BandedMatrix` | `band_lu`, `band_cholesky`, multifrontal for LDLᵀ |
| `PackedMatrix` | `packed_cholesky`, `tpsv!` |

**The LinearAlgebra functions are left intact.** The structured types of Julia
only receive `OxiBLAS.*` methods: `LinearAlgebra.lu(::Tridiagonal)` stays the
one of Julia. Only `BandedMatrix` and `PackedMatrix`, which are OxiBLAS types,
extend `LinearAlgebra` functions.

## Limitations

- **No structured engine** for QR, LQ, Schur, Hessenberg and the
  non-symmetric spectra of banded and tridiagonal types. These cases raise a
  `MethodError`, without a hidden dense conversion.
- **Indefinite `PackedMatrix`:** no packed Bunch–Kaufman. `factorize` raises an
  `ArgumentError`.
- **No `lu!` on banded storage:** it would need a buffer for the fill-in of
  pivoting.
