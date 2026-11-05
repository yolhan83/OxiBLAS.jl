# OxiBLAS.jl

[![CI](https://github.com/yolhan83/OxiBLAS.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/yolhan83/OxiBLAS.jl/actions/workflows/CI.yml)
[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://yolhan83.github.io/OxiBLAS.jl/dev/)

Dense, structured, sparse and tensor linear algebra written entirely in Julia.
OxiBLAS.jl is an adapted and extended port of the Rust library
[OxiBLAS](https://github.com/cool-japan/oxiblas) (reference pinned at commit
`8d82bb6`), under the Apache-2.0 license (see `LICENSE` and `NOTICE`).

- **Native computation.** No computation is delegated to BLAS, LAPACK,
  Accelerate, SuiteSparse or the original Rust code. Those libraries are only
  references in the tests and benchmarks.
- **SIMD kernels.** They are written with LLVM vectors, with no SIMD.jl or
  LoopVectorization.jl dependency. NEON is validated (Apple Silicon); AVX2 and
  AVX-512 are implemented but not validated on x86 hardware.
- **Coverage.** BLAS 1/2/3 and multithreaded GEMM, dense factorizations (LU,
  Cholesky, LDLᵀ, QR, Schur, QZ, SVD, MRRR…), packed and banded matrices,
  sparse formats and multifrontal direct solvers, iterative solvers and
  preconditioners, orderings and graphs, iterative spectral methods, `einsum`
  and sparse tensors.
- **High-level API.** It follows the conventions of `LinearAlgebra`:
  `OxiBLAS.lu`, `OxiBLAS.qr`, `OxiBLAS.svd`, `OxiBLAS.factorize`,
  `OxiBLAS.ldiv`, `A \ b` on the OxiBLAS sparse formats… Its functions belong
  to the package and are not exported.

## Installation

The package is not registered yet:

```julia
using Pkg
Pkg.add(url="https://github.com/yolhan83/OxiBLAS.jl")
```

Julia versions: 1.10 (LTS) and later (`[compat] julia = "1.10"`).

## Example

```julia
using OxiBLAS, SparseArrays, LinearAlgebra

A = randn(512, 512); B = randn(512, 512); C = similar(A)
gemm!(C, A, B)                        # native GEMM

F = OxiBLAS.lu(A + 512I)              # native factorization (a LinearAlgebra.Factorization)
x = OxiBLAS.ldiv(F, randn(512))       # or F \ b

S = CSRMatrix(sprand(10_000, 10_000, 1e-3) + 10I)
y = S \ ones(10_000)                  # multifrontal LU, chosen from the structure

T = einsum("abc,cd->abd", randn(32, 32, 32), randn(32, 16))
```

## Documentation

The documentation is published at <https://yolhan83.github.io/OxiBLAS.jl/dev/>.
It also builds locally:

```sh
julia --project=docs -e 'using Pkg; Pkg.instantiate()'
julia --project=docs docs/make.jl     # site in docs/build/
```

It contains one guide per family and the API reference by source family.

## Tests

```sh
julia --project=. -e 'using Pkg; Pkg.test()'                             # core tests + QA
OXIBLAS_TEST_GROUP=extended julia --project=. -e 'using Pkg; Pkg.test()'  # heavy campaigns
OXIBLAS_TEST_GROUP=sparse julia --project=. -e 'using Pkg; Pkg.test()'    # one family
```

The groups are:

- **by level:** `core` (default), `extended`, `all`;
- **by family:** `blas`, `structured`, `lapack`, `sparse`, `graphs`,
  `spectral`, `tensors`, `generic`, `api`, `qa`.

The details are in `test/groups.jl`. QA covers Aqua (piracy, ambiguities,
unbound type parameters) and ExplicitImports, plus `Test.detect_closure_boxes`
from Julia 1.14 on.

## Benchmarks

```sh
julia --project=benchmark -e 'using Pkg; Pkg.instantiate()'
julia --project=benchmark --threads=1 benchmark/run.jl          # high-level suite
julia --project=benchmark --threads=1 benchmark/run.jl gemm lu  # selection by name prefix
```

The historical campaigns and kernel micro-benchmarks are in
`benchmark/archive/` (see `benchmark/README.md`).

## Layout

```
src/        OxiBLAS.jl, then utils/ blas/ structured/ lapack/ sparse/ graphs/
            spectral/ tensors/ api/ (a single OxiBLAS module) and precompile.jl
test/       runtests.jl, groups.jl, testsetup.jl, helpers/, one folder per family
benchmark/  run.jl (high-level suite), latency/, archive/
docs/       make.jl, src/ (guides, generated reference)
```
