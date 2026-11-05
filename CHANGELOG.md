# Changelog

Campaign reports and the development journals of versions 0.1–0.6 are kept
outside the repository; the entries below summarize them.

## 0.6.0-dev — package restructuring (4 November 2026)

- **Sources.** Grouped by family under `src/`: `utils/`, `blas/`,
  `structured/`, `lapack/`, `sparse/`, `graphs/`, `spectral/`, `tensors/`,
  `api/`. There is still a single `OxiBLAS` module, and the `include` order is
  unchanged. The 3,672 method signatures and the exports are identical before
  and after.
- **Precompilation.** `src/precompile.jl` (PrecompileTools) covers GEMM
  Float32/Float64, dense LU and Cholesky, `\` and `OxiBLAS.ldiv` on the sparse
  formats, and `einsum`. The first `OxiBLAS.ldiv(A, b)` drops from 13.5 s to
  0.13 s. The workload is disabled with the preference
  `precompile_workload = false`.
- **Tests.**
  - `test/<family>/`, a shared prelude `testsetup.jl` and shared helpers in
    `test/helpers/`: every file runs on its own.
  - `core` (CI) and `extended` groups, and any family on demand
    (`OXIBLAS_TEST_GROUP`), with one progress line per file (`groups.jl`).
  - The dense SVD campaign tests are wired in.
  - Every test and fixture tied to the Rust reference is removed from the
    package.
- **Julia 1.10 to 1.13 compatibility.** `hcat`/`vcat` signatures restricted to
  `<:Number` element types and disambiguating `LinearAlgebra.mul!` and `\`
  methods (dispatch only, no runtime cost). The core tests pass on 1.10, 1.12,
  1.13 and 1.14-alpha.
- **Benchmarks.**
  - New high-level suite `benchmark/run.jl` (Chairmarks): a grid of about
    4,000 cases (each high-level operation × LinearAlgebra and OxiBLAS input
    types × Float16, Float32, Float64, ComplexF32, ComplexF64, BigFloat × three
    sizes), against OpenBLAS/Accelerate, SparseArrays and TensorOperations,
    with each result checked and the allocation count of each case recorded
    (`--allocs` runs the allocation campaign alone).
  - The historical campaigns are archived in `benchmark/archive/`.
  - The Rust programs and scripts are no longer part of the package.
- **Documentation.** Documenter (`docs/make.jl`), with a home page, eight
  guides and a reference generated per source family.
- **CI.** Three GitHub Actions workflows:
  - `CI.yml`: core tests and QA, Linux and macOS, Julia 1.10 and 1, 1 and 4
    threads, generic CPU target;
  - `Extended.yml`: manual and weekly;
  - `Documentation.yml`: build on pull requests, GitHub Pages deployment on
    `main`.

## 0.6.0-dev — linear-solve API, Core.Box campaign, SVD (4 November 2026)

- **Linear solves.** `A \ B` and `LinearAlgebra.ldiv!` on the OxiBLAS sparse
  formats and their wrappers, `OxiBLAS.ldiv` and `OxiBLAS.ldiv!` with a matrix
  or a factorization, O(nnz) `istril`/`istriu`/`ishermitian` on the OxiBLAS
  formats, and a factorization choice that follows SparseArrays (triangular,
  Hermitian, square, tall).
- **QA.** Aqua (piracy, ambiguities, unbound type parameters), ExplicitImports
  and `Test.detect_closure_boxes`. Seven methods with unbound type parameters
  were fixed (SIMD helpers, `EigenFactor`, `SVDFactor`), which also fixed
  `svd_truncated(...; vectors=false)`.
- **Closure boxes.** The 39 `Core.Box` reported by
  `detect_closure_boxes` were removed. This fixed two data races
  (`_pivot_lu_factor`, `gemm_cache_oblivious!`): rook-pivoted LU went from
  83 ms to 0.6 ms at n = 192, Float16 `syrk!` from 18 ms to 0.07 ms.
- **Dense SVD.** Chained rotation packets in the Golub–Kahan QR (the shared
  column stays in SIMD registers) and a safe-range Givens rotation (as
  LAPACK's dlartg). With vectors, at 192×192: Float32 from 5.3 to 2.5 ms,
  Float64 from 8.1 to 4.8 ms.

## 0.6.0-dev — low-level sparse solvers (4 November 2026)

- **Symbolic analysis.** Native AMD (`amd_ordering`, CSparse `cs_amd`),
  elimination tree, postorder weighted by column counts, column counts in
  O(nnz(A)·α).
- **Supernodal `multifrontal_cholesky`.** CHOLMOD relaxed supernodes,
  uncapped width, stack of contribution blocks, in-place blocked front kernel
  up to f = 192 then POTRF/TRSM/HERK; simplicial (up-looking) fallback when
  flops/nnz(L) < 40 (`supernodal=:auto`). Supernodal solve. `PosDefException`
  reports the column of A.
- **Parallel supernodal Cholesky.** Tasks across subtrees (`@spawn`,
  per-task buffers, disjoint columns of L), then the top of the tree with the
  parallel dense kernels.
- **`sparse_qr(A; pivot=false, ordering)`.** Multifrontal QR (relaxed
  supernodes of the column tree, staircase rows, whole front factorized with a
  trapezoidal contribution, WY panels); orderings `:natural`, `:amd` (AMD of
  AᵀA) or a vector; `pivot=true` unchanged. Faster `sparse_apply_q!` and
  `solve`.
- **`sparse_cholesky`.** Exact through this code in natural order (zeros of
  the relaxed supernodes removed); IC0/ICT unchanged.
- **`multifrontal_lu`.**
  - `ordering=:auto` by default.
  - UMFPACK's symmetric strategy (symmetric pattern, full diagonal): supernodes
    of A+Aᵀ, square fronts, threshold partial pivoting among fully summed rows,
    pivots delayed to the parent.
  - Unsymmetric pattern: singletons, then Gilbert–Peierls with AMD(AᵀA).
  - Explicit orderings: the row-element engine is unchanged. Solve by CSR rows.
  - Default pivot threshold 0.1. Unsymmetric cores go through a multifrontal LU
    on column fronts (threshold pivoting that prefers the sparsest row,
    remaining rows routed to the front of their first nonzero column).
- **Exact `sparse_lu`.** `ordering` argument (`:auto` by default, `:natural`,
  `:amd_ata`, vector), field `q` in `SparseLUFactor` (A[p,q] = L·U),
  left-looking Gilbert–Peierls in the given order; ILUT unchanged (natural
  order).
- **`colamd_ordering`.** Port of SuiteSparse's COLAMD, BSD-3, identical
  permutations, plus `sparse_ordering(:colamd)`. It is the core ordering of the
  unsymmetric LU and the `:colamd` option of `sparse_lu` and `sparse_qr`. Column
  front pivots: row lengths updated at each pivot, sparsest pivot column of the
  block, front width capped at 256 columns.
- **`minimum_degree_ordering`, `multiple_minimum_degree`.** Quotient graph with
  supervariables and exact external degree (MMD: multiple elimination).
- **`sparse_ldlt`.** Supernodal multifrontal, Bunch–Kaufman (1×1, 2×2) among
  fully summed variables, threshold 0.01 (MA57) when the column maximum is
  outside the candidates, delayed pivots otherwise. API unchanged
  (`hermitian=false`, `pivot=:none`, inertia). Solve by direct columns.
- **`sparse_triangular_solve`.** Native CSR/CSC kernels, without conversion.
- **BLAS 3.** No scaling for β = 1 in the packed triangular product
  (HERK/SYRK).

## 0.6.0-dev — SparseArrays, sparse tensors, `OxiBLAS.mul!` (3 November 2026)

- **SparseArrays becomes a direct dependency.**
  - Conversions both ways between `SparseMatrixCSC`/`SparseVector` and the
    OxiBLAS formats, without going through dense storage; `CSCMatrix(S)` and
    `CSRMatrix(transpose(S))` share the storage of `S`.
  - The OxiBLAS sparse routines (`sparse_mul!`, `sparse_matmul`, direct and
    iterative solvers, `ilu0`) accept SparseArrays matrices.
- **Generic element types.** Constructors of the OxiBLAS sparse formats
  extended to `GenericFloat` types (BigFloat included), without changing the
  Float32/Float64 paths.
- **`SparseTensor{T,N}`.** COO with sorted linear indices. `einsum`,
  `einsum!`, `contract`, `contract!` and the plans accept sparse operands:
  dense/sparse mixes, traces, diagonals, outer products, ellipses, with output
  `:auto`, `:dense` or `:sparse` and sparse destinations.
- **`OxiBLAS.mul!(C,A,B[,α,β])` and `OxiBLAS.mul(A,B)`.** Not exported and
  distinct from `LinearAlgebra.mul!` (neither extended nor aliased). They cover:
  - dense matrices, LinearAlgebra structured types (including sparse content,
    without densification), SparseArrays, OxiBLAS formats, sparse
    destinations;
  - type conversion that keeps the format;
  - a copy of operands whose storage overlaps `C`.
- **Place in the SparseArrays hierarchy.**
  - The sparse formats CSR, CSC, COO, DIA, ELL, BSR, BSC, HYB and SELL are
    `<: AbstractSparseMatrix`.
  - `SparseTensor{T,N} <: AbstractSparseArray{T,Int,N}`; order 1 is an
    `AbstractSparseVector`.
  - `CSCMatrix` does not inherit from `AbstractSparseMatrixCSC`, so that
    operations on these types end in OxiBLAS kernels.
  - `CSCMatrix` row indices are sorted per column; `copy`/`similar`/`zero`
    keep the format (CSC, CSR, COO).
- **`Base.*` on the OxiBLAS sparse types only.** Also on their transposes and
  structured wrappers, through `OxiBLAS.mul`: 16 methods, no dispatch
  ambiguity (dedicated test). `CSC * CSC` stays CSC; the other OxiBLAS
  products give CSR.
- **Base/LinearAlgebra operations on the OxiBLAS sparse types.** All run in
  OxiBLAS kernels (no SparseArrays or LinearAlgebra kernel, no external BLAS):
  - `LinearAlgebra.mul!` (to `OxiBLAS.mul!`), `+`/`-` (`sparse_add`),
    transposes and `permutedims` (`sparse_transpose`), `SparseArrays.permute`
    (`sparse_permute`), 1 and Inf `norm`/`opnorm` (`sparse_norm`);
  - indexing and `setindex!`, broadcast and `map` (`OxiSparseStyle`),
    reductions (`sum`, `maximum`, `count`, `any`, `all`, per dimension), `==`,
    `findall`, `findnz`, `dropzeros`/`droptol!`;
  - `triu`/`tril`, `diag`, `tr`, `issymmetric`/`ishermitian`, `dot`, `kron`,
    concatenation, `reverse`, `circshift`, `vec`/`reshape`, dense conversion,
    `zero`/`similar`.

  The cost is O(nnz) for `SparseTensor`s of any order.

## 0.6.0-dev — LAPACK, first part (3 November 2026)

- **`lu_factor` (getrf).** Recursive panel (getrf2), row swaps grouped per
  panel (laswp), lookahead (the next panel is factorized during the trailing
  update), blocks of 128, leaves of 32 (real) or 16 (complex) columns.
- **`cholesky_factor` (potrf).** Recursive (potrf2) through TRSM/HERK; leaves
  of 64 (real) or 32 (complex) columns.
- **QR (geqrf).** Compact WY form, T computed by GEMM, reused buffers,
  vectorized BLAS 1 panel, panel size from the shape. `apply_q!`
  (ormqr/unmqr) and `qr_q`/`F.Q` (orgqr/ungqr) are blocked, and run in
  parallel by splitting the right-hand sides.
- **`qr_pivoted` (geqp3).** Column norm updates with LAPACK's recomputation
  criterion.
- **`ldlt_factor` (sytrf).** Blocked Bunch–Kaufman (lasyf), same pivot
  choices.
- **Solves and inverses.** Vector solves through TRSV; `native_inverse`
  (getri) through TRTRI; new `cholesky_inverse` (potri: TRTRI then recursive
  LAUUM, a single HERK up to N = 64).
- **BLAS.** Complex transposed or adjoint `trsv!` through BLAS 1 dot products;
  `gemm!` alias test through views and transposes; the panel pool keeps the
  largest panels.

## 0.6.0-dev — BLAS 3 (3 November 2026)

- **Complex GEMM micro-kernel.** Two accumulators (`a·Re(b)`, `a·Im(b)`),
  recombined at the store; ComplexF64 register shapes NEON 1×4×3 and AVX2
  2×2×3.
- **Edge tiles.** GEMM edge tiles are computed by the SIMD kernel in a
  temporary tile, instead of the scalar kernel.
- **`syrk`/`herk`/`syr2k`/`her2k`/`gemmt`.** Packed GEMM restricted to the
  triangle (tiles outside it skipped, diagonal tiles masked); the former tile
  schedule stays as the fallback.
- **TRMM/TRSM leaves.** Real `trmm`/`trsm` leaves of order ≤ 8, left and
  right: generated, unrolled and vectorized kernels.
- **Parallelism.** Workers from the amount of work (≈ 100 µs of sequential
  work per worker), the calling thread takes part, 2-D grid for GEMM. Internal
  panels are reused from a pool (at most 4 per type) instead of being
  allocated at every call.

## 0.6.0-dev — BLAS 2 (3 November 2026)

- **`trsv`/`trmv`/`gemv`/`symv`/`hemv`.** Four columns fused per pass (4×4
  diagonal block apart); complex symmetric/Hermitian products as four real
  sums.
- **`ger`/`gerc`/`syr`/`her`/`syr2`/`her2`.** Per-column coefficient, then a
  fused SIMD loop.
- **Banded and packed kernels.** Real `gbmv` and narrow bands through direct
  band access; narrow `tbmv`/`tbsv` by rows, with real and imaginary parts
  split in ComplexF32. Packed reductions use plain `@simd`, and the packed
  rank-2 update is a single pass.
- **Parallel BLAS 2.** Workers from the estimated traffic (BLAS 1 rule,
  calling thread takes part). `symv`/`hemv`/`spmv`/`hpmv`/`sbmv`/`hbmv` work
  by columns of equal work, with partial vectors (the triangle is read once).
  `parallel_threshold` defaults to `nothing` (automatic); an explicit value
  keeps the former rule. This also fixes the `first` error of the parallel
  `symv`.

## 0.6.0-dev — BLAS 1 (3 November 2026)

- **Dot products.** Complex `dotu`/`dotc` as four real SIMD sums; real
  `dotu`/`dotc` with plain `@simd`. `dsdot`/`sdsdot`: reassociated Float64 sum
  (as OpenBLAS), with an ordered fallback when non-finite.
- **Other kernels.** Single-pass blocked `iamax` with integer keys; complex
  `rotg` without `hypot` in the safe range; Float64 `nrm2` with a vectorized
  max and a power-of-two scale.
- **`*_parallel` variants.** SIMD kernels per chunk, number of workers from
  the estimated traffic (4 MiB threshold), calling thread takes part;
  `parallel_threshold` defaults to `nothing` (automatic).

## 0.6.0-dev — implementation

- **BLAS 3 and batches.** Reusable BLAS 3 contexts, jobs and scratch buffers;
  parallel batches without nested GEMM parallelism; TRMM/TRSM on independent
  right-hand sides.
- **Packed kernels and BLAS 2.** Packed complex SIMD and fused reductions;
  parallel dense and packed BLAS 2; packing-free GEMM path for small outputs
  and ComplexF32 unrolling.
- **Dense factorizations.** Reused LU/Cholesky panels, HERK in Cholesky;
  complete/rook-pivoted LU, blocked and pivoted QR, LDL/Bunch–Kaufman, expert
  and mixed-precision complex solves.
- **Eigenvalue problems.** Definite Hermitian generalized EVD, reordered
  complex Schur and right eigenvectors, compact bidiagonal reduction.
- **Sparse.** COO/CSC, conversions, SpMM/SpGEMM, ILU0, GMRES/BiCGStab and
  Lanczos.
- **Tensors.** N-D contractions, einsum and plans, N-D batches, HOSVD, TT-SVD,
  CP-ALS, on standard Julia arrays (no dense Tensor type).
- **Autotuning.** Strictly explicit, with persistence controlled by the
  environment.

## 0.5.0 — direct packed storage and inter-tile parallelism

- **Banded kernels.** GBMV/SBMV/HBMV/TBMV/TBSV bounded to the band, without a
  dense logical sweep.
- **Packed kernels.** SPMV/HPMV/TPMV/TPSV and SPR/SPR2/HPR/HPR2 by stored
  packed columns.
- **Triangular products.** In-place triangular multiplications, pivots checked
  before substitution, packed snapshots on aliasing, paddings and unit
  diagonals ignored.
- **AXPY.** Contiguous real AXPY with LLVM SIMD, widths in the type domain to
  avoid per-column allocations, GC roots protected and explicit remainders.
- **Rank-k updates.** SYRK/HERK/SYR2K/HER2K/GEMMT run tasks on disjoint tiles,
  with panels owned by each task and global protection of the shared
  workspace. The inner GEMM is single-threaded when an outer plan already
  distributes the tiles.
- **Tests and measurements.** Dense/BLAS references, views, aliasing,
  forbidden reads, structural non-finite data and extreme scales;
  measurements on 1/2/4/8 threads.

## 0.4.0 — modified rotations and extended dot products

- **ROTM.** Float32/Float64, LLVM SIMD packets on contiguous outputs,
  immutable parameters or BLAS vector, views/strides, ignored parameters never
  read, controlled overlap of parameters and outputs.
- **ROTMG.** Pure version and state update, separate mantissas/exponents for
  extreme products, coefficients preserved across repeated rescalings.
- **DSDOT/SDSDOT.** Widened products and ordered accumulation in Float64;
  final Float32 conversion for SDSDOT, without fast-math or reassociation.
- **Tests and benchmarks.** High-precision invariant tests and an independent
  BLAS. Benchmarks on contiguous, reversed and non-unit-stride vectors, varied
  batch generation, and LLVM/assembly inspection.

## 0.3.0 — blocked structured BLAS 3 and stabilization

- **Revalidation.** P0 revalidation of v0.2 (11,574 tests), and CI on
  Linux/macOS/Windows with logs kept.
- **SYMM/HEMM.** Triangles packed directly into the GEMM panels, without
  dense symmetric/Hermitian expansion.
- **TRMM/TRSM.** Recursive decomposition, GEMM updates, in-place leaves,
  panels sized for the subproblems; contiguous right solve.
- **SYRK/SYR2K/HERK/HER2K.** Only the tiles of the selected triangle are
  computed; bounded reused diagonal buffer, exactly real Hermitian diagonal.
- **GEMMT.** Independent N/T/C, rectangular matrices, opposite triangle
  preserved.
- **Tests.** The four types, ignored triangles, unit diagonal never read,
  α/β/K without reads, aliasing, views, singularities detected before
  mutation, shared workspaces.

## 0.2.0 — native spectral decompositions

- **Hessenberg.** Real/complex compact Hessenberg reduction, implicit N/T/C
  application of Q on the left and right, Q generated by compact WY and the
  native GEMM. WY multiplications restricted to the useful regions, and the
  panel workspace reused for the whole call.
- **Schur.** Real Schur by Francis double shift and complex Schur by implicit
  QR, with relative deflation, exceptional shifts and non-convergence reported.
- **Symmetric/Hermitian eigenvalues.** Rank-2 tridiagonalization, explicit
  phase correction and eigenvalues/eigenvectors by implicit QR; values-only
  path.
- **SVD.** Economy one-sided Jacobi SVD for the four types, wide matrices,
  rank-deficient and values-only cases; scaled norms/correlations.
- **Tests.** Invariants, LAPACK references, very large/small inputs,
  subnormals, views and non-convergence.

## 0.1.0 — initial increment

- **Package and GEMM.**
  - Standalone Julia package and explicit interface for the four usual BLAS
    types.
  - Reference, cache-blocked, panel, register and LLVM-vector GEMM, with
    architecture selection and native Julia task distribution.
  - Reusable aligned buffers, α/β guarantees, views and overlap handling.
  - Real and complex NEON SIMD validated on Apple Silicon; x86 forms
    implemented and awaiting hardware validation.
- **Operations.** Dense and packed BLAS operations, blocked LU/Cholesky,
  Householder QR, solves, mixed-precision LU refinement, GEMM batches,
  CSR/SpMV and CG/Jacobi.
- **Validation.** Conformance tests, benchmarks, LLVM and assembly
  inspection, documentation of the deviations from upstream.
