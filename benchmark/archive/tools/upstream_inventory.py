#!/usr/bin/env python3
"""Inventory every upstream Rust source module; never infer completion from LOC.
Usage: python3 tools/upstream_inventory.py /path/to/oxiblas-upstream
"""
import csv, hashlib, pathlib, re, subprocess, sys

upstream=pathlib.Path(sys.argv[1]).resolve()
root=pathlib.Path(__file__).resolve().parents[3]
revision=subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD'],text=True).strip()

def status(path):
    p=str(path)
    # This stage is implementation-only: source coverage is not validation.
    if path.name in ('tensor.rs','ndtensor.rs') and 'oxiblas-blas/' in p:
        return 'partial','src/tensors.jl;src/batched.jl;Julia Base arrays/views','v0.6-dev UNVALIDATED: general einsum/ellipsis/diagonals/reductions, greedy planning, contractions, ND batched broadcast, Tucker/TT-SVD/CP-ALS; no custom dense Tensor; advanced planners/distributed tensors unfinished'
    if 'autotune' in p or 'oxiblas-core/src/tuning' in p:
        return 'partial','src/autotuning.jl','v0.6-dev UNVALIDATED: explicit opt-in native calibration and environment-checked persistent config; calibration never run in this stage; x86 hardware tuning unfinished'
    if 'oxiblas-lapack/src/lu/' in p and path.name in ('full_piv.rs','rook_piv.rs'):
        return 'partial','src/advanced_factorizations.jl','v0.6-dev UNVALIDATED: full/rook pivoting, row/column permutations and solves, task-owned trailing columns; blocked advanced variants unfinished'
    if 'oxiblas-lapack/src/cholesky/' in p and path.name in ('ldlt.rs','bunch_kaufman.rs'):
        return 'partial','src/advanced_factorizations.jl','v0.6-dev UNVALIDATED: symmetric/Hermitian LDL, Bunch-Kaufman 1x1/2x2 pivots, independent trailing updates and solve; blocked/Aasen variants unfinished'
    if 'oxiblas-lapack/src/qr/' in p and path.name=='col_pivot.rs':
        return 'partial','src/advanced_factorizations.jl','v0.6-dev UNVALIDATED: norm-recomputed column-pivoted QR and basic rank-deficient least squares; strong RRQR and minimum-norm complete decomposition unfinished'
    if 'oxiblas-lapack/src/evd/' in p and path.name=='generalized.rs':
        return 'partial','src/advanced_factorizations.jl','v0.6-dev UNVALIDATED: Hermitian-definite generalized eigenpairs through native Cholesky; general QZ/singular pencils unfinished'
    if 'oxiblas-lapack/src/svd/' in p and path.name in ('bidiag_reduce.rs','complex_bidiag.rs'):
        return 'partial','src/bidiagonal.jl','v0.6-dev UNVALIDATED: compact two-sided Householder bidiagonal reduction, lazy U/V and wide adjoint path; Golub-Kahan SVD iteration and divide-and-conquer unfinished'
    if 'oxiblas-sparse/' in p and path.name in ('csr.rs','csc.rs','coo.rs'):
        return 'partial','src/sparse.jl;src/sparse_extended.jl','v0.6-dev UNVALIDATED: owned CSR/CSC/COO conversions, canonical duplicate summation, parallel SpMV/SpMM and Gustavson SpGEMM; remaining sparse algorithms unfinished'
    if 'oxiblas-sparse/src/linalg/iterative/' in p and path.name in ('gmres.rs','bicgstab.rs'):
        return 'partial','src/sparse_extended.jl','v0.6-dev UNVALIDATED: restarted GMRES with reorthogonalization, BiCGStab, diagonal/ILU0 preconditioning and true residuals; flexible/block variants unfinished'
    if 'oxiblas-sparse/src/linalg/' in p and 'ilu' in path.stem:
        return 'partial','src/sparse_extended.jl','v0.6-dev UNVALIDATED: zero-fill ILU and solves only; threshold/fill/parallel variants unfinished'
    if 'oxiblas-sparse/src/linalg/eigenvalue/' in p and path.name=='lanczos.rs':
        return 'partial','src/sparse_extended.jl','v0.6-dev UNVALIDATED: fully reorthogonalized Hermitian Lanczos/Ritz eigenpairs and true residuals; restarted/block/general spectral variants unfinished'
    if 'oxiblas-ffi/' in p:
        return 'unfinished-optional','none','retired/excluded upstream ABI crate; separate integration phase'
    if 'oxiblas-core/src/simd' in p:
        if any(x in p for x in ('wasm32','sse')):
            return 'unfinished','none','WASM/SSE-specific kernels are not ported; native scalar fallback exists'
        return 'partial','src/hardware.jl;src/simd.jl','NEON tested/measured; AVX2/AVX512 vector shapes implemented but untested on hardware; no SSE/WASM port'
    if 'oxiblas-core/src/memory' in p:
        return 'partial','src/gemm.jl','aligned owned panels and reusable task-owned workspace only; pool/arena/NUMA unfinished'
    if 'oxiblas-core/src/scalar' in p:
        return 'partial','src/OxiBLAS.jl;src/level1.jl','four native scalar types; Kahan/pairwise dots; extended precision unfinished'
    if 'oxiblas-core/src/parallel' in p:
        return 'partial','src/gemm.jl','coarse task partitions and per-task scratch; global controls/NUMA unfinished'
    if 'oxiblas-core/src/tuning' in p or 'oxiblas-core/src/blocking' in p:
        return 'partial','src/gemm.jl','configurable upstream-derived blocks; runtime autotuning/cache-oblivious unfinished'
    if 'oxiblas-matrix/src/' in p:
        return 'partial','src/storage.jl;Julia Base arrays/views','standard Julia arrays, transpose/adjoint, packed/banded adapters; mmap/lazy/COW/interop unfinished'
    if 'oxiblas-blas/src/level1/' in p:
        if '/parallel.rs' in p:
            return 'unfinished','none','dedicated parallel Level 1 algorithms are not ported'
        return 'partial','src/level1.jl;src/modified_rotations.jl','tested native vector primitives, norms, real/complex Givens, real modified rotations with safe exponents and preserved rescaling, ordered widened Float32 dots; ROTM contiguous LLVM SIMD measured on NEON; dedicated parallel variants and more tuning unfinished'
    if 'oxiblas-blas/src/level2/' in p:
        return 'partial','src/level2.jl;src/storage.jl;src/compact_blas.jl','v0.5 direct kernels tested historically; v0.6-dev UNVALIDATED: complex LLVM AXPY/fused dot reductions and conflict-free dense/compact MV task partitions; dedicated triangular SIMD leaves unfinished'
    if 'oxiblas-blas/src/level3/' in p:
        if any(x in p for x in ('autotune','strassen','winograd','cache_oblivious')):
            return 'unfinished','none','algorithm is not ported'
        if path.name=='gemmt.rs':
            return 'partial','src/level3.jl;src/gemm.jl','v0.5 inter-tile GEMMT tested historically; v0.6-dev UNVALIDATED: cached jobs, child workspaces and diagonal scratch; dedicated ISA triangular microkernels unfinished'
        if path.name in ('symm.rs','hemm.rs','trmm.rs','syrk.rs','syr2k.rs','herk.rs','her2k.rs') or '/trsm/' in p:
            return 'partial','src/level3.jl;src/gemm.jl','v0.5 structured kernels tested historically; v0.6-dev UNVALIDATED: reusable tile contexts and parallel independent RHS in TRMM/TRSM; dedicated SIMD leaves unfinished'
        return 'partial','src/gemm.jl;src/simd.jl;src/level3.jl;src/batched.jl','v0.5 GEMM measured historically; v0.6-dev UNVALIDATED: parallel independent batches, task-owned cached contexts and no nested GEMM; other algorithm variants unfinished'
    if 'oxiblas-lapack/src/lu/' in p:
        if path.name not in ('partial_piv.rs','mod.rs'):
            return 'unfinished','none','full-pivot/rook/band LU not ported'
        return ('partial','src/factorizations.jl;src/advanced_factorizations.jl','v0.6-dev UNVALIDATED: blocked partial LU reuses panels and parallel TRSM; full/rook implemented separately; band LU unfinished')
    if 'oxiblas-lapack/src/cholesky/' in p:
        if path.name not in ('llt.rs','hermitian.rs','mod.rs'):
            return 'unfinished','none','LDLT/Bunch-Kaufman/Aasen/band/packed algorithm not ported'
        return 'partial','src/factorizations.jl;src/advanced_factorizations.jl','v0.6-dev UNVALIDATED: blocked LLH reuses panels and triangular HERK updates; LDL/Bunch-Kaufman added separately; Aasen/band/packed variants unfinished'
    if 'oxiblas-lapack/src/qr/' in p:
        if '/householder/' not in p and path.name not in ('mod.rs','unitary.rs'):
            return 'unfinished','none','pivoted QR/LQ/QL/RQ/complete orthogonal algorithm not ported'
        return 'partial','src/factorizations.jl;src/advanced_factorizations.jl','v0.6-dev UNVALIDATED: blocked compact-WY QR and column-pivoted QR; LQ/QL/RQ/complete orthogonal unfinished'
    if 'oxiblas-lapack/src/solve/' in p:
        return 'partial','src/factorizations.jl;src/advanced_factorizations.jl','v0.6-dev UNVALIDATED: equilibrated LU expert solve with backward errors, real/complex mixed refinement, pivoted basic LS; condition/error bounds and other experts unfinished'
    if 'oxiblas-sparse/src/csr.rs' in p:
        return 'partial','src/sparse.jl','owned validated CSR and native transpose-aware SpMV tested; remaining CSR algorithms unfinished'
    if 'oxiblas-lapack/src/evd/' in p:
        if path.name in ('hessenberg.rs','complex_hessenberg.rs'):
            return 'partial','src/hessenberg.jl','compact real/complex Householder, Q, implicit left/right N/T/C and compact-WY Q tested; blocked reduction and ilo/ihi range unfinished'
        if path.name in ('schur.rs','complex_schur.rs'):
            return 'partial','src/schur.jl;src/schur_advanced.jl','v0.5 Schur tested historically; v0.6-dev UNVALIDATED: complex adjacent Schur reordering and general right eigenvectors; real block reordering, balancing, condition estimates, AED/multishift unfinished'
        if path.name in ('symmetric.rs','hermitian.rs','hermitian_tridiag.rs'):
            return 'partial','src/hermitian_eigen.jl','native rank-2 tridiagonalization, complex phase correction, implicit QR eigenpairs/values-only tested; blocked and alternative algorithms unfinished'
        if path.name=='tridiag_evd.rs':
            return 'unfinished','src/hermitian_eigen.jl (alternative QR)','upstream bisection/inverse iteration/selected eigenpair algorithms not ported; a separate implicit QR eigensolver is available'
    if 'oxiblas-lapack/src/svd/' in p and path.name in ('bidiagonal.rs','complex.rs'):
        return 'partial','src/svd.jl','real/complex one-sided Jacobi economy SVD and values-only tested; scaled norms/correlations, wide adjoint path, null-column completion; full SVD/other methods unfinished'
    if 'oxiblas-sparse/src/linalg/iterative/mod.rs' in p or 'oxiblas-sparse/src/linalg/precond/jacobi.rs' in p:
        return 'partial','src/sparse.jl','CG and positive Jacobi diagonal tested; other solver variants unfinished'
    return 'unfinished','none','not ported; see docs/STATUS.md'

dest=root/'docs'/'upstream_inventory.csv'
paths=sorted(upstream.glob('crates/*/src/**/*.rs'))
with dest.open('w',newline='') as f:
    out=csv.writer(f)
    out.writerow(['revision','rust_path','sha256','lines','public_symbols','status','julia_mapping','scope'])
    for path in paths:
        data=path.read_bytes(); text=data.decode('utf-8'); rel=path.relative_to(upstream)
        symbols=re.findall(r'^\s*pub(?:\([^)]*\))?\s+(?:unsafe\s+|async\s+|const\s+)*(?:fn|struct|enum|trait|type|mod)\s+(\w+)',text,re.M)
        out.writerow([revision,str(rel),hashlib.sha256(data).hexdigest(),len(text.splitlines()),';'.join(symbols),*status(rel)])
print(f'{len(paths)} source files inventoried at {revision}: {dest}')
