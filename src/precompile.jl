# SPDX-License-Identifier: Apache-2.0
# Precompilation workload (PrecompileTools): a few common paths, on small
# deterministic and well-conditioned problems, so that their first call does
# not pay the compilation. It covers no size, type or algorithm variant beyond
# these; everything else compiles on first use, as before. The package works
# the same without it: `precompile_workload = false` in the [OxiBLAS] section of
# LocalPreferences.toml disables it (precompilation then takes about 3 s
# instead of 20, and the cache 10 MB instead of 50).
using PrecompileTools: @setup_workload, @compile_workload

@setup_workload begin
    n=8
    # Diagonally dominant: every factorization below is well conditioned.
    A64=[i==j ? Float64(2n+i) : 1.0/(i+j) for i in 1:n, j in 1:n]
    H64=A64'*A64
    A32=Float32.(A64)
    b=collect(1.0:n)
    # Tridiagonal 16×16, Hermitian positive definite, in CSR.
    m=16
    S=CSRMatrix(COOMatrix(m,m,vcat(1:m,1:m-1,2:m),vcat(1:m,2:m,1:m-1),
                          vcat(fill(4.0,m),fill(-1.0,m-1),fill(-1.0,m-1))))
    U=CSRMatrix(COOMatrix(m,m,vcat(1:m,1:m-1),vcat(1:m,2:m),vcat(fill(4.0,m),fill(-1.0,m-1))))   # nonsymmetric
    X3=reshape(collect(1.0:n^3),n,n,n)
    @compile_workload begin
        for A in (A32,A64)
            C=similar(A); gemm!(C,A,A); gemm!(C,A,A;threaded=false)
        end
        solve(lu_factor(A64),b)
        solve(cholesky_factor(H64),b)
        lu(A64)\b
        cholesky(_LA.Hermitian(H64))\b
        ldiv(A64,b)
        y=similar(b); ldiv!(y,lu(A64),b)
        bs=ones(m)
        S\bs                                  # Cholesky chosen by the structure
        U\bs                                  # sparse LU
        ldiv(factorize(U),bs)
        mul(S,bs)
        einsum("ij,jk->ik",A64,A64)
        einsum("abc,cd->abd",X3,A64)
    end
    # The GEMM workspace pool is the package's only global mutable state:
    # leave it empty, so that the image holds no workspace.
    foreach(empty!,_WORKSPACE_POOL)
end
