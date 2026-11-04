# `A \ b` on the OxiBLAS sparse formats, before and after linalg_api_solve.jl.
#
# Former path (generic AbstractMatrix `\` of LinearAlgebra):
#   1. istril(A), istriu(A) through getindex. They stop at the first entry
#      outside the triangle, so they are cheap on a general matrix and read
#      ~n²/2 positions on a triangular one.
#   2. triangular: LowerTriangular(A) \ b, a generic substitution that also
#      reads A through getindex (O(n²)); otherwise lu(A), already OxiBLAS.
# It is reproduced here without the new methods: `invoke` on the generic
# istril/istriu, and the generic substitution on an opaque AbstractMatrix
# wrapper that only forwards size and getindex to A.
#
# New path: O(nnz) istril/istriu, then OxiBLAS.factorize (native sparse
# substitution, Cholesky or LU) and the solve.
#
# Run: julia -t1 --project=. benchmark/archive/sparse_backslash.jl
using OxiBLAS, SparseArrays, Random
import LinearAlgebra as LA
LA.BLAS.set_num_threads(1)
struct Opaque{T,M} <: AbstractMatrix{T}; A::M; end
Opaque(A)=Opaque{eltype(A),typeof(A)}(A)
Base.size(X::Opaque)=size(X.A)
Base.getindex(X::Opaque,i::Int,j::Int)=X.A[i,j]

med(f;k=5)=(f(); t=sort([(@elapsed f()) for _ in 1:k]); t[(k+1)÷2])
ms(t)=isnan(t) ? lpad("—",10) : lpad(string(round(1e3t;sigdigits=3)),10)
function laplace2d(k;convection=0.0)
    e=ones(k); T=spdiagm(-1=>-e[2:end],0=>2e,1=>-e[2:end]); C=spdiagm(-1=>-e[2:end],1=>e[2:end])
    I=sparse(1.0LA.I,k,k)
    kron(I,T)+kron(T,I)+convection*(kron(I,C)+kron(C,I))
end
# The two-argument methods: the one-argument ones call istril(A, 0), which
# would dispatch to the new O(nnz) method.
generic_tests(A)=(invoke(LA.istril,Tuple{AbstractMatrix,Integer},A,0); invoke(LA.istriu,Tuple{AbstractMatrix,Integer},A,0))

println(rpad("matrix",22),rpad("n",8),lpad("old tests",10),lpad("old solve",10),lpad("new tests",10),
        lpad("new A\\b",10),lpad("SA S\\b",10),"   (ms)")
for (label,make) in (("laplace2d, Cholesky",k->laplace2d(k)),("convection, LU",k->laplace2d(k;convection=0.3)),
                     ("lower triangular",k->sparse(LA.tril(laplace2d(k;convection=0.3)))))
    for k in (32,64,100,300)
        S=make(k); A=CSRMatrix(S); n=size(S,1); b=randn(Xoshiro(1),n)
        tri=LA.istril(S)
        small=k<=100
        old_tests=small || !tri ? med(()->generic_tests(A);k=3) : NaN
        # Former solve after the tests: generic substitution for a triangular A,
        # otherwise the OxiBLAS LU that lu(A) already reached.
        old_solve=tri ? (small ? med(()->LA.LowerTriangular(Opaque(A))\b;k=3) : NaN) : med(()->OxiBLAS.lu(A)\b)
        new_tests=med(()->(LA.istril(A); LA.istriu(A)))
        new_solve=med(()->A\b); ref=med(()->S\b)
        LA.norm(S*(A\b)-b)/LA.norm(b)<1e-8 || error("residual check failed")
        println(rpad(label,22),rpad(n,8),ms(old_tests),ms(old_solve),ms(new_tests),ms(new_solve),ms(ref))
    end
end
