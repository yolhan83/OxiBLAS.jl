using OxiBLAS, Random
import LinearAlgebra as L
for T in (Float32,Float64),alg in (:jacobi,:parallel_jacobi,:golub_kahan)
    f=alg===:golub_kahan ? svd_bidiagonal : svd_factor
    F=f(ones(T,144,64);algorithm=alg);G=adjoint(F.U)*F.U;d=L.diag(G)
    println(T," ",alg," normerr=",L.norm(G-L.I)," diag=",extrema(d),
            " off=",maximum(abs,G-L.Diagonal(d))," minS=",minimum(F.S))
    flush(stdout)
end
