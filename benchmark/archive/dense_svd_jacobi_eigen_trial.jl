# Experimental local solver only: override inside this benchmark process.
# The shared Hermitian eigensolver and all package sources remain unchanged.
using OxiBLAS
@eval OxiBLAS function _svd_block_rotation!(G,W,rtol,m)
    R=real(eltype(G));q=size(G,1)
    lo=minimum(i->real(G[i,i]),1:q);hi=maximum(i->real(G[i,i]),1:q)
    lo>max((floatmin(R)/eps(R))*max(1,m),sqrt(eps(R))*hi) || return false
    changed=false
    @inbounds for j in 2:q,i in 1:j-1
        if abs(G[i,j])>rtol*sqrt(real(G[i,i]))*sqrt(real(G[j,j]))
            changed=true;break
        end
    end
    changed || return false
    F=try
        eigen_hermitian(G)
    catch err
        err isa ConvergenceError || rethrow()
        return false
    end
    minimum(F.values)>sqrt(eps(R))*maximum(F.values) || return false
    copyto!(W,F.vectors)
    true
end
include("dense_svd_jacobi_tuning.jl")
