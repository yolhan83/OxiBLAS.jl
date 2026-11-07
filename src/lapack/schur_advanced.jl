# SPDX-License-Identifier: Apache-2.0
"""
    schur_reorder(F::SchurFactor{T}, select::Function) -> SchurFactor

Stably move selected eigenvalues to the leading complex Schur block.
Adjacent unitary swaps preserve A=Z*T*Z'. Repeated inseparable eigenvalues
are rejected if a requested swap cannot distinguish their invariant subspaces.
"""
function schur_reorder(F::SchurFactor{T},select::Function) where {T<:Complex}
    A=copy(F.T); Z=copy(F.Z); n=size(A,1)
    chosen=[Bool(select(A[i,i])) for i in 1:n]; next=1
    for j in 1:n
        chosen[j] || continue
        for k in j-1:-1:next
            a=A[k,k]; d=A[k+1,k+1]; b=A[k,k+1]
            scale=max(abs(b),abs(d-a))
            if iszero(scale)
                u=zero(T); v=one(T)
            else
                x=b/scale; y=(d-a)/scale; r=hypot(abs(x),abs(y))
                iszero(y) && !iszero(x) && throw(ArgumentError("inseparable repeated Schur eigenvalues"))
                u=x/r; v=y/r
            end
            _unitary2_similarity!(A,Z,k,u,v); A[k+1,k]=zero(T)
            chosen[k],chosen[k+1]=chosen[k+1],chosen[k]
        end
        next+=1
    end
    SchurFactor(A,Z,[A[i,i] for i in 1:n],F.iterations)
end

"""
    eigen_general(A; algorithm=:auto, threaded=true, balance=false, left=false, maxiter=...) -> NamedTuple

General eigenvalues and eigenvectors via the Schur path and block TREVC substitution:
`(; values, vectors, left_vectors, schur, balancing)`.

Real 2×2 blocks are preserved during factorization; the returned eigenvectors have one
complex column per eigenvalue. `algorithm=:auto` selects between standard and multishift
Schur decomposition. `balance=true` preconditions the matrix via diagonal similarity.
"""
function eigen_general(A::AbstractMatrix{T};maxiter=100max(1,size(A,1)),threaded=true,balance=false,left=false,algorithm=:auto) where {T<:GenericFloat}
    threaded=threaded && generic_threadsafe(T)
    n=_square(A); C=Complex{real(T)}
    if balance
        B=balance_matrix(A); E=eigen_general(B.matrix;maxiter,threaded,balance=false,left,algorithm)
        V=balance_backtransform(B,E.vectors)
        W=left ? balance_backtransform(B,E.left_vectors;left=true) : nothing
        return (;values=E.values,vectors=V,left_vectors=W,schur=E.schur,balancing=B)
    end
    F=schur_factor(A;maxiter,algorithm,threaded); X=_ordinary_vectors(F.T)
    vectors=_mm(C.(F.Z),X;threaded)
    left_vectors=left ? _mm(C.(F.Z),_ordinary_vectors(F.T;left=true);threaded) : nothing
    (;values=F.values,vectors,left_vectors,schur=F,balancing=nothing)
end

_general_left_vectors(F;threaded=true)=_mm(Complex{real(eltype(F.T))}.(F.Z),_ordinary_vectors(F.T;left=true);threaded)
