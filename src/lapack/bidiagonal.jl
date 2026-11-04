# SPDX-License-Identifier: Apache-2.0
"""Compact two-sided Householder reduction. A=U*B*V'; B is upper bidiagonal
for tall inputs and lower bidiagonal for wide inputs. Reflector storage is O(m*n).
U and V are materialized only when the corresponding property is requested.
"""
struct BidiagonalFactor{T,R}
    factors::Matrix{T}
    tauleft::Vector{R}
    tauright::Vector{R}
    scale::R
    wide::Bool
end
function bidiagonal_factor(A::AbstractMatrix{T}) where {T<:GenericFloat}
    Base.require_one_based_indexing(A)
    wide=size(A,1)<size(A,2); M=wide ? adjoint(A) : A
    F,scale=_normalized_copy(M); m,n=size(F); R=real(T)
    tl=zeros(R,n); tr=zeros(R,max(0,n-1)); scratch=Vector{T}(undef,m); work=similar(scratch)
    for k in 1:n
        v=view(scratch,1:m-k+1); copyto!(v,view(F,k:m,k)); β,τ=_householder!(v); tl[k]=τ
        _reflect_left!(F,v,τ,k,k+1:n); F[k,k]=β
        for i in k+1:m
            F[i,k]=v[i-k+1]
        end
        if k<n
            v=view(scratch,1:n-k)
            for j in k+1:n
                v[j-k]=conj(F[k,j])
            end
            β,τ=_householder!(v); tr[k]=τ
            _reflect_right!(F,v,τ,k+1,k+1:m,work); F[k,k+1]=conj(β)
            for j in k+2:n
                F[k,j]=v[j-k]
            end
        end
    end
    BidiagonalFactor(F,tl,tr,scale,wide)
end
function _bidiagonal_vectors(F,left)
    A=F.factors; m,n=size(A); T=eltype(A); q=left ? m : n
    Q=_identity(T,q); scratch=Vector{T}(undef,q); work=similar(scratch)
    for k in 1:(left ? n : max(0,n-1))
        start=left ? k : k+1; len=q-start+1; v=view(scratch,1:len); v[1]=one(T)
        for i in 2:len
            v[i]=left ? A[start+i-1,k] : A[k,start+i-1]
        end
        τ=left ? F.tauleft[k] : F.tauright[k]
        _reflect_right!(Q,v,τ,start,1:q,work)
    end
    Q
end
function Base.getproperty(F::BidiagonalFactor,s::Symbol)
    if s===:U || s===:V
        return _bidiagonal_vectors(F,(s===:U)!=getfield(F,:wide))
    elseif s===:B
        A=getfield(F,:factors); m,n=size(A); B=zeros(eltype(A),m,n); scale=getfield(F,:scale)
        for k in 1:n
            B[k,k]=A[k,k]*scale
            k<n && (B[k,k+1]=A[k,k+1]*scale)
        end
        return getfield(F,:wide) ? Matrix(adjoint(B)) : B
    elseif s===:diagonal
        # A separate name per branch: one `A` assigned in three branches and
        # captured by the comprehensions was a Core.Box.
        Ad=getfield(F,:factors)
        d=[Ad[k,k]*getfield(F,:scale) for k in 1:size(Ad,2)]
        return getfield(F,:wide) ? conj.(d) : d
    elseif s===:offdiagonal
        Ae=getfield(F,:factors)
        e=[Ae[k,k+1]*getfield(F,:scale) for k in 1:max(0,size(Ae,2)-1)]
        return getfield(F,:wide) ? conj.(e) : e
    end
    getfield(F,s)
end
