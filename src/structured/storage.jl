# SPDX-License-Identifier: Apache-2.0
# Compact BLAS storage is a compelling reason for an AbstractMatrix adapter:
# no dense expansion; direct stored-column kernels live in compact_blas.jl.
struct PackedMatrix{T,V<:AbstractVector{T}} <: AbstractMatrix{T}
    data::V
    n::Int
    upper::Bool
    kind::Symbol
    unit::Bool
end
function PackedMatrix(data::AbstractVector,n::Integer;uplo=:U,kind=:symmetric,unit=false)
    Base.require_one_based_indexing(data)
    n>=0 || throw(ArgumentError("negative order"))
    length(data)==Base.checked_mul(n,Base.checked_add(n,1))÷2 || throw(DimensionMismatch("invalid packed buffer length"))
    kind in (:symmetric,:hermitian,:triangular) || throw(ArgumentError("invalid packed matrix kind"))
    PackedMatrix(data,Int(n),_uplo(uplo),kind,unit)
end
Base.size(A::PackedMatrix)=(A.n,A.n)
Base.dataids(A::PackedMatrix)=Base.dataids(A.data)
@inline _pindex(A::PackedMatrix,i,j)=A.upper ? j*(j-1)÷2+i : (j-1)*A.n-(j-1)*(j-2)÷2+i-j+1
@inline function Base.getindex(A::PackedMatrix{T},i::Int,j::Int) where T
    @boundscheck checkbounds(A,i,j)
    i==j && A.kind===:triangular && A.unit && return one(T)
    stored=A.upper ? i<=j : i>=j
    !stored && A.kind===:triangular && return zero(T)
    if A.kind===:hermitian && i==j
        return convert(T,real(@inbounds A.data[_pindex(A,i,j)]))
    end
    v=@inbounds A.data[stored ? _pindex(A,i,j) : _pindex(A,j,i)]
    !stored && A.kind===:hermitian ? conj(v) : v
end
@inline function Base.setindex!(A::PackedMatrix,v,i::Int,j::Int)
    @boundscheck checkbounds(A,i,j)
    stored=A.upper ? i<=j : i>=j
    !stored && A.kind===:triangular && throw(ArgumentError("cannot write unstored triangle"))
    value=A.kind===:hermitian ? (i==j ? real(v) : stored ? v : conj(v)) : v
    @inbounds A.data[stored ? _pindex(A,i,j) : _pindex(A,j,i)]=value
    A
end
function pack_matrix(A::AbstractMatrix{T};uplo=:U,kind=:symmetric,unit=false) where T
    n=_square(A); upper=_uplo(uplo)
    P=PackedMatrix(Vector{T}(undef,n*(n+1)÷2),n;uplo,kind,unit)
    @inbounds for j in 1:n, i in (upper ? (1:j) : (j:n))
        P[i,j]=kind===:triangular && unit && i==j ? one(T) : A[i,j]
    end
    P
end

struct BandedMatrix{T,V<:AbstractMatrix{T}} <: AbstractMatrix{T}
    data::V
    m::Int
    kl::Int
    ku::Int
    kind::Symbol
    upper::Bool
    unit::Bool
end
"""
    BandedMatrix(data, m; kl=0, ku=0, kind=:general, uplo=:U, unit=false)

BLAS band storage: general A[i,j]=data[ku+1+i-j,j].
"""
function BandedMatrix(data::AbstractMatrix,m::Integer;kl=0,ku=0,kind=:general,uplo=:U,unit=false)
    Base.require_one_based_indexing(data)
    min(m,kl,ku)>=0 || throw(ArgumentError("negative band dimension"))
    kind in (:general,:symmetric,:hermitian,:triangular) || throw(ArgumentError("invalid band matrix kind"))
    upper=_uplo(uplo)
    rows=kind===:general ? Base.checked_add(Base.checked_add(kl,ku),1) : max(kl,ku)+1
    size(data,1)>=rows || throw(DimensionMismatch("band buffer has too few rows"))
    kind!==:general && m!=size(data,2) && throw(DimensionMismatch("structured band matrix must be square"))
    if kind!==:general
        kd=max(kl,ku); kl=upper ? 0 : kd; ku=upper ? kd : 0
    end
    BandedMatrix(data,Int(m),Int(kl),Int(ku),kind,upper,unit)
end
Base.size(A::BandedMatrix)=(A.m,size(A.data,2))
Base.dataids(A::BandedMatrix)=Base.dataids(A.data)
@inline function Base.getindex(A::BandedMatrix{T},i::Int,j::Int) where T
    @boundscheck checkbounds(A,i,j)
    i==j && A.kind===:triangular && A.unit && return one(T)
    reflect=A.kind in (:symmetric,:hermitian) && (A.upper ? i>j : i<j)
    ii,jj=reflect ? (j,i) : (i,j)
    (ii-jj>A.kl || jj-ii>A.ku) && return zero(T)
    row=A.ku+1+ii-jj
    v=@inbounds A.data[row,jj]
    A.kind===:hermitian && i==j && return convert(T,real(v))
    reflect && A.kind===:hermitian ? conj(v) : v
end
function band_matrix(A::AbstractMatrix{T};kl=0,ku=0,kind=:general,uplo=:U,unit=false) where T
    m,n=size(A); upper=_uplo(uplo)
    rows=kind===:general ? kl+ku+1 : max(kl,ku)+1
    data=zeros(T,rows,n)
    P=BandedMatrix(data,m;kl,ku,kind,uplo,unit)
    @inbounds for j in 1:n, i in max(1,j-P.ku):min(m,j+P.kl)
        data[P.ku+1+i-j,j]=kind===:triangular && unit && i==j ? one(T) : A[i,j]
    end
    P
end

function _pkcheck(P,kind)
    P.kind===kind || throw(ArgumentError("expected $kind compact storage"))
    P
end
