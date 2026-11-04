# SPDX-License-Identifier: Apache-2.0
struct BandLUFactor{T}
    data::Matrix{T}
    pivots::Vector{Int}
    n::Int
    kl::Int
    ku::Int
end
@inline function _bandlu_get(F,i,j)
    row=F.kl+F.ku+1+i-j
    1<=row<=size(F.data,1) ? F.data[row,j] : zero(eltype(F.data))
end
@inline function _bandlu_set!(F,value,i,j)
    row=F.kl+F.ku+1+i-j
    1<=row<=size(F.data,1) || throw(ArgumentError("band LU fill exceeded storage"))
    F.data[row,j]=value
end
"""Partial-pivoted band LU in (2kl+ku+1)×n compact storage. Local pivot
operations and multipliers are retained in elimination order, as in GBTRF.
No full matrix expansion is made.
"""
function band_lu(A::BandedMatrix{T}) where {T<:GenericFloat}
    _pkcheck(A,:general); n=_square(A); kl=A.kl; ku=A.ku
    F=BandLUFactor(zeros(T,2kl+ku+1,n),collect(1:n),n,kl,ku)
    for j in 1:n, i in max(1,j-ku):min(n,j+kl)
        _bandlu_set!(F,A.data[ku+1+i-j,j],i,j)
    end
    lastcol=0
    for j in 1:n
        pivot=j
        for i in j+1:min(n,j+kl)
            abs(_bandlu_get(F,i,j))>abs(_bandlu_get(F,pivot,j)) && (pivot=i)
        end
        iszero(_bandlu_get(F,pivot,j)) && throw(SingularException(j))
        F.pivots[j]=pivot; lastcol=max(lastcol,min(n,j+ku+pivot-j))
        if pivot!=j
            for col in j:lastcol
                a,b=_bandlu_get(F,j,col),_bandlu_get(F,pivot,col)
                _bandlu_set!(F,b,j,col); _bandlu_set!(F,a,pivot,col)
            end
        end
        diagonal=_bandlu_get(F,j,j)
        for i in j+1:min(n,j+kl)
            _bandlu_set!(F,_bandlu_get(F,i,j)/diagonal,i,j)
        end
        for col in j+1:lastcol
            u=_bandlu_get(F,j,col)
            for i in j+1:min(n,j+kl)
                _bandlu_set!(F,_native_madd(-_bandlu_get(F,i,j),u,_bandlu_get(F,i,col)),i,col)
            end
        end
    end
    F
end
function solve(F::BandLUFactor{T},b::AbstractVecOrMat{T};trans=:N) where T
    Base.require_one_based_indexing(b); size(b,1)==F.n || throw(DimensionMismatch("band solve dimensions differ"))
    tr=_trans_kind(trans); n=F.n; X=ndims(b)==1 ? reshape(copy(b),n,1) : copy(b)
    if tr==0
        for j in 1:n
            p=F.pivots[j]
            if p!=j
                for rhs in axes(X,2)
                    X[j,rhs],X[p,rhs]=X[p,rhs],X[j,rhs]
                end
            end
            for rhs in axes(X,2), i in j+1:min(n,j+F.kl)
                X[i,rhs]-=_bandlu_get(F,i,j)*X[j,rhs]
            end
        end
        for rhs in axes(X,2), i in n:-1:1
            value=X[i,rhs]
            for j in i+1:min(n,i+F.ku+F.kl)
                value-=_bandlu_get(F,i,j)*X[j,rhs]
            end
            X[i,rhs]=value/_bandlu_get(F,i,i)
        end
    else
        for rhs in axes(X,2), i in 1:n
            value=X[i,rhs]
            for j in max(1,i-F.ku-F.kl):i-1
                a=_bandlu_get(F,j,i); value-=(tr==2 ? conj(a) : a)*X[j,rhs]
            end
            a=_bandlu_get(F,i,i); X[i,rhs]=value/(tr==2 ? conj(a) : a)
        end
        for j in n:-1:1
            for rhs in axes(X,2), i in j+1:min(n,j+F.kl)
                a=_bandlu_get(F,i,j); X[j,rhs]-=(tr==2 ? conj(a) : a)*X[i,rhs]
            end
            p=F.pivots[j]
            if p!=j
                for rhs in axes(X,2)
                    X[j,rhs],X[p,rhs]=X[p,rhs],X[j,rhs]
                end
            end
        end
    end
    ndims(b)==1 ? vec(X) : X
end

struct BandCholeskyFactor{T}
    data::Matrix{T}
    n::Int
    kd::Int
end
@inline _bandchol_get(F,i,j)=0<=i-j<=F.kd ? F.data[1+i-j,j] : zero(eltype(F.data))
function band_cholesky(A::BandedMatrix{T}) where {T<:GenericFloat}
    A.kind in (:hermitian,:symmetric) && (T<:Real || A.kind===:hermitian) || throw(ArgumentError("band Cholesky requires Hermitian storage"))
    n=_square(A); kd=max(A.kl,A.ku); F=BandCholeskyFactor(zeros(T,kd+1,n),n,kd)
    for j in 1:n
        d=real(A[j,j])
        for k in max(1,j-kd):j-1
            d-=abs2(_bandchol_get(F,j,k))
        end
        isfinite(d) && d>0 || throw(PosDefException(j))
        root=sqrt(d); F.data[1,j]=root
        for i in j+1:min(n,j+kd)
            value=A[i,j]
            for k in max(1,i-kd,j-kd):j-1
                value-=_bandchol_get(F,i,k)*conj(_bandchol_get(F,j,k))
            end
            F.data[1+i-j,j]=value/root
        end
    end
    F
end
function solve(F::BandCholeskyFactor{T},b::AbstractVecOrMat{T}) where T
    Base.require_one_based_indexing(b); n=F.n
    size(b,1)==n || throw(DimensionMismatch("band Cholesky solve dimensions differ"))
    X=ndims(b)==1 ? reshape(copy(b),n,1) : copy(b)
    for rhs in axes(X,2), i in 1:n
        s=X[i,rhs]
        for j in max(1,i-F.kd):i-1
            s-=_bandchol_get(F,i,j)*X[j,rhs]
        end
        X[i,rhs]=s/F.data[1,i]
    end
    for rhs in axes(X,2), i in n:-1:1
        s=X[i,rhs]
        for j in i+1:min(n,i+F.kd)
            s-=conj(_bandchol_get(F,j,i))*X[j,rhs]
        end
        X[i,rhs]=s/conj(F.data[1,i])
    end
    ndims(b)==1 ? vec(X) : X
end
struct PackedCholeskyFactor{T}
    L::PackedMatrix{T,Vector{T}}
end
"""Native packed Cholesky, retaining only n(n+1)/2 lower entries."""
function packed_cholesky(A::PackedMatrix{T}) where {T<:GenericFloat}
    A.kind in (:hermitian,:symmetric) && (T<:Real || A.kind===:hermitian) || throw(ArgumentError("packed Cholesky requires Hermitian storage"))
    n=size(A,1); L=PackedMatrix(zeros(T,length(A.data)),n;uplo=:L,kind=:triangular)
    for j in 1:n
        d=real(A[j,j])
        for k in 1:j-1
            d-=abs2(L[j,k])
        end
        isfinite(d) && d>0 || throw(PosDefException(j))
        L[j,j]=sqrt(d)
        for i in j+1:n
            s=A[i,j]
            for k in 1:j-1
                s-=L[i,k]*conj(L[j,k])
            end
            L[i,j]=s/L[j,j]
        end
    end
    PackedCholeskyFactor(L)
end
function solve(F::PackedCholeskyFactor{T},b::AbstractVecOrMat{T}) where T
    n=size(F.L,1); Base.require_one_based_indexing(b)
    size(b,1)==n || throw(DimensionMismatch("packed solve dimensions differ"))
    X=ndims(b)==1 ? reshape(copy(b),n,1) : copy(b)
    for j in axes(X,2)
        tpsv!(view(X,:,j),F.L); tpsv!(view(X,:,j),F.L;trans=:C)
    end
    ndims(b)==1 ? vec(X) : X
end

struct AasenFactor{T}
    L::Matrix{T}
    diagonal::Vector{T}
    offdiagonal::Vector{T}
    p::Vector{Int}
    hermitian::Bool
end
function Base.getproperty(F::AasenFactor,s::Symbol)
    if s===:T
        d=getfield(F,:diagonal); e=getfield(F,:offdiagonal); n=length(d); A=zeros(eltype(d),n,n)
        for i in 1:n
            A[i,i]=d[i]
            if i<n
                A[i+1,i]=e[i]; A[i,i+1]=getfield(F,:hermitian) ? conj(e[i]) : e[i]
            end
        end
        return A
    end
    getfield(F,s)
end
"""Partially pivoted Aasen tridiagonal congruence A[p,p]=L*T*L' (or L*T*L^T).
The first column of the unit lower L stays e1. Only uplo is read.
"""
function aasen_factor(A::AbstractMatrix{T};uplo=:L,hermitian=true,threaded=true) where {T<:GenericFloat}
    threaded=threaded && generic_threadsafe(T)
    n=_square(A); upper=_uplo(uplo); S=Matrix{T}(undef,n,n); L=_identity(T,n); perm=collect(1:n)
    for j in 1:n, i in 1:n
        S[i,j]=_symget(A,i,j,upper,hermitian)
    end
    d=zeros(T,n); e=zeros(T,max(0,n-1)); h=zeros(T,n); residual=zeros(T,n); w=zeros(T,n)
    for j in 1:n
        for k in 1:j-1
            h[k]=d[k]*_ldlconj(L[j,k],hermitian)+_ldlconj(e[k],hermitian)*_ldlconj(L[j,k+1],hermitian)
            k>1 && (h[k]+=e[k-1]*_ldlconj(L[j,k-1],hermitian))
        end
        workers=_operation_workers(n-j+1,Float64(n-j+1)*j,threaded,262144)
        _foreach_range(n-j+1,workers) do rows,_
            for offset in rows
                i=j+offset-1; value=S[i,j]
                for k in 1:j-1
                    value-=L[i,k]*h[k]
                end
                residual[i]=value
            end
        end
        hjj=residual[j]; d[j]=hjj-(j>1 ? e[j-1]*_ldlconj(L[j,j-1],hermitian) : zero(T))
        hermitian && (d[j]=real(d[j]))
        j==n && continue
        for i in j+1:n
            w[i]=residual[i]-L[i,j]*hjj
        end
        pivot=j+1
        for i in j+2:n
            abs(w[i])>abs(w[pivot]) && (pivot=i)
        end
        if pivot!=j+1
            _symmetric_swap!(S,L,perm,j+1,pivot,j+1)
            w[j+1],w[pivot]=w[pivot],w[j+1]
        end
        e[j]=w[j+1]
        if !iszero(e[j])
            for i in j+2:n
                L[i,j+1]=w[i]/e[j]
            end
        end
    end
    AasenFactor(L,d,e,perm,hermitian)
end
function solve(F::AasenFactor{T},b::AbstractVecOrMat{T}) where T
    n=length(F.diagonal); Base.require_one_based_indexing(b)
    size(b,1)==n || throw(DimensionMismatch("Aasen solve dimensions differ"))
    X=ndims(b)==1 ? reshape(copy(b[F.p]),n,1) : copy(b[F.p,:]); trsm!(X,F.L;uplo=:L,unit=true)
    storage=zeros(T,3,n)
    for i in 1:n
        storage[2,i]=F.diagonal[i]
        if i<n
            storage[3,i]=F.offdiagonal[i]; storage[1,i+1]=_ldlconj(F.offdiagonal[i],F.hermitian)
        end
    end
    X=solve(band_lu(BandedMatrix(storage,n;kl=1,ku=1,kind=:general)),X)
    trsm!(X,F.L;uplo=:L,trans=F.hermitian ? :C : :T,unit=true)
    result=similar(X)
    for j in axes(X,2), i in 1:n
        result[F.p[i],j]=X[i,j]
    end
    ndims(b)==1 ? vec(result) : result
end
