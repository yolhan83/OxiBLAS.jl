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
function band_lu(A::BandedMatrix{T};check::Bool=true) where {T<:GenericFloat}
    _pkcheck(A,:general); n=_square(A); kl=A.kl; ku=A.ku
    all(isfinite,A.data) || throw(ArgumentError("non-finite band LU entries"))
    F=BandLUFactor(zeros(T,2kl+ku+1,n),collect(1:n),n,kl,ku)
    for j in 1:n, i in max(1,j-ku):min(n,j+kl)
        _bandlu_set!(F,A.data[ku+1+i-j,j],i,j)
    end
    lastcol=0
    for j in 1:n
        pivot=j
        for i in j+1:min(n,j+kl)
            v=_bandlu_get(F,i,j); isfinite(v) || throw(ArgumentError("band LU arithmetic overflow"))
            abs(v)>abs(_bandlu_get(F,pivot,j)) && (pivot=i)
        end
        pivval=_bandlu_get(F,pivot,j)
        isfinite(pivval) || throw(ArgumentError("band LU arithmetic overflow"))
        if iszero(pivval)
            check && throw(SingularException(j))
            continue
        end
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
            u=_bandlu_get(F,i,i); iszero(u) && throw(SingularException(i))
            value=X[i,rhs]
            for j in i+1:min(n,i+F.ku+F.kl)
                value-=_bandlu_get(F,i,j)*X[j,rhs]
            end
            X[i,rhs]=value/u
        end
    else
        for rhs in axes(X,2), i in 1:n
            a=_bandlu_get(F,i,i); iszero(a) && throw(SingularException(i))
            value=X[i,rhs]
            for j in max(1,i-F.ku-F.kl):i-1
                aij=_bandlu_get(F,j,i); value-=(tr==2 ? conj(aij) : aij)*X[j,rhs]
            end
            X[i,rhs]=value/(tr==2 ? conj(a) : a)
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
    F,_=_band_cholesky_info(A,true)
    F
end
function _band_cholesky_info(A::BandedMatrix{T},check::Bool) where {T<:GenericFloat}
    A.kind in (:hermitian,:symmetric) && (T<:Real || A.kind===:hermitian) || throw(ArgumentError("band Cholesky requires Hermitian storage"))
    n=_square(A); kd=max(A.kl,A.ku); F=BandCholeskyFactor(zeros(T,kd+1,n),n,kd)
    @inbounds for j in 1:n
        d=real(A[j,j])
        for k in max(1,j-kd):j-1
            d-=abs2(_bandchol_get(F,j,k))
        end
        if !(isfinite(d) && d>0)
            check && throw(PosDefException(j))
            fill!(F.data,zero(T))
            return F,j
        end
        root=sqrt(d); F.data[1,j]=root
        for i in j+1:min(n,j+kd)
            value=A[i,j]
            for k in max(1,i-kd,j-kd):j-1
                value-=_bandchol_get(F,i,k)*conj(_bandchol_get(F,j,k))
            end
            F.data[1+i-j,j]=value/root
        end
    end
    F,0
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
@inline _pcol_base(n::Int,j::Int)=(j-1)*n-((j-1)*(j-2))>>1
@inline _cmadd_neg(a,b,c)=muladd(-a,b,c)
@inline _cmadd_neg(a::Complex{R},b::Complex{R},c::Complex{R}) where {R<:RealFloat}=
    Complex(muladd(-real(a),real(b),muladd(imag(a),imag(b),real(c))),
            muladd(-real(a),imag(b),muladd(-imag(a),real(b),imag(c))))
@inline function _packed_update_col1_panel!(d::Vector{T},n::Int,k0::Int,k1::Int,j::Int) where T
    @inbounds begin
        bj=_pcol_base(n,j)+1; len_j=n-j+1; k=k0
        while k+3<=k1
            b1=_pcol_base(n,k);   o1=b1+j-k+1; c1=conj(d[o1])
            b2=_pcol_base(n,k+1); o2=b2+j-k;   c2=conj(d[o2])
            b3=_pcol_base(n,k+2); o3=b3+j-k-1; c3=conj(d[o3])
            b4=_pcol_base(n,k+3); o4=b4+j-k-2; c4=conj(d[o4])
            @simd ivdep for p in 0:(len_j-1)
                acc=d[bj+p]
                acc=_cmadd_neg(d[o1+p],c1,acc)
                acc=_cmadd_neg(d[o2+p],c2,acc)
                acc=_cmadd_neg(d[o3+p],c3,acc)
                acc=_cmadd_neg(d[o4+p],c4,acc)
                d[bj+p]=acc
            end
            k+=4
        end
        while k<=k1
            bk=_pcol_base(n,k); off_k=bk+(j-k)+1; c=conj(d[off_k])
            @simd ivdep for p in 0:(len_j-1)
                d[bj+p]=_cmadd_neg(d[off_k+p],c,d[bj+p])
            end
            k+=1
        end
    end
    nothing
end
@inline function _packed_update_col2_panel!(d::Vector{T},n::Int,k0::Int,k1::Int,j::Int) where T
    @inbounds begin
        bj0=_pcol_base(n,j)+1; bj1=_pcol_base(n,j+1)+1; len1=n-j; k=k0
        while k+3<=k1
            b1=_pcol_base(n,k);   o1=b1+j-k+1
            b2=_pcol_base(n,k+1); o2=b2+j-k
            b3=_pcol_base(n,k+2); o3=b3+j-k-1
            b4=_pcol_base(n,k+3); o4=b4+j-k-2
            c10=conj(d[o1]); c11=conj(d[o1+1])
            c20=conj(d[o2]); c21=conj(d[o2+1])
            c30=conj(d[o3]); c31=conj(d[o3+1])
            c40=conj(d[o4]); c41=conj(d[o4+1])
            acc0=d[bj0]
            acc0=_cmadd_neg(d[o1],c10,acc0)
            acc0=_cmadd_neg(d[o2],c20,acc0)
            acc0=_cmadd_neg(d[o3],c30,acc0)
            acc0=_cmadd_neg(d[o4],c40,acc0)
            d[bj0]=acc0
            @simd ivdep for p in 1:len1
                v1=d[o1+p]; v2=d[o2+p]; v3=d[o3+p]; v4=d[o4+p]
                a0=d[bj0+p]
                a0=_cmadd_neg(v1,c10,a0)
                a0=_cmadd_neg(v2,c20,a0)
                a0=_cmadd_neg(v3,c30,a0)
                a0=_cmadd_neg(v4,c40,a0)
                d[bj0+p]=a0
                a1=d[bj1+p-1]
                a1=_cmadd_neg(v1,c11,a1)
                a1=_cmadd_neg(v2,c21,a1)
                a1=_cmadd_neg(v3,c31,a1)
                a1=_cmadd_neg(v4,c41,a1)
                d[bj1+p-1]=a1
            end
            k+=4
        end
        while k<=k1
            bk=_pcol_base(n,k); off=bk+(j-k)+1
            c0=conj(d[off]); c1=conj(d[off+1])
            d[bj0]=_cmadd_neg(d[off],c0,d[bj0])
            @simd ivdep for p in 1:len1
                v=d[off+p]
                d[bj0+p]=_cmadd_neg(v,c0,d[bj0+p])
                d[bj1+p-1]=_cmadd_neg(v,c1,d[bj1+p-1])
            end
            k+=1
        end
    end
    nothing
end
function _packed_panel_threaded!(d::Vector{T},n::Int,k0::Int,k1::Int,workers::Int) where T
    npairs=(n-k1)>>1
    @sync for w in 1:workers
        Threads.@spawn begin
            for pair in w:workers:npairs
                j=k1+2*pair-1
                _packed_update_col2_panel!(d,n,k0,k1,j)
            end
            if w==1 && isodd(n-k1)
                _packed_update_col1_panel!(d,n,k0,k1,n)
            end
        end
    end
    nothing
end
function _packed_chol_kernel!(d::Vector{T},n::Int,threaded::Bool,check::Bool) where T
    R=real(T); nb=T<:Complex ? 32 : 64; k0=1
    while k0<=n
        k1=min(n,k0+nb-1)
        @inbounds for k in k0:k1
            _packed_update_col1_panel!(d,n,k0,k-1,k)
            bk=_pcol_base(n,k)
            diag_val=real(d[bk+1])
            if !(isfinite(diag_val) && diag_val>zero(R))
                check && throw(PosDefException(k))
                return k
            end
            root=sqrt(diag_val)
            d[bk+1]=root
            inv_root=Base.inv(root)
            len=n-k
            @simd ivdep for p in 1:len
                d[bk+1+p]*=inv_root
            end
        end
        rem=n-k1
        if rem>0
            work=Float64(rem)*Float64(rem+1)*0.5*Float64(k1-k0+1)
            workers=threaded && rem>=256 && work>=1_500_000.0 ? min(Threads.nthreads(),rem>>5) : 1
            if workers<=1
                j=k1+1
                while j+1<=n
                    _packed_update_col2_panel!(d,n,k0,k1,j)
                    j+=2
                end
                j<=n && _packed_update_col1_panel!(d,n,k0,k1,j)
            else
                _packed_panel_threaded!(d,n,k0,k1,workers)
            end
        end
        k0=k1+1
    end
    0
end
"""Native packed Cholesky, retaining only n(n+1)/2 lower entries."""
function packed_cholesky(A::PackedMatrix{T};threaded::Bool=true) where {T<:GenericFloat}
    F,_=_packed_cholesky_info(A,true;threaded)
    F
end
function _packed_cholesky_info(A::PackedMatrix{T},check::Bool;threaded::Bool=true) where {T<:GenericFloat}
    A.kind in (:hermitian,:symmetric) && (T<:Real || A.kind===:hermitian) || throw(ArgumentError("packed Cholesky requires Hermitian storage"))
    threaded=threaded && generic_threadsafe(T)
    n=size(A,1); d=Vector{T}(undef,length(A.data))
    L=PackedMatrix(d,n;uplo=:L,kind=:triangular)
    upper=A.upper; herm=A.kind===:hermitian; adata=A.data
    if upper
        idx=1
        @inbounds for j in 1:n
            diag_v=real(adata[(j*(j+1))>>1])
            if !check && !(isfinite(diag_v) && diag_v>0)
                fill!(d,zero(T))
                return PackedCholeskyFactor(L),j
            end
            d[idx]=diag_v; idx+=1
            if herm
                for i in (j+1):n
                    d[idx]=conj(adata[((i*(i-1))>>1)+j]); idx+=1
                end
            else
                for i in (j+1):n
                    d[idx]=adata[((i*(i-1))>>1)+j]; idx+=1
                end
            end
        end
    else
        if !check
            @inbounds for j in 1:n
                bk=_pcol_base(n,j)
                diag_v=real(adata[bk+1])
                if !(isfinite(diag_v) && diag_v>0)
                    fill!(d,zero(T))
                    return PackedCholeskyFactor(L),j
                end
            end
        end
        copyto!(d,adata)
    end
    info=_packed_chol_kernel!(d,n,threaded,check)
    info!=0 && fill!(d,zero(T))
    PackedCholeskyFactor(L),info
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
    X=ndims(b)==1 ? reshape(b[F.p],n,1) : b[F.p,:]; trsm!(X,F.L;uplo=:L,unit=true)
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
