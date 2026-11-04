# SPDX-License-Identifier: Apache-2.0
# Adapted from the Householder/compact-factor architecture in upstream
# evd/hessenberg.rs and evd/complex_hessenberg.rs. See NOTICE.

struct HessenbergFactor{T,R}
    factors::Matrix{T}
    tau::Vector{R}
    scale::R
end

function _identity(::Type{T},n) where T
    Q=zeros(T,n,n)
    @inbounds for i in 1:n
        Q[i,i]=one(T)
    end
    Q
end

# Maximum component, not complex abs: even finite complex inputs can have a
# magnitude outside the real type's range. Division is componentwise to avoid
# complex reciprocal overflow on subnormal scales.
function _finite_scale(A::AbstractArray{T}) where T
    R=real(T); scale=zero(R)
    for z in A
        isfinite(z) || throw(ArgumentError("factorization requires finite input"))
        scale=max(scale,abs(real(z)),abs(imag(z)))
    end
    iszero(scale) ? one(R) : scale
end
@inline _realdiv(x::Real,s::Real)=x/s
@inline _realdiv(x::Complex,s::Real)=complex(real(x)/s,imag(x)/s)

function _normalized_copy(A::AbstractMatrix{T}) where T
    scale=_finite_scale(A)
    F=Matrix{T}(undef,size(A))
    @inbounds for j in axes(F,2), i in axes(F,1)
        F[i,j]=_realdiv(A[i,j],scale)
    end
    F,scale
end

# H=I-tau*v*v', H=H', v[1]=1, H*x=beta*e1. Returns real tau.
# x is overwritten in owned scratch, so there is no allocation per reflector.
function _householder!(v::AbstractVector{T}) where T
    R=real(T)
    isempty(v) && return (zero(T),zero(R))
    x1=v[1]
    tail=nrm2(view(v,2:length(v)))
    if iszero(tail)
        v[1]=one(T)
        return (x1,zero(R))
    end
    r=hypot(abs(x1),tail)
    phase=iszero(x1) ? one(T) : _realdiv(x1,abs(x1))
    den=_realdiv(x1,r)+phase
    τ=one(R)+abs(x1)/r
    @inbounds for i in 2:length(v)
        v[i]=_realdiv(v[i],r)/den
    end
    v[1]=one(T)
    (-phase*r,τ)
end

function _reflect_left!(A,v,τ,firstrow,cols)
    iszero(τ) && return A
    @inbounds for j in cols
        s=zero(eltype(A))
        for t in eachindex(v)
            s+=conj(v[t])*A[firstrow+t-1,j]
        end
        s*=τ
        for t in eachindex(v)
            A[firstrow+t-1,j]-=v[t]*s
        end
    end
    A
end

function _reflect_right!(A,v,τ,firstcol,rows,w)
    iszero(τ) && return A
    # Column traversal permits SIMD and avoids a strided row inner loop.
    @inbounds for i in rows
        w[i]=zero(eltype(w))
    end
    @inbounds for t in eachindex(v)
        z=v[t]
        @simd for i in rows
            w[i]=_arithmetic_madd(A[i,firstcol+t-1],z,w[i])
        end
    end
    @inbounds for t in eachindex(v)
        z=τ*conj(v[t])
        @simd for i in rows
            A[i,firstcol+t-1]-=w[i]*z
        end
    end
    A
end

"""
    hessenberg_factor(A)

Compact real/complex Householder reduction A=Q*H*Q'. Neither A nor its views
are modified. F.H materializes upper Hessenberg H; F.Q materializes Q. Reflector
tails and real tau are retained for implicit application. Input is normalized
before reduction to protect finite very large/small matrices. The stored matrix
is normalized; F.scale restores the upper Hessenberg entries, not the tails.
"""
function hessenberg_factor(A::AbstractMatrix{T}) where {T<:GenericFloat}
    n=_square(A)
    F,scale=_normalized_copy(A)
    τ=zeros(real(T),max(0,n-2)); v=Vector{T}(undef,n); w=Vector{T}(undef,n)
    @inbounds for k in 1:n-2
        len=n-k; vv=view(v,1:len)
        for t in 1:len
            vv[t]=F[k+t,k]
        end
        β,τk=_householder!(vv)
        τ[k]=τk
        _reflect_left!(F,vv,τk,k+1,k+1:n)
        _reflect_right!(F,vv,τk,k+1,1:n,w)
        F[k+1,k]=β
        for t in 2:len
            F[k+t,k]=vv[t]
        end
    end
    HessenbergFactor(F,τ,scale)
end

function _hessenberg_matrix(F::HessenbergFactor;rescale=true)
    a=F.factors; n=size(a,1); H=zeros(eltype(a),n,n)
    @inbounds for j in 1:n, i in 1:min(n,j+1)
        H[i,j]=rescale ? a[i,j]*F.scale : a[i,j]
    end
    H
end

# Compact WY: H_first...H_last = I - V*T*V'. Upper-triangular T is formed
# from the same compact reflectors used by the unblocked application.
function _hessenberg_wy!(V,Tmat,F,first,last)
    fill!(V,zero(eltype(V))); fill!(Tmat,zero(eltype(Tmat)))
    n=size(F.factors,1); width=last-first+1
    dots=Vector{eltype(V)}(undef,width)
    @inbounds for j in 1:width
        k=first+j-1
        V[k+1,j]=one(eltype(V))
        for i in k+2:n
            V[i,j]=F.factors[i,k]
        end
        τ=F.tau[k]; Tmat[j,j]=τ
        for l in 1:j-1
            inner=zero(eltype(V))
            for row in k+1:n
                inner+=conj(V[row,l])*V[row,j]
            end
            dots[l]=inner
        end
        # This column is built top to bottom, so earlier columns of T are fixed.
        for i in 1:j-1
            s=zero(eltype(V))
            for l in i:j-1
                s+=Tmat[i,l]*dots[l]
            end
            Tmat[i,j]=-τ*s
        end
    end
    nothing
end

function _apply_hessenberg_blocked!(B,F,side,kind,blocksize)
    n=size(F.factors,1); p=length(F.tau); bs=min(blocksize,p)
    p==0 && return B
    V=zeros(eltype(B),n,bs); TT=zeros(eltype(B),bs,bs)
    W=side===:L ? Matrix{eltype(B)}(undef,bs,size(B,2)) : Matrix{eltype(B)}(undef,size(B,1),bs)
    Z=similar(W)
    config=GemmConfig(eltype(B))
    workspace=GemmWorkspace(eltype(B),max(n,size(B,1)),max(n,size(B,2)),n;config,workers=1)
    panels=cld(p,bs)
    reverse=(side===:L && kind==0) || (side===:R && kind!=0)
    for panel in (reverse ? (panels:-1:1) : (1:panels))
        first=(panel-1)*bs+1; last=min(p,first+bs-1); width=last-first+1
        vf=view(V,:,1:width); t=view(TT,1:width,1:width)
        _hessenberg_wy!(vf,t,F,first,last)
        # Reflectors have no support before first+1. Restrict the multiplication
        # to that trailing region rather than multiplying their leading zeros.
        v=view(V,first+1:n,1:width)
        C=side===:L ? view(B,first+1:n,:) : view(B,:,first+1:n)
        # For Q^T, conjugate V and transpose T; Q^H conjugates T only.
        if kind==1
            @inbounds for idx in eachindex(v)
                v[idx]=conj(v[idx])
            end
        end
        tt=kind==0 ? t : kind==1 ? transpose(t) : adjoint(t)
        if side===:L
            w=view(W,1:width,:); z=view(Z,1:width,:)
            gemm!(w,adjoint(v),C;threaded=false,workspace,config)
            gemm!(z,tt,w;threaded=false,workspace,config)
            gemm!(C,v,z,-one(eltype(B)),one(eltype(B));threaded=false,workspace,config)
        else
            w=view(W,:,1:width); z=view(Z,:,1:width)
            gemm!(w,C,v;threaded=false,workspace,config)
            gemm!(z,w,tt;threaded=false,workspace,config)
            gemm!(C,z,adjoint(v),-one(eltype(B)),one(eltype(B));threaded=false,workspace,config)
        end
    end
    B
end

"""
    apply_hessenberg_q!(B,F;side=:L,trans=:N,algorithm=:auto,blocksize=32)

Apply Q, Qᵀ or Qᴴ from compact F, on either side, without materializing Q.
Vectors support left application. Algorithm :blocked uses compact WY and this
package's GEMM; :unblocked applies individual reflectors. Overlap with F is
snapshotted. F and B must have identical element types.
"""
function apply_hessenberg_q!(B::AbstractVecOrMat{T},F::HessenbergFactor{T};
                              side=:L,trans=:N,algorithm=:auto,blocksize=32) where T
    Base.require_one_based_indexing(B)
    side in (:L,:R) || throw(ArgumentError("side must be L or R"))
    kind=_trans_kind(trans)
    algorithm in (:auto,:blocked,:unblocked) || throw(ArgumentError("unknown reflector application algorithm"))
    blocksize>0 || throw(ArgumentError("blocksize must be positive"))
    B isa AbstractVector && side===:R && throw(ArgumentError("right application requires a matrix"))
    n=size(F.factors,1)
    n==(side===:L ? size(B,1) : size(B,2)) || throw(DimensionMismatch("Hessenberg Q application dimensions differ"))
    C=B isa AbstractVector ? reshape(B,:,1) : B
    _check_output_layout(C)
    ff=Base.mightalias(B,F.factors) || Base.mightalias(B,F.tau) ?
       HessenbergFactor(copy(F.factors),copy(F.tau),F.scale) : F
    (isempty(ff.tau) || isempty(C)) && return B
    useblocked=algorithm===:blocked || (algorithm===:auto && n>=96 && min(size(C)...)>=16)
    if useblocked
        _apply_hessenberg_blocked!(C,ff,side,kind,blocksize)
        return B
    end
    v=Vector{T}(undef,n); w=Vector{T}(undef,size(C,1))
    p=length(ff.tau)
    reverse=(side===:L && kind==0) || (side===:R && kind!=0)
    @inbounds for k in (reverse ? (p:-1:1) : (1:p))
        len=n-k; vv=view(v,1:len); vv[1]=one(T)
        for t in 2:len
            z=ff.factors[k+t,k]
            vv[t]=kind==1 ? conj(z) : z
        end
        if side===:L
            _reflect_left!(C,vv,ff.tau[k],k+1,axes(C,2))
        else
            _reflect_right!(C,vv,ff.tau[k],k+1,axes(C,1),w)
        end
    end
    B
end

function hessenberg_q(F::HessenbergFactor{T};algorithm=:auto,blocksize=32) where T
    Q=_identity(T,size(F.factors,1))
    apply_hessenberg_q!(Q,F;side=:R,algorithm,blocksize)
end
function Base.getproperty(F::HessenbergFactor,s::Symbol)
    s===:H && return _hessenberg_matrix(F)
    s===:Q && return hessenberg_q(F)
    getfield(F,s)
end
Base.propertynames(::HessenbergFactor)=(:H,:Q,:factors,:tau,:scale)
