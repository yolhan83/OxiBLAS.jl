# SPDX-License-Identifier: Apache-2.0
# Householder Hermitian rank-2 reduction, phase correction and implicit shifted
# tridiagonal QR, adapting upstream evd/{symmetric,hermitian,hermitian_tridiag}.rs.

struct TridiagonalFactor{T,R}
    reflectors::HessenbergFactor{T,R}
    diag::Vector{R}              # normalized, not in input units
    offdiag::Vector{R}           # real, nonnegative, normalized
    phases::Vector{T}
end

"""
    tridiagonal_factor(A;uplo=:U)

Reduce a symmetric/Hermitian matrix to a real symmetric tridiagonal:
A=Q*T*Q'. Only the chosen triangle and real diagonal are read. The compact
Householder representation retains the diagonal phase correction needed for
complex eigenvectors. F.diagonal/F.offdiagonal are in input units; the fields
diag/offdiag are normalized by F.reflectors.scale.
"""
function tridiagonal_factor(A::AbstractMatrix{T};uplo=:U) where {T<:GenericFloat}
    n=_square(A); upper=_uplo(uplo); R=real(T)
    S=Matrix{T}(undef,n,n)
    @inbounds for j in 1:n, i in 1:n
        S[i,j]=_symget(A,i,j,upper,true)
    end
    scale=_finite_scale(S)
    @inbounds for idx in eachindex(S)
        S[idx]=_realdiv(S[idx],scale)
    end
    tau=zeros(R,max(0,n-2)); v=Vector{T}(undef,n); w=Vector{T}(undef,n)
    @inbounds for k in 1:n-2
        len=n-k; vv=view(v,1:len); ww=view(w,1:len)
        for t in 1:len
            vv[t]=S[k+t,k]
        end
        β,τ=_householder!(vv); tau[k]=τ
        if !iszero(τ)
            trailing=view(S,k+1:n,k+1:n)
            gemv!(ww,trailing,vv,T(τ),zero(T))
            correction=(τ/2)*real(dotc(vv,ww))
            for t in 1:len
                ww[t]-=correction*vv[t]
            end
            # Preserve Hermitian symmetry exactly rather than separately
            # rounding both halves of the rank-2 update.
            her2!(trailing,vv,ww,-one(T);uplo=:L)
            for j in 1:len, i in j+1:len
                trailing[j,i]=conj(trailing[i,j])
            end
        end
        S[k+1,k]=β; S[k,k+1]=conj(β)
        for t in 2:len
            S[k+t,k]=vv[t]
            S[k,k+t]=zero(T)
        end
    end
    diagonal=[real(S[i,i]) for i in 1:n]
    offdiag=Vector{R}(undef,max(0,n-1)); phases=ones(T,n)
    @inbounds for i in 1:n-1
        e=S[i+1,i]; mag=abs(e); offdiag[i]=mag
        phase=iszero(mag) ? phases[i] : phases[i]*_realdiv(e,mag)
        phases[i+1]=_realdiv(phase,abs(phase))
    end
    TridiagonalFactor(HessenbergFactor(S,tau,scale),diagonal,offdiag,phases)
end

function tridiagonal_q(F::TridiagonalFactor;algorithm=:auto,blocksize=32)
    Q=hessenberg_q(F.reflectors;algorithm,blocksize)
    @inbounds for j in axes(Q,2), i in axes(Q,1)
        Q[i,j]*=F.phases[j]
    end
    Q
end
function Base.getproperty(F::TridiagonalFactor,s::Symbol)
    if s===:T
        d=getfield(F,:diag); e=getfield(F,:offdiag); n=length(d)
        scale=getfield(F,:reflectors).scale; R=eltype(d)
        H=zeros(R,n,n)
        @inbounds for i in 1:n
            H[i,i]=d[i]*scale
            if i<n
                H[i,i+1]=e[i]*scale; H[i+1,i]=e[i]*scale
            end
        end
        return H
    elseif s===:Q
        return tridiagonal_q(F)
    elseif s===:diagonal
        return getfield(F,:diag).*getfield(F,:reflectors).scale
    elseif s===:offdiagonal
        return getfield(F,:offdiag).*getfield(F,:reflectors).scale
    end
    getfield(F,s)
end
Base.propertynames(::TridiagonalFactor)=(:T,:Q,:diagonal,:offdiagonal,:reflectors,:diag,:offdiag,:phases)

struct EigenFactor{T,R}
    values::Vector{R}
    vectors::Union{Nothing,Matrix{T}}
    iterations::Int
    # Inner constructor: replaces the default outer one, whose T was unbound
    # when vectors === nothing.
    EigenFactor{T,R}(values,vectors,iterations) where {T,R}=new{T,R}(values,vectors,iterations)
end
EigenFactor(values::Vector{R},vectors::Matrix{T},iterations::Int) where {T,R}=EigenFactor{T,R}(values,vectors,iterations)
# There is no vector element type to infer when eigenvectors were not requested.
EigenFactor(values::Vector{R},::Nothing,iterations::Int) where R=EigenFactor{R,R}(values,nothing,iterations)

@inline function _tridiag_deflatable(d,e,i)
    R=eltype(d)
    abs(e[i])<=max(floatmin(R)/eps(R),eps(R)*(abs(d[i])+abs(d[i+1])))
end

function _rotate_eigenvectors!(Z,i,c,s)
    Z===nothing && return nothing
    @inbounds @simd for row in axes(Z,1)
        x,y=Z[row,i],Z[row,i+1]
        Z[row,i]=c*x-s*y
        Z[row,i+1]=s*x+c*y
    end
    nothing
end

function _tridiagonal_qr!(d,e,Z,maxiter)
    R=eltype(d); hi=length(d); iterations=0
    while hi>1
        lo=hi
        while lo>1
            if _tridiag_deflatable(d,e,lo-1)
                e[lo-1]=zero(R)
                break
            end
            lo-=1
        end
        if lo==hi
            hi-=1
            continue
        elseif lo==hi-1
            # Closed form symmetric 2x2; also handles equal/zero diagonals.
            delta=(d[hi]-d[lo])/2
            den=delta+copysign(hypot(delta,e[lo]),delta)
            t=e[lo]/den; c=Base.inv(hypot(one(R),t)); s=t*c
            shift=t*e[lo]; d[lo]-=shift; d[hi]+=shift; e[lo]=zero(R)
            _rotate_eigenvectors!(Z,lo,c,s)
            hi-=2
            continue
        end
        iterations>=maxiter && throw(ConvergenceError(:tridiagonal_qr,iterations,lo:hi))
        delta=(d[hi-1]-d[hi])/2
        den=delta+copysign(hypot(delta,e[hi-1]),delta)
        μ=d[hi]-e[hi-1]*(e[hi-1]/den)
        x=d[lo]-μ; z=e[lo]
        @inbounds for k in lo:hi-1
            r=hypot(x,z)
            c=iszero(r) ? one(R) : x/r
            s=iszero(r) ? zero(R) : -z/r
            k>lo && (e[k-1]=r)
            d1,d2,off=d[k],d[k+1],e[k]
            cc,ss,cs=c*c,s*s,c*s
            d[k]=cc*d1+ss*d2-2cs*off
            d[k+1]=ss*d1+cc*d2+2cs*off
            e[k]=cs*(d1-d2)+(cc-ss)*off
            if k<hi-1
                x=e[k]; z=-s*e[k+1]; e[k+1]*=c
            end
            _rotate_eigenvectors!(Z,k,c,s)
        end
        iterations+=1
    end
    iterations
end

function _validate_iteration_budget(maxiter)
    maxiter isa Integer && maxiter>=0 || throw(ArgumentError("maxiter must be a nonnegative integer"))
    nothing
end

"""Eigenpairs of a real symmetric tridiagonal, sorted ascending; implicit QR."""
function eigen_tridiagonal(diagonal::AbstractVector{R},offdiagonal::AbstractVector{R};
                           vectors=true,maxiter=100max(1,length(diagonal))) where {R<:AbstractFloat}
    Base.require_one_based_indexing(diagonal,offdiagonal)
    n=length(diagonal)
    length(offdiagonal)==max(0,n-1) || throw(DimensionMismatch("offdiagonal must have length max(n-1,0)"))
    _validate_iteration_budget(maxiter)
    scale=max(_finite_scale(diagonal),_finite_scale(offdiagonal))
    # _finite_scale(empty/zero)=1 should not destroy tiny nonzero scales.
    raw=max(maximum(abs,diagonal;init=zero(R)),maximum(abs,offdiagonal;init=zero(R)))
    scale=iszero(raw) ? one(R) : raw
    d=diagonal./scale; e=offdiagonal./scale
    Z=vectors ? _identity(R,n) : nothing
    iterations=_tridiagonal_qr!(d,e,Z,maxiter)
    perm=sortperm(d)
    EigenFactor{R,R}(d[perm].*scale,Z===nothing ? nothing : Z[:,perm],iterations)
end

"""
    eigen_hermitian(A;uplo=:U,vectors=true,maxiter=100*max(1,size(A,1)))

Real symmetric/complex Hermitian eigenpairs, ascending eigenvalues. Uses native
rank-2 Householder tridiagonalization, diagonal phase correction and implicit QR.
Only uplo and the real diagonal are read. vectors=false skips Q construction.
Convergence failure throws ConvergenceError; no external BLAS/LAPACK backend.
"""
function eigen_hermitian(A::AbstractMatrix{T};uplo=:U,vectors=true,
                         maxiter=100max(1,size(A,1))) where {T<:GenericFloat}
    _validate_iteration_budget(maxiter)
    F=tridiagonal_factor(A;uplo)
    d=copy(F.diag); e=copy(F.offdiag)
    Z=vectors ? tridiagonal_q(F) : nothing
    iterations=_tridiagonal_qr!(d,e,Z,maxiter)
    perm=sortperm(d)
    EigenFactor{T,real(T)}(d[perm].*F.reflectors.scale,Z===nothing ? nothing : Z[:,perm],iterations)
end
