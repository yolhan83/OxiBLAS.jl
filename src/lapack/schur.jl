# SPDX-License-Identifier: Apache-2.0
# Native implicit QR/bulge chasing, adapting upstream evd/schur.rs and
# evd/complex_schur.rs. No external eigensolver is called.

struct ConvergenceError <: Exception
    algorithm::Symbol
    iterations::Int
    active::UnitRange{Int}
end
Base.showerror(io::IO,e::ConvergenceError)=print(io,e.algorithm," did not converge after ",e.iterations," iterations; active block ",e.active)

struct SchurFactor{T,C}
    T::Matrix{T}
    Z::Matrix{T}
    values::Vector{C}
    iterations::Int
end

# Relative two-stage small-subdiagonal test (Ahues--Tisseur style). The second
# condition guards highly unbalanced off-diagonal pairs; no absolute '+1' scale.
@inline function _deflatable(H,i,lo,hi;tolerance=eps(real(eltype(H))))
    R=real(eltype(H)); epsR=tolerance; small=floatmin(R)/eps(R)
    h=abs(H[i,i-1])
    iszero(h) && return true
    tst=abs(H[i-1,i-1])+abs(H[i,i])
    if iszero(tst)
        i>lo+1 && (tst+=abs(H[i-1,i-2]))
        i<hi && (tst+=abs(H[i+1,i]))
    end
    h>max(small,epsR*tst) && return false
    ab=max(h,abs(H[i-1,i])); ba=min(h,abs(H[i-1,i]))
    aa=max(abs(H[i,i]),abs(H[i-1,i-1]-H[i,i]))
    bb=min(abs(H[i,i]),abs(H[i-1,i-1]-H[i,i]))
    s=aa+ab
    iszero(s) || ba*(ab/s)<=max(small,epsR*(bb*(aa/s)))
end

function _unitary2_similarity!(H,Q,i,u,v)
    n=size(H,1)
    @inbounds for j in 1:n
        x,y=H[i,j],H[i+1,j]
        H[i,j]=conj(u)*x+conj(v)*y
        H[i+1,j]=-v*x+u*y
    end
    @inbounds for row in 1:n
        x,y=H[row,i],H[row,i+1]
        H[row,i]=x*u+y*v
        H[row,i+1]=-x*conj(v)+y*conj(u)
        x,y=Q[row,i],Q[row,i+1]
        Q[row,i]=x*u+y*v
        Q[row,i+1]=-x*conj(v)+y*conj(u)
    end
    nothing
end

# Return normalized 2x2 eigenvalue data, avoiding trace^2-4det cancellation.
function _real_eig2(a,b,c,d)
    scale=max(abs(a),abs(b),abs(c),abs(d))
    iszero(scale) && return (zero(a),zero(a),zero(a),one(a))
    a,b,c,d=a/scale,b/scale,c/scale,d/scale
    mid=a/2+d/2; delta=a/2-d/2
    (mid,delta,delta*delta+b*c,scale)
end

function _finish_real_pair!(H,Q,i)
    a,b,c,d=H[i,i],H[i,i+1],H[i+1,i],H[i+1,i+1]
    mid,delta,disc,scale=_real_eig2(a,b,c,d)
    if disc>=0
        # Use the longer row-derived eigenvector, without divisions by tiny b/c.
        iszero(c) && return nothing
        λ=mid+copysign(sqrt(disc),delta)
        an,bn,cn,dn=a/scale,b/scale,c/scale,d/scale
        u1,v1=bn,λ-an; u2,v2=λ-dn,cn
        if hypot(u1,v1)>=hypot(u2,v2)
            r=hypot(u1,v1); u,v=u1/r,v1/r
        else
            r=hypot(u2,v2); u,v=u2/r,v2/r
        end
        _unitary2_similarity!(H,Q,i,u,v)
        H[i+1,i]=zero(a)
    else
        # Canonical real conjugate-pair block: equal diagonal entries.
        x=(a-d)/scale; y=(b+c)/scale; r=hypot(x,y)
        if !iszero(r)
            co=y/r
            u=sqrt(max(zero(co),(one(co)+co)/2))
            v=copysign(sqrt(max(zero(co),(one(co)-co)/2)),-x)
            _unitary2_similarity!(H,Q,i,u,v)
        end
        center=H[i,i]/2+H[i+1,i+1]/2
        H[i,i]=center; H[i+1,i+1]=center
    end
    nothing
end

function _francis_step!(H,Q,lo,hi,v,w,stagnation)
    a,b,c,d=H[hi-1,hi-1],H[hi-1,hi],H[hi,hi-1],H[hi,hi]
    if stagnation%20==0
        μ=d+eltype(H)(0.75)*(abs(c)+abs(H[hi-1,hi-2]))
        a,d,b,c=μ,μ,zero(μ),zero(μ)
    end
    h=H[lo,lo]; h10=H[lo+1,lo]
    v[1]=(h-a)*(h-d)-b*c+H[lo,lo+1]*h10
    v[2]=h10*((h-a)+(H[lo+1,lo+1]-d))
    v[3]=h10*H[lo+2,lo+1]
    @inbounds for k in lo:hi-1
        r=min(3,hi-k+1); vv=view(v,1:r)
        if k>lo
            for t in 1:r
                vv[t]=H[k+t-1,k-1]
            end
        end
        β,τ=_householder!(vv)
        _reflect_left!(H,vv,τ,k,max(lo,k-1):size(H,2))
        _reflect_right!(H,vv,τ,k,1:min(hi,k+r),w)
        _reflect_right!(Q,vv,τ,k,axes(Q,1),w)
        if k>lo
            H[k,k-1]=β
            for t in 2:r
                H[k+t-1,k-1]=zero(eltype(H))
            end
        end
    end
    nothing
end

function _complex_shift(H,hi)
    a,b,c,d=H[hi-1,hi-1],H[hi-1,hi],H[hi,hi-1],H[hi,hi]
    scale=max(abs(a),abs(b),abs(c),abs(d))
    iszero(scale) && return zero(d)
    a,b,c,d=a/scale,b/scale,c/scale,d/scale
    delta=(a-d)/2
    root=sqrt(delta*delta+b*c)
    # Stable root near d: avoid cancellation in mid-root when det is small.
    plus,minus=delta+root,delta-root
    den=abs(plus)>=abs(minus) ? plus : minus
    iszero(den) ? d*scale : (d-b*c/den)*scale
end

# Analytic 2x2 eigenvector plus unitary completion, as complex_schur.rs.
# Row-derived vectors avoid divisions by a small off-diagonal coefficient.
function _finish_complex_pair!(H,Q,i)
    iszero(H[i+1,i]) && return
    scale=maximum(abs,view(H,i:i+1,i:i+1))
    a,b,c,d=H[i,i]/scale,H[i,i+1]/scale,H[i+1,i]/scale,H[i+1,i+1]/scale
    lambda=_complex_shift(H,i+1)/scale
    u1,v1=b,lambda-a; u2,v2=lambda-d,c
    u,v=hypot(abs(u1),abs(v1))>=hypot(abs(u2),abs(v2)) ? (u1,v1) : (u2,v2)
    norm=hypot(abs(u),abs(v))
    iszero(norm) && return
    _unitary2_similarity!(H,Q,i,u/norm,v/norm); H[i+1,i]=zero(eltype(H))
end

function _complex_qr_step!(H,Q,lo,hi,v,w,stagnation)
    μ=_complex_shift(H,hi)
    if stagnation%11==0
        μ=eltype(H)(abs(H[hi,hi])+abs(H[hi,hi-1]))
    end
    @inbounds for k in lo:hi-1
        x=k==lo ? H[k,k]-μ : H[k,k-1]
        y=k==lo ? H[k+1,k] : H[k+1,k-1]
        c,s,_=rotg(x,y)
        _unitary2_similarity!(H,Q,k,c,conj(s))
        if k>lo
            H[k+1,k-1]=zero(eltype(H))
        end
    end
    nothing
end

function _schur_values(H::AbstractMatrix{T},scale) where {T<:AbstractFloat}
    n=size(H,1); values=Vector{Complex{T}}(undef,n); i=1
    @inbounds while i<=n
        if i<n && !iszero(H[i+1,i])
            mid,delta,disc,s=_real_eig2(H[i,i],H[i,i+1],H[i+1,i],H[i+1,i+1])
            root=sqrt(complex(disc))
            values[i]=(complex(mid)+root)*s*scale
            values[i+1]=(complex(mid)-root)*s*scale
            i+=2
        else
            values[i]=H[i,i]*scale
            i+=1
        end
    end
    values
end
_schur_values(H::AbstractMatrix{T},scale) where {T<:Complex}=[H[i,i]*scale for i in axes(H,1)]

"""
    schur_factor(A; algorithm=:auto, threaded=true, maxiter=100*max(1,size(A,1)), kwargs...) -> SchurFactor

Native Schur decomposition `A = Z·T·Zᴴ`.

- `algorithm=:auto` (default): dynamically chooses between shifted QR and multishift
  BLAS-3 based on matrix dimension and active [`SpectralConfig`](@ref). For native
  floats, matrices with `n >= 192` (complex) or `n >= 256` (real) route to
  [`schur_multishift`](@ref).
- `algorithm=:standard`: Francis double-shift bulge chasing for real inputs (returning
  real quasi-triangular `T` with 1×1 and 2×2 blocks) and implicit single-shift QR
  for complex inputs.
- `algorithm=:multishift`: multi-shift QR with Aggressive Early Deflation (AED) and
  blocked GEMM accumulations across threads.
"""
function schur_factor(A::AbstractMatrix{T};maxiter=100max(1,size(A,1)),algorithm=:auto,threaded=true,kwargs...) where {T<:GenericFloat}
    threaded=threaded && generic_threadsafe(T)
    if algorithm===:auto
        cfg=get_spectral_config(T)
        n=_square(A)
        cutoff=cfg!==nothing ? cfg.schur_multishift_threshold : ((T<:Complex) ? 192 : 256)
        preferred_algo=cfg!==nothing ? cfg.schur_algorithm : :multishift
        if T<:NativeFloat && n>=cutoff && maxiter>0 && preferred_algo===:multishift
            return schur_multishift(A;maxiter,threaded,kwargs...)
        else
            algorithm=:standard
        end
    end
    algorithm===:multishift && return schur_multishift(A;maxiter,threaded,kwargs...)
    algorithm===:standard && isempty(kwargs) || throw(ArgumentError("unknown Schur algorithm/options"))
    n=_square(A)
    maxiter isa Integer && maxiter>=0 || throw(ArgumentError("maxiter must be a nonnegative integer"))
    F=hessenberg_factor(A)
    H=_hessenberg_matrix(F;rescale=false); Q=hessenberg_q(F)
    v=Vector{T}(undef,3); w=Vector{T}(undef,n)
    hi=n; iterations=0; stagnation=0; oldlo=0
    while hi>1
        lo=hi
        while lo>1
            if _deflatable(H,lo,1,hi)
                H[lo,lo-1]=zero(T)
                break
            end
            lo-=1
        end
        if lo==hi
            hi-=1; stagnation=0; oldlo=0
        elseif T<:Real && lo==hi-1
            _finish_real_pair!(H,Q,lo)
            hi-=2; stagnation=0; oldlo=0
        elseif T<:Complex && lo==hi-1
            _finish_complex_pair!(H,Q,lo)
            hi-=2; stagnation=0; oldlo=0
        else
            iterations>=maxiter && throw(ConvergenceError(:schur,iterations,lo:hi))
            stagnation=lo==oldlo ? stagnation+1 : 1; oldlo=lo
            if T<:Real
                _francis_step!(H,Q,lo,hi,v,w,stagnation)
            else
                _complex_qr_step!(H,Q,lo,hi,v,w,stagnation)
            end
            iterations+=1
        end
    end
    values=_schur_values(H,F.scale)
    @inbounds for idx in eachindex(H)
        H[idx]*=F.scale
    end
    SchurFactor(H,Q,values,iterations)
end

eigenvalues(A::AbstractMatrix;kwargs...)=schur_factor(A;kwargs...).values
