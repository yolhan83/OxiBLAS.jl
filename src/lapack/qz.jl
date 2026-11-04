# SPDX-License-Identifier: Apache-2.0
struct QZFactor{T,C}
    S::Matrix{T}
    T::Matrix{T}
    Q::Matrix{T}
    Z::Matrix{T}
    alpha::Vector{C}
    beta::Vector{C}
    iterations::Int
end
function Base.getproperty(F::QZFactor,s::Symbol)
    if s===:values
        a=getfield(F,:alpha); b=getfield(F,:beta)
        return [iszero(b[i]) ? (iszero(a[i]) ? oftype(a[i],NaN) : oftype(a[i],Inf)) : a[i]/b[i] for i in eachindex(a)]
    end
    getfield(F,s)
end

# Left G=[c s; -conj(s) c], with Q <- Q*G'.
function _qz_left!(A,B,Q,i,j,c,s)
    for M in (A,B)
        @inbounds for col in axes(M,2)
            x,y=M[i,col],M[j,col]
            M[i,col]=c*x+s*y; M[j,col]=-conj(s)*x+c*y
        end
    end
    @inbounds for row in axes(Q,1)
        x,y=Q[row,i],Q[row,j]
        Q[row,i]=c*x+conj(s)*y; Q[row,j]=-s*x+c*y
    end
end
# Right G, with Z <- Z*G.
function _qz_right!(A,B,Z,i,j,c,s)
    for M in (A,B,Z)
        @inbounds for row in axes(M,1)
            x,y=M[row,i],M[row,j]
            M[row,i]=c*x-conj(s)*y; M[row,j]=s*x+c*y
        end
    end
end
function _qz_reduce!(A,B,Q,Z,lo,hi)
    T=eltype(A); scratch=Vector{T}(undef,size(A,1)); work=similar(scratch)
    # Householder QR of the active B block, applied on the left to both sides.
    for j in lo:hi
        v=view(scratch,1:hi-j+1); copyto!(v,view(B,j:hi,j)); beta,tau=_householder!(v)
        _reflect_left!(A,v,tau,j,axes(A,2)); _reflect_left!(B,v,tau,j,axes(B,2))
        _reflect_right!(Q,v,tau,j,axes(Q,1),work)
        B[j,j]=beta
        for i in j+1:hi
            B[i,j]=zero(T)
        end
    end
    # Hessenberg/triangular reduction by paired left and right rotations.
    for j in lo:hi-2, i in hi:-1:j+2
        c,s,_=rotg(A[i-1,j],A[i,j]); _qz_left!(A,B,Q,i-1,i,c,s); A[i,j]=zero(T)
        c,s,_=rotg(B[i,i],B[i,i-1]); _qz_right!(A,B,Z,i-1,i,c,s); B[i,i-1]=zero(T)
    end
end

function _qz_shift(A,B,lo,hi,iteration)
    T=eltype(A); R=real(T)
    if hi==lo
        return A[hi,hi],B[hi,hi]
    end
    a,b,c,d=A[hi-1,hi-1],A[hi-1,hi],A[hi,hi-1],A[hi,hi]
    e,f,h=B[hi-1,hi-1],B[hi-1,hi],B[hi,hi]
    ascale=max(abs(a),abs(b),abs(c),abs(d)); bscale=max(abs(e),abs(f),abs(h))
    iszero(ascale) && return zero(T),one(T)
    iszero(bscale) && return one(T),zero(T)
    a/=ascale; b/=ascale; c/=ascale; d/=ascale; e/=bscale; f/=bscale; h/=bscale
    if iteration%20==0
        alpha=d+R(.75)*abs(c); beta=h
    else
        p=e*h; q=c*f-a*h-d*e; r=a*d-b*c
        if iszero(p)
            alpha=-r; beta=q
        else
            disc=sqrt(q*q-4p*r)
            alpha1=-q+disc; alpha2=-q-disc; beta=2p
            norm1=hypot(abs(alpha1),abs(beta)); norm2=hypot(abs(alpha2),abs(beta))
            distance1=iszero(norm1) ? R(Inf) : abs(alpha1*h-beta*d)/norm1
            distance2=iszero(norm2) ? R(Inf) : abs(alpha2*h-beta*d)/norm2
            alpha=distance1<=distance2 ? alpha1 : alpha2
        end
    end
    # Account for independently scaled A/B while retaining a homogeneous pair.
    scale=max(ascale,bscale)
    alpha*=ascale/scale; beta*=bscale/scale
    norm=max(abs(alpha),abs(beta)); iszero(norm) ? (one(T),one(T)) : (alpha/norm,beta/norm)
end

function _qz_step!(A,B,Q,Z,lo,hi,alpha,beta)
    f=beta*A[lo,lo]-alpha*B[lo,lo]; g=beta*A[lo+1,lo]
    for k in lo:hi-1
        if k>lo
            f=A[k,k-1]; g=A[k+1,k-1]
        end
        c,s,_=rotg(f,g); _qz_left!(A,B,Q,k,k+1,c,s)
        k>lo && (A[k+1,k-1]=zero(eltype(A)))
        c,s,_=rotg(B[k+1,k+1],B[k+1,k]); _qz_right!(A,B,Z,k,k+1,c,s)
        B[k+1,k]=zero(eltype(B))
    end
end

# A zero beta can be isolated as a leading infinite eigenvalue without B^-1.
# Construct a right null vector of the triangular active B, align it with e_lo,
# then align A*v from the left. The remaining pencil is reduced afresh.
function _qz_infinite!(A,B,Q,Z,lo,hi,j)
    T=eltype(A); n=size(A,1); v=zeros(T,hi-lo+1); v[j-lo+1]=one(T)
    for i in j-1:-1:lo
        s=zero(T)
        for k in i+1:j
            s+=B[i,k]*v[k-lo+1]
        end
        den=B[i,i]
        iszero(den) && throw(ArgumentError("unresolved nullspace of singular pencil"))
        scale=max(abs(s),abs(den))
        if scale>0 && abs(den)<abs(s)*sqrt(floatmin(real(T)))
            for k in i+1-lo+1:length(v)
                v[k]*=abs(den)/max(abs(s),one(real(T)))
            end
            s*=abs(den)/max(abs(s),one(real(T)))
        end
        v[i-lo+1]=-s/den
    end
    norm=nrm2(v)
    for i in eachindex(v)
        v[i]=_realdiv(v[i],norm)
    end
    _,tau=_householder!(v); work=zeros(T,n)
    _reflect_right!(A,v,tau,lo,1:n,work); _reflect_right!(B,v,tau,lo,1:n,work); _reflect_right!(Z,v,tau,lo,1:n,work)
    copyto!(v,view(A,lo:hi,lo)); alpha,tau=_householder!(v)
    _reflect_left!(A,v,tau,lo,1:n); _reflect_left!(B,v,tau,lo,1:n); _reflect_right!(Q,v,tau,lo,1:n,work)
    A[lo,lo]=alpha
    for i in lo+1:hi
        A[i,lo]=zero(T)
    end
    for i in lo:hi
        B[i,lo]=zero(T)
    end
    lo<hi && _qz_reduce!(A,B,Q,Z,lo+1,hi)
end

"""Complex generalized Schur decomposition of a real or complex matrix pencil.
A=Q*S*Z', B=Q*T*Z', with S/T triangular and Q/Z unitary. Eigenvalues are
represented by homogeneous alpha/beta pairs, including beta=0 (infinity) and
alpha=beta=0 (indeterminate singular pencil). No inverse of B is formed.
"""
function _qz_complex_factor(A::AbstractMatrix{T},B::AbstractMatrix{T};maxiter=200max(1,size(A,1))^2,rtol=16eps(real(T))) where {T<:GenericFloat}
    n=_square(A); Base.require_one_based_indexing(B)
    size(B)==(n,n) || throw(DimensionMismatch("QZ pencil dimensions differ"))
    maxiter>=0 && 0<rtol<1 || throw(ArgumentError("invalid QZ budget/tolerance"))
    C=Complex{real(T)}; ascale=_finite_scale(A); bscale=_finite_scale(B)
    H=C.(_realdiv.(A,ascale)); R=C.(_realdiv.(B,bscale)); Q=_identity(C,n); Z=_identity(C,n)
    _qz_reduce!(H,R,Q,Z,1,n); hi=n; iterations=0
    while hi>1
        for i in 2:hi
            if abs(H[i,i-1])<=rtol*(abs(H[i-1,i-1])+abs(H[i,i]))
                H[i,i-1]=zero(C)
            end
        end
        if iszero(H[hi,hi-1])
            hi-=1; continue
        end
        lo=hi-1
        while lo>1 && !iszero(H[lo,lo-1])
            lo-=1
        end
        iterations>=maxiter && throw(ConvergenceError(:qz,iterations,lo:hi))
        iterations+=1
        singular=findfirst(j->iszero(R[j,j]),lo:hi)
        if singular!==nothing
            j=lo+singular-1; R[j,j]=zero(C)
            _qz_infinite!(H,R,Q,Z,lo,hi,j)
        else
            alpha,beta=_qz_shift(H,R,lo,hi,iterations)
            _qz_step!(H,R,Q,Z,lo,hi,alpha,beta)
        end
    end
    H .*= ascale; R .*= bscale
    QZFactor(H,R,Q,Z,[H[i,i] for i in 1:n],[R[i,i] for i in 1:n],iterations)
end

function eigen_pencil(A::AbstractMatrix{T},B::AbstractMatrix{T};vectors=true,left=false,kwargs...) where {T<:GenericFloat}
    F=qz_factor(A,B;kwargs...); n=size(A,1)
    !vectors && return (;values=F.values,alpha=F.alpha,beta=F.beta,vectors=nothing,qz=F)
    X=_pencil_vectors(F.S,F.T,F.alpha,F.beta)
    C=eltype(X); V=_mm(C.(F.Z),X)
    W=left ? _mm(C.(F.Q),_pencil_vectors(F.S,F.T,F.alpha,F.beta;left=true)) : nothing
    (;values=F.values,alpha=F.alpha,beta=F.beta,vectors=V,left_vectors=W,qz=F)
end
