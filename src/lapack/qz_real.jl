# SPDX-License-Identifier: Apache-2.0
# Port of the QR -> Hessenberg/triangular -> implicit QZ strategy in
# evd/qz.rs, with corrections recorded in docs/src/dev/QZ_SCHUR_PORT.md.

# Homogeneous roots of det(S-lambda*T), retaining the off-diagonal of T.
# Independent scaling of the two matrices avoids squared input magnitudes.
function _pencil_pair(S,T,i)
    R=real(eltype(S)); C=Complex{R}
    # One assignment each: sa and sb are captured by the map closure below.
    ma=maximum(abs,view(S,i:i+1,i:i+1)); mb=maximum(abs,view(T,i:i+1,i:i+1))
    sa=iszero(ma) ? one(R) : ma; sb=iszero(mb) ? one(R) : mb
    a,b,c,d=S[i,i]/sa,S[i,i+1]/sa,S[i+1,i]/sa,S[i+1,i+1]/sa
    e,f,h=T[i,i]/sb,T[i,i+1]/sb,T[i+1,i+1]/sb
    p=e*h; q=c*f-a*h-d*e; r=a*d-b*c
    iszero(p) && !iszero(e) && !iszero(h) && throw(OverflowError("2x2 pencil determinant coefficient underflow; use a wider scalar type"))
    if iszero(p)
        if iszero(q)
            roots=iszero(r) ? ((zero(C),zero(C)),(zero(C),zero(C))) : ((one(C),zero(C)),(one(C),zero(C)))
        else
            roots=((-C(r),C(q)),(one(C),zero(C)))
        end
    else
        disc=C(q*q-4p*r); root=sqrt(disc)
        # Product relation supplies the cancellation-prone second root.
        u=-C(q)+(real(conj(C(q))*root)<=0 ? root : -root)
        if real(disc)<0
            roots=((-C(q)+root,C(2p)),(-C(q)-root,C(2p)))
        else
            roots=iszero(u) ? ((-C(q),C(2p)),(-C(q),C(2p))) : ((u,C(2p)),(C(2r),u))
        end
    end
    map(roots) do pair
        alpha,beta=pair
        t=max(abs(alpha),abs(beta))
        if !iszero(t)
            alpha/=t; beta/=t
        end
        alpha*=sa; beta*=sb
        real(beta)<0 && (alpha=-alpha; beta=-beta)
        alpha,beta
    end
end

function _qz_pairs(S::AbstractMatrix{T},B) where T
    n=size(S,1); C=Complex{real(T)}; alpha=zeros(C,n); beta=similar(alpha); i=1
    while i<=n
        if T<:Real && i<n && !iszero(S[i+1,i])
            pairs=_pencil_pair(S,B,i)
            alpha[i],beta[i]=pairs[1]; alpha[i+1],beta[i+1]=pairs[2]; i+=2
        else
            alpha[i]=C(S[i,i]); beta[i]=C(B[i,i]); i+=1
        end
    end
    alpha,beta
end

# A genuine real double shift. The initial vector is the first column of
# p2*(H*B^-1)^2 + p1*(H*B^-1) + p0*I, multiplied by b11^2*b22.
# Only local entries enter; B is never inverted or used to reduce the pencil
# to an ordinary eigenproblem. Right RQ rotations restore triangular B after
# each length-three left reflector.
function _qz_double_step!(H,B,Q,Z,lo,hi,iteration)
    T=eltype(H); n=size(H,1); v=zeros(T,3); w=zeros(T,n)
    a,b,c,d=H[hi-1,hi-1],H[hi-1,hi],H[hi,hi-1],H[hi,hi]
    e,f,h=B[hi-1,hi-1],B[hi-1,hi],B[hi,hi]
    p2=e*h; p1=c*f-a*h-d*e; p0=a*d-b*c
    if iteration%20==0
        # Repeated exceptional homogeneous shift (same strategy as Francis).
        mu=d+T(3)/4*abs(c); p2=h*h; p1=-2mu*h; p0=mu*mu
    end
    z=max(abs(p2),abs(p1),abs(p0))
    iszero(z) && throw(ArgumentError("indeterminate QZ shift polynomial"))
    p2/=z; p1/=z; p0/=z
    # Scale the leading local H and B independently, adjusting the polynomial.
    hs=maximum(abs,view(H,lo:lo+2,lo:lo+1)); bs=maximum(abs,view(B,lo:lo+1,lo:lo+1))
    hs=iszero(hs) ? one(T) : hs; bs=iszero(bs) ? one(T) : bs
    sc=max(hs,bs); ah=hs/sc; bh=bs/sc
    p2*=ah*ah; p1*=ah*bh; p0*=bh*bh
    h11=H[lo,lo]/hs; h21=H[lo+1,lo]/hs
    b11=B[lo,lo]/bs; b12=B[lo,lo+1]/bs; b22=B[lo+1,lo+1]/bs
    x=h11*b22-b12*h21; y=h21*b11
    for j in 1:3
        v[j]=p2*((H[lo+j-1,lo]/hs)*x+(H[lo+j-1,lo+1]/hs)*y)+p1*(H[lo+j-1,lo]/hs)*b11*b22
    end
    v[1]+=p0*b11*b11*b22
    for k in lo:hi-1
        r=min(3,hi-k+1); vv=view(v,1:r)
        k>lo && copyto!(vv,view(H,k:k+r-1,k-1))
        beta,tau=_householder!(vv)
        _reflect_left!(H,vv,tau,k,1:n); _reflect_left!(B,vv,tau,k,1:n)
        _reflect_right!(Q,vv,tau,k,1:n,w)
        if k>lo
            H[k,k-1]=beta
            for i in k+1:k+r-1
                H[i,k-1]=zero(T)
            end
        end
        for row in k+r-1:-1:k+1, col in k:row-1
            co,si,_=rotg(B[row,row],B[row,col])
            _qz_right!(H,B,Z,col,row,co,si); B[row,col]=zero(T)
        end
    end
end

function _qz_real_finish!(H,B,Q,Z,i,pairs)
    # Complex conjugates stay a real block. Real roots are split using their
    # local homogeneous null vector, followed by the independent left rotation.
    alpha,beta=pairs[1]
    !iszero(imag(alpha*conj(beta))) && return
    a=real(beta)*H[i,i]-real(alpha)*B[i,i]
    b=real(beta)*H[i,i+1]-real(alpha)*B[i,i+1]
    c=real(beta)*H[i+1,i]
    d=real(beta)*H[i+1,i+1]-real(alpha)*B[i+1,i+1]
    if hypot(a,b)>=hypot(c,d)
        u,v=-b,a
    else
        u,v=-d,c
    end
    norm=hypot(u,v)
    iszero(norm) && return # repeated scalar block; already a valid 2x2 block
    u/=norm; v/=norm
    # Right matrix [u -v; v u] in the _qz_right! convention.
    _qz_right!(H,B,Z,i,i+1,u,-v)
    if abs(B[i,i])+abs(B[i+1,i])>=abs(H[i,i])+abs(H[i+1,i])
        co,si,_=rotg(B[i,i],B[i+1,i])
    else
        co,si,_=rotg(H[i,i],H[i+1,i])
    end
    _qz_left!(H,B,Q,i,i+1,co,si)
    H[i+1,i]=zero(eltype(H)); B[i+1,i]=zero(eltype(B))
end

"""
    qz_factor(A::AbstractMatrix{T}, B::AbstractMatrix{T}; form=:real, maxiter=..., rtol=...) -> GeneralizedSchurFactor

Generalized Schur A=Q*S*Z', B=Q*T*Z'. Real input returns real S with
isolated 1x1/2x2 blocks and triangular T; complex input returns triangular S/T.
`form=:complex` explicitly requests the complex path for real input.
Homogeneous alpha/beta pairs include infinite and indeterminate eigenvalues.
"""
function qz_factor(A::AbstractMatrix{T},B::AbstractMatrix{T};form=:real,
                   maxiter=200max(1,size(A,1))^2,rtol=16eps(real(T))) where {T<:GenericFloat}
    form in (:real,:complex) || throw(ArgumentError("QZ form must be :real or :complex"))
    (T<:Complex || form===:complex) && return _qz_complex_factor(A,B;maxiter,rtol)
    n=_square(A); Base.require_one_based_indexing(B)
    size(B)==(n,n) || throw(DimensionMismatch("QZ pencil dimensions differ"))
    maxiter isa Integer && maxiter>=0 && 0<rtol<1 || throw(ArgumentError("invalid QZ budget/tolerance"))
    H,sa=_normalized_copy(A); R,sb=_normalized_copy(B); Q=_identity(T,n); Z=_identity(T,n)
    _qz_reduce!(H,R,Q,Z,1,n); hi=n; iterations=0; stagnation=0; oldlo=0
    while hi>1
        lo=hi
        while lo>1
            # Two-stage QR test also protects a highly unbalanced real pair.
            if _deflatable(H,lo,1,hi;tolerance=rtol)
                H[lo,lo-1]=zero(T); break
            end
            lo-=1
        end
        if lo==hi
            hi-=1; stagnation=0; oldlo=0; continue
        end
        # Do not classify a small but representable beta as exactly infinite.
        # Only exact zeros enter the null-vector isolation path.
        singular=findfirst(j->iszero(R[j,j]),lo:hi)
        if singular!==nothing
            iterations>=maxiter && throw(ConvergenceError(:qz,iterations,lo:hi))
            _qz_infinite!(H,R,Q,Z,lo,hi,lo+singular-1); iterations+=1; continue
        end
        if hi==lo+1
            _qz_real_finish!(H,R,Q,Z,lo,_pencil_pair(H,R,lo)); hi-=2; continue
        end
        iterations>=maxiter && throw(ConvergenceError(:qz,iterations,lo:hi))
        stagnation=lo==oldlo ? stagnation+1 : 1; oldlo=lo
        _qz_double_step!(H,R,Q,Z,lo,hi,stagnation); iterations+=1
    end
    H .*= sa; R .*= sb
    alpha,beta=_qz_pairs(H,R)
    QZFactor(H,R,Q,Z,alpha,beta,iterations)
end
