# SPDX-License-Identifier: Apache-2.0
_apply_preconditioner!(z,r,P,iteration)=P isa Function ? P(z,r,iteration) : _precondition!(z,r,P)

"""
    fgmres(A, b; x0=zeros(T,length(b)), restart=min(30,max(1,length(b))), maxiter=2length(b), rtol=sqrt(eps(real(T))), atol=zero(real(T)), preconditioner=nothing) -> NamedTuple

Flexible right-preconditioned restarted GMRES. A callable preconditioner
has signature P(z,r,iteration), so it may change between Arnoldi steps.
Stored preconditioned basis vectors make this flexibility mathematically valid.
"""
function fgmres(A::AbstractMatrix{T},b::AbstractVector{T};x0=zeros(T,length(b)),restart=min(30,max(1,length(b))),
                maxiter=2length(b),rtol=sqrt(eps(real(T))),atol=zero(real(T)),preconditioner=nothing) where {T<:NativeFloat}
    target=_krylov_setup(A,b,x0,rtol,atol,maxiter)
    preconditioner isa Function || _check_preconditioner(preconditioner,length(b))
    restart>=1 || throw(ArgumentError("positive GMRES restart required"))
    n=length(b); q=min(n,restart); x=copy(x0); residualvector=similar(b); product=similar(b)
    residual=_true_residual!(residualvector,A,x,b,product); iterations=0
    residual<=target && return (;x,converged=true,iterations,residual,reason=:converged)
    V=zeros(T,n,q+1); Z=zeros(T,n,q); H=zeros(T,q+1,q); g=zeros(T,q+1); cs=zeros(real(T),q); sn=zeros(T,q)
    while iterations<maxiter
        for i in 1:n
            V[i,1]=residualvector[i]/residual
        end
        fill!(H,zero(T)); fill!(g,zero(T)); g[1]=residual; used=0; happy=false
        for j in 1:min(q,maxiter-iterations)
            _apply_preconditioner!(view(Z,:,j),view(V,:,j),preconditioner,iterations+1)
            _native_mv!(product,A,view(Z,:,j))
            for _ in 1:2, i in 1:j
                h=dotc(view(V,:,i),product); H[i,j]+=h; axpy!(product,view(V,:,i),-h)
            end
            beta=nrm2(product); H[j+1,j]=beta
            if !iszero(beta)
                for i in 1:n
                    V[i,j+1]=product[i]/beta
                end
            end
            for i in 1:j-1
                a,c=H[i,j],H[i+1,j]
                H[i,j]=cs[i]*a+sn[i]*c; H[i+1,j]=cs[i]*c-conj(sn[i])*a
            end
            c,s,r=rotg(H[j,j],H[j+1,j]); cs[j]=c; sn[j]=s; H[j,j]=r; H[j+1,j]=zero(T)
            g[j+1]=-conj(s)*g[j]; g[j]*=c; used=j; iterations+=1; happy=iszero(beta)
            (abs(g[j+1])<=target || happy) && break
        end
        if any(iszero(H[i,i]) for i in 1:used)
            return (;x,converged=false,iterations,residual,reason=:breakdown)
        end
        y=g[1:used]; trsv!(y,view(H,1:used,1:used);uplo=:U)
        for j in 1:used
            axpy!(x,view(Z,:,j),y[j])
        end
        residual=_true_residual!(residualvector,A,x,b,product)
        residual<=target && return (;x,converged=true,iterations,residual,reason=:converged)
        happy && return (;x,converged=false,iterations,residual,reason=:breakdown)
    end
    (;x,converged=false,iterations,residual,reason=:maxiter)
end

"""
    pcg(A, b; x0=zeros(T,length(b)), maxiter=2length(b), rtol=sqrt(eps(real(T))), atol=zero(real(T)), preconditioner=nothing) -> NamedTuple

Preconditioned conjugate gradients with an explicit positive-definite
preconditioner and true residual checks. Breakdowns are never convergence.
"""
function pcg(A::AbstractMatrix{T},b::AbstractVector{T};x0=zeros(T,length(b)),maxiter=2length(b),
             rtol=sqrt(eps(real(T))),atol=zero(real(T)),preconditioner=nothing) where {T<:NativeFloat}
    target=_krylov_setup(A,b,x0,rtol,atol,maxiter); _check_preconditioner(preconditioner,length(b))
    x=copy(x0); r=similar(b); z=similar(b); p=similar(b); product=similar(b); check=similar(b); temp=similar(b)
    residual=_true_residual!(r,A,x,b,product); iterations=0
    residual<=target && return (;x,converged=true,iterations,residual,reason=:converged)
    _precondition!(z,r,preconditioner); copyto!(p,z); rho=real(dotc(r,z))
    for iteration in 1:maxiter
        _native_mv!(product,A,p); denominator=real(dotc(p,product))
        if !(isfinite(rho) && rho>0 && isfinite(denominator) && denominator>0)
            return (;x,converged=false,iterations,residual,reason=:breakdown)
        end
        alpha=rho/denominator; axpy!(x,p,alpha); axpy!(r,product,-alpha); iterations=iteration
        residual=_true_residual!(check,A,x,b,temp)
        residual<=target && return (;x,converged=true,iterations,residual,reason=:converged)
        # Periodic true-residual replacement controls accumulated roundoff.
        iteration%50==0 && copyto!(r,check)
        _precondition!(z,r,preconditioner); next=real(dotc(r,z)); beta=next/rho
        for i in eachindex(p)
            p[i]=z[i]+beta*p[i]
        end
        rho=next
    end
    (;x,converged=false,iterations,residual,reason=:maxiter)
end

"""
    minres(A, b; x0=zeros(T,length(b)), maxiter=2length(b), rtol=sqrt(eps(real(T))), atol=zero(real(T)), preconditioner=nothing) -> NamedTuple

Paige–Saunders short-recurrence MINRES for Hermitian systems, including
indefinite matrices. The optional fixed preconditioner must be Hermitian positive
definite. Every reported convergence uses the true residual.
"""
function minres(A::AbstractMatrix{T},b::AbstractVector{T};x0=zeros(T,length(b)),maxiter=2length(b),
                rtol=sqrt(eps(real(T))),atol=zero(real(T)),preconditioner=nothing) where {T<:NativeFloat}
    target=_krylov_setup(A,b,x0,rtol,atol,maxiter); _check_preconditioner(preconditioner,length(b))
    n=length(b); R=real(T); x=copy(x0); r1=zeros(T,n); r2=similar(b); y=similar(b); product=similar(b)
    check=similar(b); temp=similar(b); v=similar(b); w=zeros(T,n); w1=similar(w); fill!(w1,zero(T)); w2=copy(w)
    residual=_true_residual!(r2,A,x,b,product); iterations=0
    residual<=target && return (;x,converged=true,iterations,residual,reason=:converged)
    _precondition!(y,r2,preconditioner); beta2=real(dotc(r2,y))
    beta2>0 && isfinite(beta2) || return (;x,converged=false,iterations,residual,reason=:nonpositive_preconditioner)
    beta=sqrt(beta2); oldbeta=zero(R); phibar=beta; dbar=epsln=zero(R); c=-one(R); s=zero(R)
    for iteration in 1:maxiter
        for i in 1:n
            v[i]=y[i]/beta
        end
        _native_mv!(product,A,v)
        iteration>1 && axpy!(product,r1,-beta/oldbeta)
        alpha=real(dotc(v,product)); axpy!(product,r2,-alpha/beta)
        r1,r2,product=r2,product,r1
        _precondition!(y,r2,preconditioner); next2=real(dotc(r2,y))
        if !isfinite(next2) || next2<0
            return (;x,converged=false,iterations,residual,reason=:nonpositive_preconditioner)
        end
        oldbeta=beta; beta=sqrt(next2); oldeps=epsln
        delta=c*dbar+s*alpha; gbar=s*dbar-c*alpha; epsln=s*beta; dbar=-c*beta
        gamma=hypot(gbar,beta)
        iszero(gamma) && return (;x,converged=false,iterations,residual,reason=:breakdown)
        c=gbar/gamma; s=beta/gamma; phi=c*phibar; phibar*=s
        w1,w2,w=w2,w,w1
        for i in 1:n
            w[i]=(v[i]-oldeps*w1[i]-delta*w2[i])/gamma
        end
        axpy!(x,w,phi); iterations=iteration
        residual=_true_residual!(check,A,x,b,temp)
        residual<=target && return (;x,converged=true,iterations,residual,reason=:converged)
        iszero(beta) && return (;x,converged=false,iterations,residual,reason=:breakdown)
    end
    (;x,converged=false,iterations,residual,reason=:maxiter)
end

"""
    tfqmr(A, b; x0=zeros(T,length(b)), maxiter=2length(b), rtol=sqrt(eps(real(T))), atol=zero(real(T)), preconditioner=nothing) -> NamedTuple

Transpose-free quasi-minimal residual with optional fixed right
preconditioning. `maxiter` counts half-steps; the stopping residual is exact.
"""
function tfqmr(A::AbstractMatrix{T},b::AbstractVector{T};x0=zeros(T,length(b)),maxiter=2length(b),
               rtol=sqrt(eps(real(T))),atol=zero(real(T)),preconditioner=nothing) where {T<:NativeFloat}
    target=_krylov_setup(A,b,x0,rtol,atol,maxiter); _check_preconditioner(preconditioner,length(b))
    n=length(b); R=real(T); x=copy(x0); r=similar(b); check=similar(b); temp=similar(b); pre=similar(b)
    residual=_true_residual!(r,A,x,b,temp); iterations=0
    residual<=target && return (;x,converged=true,iterations,residual,reason=:converged)
    shadow=copy(r); w=copy(r); y1=copy(r); y2=similar(r); direction=zeros(T,n); v=similar(r); u1=similar(r); u2=similar(r)
    function apply!(out,input)
        _precondition!(pre,input,preconditioner); _native_mv!(out,A,pre)
    end
    apply!(v,y1); copyto!(u1,v); rho=dotc(shadow,r); tau=residual; theta=zero(R); eta=zero(T)
    while iterations<maxiter
        sigma=dotc(shadow,v)
        (!isfinite(sigma) || iszero(sigma) || iszero(rho)) && return (;x,converged=false,iterations,residual,reason=:breakdown)
        alpha=rho/sigma
        iszero(alpha) && return (;x,converged=false,iterations,residual,reason=:breakdown)
        for half in 1:2
            if half==2
                for i in 1:n
                    y2[i]=y1[i]-alpha*v[i]
                end
                apply!(u2,y2)
            end
            yy=half==1 ? y1 : y2; uu=half==1 ? u1 : u2
            coefficient=theta*theta*eta/alpha
            for i in 1:n
                w[i]-=alpha*uu[i]; direction[i]=yy[i]+coefficient*direction[i]
            end
            iszero(tau) && return (;x,converged=false,iterations,residual,reason=:breakdown)
            theta=nrm2(w)/tau; c=Base.inv(hypot(one(R),theta)); tau*=theta*c; eta=c*c*alpha
            _precondition!(pre,direction,preconditioner); axpy!(x,pre,eta); iterations+=1
            residual=_true_residual!(check,A,x,b,temp)
            residual<=target && return (;x,converged=true,iterations,residual,reason=:converged)
            iterations>=maxiter && return (;x,converged=false,iterations,residual,reason=:maxiter)
        end
        next=dotc(shadow,w); beta=next/rho
        for i in 1:n
            y1[i]=w[i]+beta*y2[i]
        end
        apply!(u1,y1)
        for i in 1:n
            v[i]=u1[i]+beta*(u2[i]+beta*v[i])
        end
        rho=next
    end
    (;x,converged=false,iterations,residual,reason=:maxiter)
end

_native_adjoint_mv!(y,A::AbstractMatrix,x)=gemv!(y,A,x;trans=:C)
_native_adjoint_mv!(y,A::NativeSparse,x)=sparse_mul!(y,A,x;trans=:C)
"""
    qmr(A, b; x0=zeros(T,length(b)), maxiter=2length(b), rtol=sqrt(eps(real(T))), atol=zero(real(T))) -> NamedTuple

Unpreconditioned two-sided QMR, using A and A' products and short
bi-Lanczos recurrences. True residuals govern all termination decisions.
"""
function qmr(A::AbstractMatrix{T},b::AbstractVector{T};x0=zeros(T,length(b)),maxiter=2length(b),
             rtol=sqrt(eps(real(T))),atol=zero(real(T))) where {T<:NativeFloat}
    target=_krylov_setup(A,b,x0,rtol,atol,maxiter); n=length(b); R=real(T)
    x=copy(x0); r=similar(b); temp=similar(b); residual=_true_residual!(r,A,x,b,temp); iterations=0
    residual<=target && return (;x,converged=true,iterations,residual,reason=:converged)
    vt=copy(r); wt=copy(r); v=similar(r); w=similar(r); p=zeros(T,n); q=zeros(T,n); ap=similar(r); ahq=similar(r); direction=zeros(T,n)
    rho=nrm2(vt); xi=nrm2(wt); gamma=one(R); theta=zero(R); eta=-one(T); epsilon=one(T)
    for iteration in 1:maxiter
        (iszero(rho) || iszero(xi)) && return (;x,converged=false,iterations,residual,reason=:breakdown)
        for i in 1:n
            v[i]=vt[i]/rho; w[i]=wt[i]/xi
        end
        delta=dotc(w,v)
        iszero(delta) && return (;x,converged=false,iterations,residual,reason=:breakdown)
        for i in 1:n
            p[i]=iteration==1 ? v[i] : v[i]-(xi*delta/epsilon)*p[i]
            q[i]=iteration==1 ? w[i] : w[i]-conj(rho*delta/epsilon)*q[i]
        end
        _native_mv!(ap,A,p); epsilon=dotc(q,ap)
        iszero(epsilon) && return (;x,converged=false,iterations,residual,reason=:breakdown)
        beta=epsilon/delta; _native_adjoint_mv!(ahq,A,q)
        for i in 1:n
            vt[i]=ap[i]-beta*v[i]; wt[i]=ahq[i]-conj(beta)*w[i]
        end
        next_rho=nrm2(vt); next_xi=nrm2(wt); next_theta=next_rho/(gamma*abs(beta)); next_gamma=Base.inv(hypot(one(R),next_theta))
        iszero(next_gamma) && return (;x,converged=false,iterations,residual,reason=:breakdown)
        next_eta=-eta*rho*(next_gamma/gamma)^2/beta; coefficient=theta*theta*next_gamma*next_gamma
        for i in 1:n
            direction[i]=next_eta*p[i]+coefficient*direction[i]
        end
        axpy!(x,direction,one(T)); iterations=iteration; residual=_true_residual!(r,A,x,b,temp)
        residual<=target && return (;x,converged=true,iterations,residual,reason=:converged)
        rho=next_rho; xi=next_xi; gamma=next_gamma; theta=next_theta; eta=next_eta
    end
    (;x,converged=false,iterations,residual,reason=:maxiter)
end

"""
    idrs(A, b; s=4, x0=zeros(T,length(b)), maxiter=2length(b), rtol=sqrt(eps(real(T))), atol=zero(real(T)), preconditioner=nothing, seed=0x0b1a5, angle=real(T)(.7)) -> NamedTuple

IDR(s) with an orthonormal local shadow space, oblique Sonneveld-space
updates, angle-stabilized omega and optional fixed right preconditioning.
"""
function idrs(A::AbstractMatrix{T},b::AbstractVector{T};s=4,x0=zeros(T,length(b)),maxiter=2length(b),
              rtol=sqrt(eps(real(T))),atol=zero(real(T)),preconditioner=nothing,seed=0x0b1a5,angle=real(T)(.7)) where {T<:NativeFloat}
    target=_krylov_setup(A,b,x0,rtol,atol,maxiter); _check_preconditioner(preconditioner,length(b))
    s>=1 && 0<angle<=1 || throw(ArgumentError("invalid IDR shadow-space options"))
    n=length(b); x=copy(x0); r=similar(b); check=similar(b); temp=similar(b); residual=_true_residual!(r,A,x,b,temp); iterations=0
    residual<=target && return (;x,converged=true,iterations,residual,reason=:converged)
    q=min(s,n); P=_orthogonal_basis(_random_projection(T,n,q,seed)); size(P,2)==q || throw(ArgumentError("degenerate IDR shadow space"))
    G=zeros(T,n,q); U=zeros(T,n,q); M=_identity(T,q); omega=one(T); v=similar(b); z=similar(b); u=similar(b); g=similar(b); t=similar(b)
    while iterations<maxiter
        f=T[dotc(view(P,:,j),r) for j in 1:q]
        for k in 1:q
            any(iszero(M[j,j]) for j in k:q) && return (;x,converged=false,iterations,residual,reason=:breakdown)
            c=f[k:q]; trsv!(c,view(M,k:q,k:q);uplo=:L); copyto!(v,r)
            for j in k:q
                axpy!(v,view(G,:,j),-c[j-k+1])
            end
            _precondition!(z,v,preconditioner)
            for i in 1:n
                u[i]=omega*z[i]
            end
            for j in k:q
                axpy!(u,view(U,:,j),c[j-k+1])
            end
            _native_mv!(g,A,u)
            for j in 1:k-1
                alpha=dotc(view(P,:,j),g)/M[j,j]; axpy!(g,view(G,:,j),-alpha); axpy!(u,view(U,:,j),-alpha)
            end
            copyto!(view(G,:,k),g); copyto!(view(U,:,k),u)
            for j in k:q
                M[j,k]=dotc(view(P,:,j),g)
            end
            iszero(M[k,k]) && return (;x,converged=false,iterations,residual,reason=:breakdown)
            beta=f[k]/M[k,k]; axpy!(r,g,-beta); axpy!(x,u,beta); iterations+=1
            for j in k+1:q
                f[j]-=beta*M[j,k]
            end
            residual=_true_residual!(check,A,x,b,temp)
            residual<=target && return (;x,converged=true,iterations,residual,reason=:converged)
            iterations>=maxiter && return (;x,converged=false,iterations,residual,reason=:maxiter)
        end
        copyto!(r,check); _precondition!(z,r,preconditioner); _native_mv!(t,A,z)
        tt=real(dotc(t,t)); tr=dotc(t,r); nr=nrm2(r)
        (iszero(tt) || iszero(tr) || !isfinite(tt)) && return (;x,converged=false,iterations,residual,reason=:breakdown)
        omega=tr/tt; cosine=abs(tr)/(sqrt(tt)*nr)
        cosine<angle && (omega*=angle/cosine)
        axpy!(x,z,omega); axpy!(r,t,-omega); iterations+=1
        residual=_true_residual!(check,A,x,b,temp)
        residual<=target && return (;x,converged=true,iterations,residual,reason=:converged)
    end
    (;x,converged=false,iterations,residual,reason=:maxiter)
end
