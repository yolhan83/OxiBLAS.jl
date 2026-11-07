# SPDX-License-Identifier: Apache-2.0
# R ← B - A X; the column norms go to `norms` (reused across iterations).
function _block_residual!(norms,R,A,X,B,product)
    _native_mm!(product,A,X)
    for i in eachindex(R,B)
        R[i]=B[i]-product[i]
    end
    for j in axes(R,2)
        norms[j]=nrm2(view(R,:,j))
    end
    norms
end
_within(residuals,targets)=all(i->residuals[i]<=targets[i],eachindex(residuals,targets))
function _block_targets(B,rtol,atol)
    [max(atol,rtol*nrm2(view(B,:,j))) for j in axes(B,2)]
end
function _block_fallback(A,B,X,targets,preconditioner,budget,solver)
    results=Any[]
    for j in axes(B,2)
        result=solver(A,view(B,:,j);x0=copy(view(X,:,j)),maxiter=budget,rtol=zero(eltype(targets)),atol=targets[j],preconditioner)
        X[:,j]=result.x; push!(results,result)
    end
    residuals=[result.residual for result in results]
    (;x=X,converged=all(result.converged for result in results),residuals,fallback=true,column_results=results)
end
function _block_full_rank(A)
    isempty(A) && return true
    F=qr_pivoted(A;threaded=false); d=[abs(F.factors[i,i]) for i in 1:min(size(A)...)]; scale=maximum(d;init=zero(eltype(d)))
    all(x->x>64eps(real(eltype(A)))*max(size(A)...)*scale,d)
end
"""
    block_cg(A, B; x0=zeros(T,size(B)), maxiter=2size(A,1), rtol=sqrt(eps(real(T))), atol=zero(real(T)), preconditioner=nothing) -> NamedTuple

Coupled block preconditioned CG for multiple Hermitian positive-definite
systems. Rank loss in the block Gram system triggers explicit scalar PCG
continuation from the current iterates, rather than inverting a singular block.
"""
function block_cg(A::AbstractMatrix{T},B::AbstractMatrix{T};x0=zeros(T,size(B)),maxiter=2size(A,1),
                  rtol=sqrt(eps(real(T))),atol=zero(real(T)),preconditioner=nothing) where {T<:NativeFloat}
    n=_square(A); Base.require_one_based_indexing(B,x0)
    size(B,1)==n && size(x0)==size(B) && min(maxiter,rtol,atol)>=0 || throw(ArgumentError("invalid block CG dimensions/options"))
    _check_preconditioner(preconditioner,n); targets=_block_targets(B,rtol,atol); X=copy(x0); R=similar(B); product=similar(B); s=size(B,2)
    residuals=_block_residual!(similar(targets),R,A,X,B,product); iterations=0
    _within(residuals,targets) && return (;x=X,converged=true,residuals,iterations,fallback=false)
    Z=similar(B)
    for j in 1:s
        _precondition!(view(Z,:,j),view(R,:,j),preconditioner)
    end
    # n×s and s×s buffers reused by every iteration; P and Pnext, gram and
    # next swap roles.
    P=copy(Z); Pnext=similar(B); gram=_mm(adjoint(R),Z); next=similar(gram); H=similar(gram)
    for iteration in 1:maxiter
        _native_mm!(product,A,P); gemm!(H,adjoint(P),product)
        if !_block_full_rank(H) || !_block_full_rank(gram)
            return merge(_block_fallback(A,B,X,targets,preconditioner,max(0,maxiter-iterations),pcg),(;iterations))
        end
        alpha=solve(lu_factor(H;threaded=false),gram); gemm!(X,P,alpha,one(T),one(T))
        _block_residual!(residuals,R,A,X,B,product); iterations=iteration
        _within(residuals,targets) && return (;x=X,converged=true,residuals,iterations,fallback=false)
        for j in 1:s
            _precondition!(view(Z,:,j),view(R,:,j),preconditioner)
        end
        gemm!(next,adjoint(R),Z); beta=solve(lu_factor(gram;threaded=false),next)
        copyto!(Pnext,Z); gemm!(Pnext,P,beta,one(T),one(T))
        P,Pnext=Pnext,P; gram,next=next,gram
    end
    (;x=X,converged=false,residuals,iterations,fallback=false)
end

"""
    block_gmres(A, B; x0=zeros(T,size(B)), restart=10, maxiter=2size(A,1), rtol=sqrt(eps(real(T))), atol=zero(real(T)), preconditioner=nothing) -> NamedTuple

Restarted right-preconditioned block GMRES with block Arnoldi and native
minimum-norm projected least squares. Rank loss falls back to independent
flexible GMRES continuations; the returned `fallback` flag makes this explicit.
`maxiter` counts block Arnoldi steps.
"""
function block_gmres(A::AbstractMatrix{T},B::AbstractMatrix{T};x0=zeros(T,size(B)),restart=10,maxiter=2size(A,1),
                     rtol=sqrt(eps(real(T))),atol=zero(real(T)),preconditioner=nothing) where {T<:NativeFloat}
    n=_square(A); Base.require_one_based_indexing(B,x0)
    size(B,1)==n && size(B)==size(x0) && min(maxiter,rtol,atol)>=0 && restart>=1 || throw(ArgumentError("invalid block GMRES dimensions/options"))
    preconditioner isa Function || _check_preconditioner(preconditioner,n)
    s=size(B,2); targets=_block_targets(B,rtol,atol); X=copy(x0); R=similar(B); product=similar(B); iterations=0
    residuals=_block_residual!(similar(targets),R,A,X,B,product)
    _within(residuals,targets) && return (;x=X,converged=true,residuals,iterations,fallback=false)
    # Krylov bases, Hessenberg matrix and n×s work blocks sized once for the
    # longest cycle and reused by every restart.
    qmax=min(restart,maxiter); Vbuf=zeros(T,n,(qmax+1)*s); Zbuf=zeros(T,n,qmax*s); Hbuf=zeros(T,(qmax+1)*s,qmax*s); rightbuf=zeros(T,(qmax+1)*s,s)
    W=similar(B); hij=similar(B,s,s); base=similar(X); trial=similar(X)
    while iterations<maxiter
        if n<s || !_block_full_rank(R)
            return merge(_block_fallback(A,B,X,targets,preconditioner,(maxiter-iterations)*s,fgmres),(;iterations))
        end
        q=min(restart,maxiter-iterations)
        V=view(Vbuf,:,1:(q+1)*s); Z=view(Zbuf,:,1:q*s); H=view(Hbuf,1:(q+1)*s,1:q*s)
        fill!(V,zero(T)); fill!(Z,zero(T)); fill!(H,zero(T))
        F=qr_factor(R;threaded=false); V[:,1:s]=qr_q(F;columns=s,threaded=false)
        @inbounds for j in 1:s, i in 1:s
            rightbuf[i,j]=i<=j ? F.factors[i,j] : zero(T)
        end
        copyto!(base,X); copyto!(trial,X)
        for j in 1:q
            current=(j-1)*s+1:j*s; following=j*s+1:(j+1)*s
            for (k,col) in enumerate(current)
                _apply_preconditioner!(view(Z,:,col),view(V,:,col),preconditioner,iterations*s+k)
            end
            _native_mm!(W,A,view(Z,:,current))
            for _ in 1:2, i in 1:j
                rows=(i-1)*s+1:i*s; gemm!(hij,adjoint(view(V,:,rows)),W)
                for c in 1:s, r in 1:s
                    H[rows[r],current[c]]+=hij[r,c]
                end
                gemm!(W,view(V,:,rows),hij,-one(T),one(T))
            end
            fullrank=n>=(j+1)*s && _block_full_rank(W); factor=qr_factor(W;threaded=false)
            @inbounds for c in 1:s, r in 1:s
                H[following[r],current[c]]=r<=c ? factor.factors[r,c] : zero(T)
            end
            fullrank && (V[:,following]=qr_q(factor;columns=s,threaded=false))
            right=view(rightbuf,1:(j+1)*s,1:s)
            projected=complete_orthogonal(view(H,1:(j+1)*s,1:j*s);threaded=false)
            coefficients=solve(projected,right); copyto!(trial,base); gemm!(trial,view(Z,:,1:j*s),coefficients,one(T),one(T))
            _block_residual!(residuals,R,A,trial,B,product); iterations+=1
            _within(residuals,targets) && return (;x=trial,converged=true,residuals,iterations,fallback=false)
            iterations>=maxiter && return (;x=trial,converged=false,residuals,iterations,fallback=false)
            if !fullrank
                return merge(_block_fallback(A,B,trial,targets,preconditioner,(maxiter-iterations)*s,fgmres),(;iterations))
            end
        end
        copyto!(X,trial)
    end
    (;x=X,converged=false,residuals,iterations,fallback=false)
end
