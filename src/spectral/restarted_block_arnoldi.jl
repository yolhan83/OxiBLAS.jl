# SPDX-License-Identifier: Apache-2.0
# Block Arnoldi port completed with a real/complex Krylov-Schur thick restart.
# Cached images of retained vectors avoid repeating their operator products.
function _arnoldi_basis(X,V,limit,tol,mode,threaded)
    mode===:full && return _block_basis(X,V;limit,rtol=tol,threaded)
    scale=_block_column_scale(X); W=copy(X)
    if size(V,2)>0
        H=_mm(adjoint(V),W;threaded); gemm!(W,V,H,-one(eltype(W)),one(eltype(W));threaded)
        check=_mm(adjoint(V),W;threaded); after=_block_column_scale(W)
        if after<(real(eltype(W))(7)/10)*scale || _block_column_scale(check)>8eps(real(eltype(W)))*after
            gemm!(W,V,check,-one(eltype(W)),one(eltype(W));threaded)
        end
    end
    _block_basis(W,zeros(eltype(W),size(W,1),0);limit,rtol=tol,scale,threaded)
end
function _arnoldi_continue(A,V0,AV0,frontier,subspace,blocksize,tol,mode,threaded)
    T=eltype(V0); n=size(V0,1); kept=size(V0,2); V=Matrix{T}(undef,n,subspace); AV=similar(V)
    V[:,1:kept]=V0; AV[:,1:kept]=AV0; blocks=UnitRange{Int}[1:kept]; used=kept; products=0
    Q=_arnoldi_basis(frontier,V0,min(blocksize,subspace-kept),tol,mode,threaded)
    while size(Q,2)>0 && used<subspace
        current=used+1:used+size(Q,2); V[:,current]=Q; push!(blocks,current); used=last(current)
        _block_apply!(view(AV,:,current),A,view(V,:,current);threaded); products+=length(current)
        all(isfinite,view(AV,:,current)) || throw(ArgumentError("nonfinite Arnoldi image"))
        used==subspace && break
        Q=_arnoldi_basis(view(AV,:,current),view(V,:,1:used),min(blocksize,subspace-used),tol,mode,threaded)
    end
    basis=copy(view(V,:,1:used)); images=copy(view(AV,:,1:used)); H=_mm(adjoint(basis),images;threaded)
    # The residual after a restart can couple to the whole retained Schur
    # block, so obtain its basis from the full residual, not a scalar tail row.
    residual=images-_mm(basis,H;threaded)
    boundary=_block_basis(residual,basis;limit=min(blocksize,n-used),rtol=tol,scale=_block_column_scale(images),threaded)
    coupling=_mm(adjoint(boundary),images;threaded)
    F=BlockKrylovFactor(basis,images,H,boundary,coupling,blocks,length(blocks),products,
        size(boundary,2)==0 ? :invariant_subspace : :subspace_limit,false)
    F
end

# Order entire Schur blocks. Real conjugates are retained together, including
# when the requested nev cuts a pair; the public result still has nev values.
function _arnoldi_keep(F,keep,capacity,which,target,locked_values,threaded)
    S=schur_factor(F.projected); blocks=_schur_blocks(S.T)
    order=_block_order(S.values,which,target,false); position=invperm(order)
    locked=falses(length(S.values))
    for value in locked_values
        candidates=findall(!,locked)
        isempty(candidates) && break
        j=candidates[argmin(abs.(S.values[candidates].-value))]; locked[j]=true
    end
    mandatory=count(b->any(locked[b]),blocks)
    ids=sortperm([(any(locked[b]) ? 0 : 1,minimum(position[b])) for b in blocks]); chosen=Int[]; total=0
    for (slot,id) in enumerate(ids)
        total>=keep && slot>mandatory && break
        total+length(blocks[id])<=capacity || throw(ArgumentError("Arnoldi subspace needs more room to retain a real pair and expand"))
        push!(chosen,id); total+=length(blocks[id])
    end
    H=copy(S.T); Z=copy(S.Z); sizes=length.(blocks); current=collect(eachindex(blocks)); next=1
    for id in chosen
        j=findfirst(==(id),current)
        for b in j-1:-1:next
            k=1+sum(view(sizes,1:b-1))
            if eltype(H)<:Real
                _real_block_swap!(H,Z,k,sizes[b],sizes[b+1])
            else
                _schur_swap!(H,Z,k)
            end
            sizes[b],sizes[b+1]=sizes[b+1],sizes[b]; current[b],current[b+1]=current[b+1],current[b]
        end
        next+=1
    end
    coefficients=view(Z,:,1:total)
    _block_lift(F.basis,coefficients;threaded),_block_lift(F.images,coefficients;threaded),total
end

"""
    thick_restart_arnoldi(A; nev=1, subspace=..., keep=..., maxrestarts=30, kwargs...)

Thick-restarted block Arnoldi / Krylov-Schur for general dense or sparse A.
Retains whole real Schur blocks and cached operator images. `locking=true`
soft-locks wanted converged Ritz pairs by retaining their Schur subspace; it
never projects with nonorthogonal individual eigenvectors. `:selective` uses
norm-loss/orthogonality checks to trigger a second projection; default :full
always uses two passes. True operator residuals decide convergence.
"""
function thick_restart_arnoldi(A::AbstractMatrix{T};nev::Integer=1,
        subspace::Integer=min(size(A,1),max(20,4nev)),keep::Integer=min(subspace-1,2nev),
        maxrestarts::Integer=30,blocksize::Integer=min(2,subspace),initial=nothing,
        which=:largest_magnitude,target=zero(real(T)),rtol=sqrt(eps(real(T))),atol=zero(real(T)),
        deflation_tol=_block_deflation_default(T,size(A,1)),seed::Integer=0x0b1a5,
        reorthogonalization=:full,locking=true,threaded=true) where {T<:NativeFloat}
    n=_square(A)
    0<nev<=subspace<=n && 1<=blocksize<=subspace && maxrestarts>=0 || throw(ArgumentError("invalid restarted Arnoldi sizes"))
    0<=keep<subspace && (subspace==n || (nev<=keep<=subspace-blocksize)) || throw(ArgumentError("retain nev vectors and room for a block"))
    reorthogonalization in (:full,:selective) || throw(ArgumentError("unknown Arnoldi reorthogonalization"))
    isfinite(rtol) && isfinite(atol) && min(rtol,atol)>=0 && isfinite(target) && isfinite(deflation_tol) && 0<=deflation_tol<1 && seed>=0 || throw(ArgumentError("invalid Arnoldi tolerances/seed"))
    _block_order(real(T)[],which,target,false); A=_block_operator(A)
    F=_block_factor(A;blocksize,maxblocks=subspace,subspace,initial,deflation_tol,seed,threaded)
    products=0; verification=0; iterations=0; history=NamedTuple[]
    for restart in 0:maxrestarts
        result,_,_=_block_ritz(A,F,nev,which,target,rtol,atol;threaded)
        products+=F.matvecs; verification+=result.verification_matvecs; iterations+=F.iterations
        # Soft locks are refreshed every cycle; a pair whose actual residual
        # deteriorates is unlocked instead of being reported permanently valid.
        locked=locking ? count(identity,result.converged_pairs) : 0
        push!(history,(;restart,dimension=size(F.basis,2),residuals=copy(result.residuals),soft_locked=locked,new_matvecs=F.matvecs))
        invariant=size(F.residual_basis,2)==0
        if result.converged || restart==maxrestarts || invariant || subspace==n
            reason=result.converged ? :converged : invariant ? :invariant_subspace : :maxrestarts
            return merge(result,(;restarts=restart,iterations,matvecs=products,verification_matvecs=verification,
                reason,history,soft_locked=locked,reorthogonalization,locking))
        end
        locked_values=locking ? result.values[result.converged_pairs] : eltype(result.values)[]
        V,AV,retained=_arnoldi_keep(F,keep,subspace-blocksize,which,target,locked_values,threaded)
        F=_arnoldi_continue(A,V,AV,F.residual_basis,subspace,blocksize,deflation_tol,reorthogonalization,threaded)
    end
    error("unreachable restarted Arnoldi state")
end
