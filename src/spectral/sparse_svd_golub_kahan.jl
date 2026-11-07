# SPDX-License-Identifier: Apache-2.0
# Two-sided restarted Golub-Kahan. No normal matrix or Hermitian dilation is
# built. The old dilation implementation remains an explicit alternative.
struct SparseSVDWorkspace{T,R,G}
    U::Matrix{T}
    V::Matrix{T}
    projected::Matrix{T}
    left::Matrix{T}
    right::Matrix{T}
    coefficients::Matrix{T}
    triangular::Matrix{T}
    permutation::Vector{Int}
    ritzU::Matrix{T}
    ritzV::Matrix{T}
    rows::Matrix{T}
    sigma::Vector{R}
    left_errors::Vector{R}
    right_errors::Vector{R}
    gemm::G
    config::GemmConfig
    guard::ReentrantLock
    busy::Base.RefValue{Bool}
end
function SparseSVDWorkspace(::Type{T},m::Integer,n::Integer,rank::Integer;
        subspace=min(m,n,max(20,4rank+8)),blocksize=min(2,max(0,subspace)),rowblock=64) where {T<:NativeFloat}
    isconcretetype(T) && min(m,n,rank,subspace,blocksize)>=0 && rank<=min(m,n) && rowblock>0 || throw(ArgumentError("invalid sparse SVD workspace dimensions"))
    cap=min(subspace,m,n); rank<=cap || throw(ArgumentError("workspace subspace must hold requested rank"))
    (cap==0 ? blocksize==0 : 1<=blocksize<=cap) || throw(ArgumentError("invalid sparse SVD block width"))
    b=Int(blocksize); s=Int(cap); nv=min(n,s+max(b,rank)); R=real(T)
    config=GemmConfig(T;mc=64,kc=64,nc=max(8,min(64,nv)),parallel_threshold=typemax(Int))
    packing=GemmWorkspace(T,max(m,n,s),max(nv,b),max(m,n,s);config,workers=1)
    SparseSVDWorkspace(zeros(T,m,s),zeros(T,n,nv),zeros(T,s,nv),zeros(T,m,b),zeros(T,n,b),
        zeros(T,max(s,nv),b),zeros(T,b,b),collect(1:b),zeros(T,m,rank),zeros(T,n,rank),
        zeros(T,min(rowblock,max(m,n)),s),zeros(R,rank),zeros(R,rank),zeros(R,rank),packing,config,ReentrantLock(),Ref(false))
end
SparseSVDWorkspace(A::NativeSparse,rank::Integer;kwargs...)=SparseSVDWorkspace(eltype(A),size(A,1),size(A,2),rank;kwargs...)
function _gk_arrays(w)
    (w.U,w.V,w.projected,w.left,w.right,w.coefficients,w.triangular,w.permutation,
     w.ritzU,w.ritzV,w.rows,w.sigma,w.left_errors,w.right_errors)
end
"""
    sparse_svd_workspace_bytes(w::SparseSVDWorkspace) -> Int

Owned numeric payload bytes (including packing); excludes outputs, sparse
input, locks/headers, transient small projected factorizations and task metadata.
"""
function sparse_svd_workspace_bytes(w::SparseSVDWorkspace)
    sum(sizeof,_gk_arrays(w))+sum(sizeof(b.a)+sizeof(b.b) for b in w.gemm.buffers)
end
function _gk_check(w::SparseSVDWorkspace{T},A,rank,cap,b,initial) where T
    eltype(A)===T || throw(ArgumentError("sparse SVD workspace scalar differs"))
    m,n=size(A); nv=min(n,cap+max(b,rank))
    size(w.U,1)==m && size(w.U,2)>=cap && size(w.V,1)==n && size(w.V,2)>=nv &&
        size(w.left,1)==m && size(w.left,2)>=b && size(w.right,1)==n && size(w.right,2)>=b &&
        size(w.projected,1)>=cap && size(w.projected,2)>=nv && size(w.coefficients,1)>=max(cap,nv) &&
        size(w.coefficients,2)>=b && min(size(w.triangular)...)>=b && length(w.permutation)>=b &&
        size(w.ritzU,1)==m && size(w.ritzV,1)==n && min(size(w.ritzU,2),size(w.ritzV,2),length(w.sigma),length(w.left_errors),length(w.right_errors))>=rank &&
        size(w.rows,1)>0 && size(w.rows,2)>=cap || throw(DimensionMismatch("sparse SVD workspace capacity differs"))
    arrays=_gk_arrays(w)
    for i in eachindex(arrays)
        Base.mightalias(A.values,arrays[i]) && throw(ArgumentError("sparse values alias SVD workspace"))
        initial===nothing || !Base.mightalias(initial,arrays[i]) || throw(ArgumentError("initial block aliases SVD workspace"))
        for j in 1:i-1
            Base.mightalias(arrays[i],arrays[j]) && throw(ArgumentError("SVD workspace arrays alias"))
        end
        _workspace_operands_check(w.gemm,arrays[i])
    end
    _workspace_operands_check(w.gemm,A.values)
    initial===nothing || _workspace_operands_check(w.gemm,initial)
end
_gk_mul!(C,A,B,w,alpha=one(eltype(C)),beta=zero(eltype(C)))=gemm!(C,A,B,alpha,beta;workspace=w.gemm,config=w.config,threaded=false)

# Row ownership exposes parallelism even for one or two RHSs, without allocating
# a transpose plan or introducing a concurrent scatter in A'*X.
function _gk_apply!(Y,A::CSRMatrix{T},X,threaded) where T
    workers=_operation_workers(A.m,Float64(length(A.values))*size(X,2),threaded,262144)
    _foreach_range(A.m,workers) do rows,_
        @inbounds for i in rows, j in axes(X,2)
            value=zero(T)
            for slot in A.rowptr[i]:A.rowptr[i+1]-1
                value=_native_madd(A.values[slot],X[A.colind[slot],j],value)
            end
            Y[i,j]=value
        end
    end
    Y
end

# Pivoted, twice-reorthogonalized MGS in owned block scratch. R columns retain
# their permutation so the projected coupling is filled in the original order.
function _gk_qr!(X,w,limit,tol,scale)
    b=size(X,2); R=w.triangular; fill!(R,zero(eltype(R)))
    for j in 1:b
        w.permutation[j]=j
    end
    q=0
    for k in 1:min(b,limit)
        pivot=k; norm=nrm2(view(X,:,k))
        for j in k+1:b
            candidate=nrm2(view(X,:,j))
            candidate>norm && (norm=candidate;pivot=j)
        end
        isfinite(norm) || throw(ArgumentError("nonfinite Golub-Kahan residual"))
        norm>tol*scale && norm>0 || break
        if pivot!=k
            for i in axes(X,1)
                X[i,k],X[i,pivot]=X[i,pivot],X[i,k]
            end
            for i in 1:k-1
                R[i,k],R[i,pivot]=R[i,pivot],R[i,k]
            end
            w.permutation[k],w.permutation[pivot]=w.permutation[pivot],w.permutation[k]
        end
        R[k,k]=norm
        for i in axes(X,1)
            X[i,k]=_realdiv(X[i,k],norm)
        end
        for _ in 1:2, j in k+1:b
            h=dotc(view(X,:,k),view(X,:,j)); R[k,j]+=h; axpy!(view(X,:,j),view(X,:,k),-h)
        end
        q=k
    end
    q
end
function _gk_random!(X,state)
    for j in axes(X,2)
        _probe!(view(X,:,j),state,:gaussian)
    end
    X
end
function _gk_remove!(X,V,coeff,w)
    size(V,2)==0 && return X
    for _ in 1:2
        _gk_mul!(coeff,adjoint(V),X,w); _gk_mul!(X,V,coeff,w,-one(eltype(X)),one(eltype(X)))
    end
    X
end

# Right-subspace thick restart: retain selected right Ritz directions, rebuild
# the left recurrence from them. This trades recomputation (counted) for not
# storing A*V/A'*U or a second pair of full restart bases.
function _gk_compress_right!(w,coeff,nv,kept)
    rows=size(w.rows,1); n=size(w.V,1)
    for first in 1:rows:n
        I=first:min(n,first+rows-1); tmp=view(w.rows,1:length(I),1:kept)
        _gk_mul!(tmp,view(w.V,I,1:nv),coeff,w)
        copyto!(view(w.V,I,1:kept),tmp)
    end
end
function _gk_cycle!(A,w,nv,cap,b,tol,state,threaded,needed)
    T=eltype(A); m,n=size(A); nu=0; processed=0; products=0; iterations=0; vmax=min(n,cap+max(b,needed))
    fill!(w.projected,zero(T))
    while nu<cap
        if processed==nv
            nu>=needed && break
            nv>=vmax && break
            width=min(b,vmax-nv); X=view(w.right,:,1:width); _gk_random!(X,state)
            _gk_remove!(X,view(w.V,:,1:nv),view(w.coefficients,1:nv,1:width),w)
            added=_gk_qr!(X,w,width,tol,one(real(T)))
            added==0 && break
            w.V[:,nv+1:nv+added]=view(X,:,1:added); nv+=added
        end
        J=processed+1:min(nv,processed+b); width=length(J); X=view(w.left,:,1:width)
        _gk_apply!(X,A,view(w.V,:,J),threaded); products+=width; iterations+=1
        scale=_block_column_scale(X)
        if nu>0
            C=view(w.coefficients,1:nu,1:width)
            for pass in 1:2
                _gk_mul!(C,adjoint(view(w.U,:,1:nu)),X,w)
                for (jj,j) in enumerate(J), i in 1:nu
                    w.projected[i,j]=pass==1 ? C[i,jj] : w.projected[i,j]+C[i,jj]
                end
                _gk_mul!(X,view(w.U,:,1:nu),C,w,-one(T),one(T))
            end
        end
        added=_gk_qr!(X,w,cap-nu,tol,scale); processed=last(J)
        added==0 && continue
        I=nu+1:nu+added; w.U[:,I]=view(X,:,1:added); nu+=added
        Z=view(w.right,:,1:added); _block_apply_adjoint!(Z,A,view(w.U,:,I);threaded); products+=added
        scale=_block_column_scale(Z); C=view(w.coefficients,1:nv,1:added)
        for pass in 1:2
            _gk_mul!(C,adjoint(view(w.V,:,1:nv)),Z,w)
            for (ii,i) in enumerate(I), j in 1:nv
                w.projected[i,j]=pass==1 ? conj(C[j,ii]) : w.projected[i,j]+conj(C[j,ii])
            end
            _gk_mul!(Z,view(w.V,:,1:nv),C,w,-one(T),one(T))
        end
        appended=_gk_qr!(Z,w,min(b,vmax-nv),tol,scale)
        if appended>0
            K=nv+1:nv+appended; w.V[:,K]=view(Z,:,1:appended)
            for pos in 1:added, j in 1:appended
                w.projected[first(I)+w.permutation[pos]-1,first(K)+j-1]=conj(w.triangular[j,pos])
            end
            nv+=appended
        end
        # A thick right starting block can fill V before U. Stop the cycle
        # at either capacity; do not continue a truncated adjoint recurrence.
        nv>=vmax && nv<n && break
    end
    (;nu,nv,processed,products,iterations)
end
function _gk_ritz!(A,w,cycle,rank,singular_tol,rtol,atol,b,threaded)
    T=eltype(A); R=real(T); nu,nv=cycle.nu,cycle.nv
    if nu==0
        return (;q=0,converged=false,flags=Bool[],small=nothing,selected=Int[],verification=0)
    end
    small=svd_bidiagonal(view(w.projected,1:nu,1:nv);threaded=false)
    all(isfinite,small.S) || throw(OverflowError("projected singular values cannot be represented"))
    scale=maximum(small.S;init=zero(R)); selected=findall(s->s>0 && s>singular_tol*scale,small.S)
    resize!(selected,min(rank,length(selected))); q=length(selected)
    _gk_mul!(view(w.ritzU,:,1:q),view(w.U,:,1:nu),view(small.U,:,selected),w)
    _gk_mul!(view(w.ritzV,:,1:q),view(w.V,:,1:nv),adjoint(view(small.Vt,selected,:)),w)
    w.sigma[1:q]=small.S[selected]
    for first in 1:b:q
        J=first:min(q,first+b-1); X=view(w.left,:,1:length(J)); Z=view(w.right,:,1:length(J))
        _gk_apply!(X,A,view(w.ritzV,:,J),threaded); _block_apply_adjoint!(Z,A,view(w.ritzU,:,J);threaded)
        for (jj,j) in enumerate(J)
            axpy!(view(X,:,jj),view(w.ritzU,:,j),-w.sigma[j]); axpy!(view(Z,:,jj),view(w.ritzV,:,j),-w.sigma[j])
            w.left_errors[j]=nrm2(view(X,:,jj)); w.right_errors[j]=nrm2(view(Z,:,jj))
        end
    end
    flags=[max(w.left_errors[j],w.right_errors[j])<=max(atol,rtol*w.sigma[j]) for j in 1:q]
    (;q,converged=q==rank && all(flags),flags,small,selected,verification=2q)
end

"""
    sparse_svd_lanczos(A, rank; algorithm=:golub_kahan, subspace=..., kwargs...)

Sparse singular triplets with bounded two-sided Golub-Kahan bases.
`workspace` reuses basis, projection, block scratch, Ritz and packing storage;
results own their arrays. `algorithm=:dilation` explicitly retains the prior
Hermitian-dilation solver. No automatic normal-matrix/full-spectrum fallback.
Default results do not retain the Krylov basis; `return_basis=true` copies it.
"""
function sparse_svd_lanczos(A::NativeSparse,rank::Integer;
        algorithm=:golub_kahan,subspace::Integer=min(algorithm===:dilation ? sum(size(A)) : min(size(A)...),max(20,4rank+8)),
        blocksize::Integer=min(2,max(0,subspace)),keep::Integer=max(0,min(subspace-1,2rank)),
        maxrestarts::Integer=30,initial=nothing,rtol=sqrt(eps(real(eltype(A)))),atol=zero(real(eltype(A))),
        singular_tol=eps(real(eltype(A)))*max(size(A)...),deflation_tol=_block_deflation_default(eltype(A),sum(size(A))),
        seed::Integer=0x0b1a5,threaded=true,workspace=nothing,return_basis=false)
    if algorithm===:dilation
        workspace===nothing || throw(ArgumentError("SparseSVDWorkspace belongs to the Golub-Kahan path"))
        return _sparse_svd_dilation(A,rank;subspace,blocksize,keep,maxrestarts,initial,rtol,atol,singular_tol,deflation_tol,seed,threaded)
    end
    algorithm===:golub_kahan || throw(ArgumentError("unknown sparse SVD algorithm"))
    T=eltype(A); R=real(T); m,n=size(A); cap=min(subspace,m,n)
    0<=rank<=min(m,n) && subspace>=0 && rank<=cap || throw(ArgumentError("invalid sparse SVD rank/subspace"))
    (rank==0 ? 0<=blocksize<=cap : 1<=blocksize<=cap) && keep>=0 && maxrestarts>=0 && seed>=0 || throw(ArgumentError("invalid sparse SVD sizes/budget"))
    isfinite(rtol) && isfinite(atol) && min(rtol,atol)>=0 && isfinite(singular_tol) && 0<=singular_tol<1 && isfinite(deflation_tol) && 0<=deflation_tol<1 || throw(ArgumentError("invalid sparse SVD tolerances"))
    rank>0 && cap<min(m,n) && maxrestarts>0 && !(rank<=keep<cap) && throw(ArgumentError("restart must retain rank directions and leave expansion space"))
    if initial!==nothing
        Base.require_one_based_indexing(initial)
        size(initial)==(n,blocksize) && eltype(initial)===T || throw(DimensionMismatch("SVD initial block shape/type differs"))
    end
    A=_block_operator(A); all(isfinite,A.values) || throw(ArgumentError("finite sparse values required"))
    if rank==0
        return (;U=zeros(T,m,0),S=R[],Vt=zeros(T,0,n),residuals=R[],left_residuals=R[],right_residuals=R[],
            converged=true,converged_triplets=Bool[],iterations=0,restarts=0,operator_matvecs=0,verification_matvecs=0,
            reason=:converged,krylov=nothing,algorithm,workspace_bytes=0,history=NamedTuple[])
    end
    w=workspace===nothing ? SparseSVDWorkspace(T,m,n,rank;subspace=cap,blocksize) : workspace
    w isa SparseSVDWorkspace || throw(ArgumentError("invalid sparse SVD workspace"))
    lock(w.guard)
    try
        w.busy[] && throw(ArgumentError("reentrant sparse SVD workspace use"))
        _gk_check(w,A,rank,cap,blocksize,initial); w.busy[]=true
        try
            state=Ref(UInt64(seed)); X=view(w.right,:,1:blocksize)
            initial===nothing ? _gk_random!(X,state) : copyto!(X,initial)
            nv=_gk_qr!(X,w,blocksize,deflation_tol,_block_column_scale(X))
            nv>0 || throw(ArgumentError("initial sparse SVD block has zero numerical rank"))
            w.V[:,1:nv]=view(X,:,1:nv); products=0; verification=0; iterations=0; history=NamedTuple[]
            for restart in 0:maxrestarts
                cycle=_gk_cycle!(A,w,nv,cap,blocksize,deflation_tol,state,threaded,rank)
                ritz=_gk_ritz!(A,w,cycle,rank,singular_tol,rtol,atol,blocksize,threaded)
                products+=cycle.products; verification+=ritz.verification; iterations+=cycle.iterations; q=ritz.q
                push!(history,(;restart,left_dimension=cycle.nu,right_dimension=cycle.nv,new_matvecs=cycle.products,
                    max_residual=maximum((max(w.left_errors[j],w.right_errors[j]) for j in 1:q);init=zero(R))))
                finished=ritz.converged || restart==maxrestarts || cycle.nu<rank || cap==min(m,n)
                if finished
                    S=copy(view(w.sigma,1:q)); le=copy(view(w.left_errors,1:q)); re=copy(view(w.right_errors,1:q))
                    U=copy(view(w.ritzU,:,1:q)); Vt=Matrix(adjoint(view(w.ritzV,:,1:q)))
                    reason=ritz.converged ? :converged : q<rank ? :insufficient_positive_ritz_values : restart==maxrestarts ? :maxrestarts : :singular_residual
                    basis=return_basis ? (;left=copy(view(w.U,:,1:cycle.nu)),right=copy(view(w.V,:,1:cycle.nv)),projected=copy(view(w.projected,1:cycle.nu,1:cycle.nv))) : nothing
                    return (;U,S,Vt,residuals=max.(le,re),left_residuals=le,right_residuals=re,converged=ritz.converged,
                        converged_triplets=ritz.flags,iterations,restarts=restart,operator_matvecs=products,verification_matvecs=verification,
                        reason,krylov=basis,algorithm,left_basis_dimension=cycle.nu,right_basis_dimension=cycle.nv,
                        workspace_bytes=sparse_svd_workspace_bytes(w),history)
                end
                retained=min(keep,length(ritz.small.S)); retained=max(rank,retained)
                coeff=adjoint(view(ritz.small.Vt,1:retained,:))
                _gk_compress_right!(w,coeff,cycle.nv,retained); nv=retained
            end
        finally
            w.busy[]=false
        end
    finally
        unlock(w.guard)
    end
    error("unreachable sparse SVD state")
end
