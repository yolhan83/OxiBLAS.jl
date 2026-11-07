# SPDX-License-Identifier: Apache-2.0
# Native block Krylov projections; upstream eigenvalue/block.rs and thick_restart.rs.
"""
    BlockKrylovFactor{T}

Block Krylov projection, with variable-width blocks after rank deflation.
Approximately A*basis = basis*projected + residual_basis*residual_coupling.
`images` stores A*basis; the large operator is never converted to a dense matrix.
For a Hermitian operator, the projection is block tridiagonal up to rounding
before a thick restart. Full two-pass reorthogonalization is always used.
"""
struct BlockKrylovFactor{T}
    basis::Matrix{T}
    images::Matrix{T}
    projected::Matrix{T}
    residual_basis::Matrix{T}
    residual_coupling::Matrix{T}
    blocks::Vector{UnitRange{Int}}
    iterations::Int
    matvecs::Int
    reason::Symbol
    hermitian::Bool
end

_block_operator(A)=A
_block_operator(A::Union{CSCMatrix,COOMatrix})=CSRMatrix(A)
_block_apply!(Y,A::AbstractMatrix,X;threaded=true)=gemm!(Y,A,X;threaded)
_block_apply!(Y,A::NativeSparse,X;threaded=true)=sparse_mul!(Y,A,X;threaded)
_block_apply_adjoint!(Y,A::AbstractMatrix,X;threaded=true)=gemm!(Y,A,X;transA=:C,threaded)
_block_apply_adjoint!(Y,A::NativeSparse,X;threaded=true)=sparse_mul!(Y,A,X;trans=:C,threaded)
_block_is_hermitian(A::AbstractMatrix)=all(A[i,j]==conj(A[j,i]) for j in axes(A,2) for i in 1:j)
_block_is_hermitian(A::NativeSparse)=_stochastic_hermitian(A)
function _block_column_scale(X)
    maximum((nrm2(view(X,:,j)) for j in axes(X,2));init=zero(real(eltype(X))))
end
_block_deflation_default(::Type{T},n) where T=min(64eps(real(T))*sqrt(max(1,n)),sqrt(eps(real(T)))/8)

# Pivot the remaining residual columns by norm; compare with their scale before
# projection, so roundoff left by an invariant subspace cannot create new vectors.
function _block_basis(X,against;limit=size(X,2),rtol,scale=_block_column_scale(X),threaded=true)
    T=eltype(X); n,s=size(X); W=copy(X); cap=min(n-size(against,2),s,limit)
    all(isfinite,W) && isfinite(scale) || throw(ArgumentError("finite block Krylov data required"))
    Q=Matrix{T}(undef,n,max(0,cap)); cap<=0 && return Q
    if size(against,2)>0
        H=Matrix{T}(undef,size(against,2),s)
        for _ in 1:2
            gemm!(H,against,W;transA=:C,threaded)
            gemm!(W,against,H,-one(T),one(T);threaded)
        end
    end
    used=0; norms=Vector{real(T)}(undef,s)
    for k in 1:cap
        for j in axes(W,2)
            norms[j]=nrm2(view(W,:,j))
        end
        pivot=argmax(norms); norm=norms[pivot]
        isfinite(norm) || throw(ArgumentError("nonfinite Krylov projection"))
        norm>rtol*scale || break
        used+=1
        for i in 1:n
            Q[i,used]=W[i,pivot]/norm
        end
        for _ in 1:2, j in axes(W,2)
            h=dotc(view(Q,:,used),view(W,:,j)); axpy!(view(W,:,j),view(Q,:,used),-h)
        end
        fill!(view(W,:,pivot),zero(T))
    end
    used==cap ? Q : Q[:,1:used]
end

function _block_factor(A::AbstractMatrix{T};hermitian=false,blocksize::Integer=min(2,size(A,1)),
                       maxblocks::Integer=10,
                       subspace::Integer=min(size(A,1),maxblocks*blocksize),initial=nothing,
                       deflation_tol=_block_deflation_default(T,size(A,1)),
                       seed::Integer=0x0b1a5,threaded=true,check_hermitian=true) where {T<:NativeFloat}
    n=_square(A); Base.require_one_based_indexing(A)
    1<=blocksize<=subspace<=n && maxblocks>=1 || throw(ArgumentError("invalid block Krylov sizes"))
    isfinite(deflation_tol) && 0<=deflation_tol<1 && seed>=0 || throw(ArgumentError("invalid block Krylov tolerance/seed"))
    A=_block_operator(A)
    hermitian && check_hermitian && !_block_is_hermitian(A) && throw(ArgumentError("Hermitian block Lanczos operator required"))
    if initial!==nothing
        Base.require_one_based_indexing(initial)
        size(initial)==(n,blocksize) && eltype(initial)===T || throw(DimensionMismatch("initial Krylov block shape/type differs"))
    end
    start=initial===nothing ? _random_projection(T,n,blocksize,seed) : Matrix{T}(initial)
    emptybasis=zeros(T,n,0); Q=_block_basis(start,emptybasis;rtol=deflation_tol,threaded)
    size(Q,2)>0 || throw(ArgumentError("initial Krylov block has zero numerical rank"))
    V=Matrix{T}(undef,n,subspace); AV=similar(V); used=size(Q,2)
    V[:,1:used]=Q; blocks=UnitRange{Int}[1:used]; reason=:block_limit
    for iteration in 1:maxblocks
        current=blocks[end]; _block_apply!(view(AV,:,current),A,view(V,:,current);threaded)
        all(isfinite,view(AV,:,current)) || throw(ArgumentError("nonfinite operator image"))
        if used==subspace
            reason=:subspace_limit; break
        elseif iteration==maxblocks
            break
        end
        next=_block_basis(view(AV,:,current),view(V,:,1:used);limit=subspace-used,rtol=deflation_tol,threaded)
        width=size(next,2)
        if width==0
            reason=:invariant_subspace; break
        end
        push!(blocks,used+1:used+width); V[:,blocks[end]]=next; used+=width
    end
    basis=used==subspace ? V : copy(view(V,:,1:used))
    images=used==subspace ? AV : copy(view(AV,:,1:used)); projected=Matrix{T}(undef,used,used)
    gemm!(projected,basis,images;transA=:C,threaded)
    boundary=_block_basis(view(images,:,blocks[end]),basis;limit=n-used,rtol=deflation_tol,threaded)
    coupling=Matrix{T}(undef,size(boundary,2),used)
    gemm!(coupling,boundary,images;transA=:C,threaded)
    BlockKrylovFactor(basis,images,projected,boundary,coupling,blocks,length(blocks),used,reason,hermitian)
end

"""
    block_arnoldi_factor(A; kwargs...) -> BlockKrylovFactor

Build a rank-deflating block Arnoldi projection; maxblocks counts blocks,
subspace bounds stored basis columns. Initial columns may be dependent.
The default factor reserves at most ten blocks, never n columns unconditionally.
"""
block_arnoldi_factor(A::AbstractMatrix;kwargs...)=_block_factor(A;hermitian=false,kwargs...)

"""
    block_lanczos_factor(A; kwargs...) -> BlockKrylovFactor

Fully reorthogonalized Hermitian block Lanczos projection.
check_hermitian=false is a caller assertion, not an inferred property.
"""
block_lanczos_factor(A::AbstractMatrix;kwargs...)=_block_factor(A;hermitian=true,kwargs...)

function _block_order(values,which,target,hermitian)
    which===:closest && return sortperm(abs.(values .- target))
    which in (:largest,:smallest) && (which=which===:largest ? :largest_real : :smallest_real)
    if which===:bothends
        hermitian || throw(ArgumentError(":bothends requires a Hermitian problem"))
        sorted=sortperm(real.(values)); order=Int[]; lo=1; hi=length(sorted)
        while lo<=hi
            push!(order,sorted[hi]); hi-=1
            lo<=hi && (push!(order,sorted[lo]); lo+=1)
        end
        return order
    end
    _ritz_order(values,which)
end
function _block_lift(V,C;threaded=true)
    T=eltype(V)
    if eltype(C)===T
        Y=Matrix{T}(undef,size(V,1),size(C,2)); gemm!(Y,V,C;threaded); return Y
    end
    T<:RealFloat && eltype(C)===Complex{T} || throw(ArgumentError("incompatible projected scalar type"))
    yr=Matrix{T}(undef,size(V,1),size(C,2)); yi=similar(yr)
    gemm!(yr,V,real.(C);threaded); gemm!(yi,V,imag.(C);threaded)
    complex.(yr,yi)
end
function _block_images(A,X;threaded=true)
    T=eltype(A)
    if eltype(X)===T
        Y=Matrix{T}(undef,size(A,1),size(X,2)); _block_apply!(Y,A,X;threaded); return Y
    end
    T<:RealFloat && eltype(X)===Complex{T} || throw(ArgumentError("incompatible Ritz scalar type"))
    yr=Matrix{T}(undef,size(A,1),size(X,2)); yi=similar(yr)
    _block_apply!(yr,A,real.(X);threaded); _block_apply!(yi,A,imag.(X);threaded)
    complex.(yr,yi)
end
function _block_ritz(A,F,nev,which,target,rtol,atol;threaded=true)
    E=F.hermitian ? eigen_hermitian(F.projected) : eigen_general(F.projected;threaded)
    order=_block_order(E.values,which,target,F.hermitian); selected=order[1:min(nev,length(order))]
    values=E.values[selected]; vectors=_block_lift(F.basis,E.vectors[:,selected];threaded)
    for j in axes(vectors,2)
        norm=nrm2(view(vectors,:,j)); isfinite(norm) && norm>0 || throw(ArgumentError("invalid Ritz vector"))
        view(vectors,:,j)./=norm
    end
    images=_block_images(A,vectors;threaded); R=real(eltype(A)); residuals=R[]
    for j in axes(vectors,2)
        axpy!(view(images,:,j),view(vectors,:,j),-values[j]); push!(residuals,nrm2(view(images,:,j)))
    end
    flags=[isfinite(residuals[j]) && residuals[j]<=max(atol,rtol*abs(values[j])) for j in eachindex(values)]
    result=(;values,vectors,residuals,converged=length(values)==nev && all(flags),converged_pairs=flags,
             iterations=F.iterations,matvecs=F.matvecs,
             verification_matvecs=(eltype(vectors)===eltype(A) ? 1 : 2)*length(values),
             basis_dimension=size(F.basis,2),block_widths=length.(F.blocks),
             reason=length(values)==nev && all(flags) ? :converged : F.reason,factor=F)
    result,E,order
end
function _block_eigen(A::AbstractMatrix{T};hermitian=false,nev::Integer=1,
                      which=:largest_magnitude,target=zero(real(T)),rtol=sqrt(eps(real(T))),atol=zero(real(T)),
                      blocksize::Integer=min(2,size(A,1)),maxblocks::Integer=max(10,cld(2nev,max(1,blocksize))),
                      subspace::Integer=min(size(A,1),maxblocks*blocksize),
                      threaded=true,kwargs...) where {T<:NativeFloat}
    n=_square(A); 0<nev<=n || throw(ArgumentError("invalid requested eigenpair count"))
    isfinite(rtol) && isfinite(atol) && min(rtol,atol)>=0 && isfinite(target) || throw(ArgumentError("invalid Ritz tolerance/target"))
    _block_order(real(T)[],which,target,hermitian) # validate target before operator work
    A=_block_operator(A); F=_block_factor(A;hermitian,blocksize,maxblocks,subspace,threaded,kwargs...)
    result,_,_=_block_ritz(A,F,nev,which,target,rtol,atol;threaded); result
end
"""
    block_arnoldi(A; restart=false, kwargs...)

General block Arnoldi Ritz pairs, with true operator residuals.
No restart is implied; rank loss shrinks blocks rather than launching scalar
solvers. A converged residual does not certify globally extremal selection.
Convergence requires norm(A*v-lambda*v)<=max(atol,rtol*abs(lambda)).
"""
block_arnoldi(A::AbstractMatrix;restart=false,kwargs...)=restart ? thick_restart_arnoldi(A;kwargs...) : _block_eigen(A;hermitian=false,kwargs...)

"""
    block_lanczos(A; kwargs...)

Hermitian block Lanczos Ritz pairs; algebraic, magnitude, bothends or closest
targets. Full reorthogonalization trades additional work/storage for robustness.
"""
block_lanczos(A::AbstractMatrix;kwargs...)=_block_eigen(A;hermitian=true,kwargs...)

"""
    thick_restart_lanczos(A; nev=1, subspace=..., keep=..., maxrestarts=30, kwargs...)

Thick-restarted Hermitian block Krylov eigensolver.
Retains `keep` Ritz vectors together with boundary residual directions, then
expands the retained subspace with full reorthogonalization. Uses a full small
Rayleigh–Ritz projection, not a short-recurrence tridiagonal after restarting.
No hard locking, interval filtering or generalized metric is implied.
"""
function thick_restart_lanczos(A::AbstractMatrix{T};nev::Integer=1,
                               subspace::Integer=min(size(A,1),max(20,4nev)),
                               keep::Integer=min(subspace-1,2nev),maxrestarts::Integer=30,
                               blocksize::Integer=min(2,subspace),initial=nothing,which=:largest_magnitude,
                               target=zero(real(T)),rtol=sqrt(eps(real(T))),atol=zero(real(T)),
                               deflation_tol=_block_deflation_default(T,size(A,1)),seed::Integer=0x0b1a5,
                               check_hermitian=true,threaded=true) where {T<:NativeFloat}
    n=_square(A); 0<nev<=subspace<=n && 1<=blocksize<=subspace && maxrestarts>=0 || throw(ArgumentError("invalid thick restart sizes"))
    0<=keep<subspace && (subspace==n || keep>=nev) || throw(ArgumentError("retain at least nev vectors and room for expansion"))
    isfinite(rtol) && isfinite(atol) && min(rtol,atol)>=0 && isfinite(target) || throw(ArgumentError("invalid restart tolerance/target"))
    _block_order(real(T)[],which,target,true); A=_block_operator(A)
    check_hermitian && !_block_is_hermitian(A) && throw(ArgumentError("Hermitian restart operator required"))
    current=initial; width=blocksize; matvecs=0; verification_matvecs=0; iterations=0
    history=NamedTuple[]
    for restart in 0:maxrestarts
        F=_block_factor(A;hermitian=true,blocksize=width,maxblocks=subspace,subspace,initial=current,
                        deflation_tol,seed,check_hermitian=false,threaded)
        result,E,order=_block_ritz(A,F,nev,which,target,rtol,atol;threaded)
        matvecs+=result.matvecs; verification_matvecs+=result.verification_matvecs; iterations+=result.iterations
        push!(history,(;restart,dimension=result.basis_dimension,residuals=copy(result.residuals),block_widths=copy(result.block_widths)))
        exhausted=restart==maxrestarts; invariant=size(F.residual_basis,2)==0
        if result.converged || exhausted || invariant || subspace==n
            reason=result.converged ? :converged : invariant ? :invariant_subspace : :maxrestarts
            return merge(result,(;restarts=restart,iterations,matvecs,verification_matvecs,reason,history))
        end
        retained=min(keep,size(F.basis,2)); coefficients=E.vectors[:,order[1:retained]]
        X=_block_lift(F.basis,coefficients;threaded)
        extra=min(subspace-retained,size(F.residual_basis,2))
        current=hcat(X,view(F.residual_basis,:,1:extra)); width=size(current,2)
    end
    error("unreachable thick restart state")
end
