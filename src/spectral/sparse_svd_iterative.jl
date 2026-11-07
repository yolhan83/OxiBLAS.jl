# SPDX-License-Identifier: Apache-2.0
# Hermitian dilation avoids forming A'*A and its squared condition number.
struct _SVDHermitianDilation{T,A} <: AbstractMatrix{T}
    parent::A
end
_SVDHermitianDilation(A::AbstractMatrix{T}) where T=_SVDHermitianDilation{T,typeof(A)}(A)
Base.size(D::_SVDHermitianDilation)=(sum(size(D.parent)),sum(size(D.parent)))
Base.getindex(::_SVDHermitianDilation,i::Int,j::Int)=throw(ArgumentError("SVD dilation is an implicit operator"))
_block_is_hermitian(::_SVDHermitianDilation)=true
function _block_apply!(Y,D::_SVDHermitianDilation,X;threaded=true)
    A=D.parent; m,n=size(A)
    size(X,1)==m+n && size(Y)==(m+n,size(X,2)) || throw(DimensionMismatch("SVD dilation dimensions differ"))
    XX=Base.mightalias(Y,X) ? copy(X) : X
    _block_apply!(view(Y,1:m,:),A,view(XX,m+1:m+n,:);threaded)
    _block_apply_adjoint!(view(Y,m+1:m+n,:),A,view(XX,1:m,:);threaded)
    Y
end
_block_apply_adjoint!(Y,D::_SVDHermitianDilation,X;threaded=true)=_block_apply!(Y,D,X;threaded)
function _native_mv!(y,D::_SVDHermitianDilation,x)
    xx=Base.mightalias(y,x) ? copy(x) : x
    _block_apply!(reshape(y,:,1),D,reshape(xx,:,1)); y
end

"""
    _sparse_svd_dilation(A, rank; subspace=..., blocksize=..., kwargs...)

Iterative sparse singular triplets through the Hermitian dilation [0 A; A' 0].
Uses thick-restarted block Lanczos and native small Hermitian eigensolvers.
Neither A nor A'*A is formed densely. The Krylov basis has m+n rows, a memory
tradeoff versus specialized Golub–Kahan. Only positive singular triplets are
returned; unresolved/numerically zero values can yield fewer than rank results.
`initial`, if supplied, is an n-by-blocksize right starting block. Thresholding
uses singular_tol times the largest computed positive Ritz value; it is not a
certificate of numerical rank. No zero-singular-vector completion is performed.
"""
function _sparse_svd_dilation(A::NativeSparse,rank::Integer;
                            subspace::Integer=min(sum(size(A)),max(20,4rank+8)),
                            blocksize::Integer=min(2,subspace),keep::Integer=max(0,min(subspace-1,2rank)),
                            maxrestarts::Integer=30,initial=nothing,
                            rtol=sqrt(eps(real(eltype(A)))),atol=zero(real(eltype(A))),
                            singular_tol=eps(real(eltype(A)))*max(size(A)...),
                            deflation_tol=_block_deflation_default(eltype(A),sum(size(A))),
                            seed::Integer=0x0b1a5,threaded=true)
    T=eltype(A); R=real(T); m,n=size(A)
    0<=rank<=min(m,n) || throw(ArgumentError("invalid sparse SVD rank"))
    isfinite(singular_tol) && 0<=singular_tol<1 || throw(ArgumentError("invalid singular value threshold"))
    isfinite(rtol) && isfinite(atol) && min(rtol,atol)>=0 || throw(ArgumentError("invalid singular residual tolerance"))
    maxrestarts>=0 && seed>=0 && isfinite(deflation_tol) && 0<=deflation_tol<1 || throw(ArgumentError("invalid SVD iteration options"))
    if initial!==nothing
        Base.require_one_based_indexing(initial)
        size(initial)==(n,blocksize) && eltype(initial)===T || throw(DimensionMismatch("SVD right starting block differs"))
    end
    if rank==0
        0<=subspace<=m+n && 0<=blocksize<=subspace && 0<=keep<=max(0,subspace-1) || throw(ArgumentError("invalid empty SVD subspace options"))
        return (;U=zeros(T,m,0),S=R[],Vt=zeros(T,0,n),residuals=R[],left_residuals=R[],right_residuals=R[],
                converged=true,converged_triplets=Bool[],iterations=0,restarts=0,
                operator_matvecs=0,verification_matvecs=0,reason=:converged,krylov=nothing)
    end
    A=_block_operator(A); D=_SVDHermitianDilation(A); start=nothing
    if initial!==nothing
        start=vcat(zeros(T,m,blocksize),initial)
    end
    result=thick_restart_lanczos(D;nev=rank,subspace,blocksize,keep,maxrestarts,initial=start,
                                 which=:largest,rtol=rtol/2,atol=atol/2,deflation_tol,seed,threaded,check_hermitian=false)
    scale=maximum(result.values;init=zero(R)); selected=findall(x->x>singular_tol*scale && x>0,result.values)
    S=result.values[selected]; U=Matrix{T}(result.vectors[1:m,selected]); V=Matrix{T}(result.vectors[m+1:m+n,selected])
    for j in eachindex(S)
        unorm=nrm2(view(U,:,j)); vnorm=nrm2(view(V,:,j))
        isfinite(unorm) && isfinite(vnorm) && min(unorm,vnorm)>0 || throw(ArgumentError("unresolved singular vector components"))
        view(U,:,j)./=unorm; view(V,:,j)./=vnorm
    end
    AV=Matrix{T}(undef,m,length(S)); AHU=Matrix{T}(undef,n,length(S))
    _block_apply!(AV,A,V;threaded); _block_apply_adjoint!(AHU,A,U;threaded)
    left=R[]; right=R[]
    for j in eachindex(S)
        axpy!(view(AV,:,j),view(U,:,j),-S[j]); axpy!(view(AHU,:,j),view(V,:,j),-S[j])
        push!(left,nrm2(view(AV,:,j))); push!(right,nrm2(view(AHU,:,j)))
    end
    residuals=max.(left,right)
    flags=[isfinite(residuals[j]) && residuals[j]<=max(atol,rtol*S[j]) for j in eachindex(S)]
    converged=length(S)==rank && all(flags)
    Vt=Matrix{T}(undef,length(S),n)
    for j in 1:n, i in eachindex(S)
        Vt[i,j]=conj(V[j,i])
    end
    reason=converged ? :converged : length(S)<rank ? :insufficient_positive_ritz_values : result.reason===:converged ? :singular_residual : result.reason
    # One dilation application is one A*x and one A'*y. Verification of the
    # original singular equations is additional and accounted separately.
    (;U,S,Vt,residuals,left_residuals=left,right_residuals=right,converged,converged_triplets=flags,
      iterations=result.iterations,restarts=result.restarts,operator_matvecs=2*result.matvecs,
      verification_matvecs=2*result.verification_matvecs+2*length(S),reason,krylov=result)
end
