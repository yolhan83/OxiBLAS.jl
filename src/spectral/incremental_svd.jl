# SPDX-License-Identifier: Apache-2.0
"""Mutable, rank-capped native SVD state for appending rows or columns.
Updates factor the stored approximation augmented with the new block; discarded
directions from earlier updates cannot be recovered. Use svd_snapshot for a
coherent independent copy while other tasks may update this state.
last_truncation_norm and last_projection_residual are local diagnostics, not a
certified cumulative error bound for the original data stream.
"""
mutable struct IncrementalSVDFactor{T,R}
    U::Matrix{T}
    S::Vector{R}
    Vt::Matrix{T}
    shape::Tuple{Int,Int}
    maxrank::Int
    ranktol::R
    svd_rtol::R
    maxsweeps::Int
    reorthogonalize::Bool
    initial_method::Symbol
    updates::Int
    last_truncation_norm::R
    last_projection_residual::R
    guard::ReentrantLock
    active::Bool
end
Base.eltype(::Type{IncrementalSVDFactor{T,R}}) where {T,R}=T
Base.size(F::IncrementalSVDFactor)=lock(()->F.shape,F.guard)
"""Current retained rank, read under the state guard without copying factors."""
svd_rank(F::IncrementalSVDFactor)=lock(()->length(F.S),F.guard)
function Base.size(F::IncrementalSVDFactor,d::Integer)
    d>0 || throw(ArgumentError("dimension must be positive"))
    d<=2 ? size(F)[d] : 1
end
function _isvd_locked(f,F::IncrementalSVDFactor)
    lock(F.guard) do
        F.active && throw(ArgumentError("recursive incremental SVD use is unsupported"))
        F.active=true
        try
            f()
        finally
            F.active=false
        end
    end
end
function svd_snapshot(F::IncrementalSVDFactor)
    _isvd_locked(F) do
        (;U=copy(F.U),S=copy(F.S),Vt=copy(F.Vt),shape=F.shape,maxrank=F.maxrank,
          updates=F.updates,initial_method=F.initial_method,
          last_truncation_norm=F.last_truncation_norm,last_projection_residual=F.last_projection_residual)
    end
end
function Base.copy(F::IncrementalSVDFactor)
    _isvd_locked(F) do
        IncrementalSVDFactor(copy(F.U),copy(F.S),copy(F.Vt),F.shape,F.maxrank,F.ranktol,
                             F.svd_rtol,F.maxsweeps,F.reorthogonalize,F.initial_method,F.updates,
                             F.last_truncation_norm,F.last_projection_residual,ReentrantLock(),false)
    end
end
function _isvd_count(S,cap,ranktol)
    all(x->isfinite(x) && x>=0,S) || throw(ArgumentError("singular values not representable in working precision"))
    scale=maximum(S;init=zero(eltype(S)))
    min(cap,count(x->x>ranktol*scale,S))
end
_isvd_default_rank(A)=A isa NativeSparse ? min(16,min(size(A)...)) : min(size(A)...)

"""Initialize a native incremental SVD. maxrank defaults to the requested rank,
so increasing the retained rank later requires a larger explicit cap here.
Dense input defaults to :jacobi and also accepts :randomized; sparse input
defaults to :randomized (at most rank 16 unless requested) and accepts :lanczos.
Only the factor bases are retained, not a copy
of the original matrix. Zero requested rank creates a zero approximation.
"""
function incremental_svd(A::AbstractMatrix{T};rank::Integer=_isvd_default_rank(A),maxrank::Integer=rank,
                         method=:auto,ranktol=32eps(real(T)),svd_rtol=8eps(real(T)),
                         maxsweeps::Integer=100,reorthogonalize=true,
                         oversampling::Integer=8,power_iterations::Integer=2,seed::Integer=0x0b1a5,
                         threaded=true) where {T<:NativeFloat}
    Base.require_one_based_indexing(A); m,n=size(A); R=real(T)
    0<=rank<=min(m,n) && maxrank>=rank && maxsweeps>=0 || throw(ArgumentError("invalid incremental SVD ranks/budget"))
    isfinite(ranktol) && 0<=ranktol<1 && isfinite(svd_rtol) && 0<svd_rtol<1 || throw(ArgumentError("invalid incremental SVD tolerances"))
    rt,st=R(ranktol),R(svd_rtol)
    0<=rt<1 && 0<st<1 || throw(ArgumentError("tolerances not representable in working precision"))
    min(oversampling,power_iterations,seed)>=0 || throw(ArgumentError("invalid sparse initialization options"))
    reorthogonalize isa Bool || throw(ArgumentError("reorthogonalize must be Boolean"))
    selected=method===:auto ? (A isa NativeSparse ? :randomized : :jacobi) : method
    selected in (:jacobi,:randomized,:lanczos) || throw(ArgumentError("unknown incremental SVD initialization"))
    (A isa NativeSparse ? selected!==:jacobi : selected!==:lanczos) || throw(ArgumentError("use dense :jacobi/:randomized or sparse :randomized/:lanczos initialization"))
    if rank==0
        U=zeros(T,m,0); S=R[]; Vt=zeros(T,0,n); trunc=isempty(A) ? zero(R) : R(NaN)
    elseif selected===:jacobi
        initial=svd_factor(A;rtol=st,maxsweeps,threaded)
        k=_isvd_count(initial.S,rank,rt); U=initial.U[:,1:k]; S=initial.S[1:k]; Vt=initial.Vt[1:k,:]
        trunc=nrm2(view(initial.S,k+1:length(initial.S)))
    elseif selected===:randomized
        initial=A isa NativeSparse ? sparse_svd(A,rank;oversampling,power_iterations,seed,threaded,rtol=st) :
                                    svd_randomized(A,rank;oversampling,power_iterations,seed,threaded,rtol=st)
        k=_isvd_count(initial.S,rank,rt); U=initial.U[:,1:k]; S=initial.S[1:k]; Vt=initial.Vt[1:k,:]
        trunc=R(NaN) # omitted sketch directions have no certified error here
    else
        initial=sparse_svd_lanczos(A,rank;seed,threaded,singular_tol=rt)
        all(initial.converged_triplets) && (initial.converged || initial.reason===:insufficient_positive_ritz_values) ||
            throw(ConvergenceError(:incremental_svd_init,initial.iterations,1:rank))
        k=_isvd_count(initial.S,rank,rt); U=initial.U[:,1:k]; S=initial.S[1:k]; Vt=initial.Vt[1:k,:]
        trunc=R(NaN)
    end
    IncrementalSVDFactor(U,S,Vt,(m,n),Int(maxrank),rt,st,Int(maxsweeps),Bool(reorthogonalize),
                         rank==0 ? :zero : selected,0,trunc,zero(R),ReentrantLock(),false)
end

"""Replace a state's initial matrix, preserving its cap/tolerances/options.
On success resets the update count; on failure leaves the old state intact.
Initialization keywords such as method, seed and sketch sizes are forwarded.
"""
function initialize_svd!(F::IncrementalSVDFactor{T},A::AbstractMatrix;
                         rank::Integer=min(F.maxrank,_isvd_default_rank(A)),kwargs...) where T
    _isvd_locked(F) do
        eltype(A)===T || throw(ArgumentError("replacement SVD matrix has wrong scalar type"))
        replacement=incremental_svd(A;rank,maxrank=F.maxrank,ranktol=F.ranktol,svd_rtol=F.svd_rtol,
                                    maxsweeps=F.maxsweeps,reorthogonalize=F.reorthogonalize,kwargs...)
        F.U=replacement.U; F.S=replacement.S; F.Vt=replacement.Vt; F.shape=replacement.shape
        F.initial_method=replacement.initial_method; F.updates=0
        F.last_truncation_norm=replacement.last_truncation_norm
        F.last_projection_residual=replacement.last_projection_residual
        F
    end
end

# Recondition the two bases through QR and a small SVD, preserving the represented
# matrix instead of independently normalizing columns while leaving Sigma fixed.
function _isvd_reorthogonalize(U,S,Vt,svd_rtol,maxsweeps;threaded=true)
    T=eltype(U); r=length(S); r==0 && return U,S,Vt
    left=qr_factor(U;threaded); right=qr_factor(Matrix(adjoint(Vt));threaded)
    Qu=qr_q(left;columns=r,threaded); Qv=qr_q(right;columns=r,threaded)
    Ru=zeros(T,r,r); Rv=zeros(T,r,r)
    for j in 1:r, i in 1:j
        Ru[i,j]=left.factors[i,j]*S[j]; Rv[i,j]=right.factors[i,j]
    end
    core=Matrix{T}(undef,r,r); gemm!(core,Ru,Rv;transB=:C,threaded)
    G=svd_factor(core;rtol=svd_rtol,maxsweeps,threaded)
    unew=Matrix{T}(undef,size(U)); vnew=Matrix{T}(undef,size(Vt))
    gemm!(unew,Qu,G.U;threaded); gemm!(vnew,G.Vt,Qv;transB=:C,threaded)
    unew,G.S,vnew
end
function reorthogonalize_svd!(F::IncrementalSVDFactor;threaded=true)
    _isvd_locked(F) do
        U,S,Vt=_isvd_reorthogonalize(F.U,F.S,F.Vt,F.svd_rtol,F.maxsweeps;threaded)
        F.U=U; F.S=S; F.Vt=Vt
        F
    end
end
function _isvd_bases(F;threaded=true)
    F.reorthogonalize ? _isvd_reorthogonalize(F.U,F.S,F.Vt,F.svd_rtol,F.maxsweeps;threaded) : (F.U,F.S,F.Vt)
end
function _isvd_commit!(F,U,S,Vt,shape,truncation,projection)
    # All numerical operations and allocations complete before replacing state.
    # Readers must hold the guard (svd_snapshot does) to see one coherent version.
    updates=Base.checked_add(F.updates,1)
    F.U=U; F.S=S; F.Vt=Vt; F.shape=shape
    F.last_truncation_norm=truncation; F.last_projection_residual=projection
    F.updates=updates
    F
end

"""Append a p-by-n block of rows, always refactoring the coupling SVD even if
the residual block has rank zero. The input is copied under the state lock,
so views of owned factor data are accepted without in-place corruption.
"""
function svd_add_rows!(F::IncrementalSVDFactor{T},rows::AbstractMatrix;threaded=true) where T
    _isvd_locked(F) do
        Base.require_one_based_indexing(rows); m,n=F.shape; p=size(rows,1)
        size(rows,2)==n || throw(DimensionMismatch("new SVD rows have wrong column count"))
        eltype(rows)===T || throw(ArgumentError("new SVD rows have wrong scalar type"))
        p==0 && return F
        shape=(Base.checked_add(m,p),n); B=Matrix{T}(rows)
        all(isfinite,B) || throw(ArgumentError("finite SVD update data required"))
        if F.maxrank==0 && isempty(F.S)
            return _isvd_commit!(F,zeros(T,m+p,0),real(T)[],zeros(T,0,n),shape,nrm2(vec(B)),zero(real(T)))
        end
        U,S,Vt=_isvd_bases(F;threaded); r=length(S)
        M=Matrix{T}(undef,p,r); gemm!(M,B,Vt;transB=:C,threaded)
        V=Matrix(adjoint(Vt)); Q=_block_basis(Matrix(adjoint(B)),V;rtol=F.ranktol,threaded)
        s=size(Q,2); C=Matrix{T}(undef,p,s); gemm!(C,B,Q;threaded)
        residual=copy(B); gemm!(residual,M,Vt,-one(T),one(T);threaded)
        gemm!(residual,C,Q,-one(T),one(T);transB=:C,threaded)
        projection=nrm2(vec(residual))
        K=zeros(T,r+p,r+s)
        for i in 1:r
            K[i,i]=S[i]
        end
        K[r+1:r+p,1:r]=M; K[r+1:r+p,r+1:r+s]=C
        G=svd_factor(K;rtol=F.svd_rtol,maxsweeps=F.maxsweeps,threaded)
        k=_isvd_count(G.S,F.maxrank,F.ranktol)
        unew=Matrix{T}(undef,m+p,k); vnew=Matrix{T}(undef,k,n)
        gemm!(view(unew,1:m,:),U,view(G.U,1:r,1:k);threaded)
        unew[m+1:m+p,:]=G.U[r+1:r+p,1:k]
        gemm!(vnew,view(G.Vt,1:k,1:r),Vt;threaded)
        gemm!(vnew,view(G.Vt,1:k,r+1:r+s),Q,one(T),one(T);transB=:C,threaded)
        _isvd_commit!(F,unew,G.S[1:k],vnew,shape,nrm2(view(G.S,k+1:length(G.S))),projection)
    end
end

"""Append an m-by-q column block, using a rank-revealing residual basis and a
native coupling SVD. A zero residual still changes singular values/vectors.
"""
function svd_add_columns!(F::IncrementalSVDFactor{T},columns::AbstractMatrix;threaded=true) where T
    _isvd_locked(F) do
        Base.require_one_based_indexing(columns); m,n=F.shape; q=size(columns,2)
        size(columns,1)==m || throw(DimensionMismatch("new SVD columns have wrong row count"))
        eltype(columns)===T || throw(ArgumentError("new SVD columns have wrong scalar type"))
        q==0 && return F
        shape=(m,Base.checked_add(n,q)); B=Matrix{T}(columns)
        all(isfinite,B) || throw(ArgumentError("finite SVD update data required"))
        if F.maxrank==0 && isempty(F.S)
            return _isvd_commit!(F,zeros(T,m,0),real(T)[],zeros(T,0,n+q),shape,nrm2(vec(B)),zero(real(T)))
        end
        U,S,Vt=_isvd_bases(F;threaded); r=length(S)
        P=Matrix{T}(undef,r,q); gemm!(P,U,B;transA=:C,threaded)
        Q=_block_basis(B,U;rtol=F.ranktol,threaded); s=size(Q,2)
        C=Matrix{T}(undef,s,q); gemm!(C,Q,B;transA=:C,threaded)
        residual=copy(B); gemm!(residual,U,P,-one(T),one(T);threaded)
        gemm!(residual,Q,C,-one(T),one(T);threaded); projection=nrm2(vec(residual))
        K=zeros(T,r+s,r+q)
        for i in 1:r
            K[i,i]=S[i]
        end
        K[1:r,r+1:r+q]=P; K[r+1:r+s,r+1:r+q]=C
        G=svd_factor(K;rtol=F.svd_rtol,maxsweeps=F.maxsweeps,threaded)
        k=_isvd_count(G.S,F.maxrank,F.ranktol)
        unew=Matrix{T}(undef,m,k); vnew=Matrix{T}(undef,k,n+q)
        gemm!(unew,U,view(G.U,1:r,1:k);threaded)
        gemm!(unew,Q,view(G.U,r+1:r+s,1:k),one(T),one(T);threaded)
        gemm!(view(vnew,:,1:n),view(G.Vt,1:k,1:r),Vt;threaded)
        vnew[:,n+1:n+q]=G.Vt[1:k,r+1:r+q]
        _isvd_commit!(F,unew,G.S[1:k],vnew,shape,nrm2(view(G.S,k+1:length(G.S))),projection)
    end
end
