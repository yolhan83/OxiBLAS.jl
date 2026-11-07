# SPDX-License-Identifier: Apache-2.0
struct CompleteLUFactor{T}
    factors::Matrix{T}
    p::Vector{Int}
    q::Vector{Int}
    info::Int
    pivot::Symbol
end
function Base.getproperty(F::CompleteLUFactor,s::Symbol)
    if s in (:L,:U)
        A=getfield(F,:factors); m,n=size(A); k=min(m,n); T=eltype(A)
        C=s===:L ? zeros(T,m,k) : zeros(T,k,n)
        if s===:L
            for j in 1:k, i in j:m
                C[i,j]=i==j ? one(T) : A[i,j]
            end
        else
            for j in 1:n, i in 1:min(j,k)
                C[i,j]=A[i,j]
            end
        end
        return C
    end
    getfield(F,s)
end
function _complete_pivot(A,j)
    p=q=j; best=abs(A[j,j])
    for col in j:size(A,2), row in j:size(A,1)
        v=abs(A[row,col])
        if v>best
            best=v; p=row; q=col
        end
    end
    p,q
end
function _rook_pivot(A,j)
    col=j
    for _ in 1:2*(size(A,1)+size(A,2))
        row=j
        for i in j+1:size(A,1)
            abs(A[i,col])>abs(A[row,col]) && (row=i)
        end
        nextcol=col
        for c in j:size(A,2)
            abs(A[row,c])>abs(A[row,nextcol]) && (nextcol=c)
        end
        nextcol==col && return row,col
        col=nextcol
    end
    _complete_pivot(A,j)
end
function _pivot_lu_factor(A::AbstractMatrix{T},pivot;check=true,threaded=true) where {T<:GenericFloat}
    threaded=threaded && generic_threadsafe(T)
    pivot in (:complete,:rook) || throw(ArgumentError("pivot must be :partial, :complete or :rook"))
    Base.require_one_based_indexing(A)
    F=Matrix(A); m,n=size(F); p=collect(1:m); q=collect(1:n); info=0
    for j in 1:min(m,n)
        row,col=pivot===:rook ? _rook_pivot(F,j) : _complete_pivot(F,j)
        if row!=j
            for c in 1:n
                F[j,c],F[row,c]=F[row,c],F[j,c]
            end
            p[j],p[row]=p[row],p[j]
        end
        if col!=j
            for r in 1:m
                F[r,j],F[r,col]=F[r,col],F[r,j]
            end
            q[j],q[col]=q[col],q[j]
        end
        if iszero(F[j,j])
            info==0 && (info=j)
            check && throw(SingularException(j))
            continue
        end
        for i in j+1:m
            F[i,j]/=F[j,j]
        end
        workers=_operation_workers(n-j,Float64(m-j)*(n-j),threaded,262144)
        _foreach_range(n-j,workers) do columns,_
            @inbounds for c in columns
                # Task-local column: `col` (the pivot column above) was shared
                # by every task of this closure, a data race when threaded.
                local jc=j+c; v=F[j,jc]
                @simd for i in j+1:m
                    F[i,jc]-=F[i,j]*v
                end
            end
        end
    end
    CompleteLUFactor(F,p,q,info,pivot)
end
lu_complete(A;kwargs...)=lu_factor(A;pivot=:complete,kwargs...)
lu_rook(A;kwargs...)=lu_factor(A;pivot=:rook,kwargs...)
function solve(F::CompleteLUFactor,b::AbstractVecOrMat)
    n=_square(F.factors); Base.require_one_based_indexing(b)
    size(b,1)==n || throw(DimensionMismatch("LU solve dimensions differ"))
    F.info==0 || throw(SingularException(F.info))
    x=ndims(b)==1 ? reshape(b[F.p],n,1) : b[F.p,:]
    trsm!(x,F.factors;uplo=:L,unit=true); trsm!(x,F.factors;uplo=:U)
    result=similar(x)
    for j in axes(x,2), i in 1:n
        result[F.q[i],j]=x[i,j]
    end
    ndims(b)==1 ? vec(result) : result
end

# The reflectors use a real tau for both real and complex data: H=H'=I-tau*v*v'.
function _qr_reflector!(F,tau,j)
    m=size(F,1); T=eltype(F); r=nrm2(view(F,j:m,j))
    iszero(r) && return zero(T)
    phase=iszero(F[j,j]) ? one(T) : F[j,j]/abs(F[j,j])
    den=F[j,j]/r+phase; τ=one(real(T))+abs(F[j,j])/r
    for i in j+1:m
        F[i,j]=(F[i,j]/r)/den
    end
    F[j,j]=-phase*r; tau[j]=τ
    convert(T,τ)
end
function _qr_update_columns!(F,j,τ,columns)
    iszero(τ) && return
    m=size(F,1)
    if F isa StridedMatrix{<:NativeFloat}
        # Native data: the SIMD Level 1 dot/axpy (four real sums for complex
        # dots) instead of the scalar loops below.
        v=view(F,j+1:m,j)
        @inbounds for col in columns
            c=view(F,j+1:m,col)
            s=(F[j,col]+dotc(v,c))*τ
            F[j,col]-=s
            axpy!(c,v,-s)
        end
        return
    end
    @inbounds for col in columns
        s=F[j,col]
        for i in j+1:m
            s+=conj(F[i,j])*F[i,col]
        end
        s*=τ; F[j,col]-=s
        @simd for i in j+1:m
            F[i,col]-=F[i,j]*s
        end
    end
end
# Compact WY form (LAPACK larft/larfb). Reflectors j0:j0+p-1 of a QR factor
# (real tau: H=I-tau*v*v') multiply to H_j0...H_j0+p-1 = I-V*T*V', V unit
# lower trapezoidal and T upper triangular. V is copied into a reused buffer
# so that both products are plain GEMMs.
function _qr_block_v!(V,F,j0,p)
    m=size(F,1); rows=m-j0+1; T=eltype(V)
    @inbounds for c in 1:p
        for i in 1:c-1
            V[i,c]=zero(T)
        end
        V[c,c]=one(T)
        for i in c+1:rows
            V[i,c]=F[j0+i-1,j0+c-1]
        end
    end
    view(V,1:rows,1:p)
end
# T[1:j-1,j] = -tau_j * T[1:j-1,1:j-1] * V[:,1:j-1]'*v_j; the Gram matrix V'V
# comes from one GEMM instead of p^2/2 scalar dot products.
function _qr_tfactor!(Tm,V,tau,j0,G;threaded=true)
    p=size(V,2); T=eltype(Tm)
    gemm!(G,adjoint(V),V;threaded)
    @inbounds for j in 1:p
        τ=convert(T,tau[j0+j-1]); Tm[j,j]=τ
        for i in 1:j-1
            Tm[i,j]=-τ*G[i,j]
        end
        for i in 1:j-1
            s=zero(T)
            for k in i:j-1
                s+=Tm[i,k]*Tm[k,j]
            end
            Tm[i,j]=s # ascending i keeps Tm[k,j], k>i, intact
        end
        for i in j+1:p
            Tm[i,j]=zero(T)
        end
    end
    Tm
end
# B <- (I-V*op(T)*V')*B (left) or B*(I-V*op(T)*V') (right), op = adjoint
# when adj: two GEMMs and one TRMM with the W buffer.
function _qr_block_apply!(B,V,Tm,W,left,adj;threaded=true)
    p=size(V,2); T=eltype(B); tr=adj ? :C : :N
    if left
        Wv=view(W,1:p,1:size(B,2))
        gemm!(Wv,adjoint(V),B;threaded)
        trmm!(Wv,Tm;uplo=:U,trans=tr,threaded)
        gemm!(B,V,Wv,-one(T),one(T);threaded)
    else
        Wv=view(W,1:size(B,1),1:p)
        gemm!(Wv,B,V;threaded)
        trmm!(Wv,Tm;side=:R,uplo=:U,trans=tr,threaded)
        gemm!(B,Wv,adjoint(V),-one(T),one(T);threaded)
    end
    B
end
function _qr_blocked(A::AbstractMatrix{T};blocksize=32,threaded=true) where {T<:GenericFloat}
    threaded=threaded && generic_threadsafe(T)
    Base.require_one_based_indexing(A)
    blocksize>0 || throw(ArgumentError("blocksize must be positive"))
    F=Matrix(A); m,n=size(F); q=min(m,n); tau=zeros(T,q)
    nb=min(blocksize,q)
    # Panel buffers are allocated once (V, T, Gram, W) instead of per block.
    V=Matrix{T}(undef,m,nb); Tm=Matrix{T}(undef,nb,nb); G=Matrix{T}(undef,nb,nb)
    W=Matrix{T}(undef,nb,max(0,n-nb))
    for start in 1:blocksize:q
        stop=min(q,start+blocksize-1); p=stop-start+1
        for j in start:stop
            τ=_qr_reflector!(F,tau,j)
            _qr_update_columns!(F,j,τ,j+1:stop)
        end
        stop==n && continue
        Vv=_qr_block_v!(V,F,start,p); Tv=_qr_tfactor!(view(Tm,1:p,1:p),Vv,tau,start,view(G,1:p,1:p);threaded)
        _qr_block_apply!(view(F,start:m,stop+1:n),Vv,Tv,W,true,true;threaded)
    end
    QRFactor(F,tau)
end

struct PivotedQRFactor{T}
    qr::QRFactor{T}
    p::Vector{Int}
end
function Base.getproperty(F::PivotedQRFactor,s::Symbol)
    s in (:Q,:R,:factors,:tau) && return getproperty(getfield(F,:qr),s)
    getfield(F,s)
end
"""
    qr_pivoted(A::AbstractMatrix{T}; threaded=true) -> PivotedQRFactor

Column-pivoted Householder QR; A[:,p]=Q*R. Column norms are downdated after
each reflector and recomputed when cancellation makes the downdate unreliable,
with LAPACK's geqp3/laqp2 criterion.
"""
function qr_pivoted(A::AbstractMatrix{T};threaded=true) where {T<:GenericFloat}
    threaded=threaded && generic_threadsafe(T)
    Base.require_one_based_indexing(A)
    F=Matrix(A); m,n=size(F); tau=zeros(T,min(m,n)); p=collect(1:n); R=real(T)
    # vn1: downdated norms of the trailing parts of the columns; vn2: the
    # norm at their last exact computation. Recomputing every remaining
    # norm at every step cost an extra O(mn^2) Level 1 sweep (1.2-9x behind
    # OpenBLAS geqp3).
    vn1=R[nrm2(view(F,:,k)) for k in 1:n]; vn2=copy(vn1)
    tol3z=sqrt(eps(R))
    for j in 1:min(m,n)
        col=j; best=vn1[j]
        for k in j+1:n
            vn1[k]>best && (best=vn1[k]; col=k)
        end
        if col!=j
            for i in 1:m
                F[i,j],F[i,col]=F[i,col],F[i,j]
            end
            p[j],p[col]=p[col],p[j]; vn1[col]=vn1[j]; vn2[col]=vn2[j]
        end
        τ=_qr_reflector!(F,tau,j)
        workers=_operation_workers(n-j,Float64(m-j)*(n-j),threaded,262144)
        _foreach_range(n-j,workers) do columns,_
            _qr_update_columns!(F,j,τ,(j+first(columns)):(j+last(columns)))
        end
        for k in j+1:n
            iszero(vn1[k]) && continue
            temp=max(zero(R),one(R)-(abs(F[j,k])/vn1[k])^2)
            if temp*(vn1[k]/vn2[k])^2<=tol3z
                vn1[k]=j<m ? nrm2(view(F,j+1:m,k)) : zero(R); vn2[k]=vn1[k]
            else
                vn1[k]*=sqrt(temp)
            end
        end
    end
    PivotedQRFactor(QRFactor(F,tau),p)
end
"""
    least_squares(F::PivotedQRFactor, b::AbstractVecOrMat; rtol=eps(real(eltype(F.factors)))*max(size(F.factors)...)) -> AbstractVecOrMat

Rank-revealing QR basic least-squares solution (free pivoted variables zero).
For a rank-deficient problem this is a basic solution, not the minimum norm solution.
"""
function least_squares(F::PivotedQRFactor,b::AbstractVecOrMat;rtol=eps(real(eltype(F.factors)))*max(size(F.factors)...))
    A=F.factors; m,n=size(A); Base.require_one_based_indexing(b)
    size(b,1)==m && rtol>=0 || throw(ArgumentError("invalid pivoted least-squares dimensions/tolerance"))
    scale=maximum(abs(A[i,i]) for i in 1:min(m,n);init=zero(real(eltype(A))))
    rank=0
    for i in 1:min(m,n)
        abs(A[i,i])>rtol*scale || break
        rank=i
    end
    Y=ndims(b)==1 ? reshape(copy(b),m,1) : copy(b); _apply_q!(Y,F.qr,true)
    Z=zeros(eltype(A),n,size(Y,2))
    if rank>0
        copyto!(view(Z,1:rank,:),view(Y,1:rank,:))
        trsm!(view(Z,1:rank,:),view(A,1:rank,1:rank);uplo=:U)
    end
    X=similar(Z)
    for j in axes(X,2), i in 1:n
        X[F.p[i],j]=Z[i,j]
    end
    ndims(b)==1 ? vec(X) : X
end
solve(F::PivotedQRFactor,b::AbstractVecOrMat)=least_squares(F,b)

struct LDLFactor{T}
    L::Matrix{T}
    d::Vector{T}
    dl::Vector{T}
    p::Vector{Int}
    blocks::Vector{Int}
    hermitian::Bool
end
function Base.getproperty(F::LDLFactor{T},s::Symbol) where T
    if s===:D
        d=getfield(F,:d); dl=getfield(F,:dl); herm=getfield(F,:hermitian)
        n=length(d); D=zeros(T,n,n)
        @inbounds for i in 1:n
            D[i,i]=d[i]
            if i<n && !iszero(dl[i])
                D[i+1,i]=dl[i]; D[i,i+1]=herm ? conj(dl[i]) : dl[i]
            end
        end
        return D
    end
    getfield(F,s)
end
Base.propertynames(::LDLFactor,private::Bool=false)=(:L,:D,:p,:blocks,:hermitian,(private ? (:d,:dl) : ())...)
@inline _ldlconj(x,h)=h ? conj(x) : x
function _symmetric_swap!(S,L,p,a,b,k)
    a==b && return
    n=size(S,1)
    for j in 1:n
        S[a,j],S[b,j]=S[b,j],S[a,j]
    end
    for i in 1:n
        S[i,a],S[i,b]=S[i,b],S[i,a]
    end
    for j in 1:k-1
        L[a,j],L[b,j]=L[b,j],L[a,j]
    end
    p[a],p[b]=p[b],p[a]
end
"""
    ldlt_factor(A::AbstractMatrix{T}; uplo=:L, hermitian=true, pivot=:bunchkaufman, threaded=true, blocksize=nothing) -> LDLTFactor

Symmetric/Hermitian indefinite LDL factorization with Bunch–Kaufman
1×1/2×2 diagonal pivot selection. Only the selected input triangle is read.
Set pivot=:none for unpivoted LDL. A[p,p]=L*D*(hermitian ? L' : transpose(L)).
"""
function ldlt_factor(A::AbstractMatrix{T};uplo=:L,hermitian=true,pivot=:bunchkaufman,threaded=true,blocksize=nothing) where {T<:GenericFloat}
    threaded=threaded && generic_threadsafe(T)
    n=_square(A); upper=_uplo(uplo)
    pivot in (:none,:bunchkaufman) || throw(ArgumentError("unsupported LDL pivot strategy"))
    blocksize=something(blocksize,n<=256 ? 32 : 64)
    blocksize>0 || throw(ArgumentError("blocksize must be positive"))
    # Working lower triangle only; L accumulates below its diagonal.
    S=Matrix{T}(undef,n,n)
    @inbounds for j in 1:n, i in j:n
        S[i,j]=_symget(A,i,j,upper,hermitian)
    end
    p=collect(1:n); blocks=Int[]; sizehint!(blocks,n)
    d=zeros(T,n); dl=zeros(T,max(0,n-1))
    _ldl_blocked!(S,d,dl,p,blocks,hermitian,pivot===:bunchkaufman,blocksize,threaded)
    @inbounds for j in 1:n
        for i in 1:j-1
            S[i,j]=zero(T)
        end
        S[j,j]=one(T)
    end
    LDLFactor(S,d,dl,p,blocks,hermitian)
end
# Blocked Bunch-Kaufman (LAPACK sytrf/lasyf, lower): within a panel of
# `nb` columns each candidate column is brought up to date by one GEMV with
# the panel's L columns and W=L*D, the same 1x1/2x2 decisions are taken as
# before, interchanges are applied to whole rows of the lower storage, and
# the trailing lower triangle receives A22 -= L21*W21' once per panel
# (GEMMT). The previous unblocked loop updated both triangles element by
# element with strided stores: 370-2500x behind OpenBLAS sytrf.
function _ldl_blocked!(S,d_vec,dl_vec,p,blocks,hermitian,bk,nb,threaded)
    n=size(S,1); T=eltype(S); R=real(T)
    cj(x)=hermitian ? conj(x) : x
    α=(1+sqrt(R(17)))/8
    W=Matrix{T}(undef,n,min(n,nb)+1)
    k=1
    while k<=n
        k0=k
        while k<=n && k-k0<nb
            jw=k-k0+1
            _ldl_column!(W,jw,S,k,k,k0,hermitian,threaded)   # W[k:n,jw] = updated column k
            step=1; kp=k
            if bk && k<n
                imax=k+1; colmax=abs(W[imax,jw])
                @inbounds for i in k+2:n
                    v=abs(W[i,jw]); v>colmax && (colmax=v; imax=i)
                end
                absakk=abs(W[k,jw])
                if !iszero(colmax) && absakk<α*colmax
                    _ldl_column!(W,jw+1,S,imax,k,k0,hermitian,threaded)  # updated column imax
                    rowmax=zero(R)
                    @inbounds for i in k:n
                        i!=imax && (rowmax=max(rowmax,abs(W[i,jw+1])))
                    end
                    if abs(W[imax,jw+1])>=α*rowmax
                        kp=imax
                        @inbounds for i in k:n   # the pivot column becomes column k
                            W[i,jw]=W[i,jw+1]
                        end
                    elseif absakk>=α*colmax*(colmax/rowmax)
                        kp=k
                    else
                        step=2; kp=imax
                    end
                end
            end
            kk=k+step-1
            if kp!=kk
                _ldl_interchange!(S,W,p,kk,kp,k,jw+step-1,hermitian)
            end
            if step==1
                d=W[k,jw]; hermitian && (d=real(d))
                iszero(d) && throw(SingularException(k))
                d_vec[k]=d; W[k,jw]=d
                inv_d=Base.inv(d)
                @inbounds @simd for i in k+1:n
                    S[i,k]=W[i,jw]*inv_d
                end
            else
                a=W[k,jw]; b=W[k+1,jw]; c=W[k+1,jw+1]; bh=cj(b)
                hermitian && (a=real(a); c=real(c))
                scale=max(abs(a),abs(b),abs(c)); iszero(scale) && throw(SingularException(k))
                aa=a/scale; bb=b/scale; cc=c/scale; bbh=bh/scale; det=aa*cc-bb*bbh
                iszero(det) && throw(SingularException(k))
                d_vec[k]=a; dl_vec[k]=b; d_vec[k+1]=c
                W[k,jw]=a; W[k+1,jw+1]=c
                S[k+1,k]=zero(T)   # inside the 2x2 block of D, not part of L
                inv_scale=Base.inv(scale); inv_det=Base.inv(det)
                @inbounds for i in k+2:n
                    x=W[i,jw]*inv_scale; y=W[i,jw+1]*inv_scale
                    S[i,k]=(x*cc-y*bb)*inv_det; S[i,k+1]=(y*aa-x*bbh)*inv_det
                end
            end
            push!(blocks,step); k+=step
        end
        if k<=n
            # A22 -= L21*W21' (W=L*D), lower triangle only.
            L21=view(S,k:n,k0:k-1); W21=view(W,k:n,1:k-k0)
            gemmt!(view(S,k:n,k:n),L21,hermitian ? adjoint(W21) : transpose(W21),-one(T),one(T);uplo=:L,threaded)
        end
    end
    nothing
end
# W[k0':n,jw] (rows >= kfirst) = column `col` of the active matrix brought
# up to date with the panel columns k0:kfirst-1: A[:,col]-L[:,k0:]*cj(W[col,:]).
function _ldl_column!(W,jw,S,col,kfirst,k0,hermitian,threaded)
    n=size(S,1)
    @inbounds for i in kfirst:n
        # Lower storage: row col left of the diagonal is the mirrored column.
        W[i,jw]=i<col ? (hermitian ? conj(S[col,i]) : S[col,i]) : S[i,col]
    end
    hermitian && (@inbounds W[col,jw]=real(W[col,jw]))
    if kfirst>k0
        c=k0
        @inbounds while c+3<=kfirst-1
            z1=-(hermitian ? conj(W[col,c-k0+1]) : W[col,c-k0+1])
            z2=-(hermitian ? conj(W[col,c-k0+2]) : W[col,c-k0+2])
            z3=-(hermitian ? conj(W[col,c-k0+3]) : W[col,c-k0+3])
            z4=-(hermitian ? conj(W[col,c-k0+4]) : W[col,c-k0+4])
            @simd ivdep for i in kfirst:n
                acc=W[i,jw]
                acc=_arithmetic_madd(S[i,c],  z1,acc)
                acc=_arithmetic_madd(S[i,c+1],z2,acc)
                acc=_arithmetic_madd(S[i,c+2],z3,acc)
                acc=_arithmetic_madd(S[i,c+3],z4,acc)
                W[i,jw]=acc
            end
            c+=4
        end
        @inbounds while c<=kfirst-1
            z=-(hermitian ? conj(W[col,c-k0+1]) : W[col,c-k0+1])
            @simd ivdep for i in kfirst:n
                W[i,jw]=_arithmetic_madd(S[i,c],z,W[i,jw])
            end
            c+=1
        end
        hermitian && (@inbounds W[col,jw]=real(W[col,jw]))
    end
    W
end
# Symmetric interchange of a<b in the lower storage (all columns left of a,
# the a..b segment against row b, the diagonal and the tail), in the panel
# buffer W (columns 1:jlast) and in the permutation.
function _ldl_interchange!(S,W,p,a,b,k,jlast,hermitian)
    n=size(S,1); cj(x)=hermitian ? conj(x) : x
    @inbounds begin
        for j in 1:a-1
            S[a,j],S[b,j]=S[b,j],S[a,j]
        end
        S[a,a],S[b,b]=S[b,b],S[a,a]
        for i in a+1:b-1
            S[i,a],S[b,i]=cj(S[b,i]),cj(S[i,a])
        end
        S[b,a]=cj(S[b,a])
        for i in b+1:n
            S[i,a],S[i,b]=S[i,b],S[i,a]
        end
        for j in 1:jlast
            W[a,j],W[b,j]=W[b,j],W[a,j]
        end
    end
    p[a],p[b]=p[b],p[a]
    nothing
end
bunchkaufman_factor(A;kwargs...)=ldlt_factor(A;pivot=:bunchkaufman,kwargs...)
function solve(F::LDLFactor,b::AbstractVecOrMat)
    n=size(F.L,1); Base.require_one_based_indexing(b)
    size(b,1)==n || throw(DimensionMismatch("LDL solve dimensions differ"))
    X=Array{promote_type(eltype(F.L),eltype(b))}(undef,size(b)); copyto!(X,b)
    _ldl_solve!(F,X)
end
# In-place P'L D Lᴴ P solve of the columns of X: one permutation buffer, a
# triangular vector solve for a vector right-hand side.
function _ldl_solve!(F::LDLFactor,X::AbstractVecOrMat)
    n=size(F.L,1); size(X,1)==n || throw(DimensionMismatch("LDL solve dimensions differ"))
    t=similar(X,n)
    for j in axes(X,2)
        c=view(X,:,j)
        for i in 1:n; t[i]=c[F.p[i]]; end
        copyto!(c,t)
    end
    X isa AbstractVector ? trsv!(X,F.L;uplo=:L,unit=true) : trsm!(X,F.L;uplo=:L,unit=true); k=1
    d_vec=getfield(F,:d); dl_vec=getfield(F,:dl); herm=getfield(F,:hermitian)
    for step in F.blocks
        if step==1
            dk=d_vec[k]
            for j in axes(X,2)
                X[k,j]/=dk
            end
        else
            a=d_vec[k]; b=dl_vec[k]; c=d_vec[k+1]; bh=herm ? conj(b) : b
            scale=max(abs(a),abs(b),abs(c)); aa=a/scale; bb=b/scale; cc=c/scale; bbh=bh/scale
            det=aa*cc-bb*bbh
            for j in axes(X,2)
                x=X[k,j]/scale; y=X[k+1,j]/scale
                X[k,j]=(cc*x-bbh*y)/det; X[k+1,j]=(aa*y-bb*x)/det
            end
        end
        k+=step
    end
    trans=F.hermitian ? :C : :T
    X isa AbstractVector ? trsv!(X,F.L;uplo=:L,unit=true,trans) : trsm!(X,F.L;uplo=:L,unit=true,trans)
    for j in axes(X,2)
        c=view(X,:,j)
        for i in 1:n; t[F.p[i]]=c[i]; end
        copyto!(c,t)
    end
    X
end

"""
    eigen_generalized(A::AbstractMatrix{T}, B::AbstractMatrix{T}; uplo=:U, vectors=true, threaded=true, kwargs...) -> EigenFactor

Hermitian-definite generalized eigenproblem A*V=B*V*Diagonal(values).
B must be positive definite. V'BV=I. Uses native Cholesky, TRSM and EVD.
"""
function eigen_generalized(A::AbstractMatrix{T},B::AbstractMatrix{T};uplo=:U,vectors=true,threaded=true,kwargs...) where {T<:GenericFloat}
    threaded=threaded && generic_threadsafe(T)
    n=_square(A); size(B)==(n,n) || throw(DimensionMismatch("generalized eigenproblem dimensions differ"))
    upper=_uplo(uplo)
    AA=Matrix{T}(undef,n,n); BB=similar(AA)
    for j in 1:n, i in 1:n
        AA[i,j]=_symget(A,i,j,upper,true); BB[i,j]=_symget(B,i,j,upper,true)
    end
    C=cholesky_factor(BB;threaded)
    trsm!(AA,C.L;uplo=:L,threaded)
    trsm!(AA,C.L;side=:R,uplo=:L,trans=:C,threaded)
    E=eigen_hermitian(AA;uplo=:L,vectors,kwargs...)
    if vectors
        trsm!(E.vectors,C.L;uplo=:L,trans=:C,threaded)
    end
    E
end

"""
    expert_solve(A::AbstractMatrix{T}, b::AbstractVecOrMat{T}; maxiter=8, rtol=8eps(real(T)), equilibrate=true, pivot=:partial, threaded=true) -> NamedTuple

Equilibrated LU expert solve with componentwise backward error and iterative
refinement. Returns diagnostics and never silently asserts convergence.
"""
function expert_solve(A::AbstractMatrix{T},b::AbstractVecOrMat{T};maxiter=8,rtol=8eps(real(T)),equilibrate=true,pivot=:partial,threaded=true) where {T<:GenericFloat}
    threaded=threaded && generic_threadsafe(T)
    n=_square(A); Base.require_one_based_indexing(b)
    size(b,1)==n && maxiter>=0 && rtol>=0 || throw(ArgumentError("invalid expert solve dimensions/options"))
    R=real(T); rows=ones(R,n); cols=ones(R,n); S=Matrix(A)
    if equilibrate
        for i in 1:n
            v=maximum(abs,view(S,i,:);init=zero(R)); iszero(v) && throw(SingularException(i))
            rows[i]=v
            for j in 1:n
                S[i,j]/=v
            end
        end
        for j in 1:n
            v=maximum(abs,view(S,:,j);init=zero(R)); cols[j]=iszero(v) ? one(R) : v
            for i in 1:n
                S[i,j]/=cols[j]
            end
        end
    end
    F=lu_factor(S;pivot,threaded)
    B=ndims(b)==1 ? reshape(copy(b),n,1) : copy(b)
    for j in axes(B,2), i in 1:n
        B[i,j]/=rows[i]
    end
    X=solve(F,B)
    for j in axes(X,2), i in 1:n
        X[i,j]/=cols[i]
    end
    residual=similar(X); error=R(Inf); iterations=0
    for iteration in 0:maxiter
        error=zero(R)
        for j in axes(X,2), i in 1:n
            s=zero(T); den=abs(b isa AbstractVector ? b[i] : b[i,j])
            for k in 1:n
                s+=A[i,k]*X[k,j]; den+=abs(A[i,k])*abs(X[k,j])
            end
            r=(b isa AbstractVector ? b[i] : b[i,j])-s; residual[i,j]=r
            err=!isfinite(den) ? R(Inf) : iszero(den) ? (iszero(r) ? zero(R) : R(Inf)) : abs(r)/den
            error=max(error,err)
        end
        iterations=iteration
        error<=rtol && break
        iteration==maxiter && break
        for j in axes(residual,2), i in 1:n
            residual[i,j]/=rows[i]
        end
        correction=solve(F,residual)
        for j in axes(X,2), i in 1:n
            X[i,j]+=correction[i,j]/cols[i]
        end
    end
    x=ndims(b)==1 ? vec(X) : X
    (;x,converged=error<=rtol,iterations,backward_error=error,row_scale=rows,column_scale=cols,factor=F)
end
