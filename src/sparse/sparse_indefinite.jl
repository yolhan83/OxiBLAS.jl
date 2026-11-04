# SPDX-License-Identifier: Apache-2.0
struct SparseLDLFactor{T,R}
    L::CSCMatrix{T,Int}
    D::CSRMatrix{T,Int}
    diagonal::Vector{T}
    offdiagonal::Vector{T}
    p::Vector{Int}
    blocks::Vector{Int}
    hermitian::Bool
    growth_factor::R
end
Base.size(F::SparseLDLFactor)=size(F.L)
function _ldl_block_scaled(a,b,c,hermitian,k)
    scale=max(abs(a),abs(b),abs(c))
    isfinite(scale) && scale>0 || throw(SingularException(k))
    aa,bb,cc=a/scale,b/scale,c/scale; bh=_ldlconj(bb,hermitian)
    determinant=_native_madd(aa,cc,-bb*bh)
    isfinite(determinant) && !iszero(determinant) || throw(SingularException(k))
    scale,aa,bb,cc,bh,determinant
end
"""Sparse symmetric/Hermitian LDL with Bunch–Kaufman 1x1/2x2 pivots, by
supernodal multifrontal elimination (src/sparse/supernodal_ldlt.jl). Only uplo is
read. Factor identity: A[p,p]=L*D*(hermitian ? L' : transpose(L)); p is the
elimination order (the ordering composed with a postorder and the pivots).
pivot=:none is also supported. Exact zero pivots are errors, not perturbations.
"""
function sparse_ldlt(A::NativeSparse;uplo=:L,hermitian=true,pivot=:bunchkaufman,ordering=:natural)
    _square(A); _uplo(uplo)
    pivot in (:none,:bunchkaufman) || throw(ArgumentError("unknown sparse LDL pivot mode"))
    _mf_ldlt(A,uplo,hermitian,pivot,ordering,typemax(Int))
end
# Unit lower solves with L by columns (rows sorted, the stored unit diagonal
# skipped): forward axpys, backward dot products (conjugated for a Hermitian
# factor); a column whose rows form one range is addressed directly.
function _ldl_forward!(x,L::CSCMatrix)
    colptr,rowind,values=L.colptr,L.rowind,L.values
    @inbounds for j in 1:size(L,2)
        lo,hi=colptr[j],colptr[j+1]-1
        lo<=hi && rowind[lo]==j && (lo+=1)
        lo<=hi || continue
        xj=x[j]; iszero(xj) && continue
        if rowind[hi]-rowind[lo]==hi-lo
            d=rowind[lo]-lo
            @simd for q in lo:hi
                x[q+d]=muladd(-values[q],xj,x[q+d])
            end
        else
            for q in lo:hi
                x[rowind[q]]=muladd(-values[q],xj,x[rowind[q]])
            end
        end
    end
    x
end
function _ldl_backward!(x,L::CSCMatrix,hermitian)
    colptr,rowind,values=L.colptr,L.rowind,L.values
    @inbounds for j in size(L,2):-1:1
        lo,hi=colptr[j],colptr[j+1]-1
        lo<=hi && rowind[lo]==j && (lo+=1)
        lo<=hi || continue
        acc=if !hermitian || eltype(values)<:Real
            _lu_rowdot(values,rowind,x,lo,hi)
        elseif rowind[hi]-rowind[lo]==hi-lo
            _cdot(values,lo-1,x,rowind[lo]-1,hi-lo+1)
        else
            _cdot_idx(values,x,rowind,lo,hi)
        end
        x[j]-=acc
    end
    x
end
function solve(F::SparseLDLFactor{T},b::AbstractVecOrMat{T};trans=:N,threaded=true) where T
    n=length(F.p); Base.require_one_based_indexing(b); size(b,1)==n || throw(DimensionMismatch("sparse LDL RHS differs"))
    tr=_trans_kind(trans); conjugated=T<:Complex && (F.hermitian ? tr==1 : tr==2)
    X=ndims(b)==1 ? reshape(copy(b[F.p]),n,1) : copy(b[F.p,:]); conjugated && (X .= conj.(X))
    workers=_operation_workers(size(X,2),Float64(length(F.L.values))*size(X,2),threaded,262144)
    _foreach_range(size(X,2),workers) do columns,_
        for rhs in columns
            xv=view(X,:,rhs)
            _ldl_forward!(xv,F.L)
            k=1
            for step in F.blocks
                if step==1
                    X[k,rhs]/=F.diagonal[k]
                else
                    scale,a,bb,c,bh,det=_ldl_block_scaled(F.diagonal[k],F.offdiagonal[k],F.diagonal[k+1],F.hermitian,k)
                    x,y=X[k,rhs]/scale,X[k+1,rhs]/scale
                    X[k,rhs]=(c*x-bh*y)/det; X[k+1,rhs]=(a*y-bb*x)/det
                end
                k+=step
            end
            _ldl_backward!(xv,F.L,F.hermitian)
        end
    end
    result=similar(X)
    for rhs in axes(X,2), i in 1:n
        result[F.p[i],rhs]=conjugated ? conj(X[i,rhs]) : X[i,rhs]
    end
    ndims(b)==1 ? vec(result) : result
end
"""Inertia from real/Hermitian pivot blocks in working precision. tol applies
to D-block eigenvalues, not to eigenvalues of A or a certified spectral gap.
"""
function ldl_inertia(F::SparseLDLFactor{T};tol=zero(real(T))) where T
    (F.hermitian || T<:Real) && isfinite(tol) && tol>=0 || throw(ArgumentError("real/Hermitian inertia and nonnegative tolerance required"))
    # The counters are a value threaded through a pure classifier (the former
    # closure incremented three captured counters: three Core.Box).
    counts=(0,0,0); k=1
    classify(cnt,value,threshold)=value>threshold ? (cnt[1]+1,cnt[2],cnt[3]) :
                                  value < -threshold ? (cnt[1],cnt[2]+1,cnt[3]) : (cnt[1],cnt[2],cnt[3]+1)
    for step in F.blocks
        if step==1
            counts=classify(counts,real(F.diagonal[k]),tol)
        else
            scale,a,b,c,_,det=_ldl_block_scaled(F.diagonal[k],F.offdiagonal[k],F.diagonal[k+1],true,k)
            mid=real(a)/2+real(c)/2; radius=hypot(real(a)/2-real(c)/2,abs(b))
            big=mid+copysign(radius,mid); small=real(det)/big
            counts=classify(classify(counts,big,tol/scale),small,tol/scale)
        end
        k+=step
    end
    positive,negative,zeros_count=counts
    (;positive,negative,zero=zeros_count)
end

struct SparseStaticLUFactor{T,R}
    L::CSRMatrix{T,Int}
    U::CSRMatrix{T,Int}
    p::Vector{Int}
    q::Vector{Int}
    epsilon::R
    threshold::R
    perturbations::Vector{T}
    growth_factor::R
end
Base.size(F::SparseStaticLUFactor)=size(F.L)
"""Fixed-order sparse LU with phase-preserving diagonal perturbations.
A[p,q]+Diagonal(perturbations)=L*U, with p=q from a symmetric prescribed
ordering. A perturbed factor solves the modified system, not exactly A.
relative=true scales epsilon by the largest input magnitude.
"""
function sparse_lu_static(A::NativeSparse;epsilon=sqrt(eps(real(eltype(A)))),relative=false,ordering=:natural)
    n=_square(A); T=eltype(A); R=real(T); epsilon=R(epsilon)
    isfinite(epsilon) && epsilon>0 || throw(ArgumentError("positive finite static pivot threshold required"))
    q=sparse_ordering(A,ordering); C=sparse_permute(A,q); all(isfinite,C.values) || throw(ArgumentError("finite static LU entries required"))
    scale=maximum(abs,C.values;init=zero(R)); isfinite(scale) || throw(ArgumentError("input magnitude overflow"))
    threshold=relative && !iszero(scale) ? epsilon*scale : epsilon
    isfinite(threshold) && threshold>0 || throw(ArgumentError("static threshold not representable"))
    rows=[Dict{Int,T}() for _ in 1:n]; incidence=[Set{Int}() for _ in 1:n]
    for i in 1:n, e in C.rowptr[i]:C.rowptr[i+1]-1
        j=C.colind[e]; iszero(C.values[e]) && continue
        rows[i][j]=C.values[e]; push!(incidence[j],i)
    end
    delta=zeros(T,n); growth=scale
    for j in 1:n
        diagonal=get(rows[j],j,zero(T))
        if abs(diagonal)<threshold
            replacement=iszero(diagonal) ? T(threshold) : (diagonal/abs(diagonal))*threshold
            delta[j]=replacement-diagonal; rows[j][j]=replacement; diagonal=replacement
        end
        upper=sort!([col for col in keys(rows[j]) if col>j]); growth=max(growth,abs(diagonal))
        for i in sort!([row for row in incidence[j] if row>j])
            multiplier=rows[i][j]/diagonal; isfinite(multiplier) || throw(ArgumentError("static LU multiplier overflow"))
            rows[i][j]=multiplier
            for col in upper
                value=_native_madd(-multiplier,rows[j][col],get(rows[i],col,zero(T)))
                isfinite(value) && isfinite(abs(value)) || throw(ArgumentError("static LU update overflow"))
                growth=max(growth,abs(value))
                if iszero(value)
                    delete!(rows[i],col); delete!(incidence[col],i)
                else
                    rows[i][col]=value; push!(incidence[col],i)
                end
            end
        end
        empty!(incidence[j])
    end
    lower=[Dict{Int,T}(i=>one(T)) for i in 1:n]; upper=[Dict{Int,T}() for _ in 1:n]
    for i in 1:n, (j,value) in rows[i]
        j<i ? (lower[i][j]=value) : (upper[i][j]=value)
    end
    SparseStaticLUFactor(_rows_to_csr(lower,T,n),_rows_to_csr(upper,T,n),copy(q),q,epsilon,threshold,delta,
                         iszero(scale) ? (iszero(growth) ? one(R) : R(Inf)) : growth/scale)
end
function _sparse_lu_solve(L,U,p,q,b;trans=:N)
    n=length(p); Base.require_one_based_indexing(b); size(b,1)==n || throw(DimensionMismatch("sparse LU RHS differs"))
    tr=_trans_kind(trans); input=tr==0 ? p : q; output=tr==0 ? q : p
    B=ndims(b)==1 ? copy(b[input]) : copy(b[input,:])
    X=tr==0 ? sparse_triangular_solve(U,sparse_triangular_solve(L,B;uplo=:L,unit=true);uplo=:U) :
              sparse_triangular_solve(L,sparse_triangular_solve(U,B;uplo=:U,trans);uplo=:L,unit=true,trans)
    result=similar(X)
    if ndims(b)==1
        for i in 1:n
            result[output[i]]=X[i]
        end
    else
        for rhs in axes(X,2), i in 1:n
            result[output[i],rhs]=X[i,rhs]
        end
    end
    result
end
solve(F::SparseStaticLUFactor{T},b::AbstractVecOrMat{T};kwargs...) where T=_sparse_lu_solve(F.L,F.U,F.p,F.q,b;kwargs...)
function sparse_direct_diagnostics(A::NativeSparse,F,b::AbstractVecOrMat{T},x=solve(F,b)) where T
    C=CSRMatrix(A); n=_square(C); Base.require_one_based_indexing(b,x)
    size(b)==size(x) && size(b,1)==n && eltype(C)===T && eltype(x)===T || throw(DimensionMismatch("direct diagnostic dimensions/types differ"))
    B=ndims(b)==1 ? reshape(b,n,1) : b; X=ndims(x)==1 ? reshape(x,n,1) : x
    R=real(T); residual=zeros(T,size(X)); errors=zeros(R,size(X,2))
    for rhs in axes(X,2), i in 1:n
        value=B[i,rhs]; denominator=abs(value)
        for e in C.rowptr[i]:C.rowptr[i+1]-1
            j=C.colind[e]; value=_native_madd(-C.values[e],X[j,rhs],value)
            denominator+=abs(C.values[e])*abs(X[j,rhs])
        end
        residual[i,rhs]=value
        ratio=!isfinite(denominator) ? R(Inf) : iszero(denominator) ? (iszero(value) ? zero(R) : R(Inf)) : abs(value)/denominator
        errors[rhs]=max(errors[rhs],ratio)
    end
    (;residual_norms=[nrm2(view(residual,:,j)) for j in axes(X,2)],componentwise_backward_error=errors,
      finite=all(isfinite,residual) && all(isfinite,errors),
      perturbed=F isa SparseStaticLUFactor && any(!iszero,F.perturbations))
end
