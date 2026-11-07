# SPDX-License-Identifier: Apache-2.0
function solve_transposed(F::LUFactor{T},b::AbstractVecOrMat{T};trans=:C) where T
    n=_square(F.factors); Base.require_one_based_indexing(b); tr=_trans_kind(trans)
    tr==0 && return solve(F,b)
    size(b,1)==n || throw(DimensionMismatch("transpose solve dimensions differ")); F.info==0 || throw(SingularException(F.info))
    if ndims(b)==1
        # A vector right-hand side uses TRSV (a one-column TRSM recursed into
        # tiny GEMM calls).
        x=copy(b); trsv!(x,F.factors;uplo=:U,trans); trsv!(x,F.factors;uplo=:L,unit=true,trans)
        result=similar(x)
        for i in 1:n
            result[F.p[i]]=x[i]
        end
        return result
    end
    X=copy(b)
    trsm!(X,F.factors;uplo=:U,trans); trsm!(X,F.factors;uplo=:L,unit=true,trans)
    result=similar(X)
    for j in axes(X,2), i in 1:n
        result[F.p[i],j]=X[i,j]
    end
    result
end
function solve_transposed(F::CompleteLUFactor{T},b::AbstractVecOrMat{T};trans=:C) where T
    n=_square(F.factors); Base.require_one_based_indexing(b); tr=_trans_kind(trans)
    tr==0 && return solve(F,b)
    size(b,1)==n || throw(DimensionMismatch("transpose solve dimensions differ")); F.info==0 || throw(SingularException(F.info))
    X=ndims(b)==1 ? reshape(b[F.q],n,1) : b[F.q,:]
    trsm!(X,F.factors;uplo=:U,trans); trsm!(X,F.factors;uplo=:L,unit=true,trans)
    result=similar(X)
    for j in axes(X,2), i in 1:n
        result[F.p[i],j]=X[i,j]
    end
    ndims(b)==1 ? vec(result) : result
end
_adjoint_solve(F::Union{LUFactor,CompleteLUFactor},b)=solve_transposed(F,b)
_adjoint_solve(F::CholeskyFactor,b)=solve(F,b)
_adjoint_solve(F::LDLFactor,b)=F.hermitian ? solve(F,b) : conj.(solve(F,conj.(b)))
"""
    inverse_norm_estimate(F, ::Type{T}, n::Integer; maxiter=8) -> real(T)

Hager–Higham estimate of ||A^-1||_1 using native normal/adjoint solves.
This is an estimate, not a certified upper bound.
"""
function inverse_norm_estimate(F,::Type{T},n::Integer;maxiter=8) where {T<:GenericFloat}
    n>=0 && maxiter>=1 || throw(ArgumentError("invalid inverse norm estimator options")); R=real(T)
    n==0 && return zero(R)
    x=fill(T(Base.inv(R(n))),n); estimate=zero(R); previous=0
    for _ in 1:maxiter
        y=solve(F,x); next=sum(abs,y); estimate=max(estimate,next)
        signs=T[iszero(value) ? one(T) : value/abs(value) for value in y]; z=_adjoint_solve(F,signs)
        index=argmax(abs.(z)); abs(z[index])<=real(dotc(z,x)) && break
        index==previous && break
        fill!(x,zero(T)); x[index]=one(T); previous=index
    end
    for i in 1:n
        x[i]=T((isodd(i) ? one(R) : -one(R))*(one(R)+R(i-1)/max(1,n-1)))
    end
    max(estimate,2sum(abs,solve(F,x))/(3n))
end
function reciprocal_condition(A::AbstractMatrix{T};maxiter=8) where {T<:GenericFloat}
    n=_square(A); R=real(T); n==0 && return one(R)
    F=lu_factor(A;check=false); F.info!=0 && return zero(R)
    value=matrix_norm(A,:one)*inverse_norm_estimate(F,T,n;maxiter)
    iszero(value) ? one(R) : min(one(R),Base.inv(value))
end
function _refinement_residual!(residual,A,X,B)
    T=eltype(A); R=real(T)
    W=_residual_worktype(T)
    error=zero(R)
    for j in axes(X,2), i in axes(A,1)
        value=convert(W,B[i,j]); denominator=abs(B[i,j])
        for k in axes(A,2)
            value-=convert(W,A[i,k])*convert(W,X[k,j]); denominator+=abs(A[i,k])*abs(X[k,j])
        end
        residual[i,j]=convert(T,value)
        relative=!isfinite(denominator) ? R(Inf) : iszero(denominator) ? (iszero(residual[i,j]) ? zero(R) : R(Inf)) : abs(residual[i,j])/denominator
        error=max(error,relative)
    end
    error
end
"""
    refine_solve(A::AbstractMatrix{T}, F, b::AbstractVecOrMat{T}; maxiter=8, rtol=8eps(real(T))) -> NamedTuple

Iterative refinement in the generic work type; Float32 residuals widen to Float64.
Returns componentwise backward error, true residual norm and estimated forward
error/reciprocal condition. These condition estimates are explicitly non-certified.
"""
function refine_solve(A::AbstractMatrix{T},F,b::AbstractVecOrMat{T};maxiter=8,rtol=8eps(real(T))) where {T<:GenericFloat}
    n=_square(A); Base.require_one_based_indexing(b)
    size(b,1)==n && maxiter>=0 && rtol>=0 || throw(ArgumentError("invalid refinement dimensions/options"))
    B=ndims(b)==1 ? reshape(b,n,1) : b; X=solve(F,B); residual=similar(X); error=real(T)(Inf); iterations=0
    for iteration in 0:maxiter
        error=_refinement_residual!(residual,A,X,B); iterations=iteration
        (error<=rtol || iteration==maxiter) && break
        correction=solve(F,residual); X .+= correction
    end
    inverse_norm=inverse_norm_estimate(F,T,n); anorm=matrix_norm(A,:one); condition=anorm*inverse_norm
    rcond=iszero(condition) ? one(real(T)) : min(one(real(T)),Base.inv(condition))
    residual_norm=nrm2(residual); xnorm=matrix_norm(X,:one); delta=iszero(xnorm) ? (iszero(residual_norm) ? zero(real(T)) : real(T)(Inf)) : inverse_norm*matrix_norm(residual,:one)/xnorm
    forward_error_estimate=delta<1 ? delta/(1-delta) : real(T)(Inf)
    x=ndims(b)==1 ? vec(X) : X
    (;x,converged=error<=rtol,iterations,backward_error=error,residual_norm,rcond_estimate=rcond,forward_error_estimate)
end
function _expert_structured_matrix(A,uplo,hermitian)
    n=_square(A); upper=_uplo(uplo); S=Matrix{eltype(A)}(undef,n,n)
    for j in 1:n, i in 1:n
        S[i,j]=_symget(A,i,j,upper,hermitian)
    end
    S
end
function expert_cholesky(A::AbstractMatrix,b::AbstractVecOrMat;uplo=:L,maxiter=8,rtol=8eps(real(eltype(A))),kwargs...)
    S=_expert_structured_matrix(A,uplo,true)
    refine_solve(S,cholesky_factor(S;kwargs...),b;maxiter,rtol)
end
function expert_symmetric(A::AbstractMatrix,b::AbstractVecOrMat;uplo=:L,hermitian=true,maxiter=8,rtol=8eps(real(eltype(A))),kwargs...)
    S=_expert_structured_matrix(A,uplo,hermitian)
    refine_solve(S,ldlt_factor(S;uplo=:L,hermitian,kwargs...),b;maxiter,rtol)
end
