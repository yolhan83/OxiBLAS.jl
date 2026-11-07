# SPDX-License-Identifier: Apache-2.0
function matrix_norm(A::AbstractMatrix{T},kind=:fro) where {T<:GenericFloat}
    Base.require_one_based_indexing(A); R=real(T)
    kind in (:fro,:F) && return nrm2(A)
    kind in (:one,1) && return maximum((sum(abs,view(A,:,j)) for j in axes(A,2));init=zero(R))
    kind in (:inf,Inf) && return maximum((sum(abs,view(A,i,:)) for i in axes(A,1));init=zero(R))
    kind===:max && return maximum(abs,A;init=zero(R))
    if kind in (:two,2,:nuclear)
        s=singular_values(A)
        return kind===:nuclear ? sum(s) : maximum(s;init=zero(R))
    end
    throw(ArgumentError("unknown matrix norm"))
end
function native_inverse(A::AbstractMatrix{T};threaded=true,kwargs...) where {T<:GenericFloat}
    n=_square(A); F=lu_factor(A;threaded,kwargs...)
    F.info==0 || throw(SingularException(F.info))
    # As LAPACK getri: inv(A)=inv(U)*inv(L)*P. inv(U) in place (TRTRI), then
    # X*L=inv(U) by a right TRSM with the unit lower factor, then the column
    # permutation: 2n^3 flops in all, instead of two full solves with the
    # identity (2.7n^3).
    X=zeros(T,n,n); a=F.factors
    @inbounds for j in 1:n, i in 1:j
        X[i,j]=a[i,j]
    end
    _trtri!(X,true,false,threaded)
    n>1 && trsm!(X,a;side=:R,uplo=:L,unit=true,threaded)
    result=similar(X)
    @inbounds for k in 1:n, i in 1:n
        result[i,F.p[k]]=X[i,k]
    end
    result
end
"""Inverse of a Hermitian positive definite matrix from its Cholesky factor
(LAPACK potri): inv(A)=inv(L)'*inv(L), returned as a full Hermitian matrix."""
function cholesky_inverse(F::CholeskyFactor{T};threaded=true) where {T<:GenericFloat}
    threaded=threaded && generic_threadsafe(T)
    n=size(F.L,1); X=copy(F.L)
    _trtri!(X,false,false,threaded)
    # Up to 64, one HERK of the triangular inverse is faster than the scalar
    # lauum leaves (measured: 1.1-1.25x at n=64; lauum wins from 256 on).
    if n<=64
        C=zeros(T,n,n)
        herk!(C,X,one(real(T)),zero(real(T));uplo=:L,trans=:C,threaded)
        X=C
    else
        _lauum_lower!(X,threaded)
    end
    @inbounds for j in 1:n, i in 1:j-1
        X[i,j]=conj(X[j,i])
    end
    X
end
# X <- X'*X in the lower triangle of a lower triangular X (LAPACK lauum),
# recursive: for X=[X11 0;X21 X22], the lower part of X'X is
# [X11'X11+X21'X21  .; X22'X21  X22'X22]. Each update reads the blocks it
# needs before they are overwritten (n^3/3 flops; a full HERK of the
# triangular factor took n^3 and ran 1.4-2.2x behind OpenBLAS potri).
function _lauum_lower!(X,threaded)
    n=size(X,1); T=eltype(X)
    if n<=_TRTRI_LEAF
        @inbounds for j in 1:n, i in j:n   # (X'X)[i,j]=sum_{k>=i} conj(X[k,i])*X[k,j]: row i uses rows >= i only
            s=zero(T)
            for k in i:n
                s+=conj(X[k,i])*X[k,j]
            end
            X[i,j]=s
        end
        return X
    end
    h=n÷2; lo=1:h; hi=h+1:n
    X11=view(X,lo,lo); X21=view(X,hi,lo); X22=view(X,hi,hi)
    _lauum_lower!(X11,threaded)
    herk!(X11,X21,one(real(T)),one(real(T));uplo=:L,trans=:C,threaded)
    trmm!(X21,X22;side=:L,uplo=:L,trans=:C,threaded)
    _lauum_lower!(X22,threaded)
    X
end
cholesky_inverse(A::AbstractMatrix;kwargs...)=cholesky_inverse(cholesky_factor(A;kwargs...);kwargs...)
# In-place inverse of the upper or lower triangle of X (LAPACK trtri),
# recursive: for lower X=[X11 0;X21 X22], X21 <- -inv(X22)*X21*inv(X11) by
# two TRMMs once both diagonal blocks are inverted (n^3/3 flops). The other
# triangle is neither read nor written. Diagonal zeros raise SingularException.
const _TRTRI_LEAF=16
function _trtri!(X,upper,unit,threaded,offset=1)
    n=size(X,1); T=eltype(X)
    if n<=_TRTRI_LEAF
        @inbounds for j in 1:n
            unit || iszero(X[j,j]) && throw(SingularException(offset+j-1))
        end
        # Column j of the inverse from the already inverted columns.
        @inbounds if upper
            for j in 1:n
                d=unit ? one(T) : (X[j,j]=Base.inv(X[j,j]))
                for i in 1:j-1
                    s=(unit ? X[i,j] : X[i,i]*X[i,j])
                    for k in i+1:j-1
                        s+=X[i,k]*X[k,j]
                    end
                    X[i,j]=-s*d
                end
            end
        else
            for j in n:-1:1
                d=unit ? one(T) : (X[j,j]=Base.inv(X[j,j]))
                for i in n:-1:j+1
                    s=(unit ? X[i,j] : X[i,i]*X[i,j])
                    for k in j+1:i-1
                        s+=X[i,k]*X[k,j]
                    end
                    X[i,j]=-s*d
                end
            end
        end
        return X
    end
    h=n÷2; lo=1:h; hi=h+1:n
    _trtri!(view(X,lo,lo),upper,unit,threaded,offset)
    _trtri!(view(X,hi,hi),upper,unit,threaded,offset+h)
    if upper
        B=view(X,lo,hi)   # X12 <- -inv(X11)*X12*inv(X22)
        trmm!(B,view(X,lo,lo),-one(T);side=:L,uplo=:U,unit,threaded)
        trmm!(B,view(X,hi,hi);side=:R,uplo=:U,unit,threaded)
    else
        B=view(X,hi,lo)   # X21 <- -inv(X22)*X21*inv(X11)
        trmm!(B,view(X,hi,hi),-one(T);side=:L,uplo=:L,unit,threaded)
        trmm!(B,view(X,lo,lo);side=:R,uplo=:L,unit,threaded)
    end
    X
end
function pseudoinverse(A::AbstractMatrix{T};rtol=eps(real(T))*max(size(A)...),atol=zero(real(T)),algorithm=:golub_kahan) where {T<:GenericFloat}
    min(rtol,atol)>=0 || throw(ArgumentError("invalid pseudoinverse tolerance"))
    F=svd_bidiagonal(A;algorithm); cutoff=max(atol,rtol*maximum(F.S;init=zero(real(T))))
    V=Matrix(adjoint(F.Vt))
    for j in axes(V,2)
        scale=F.S[j]>cutoff ? Base.inv(F.S[j]) : zero(real(T))
        for i in axes(V,1)
            V[i,j]*=scale
        end
    end
    P=Matrix{T}(undef,size(A,2),size(A,1)); gemm!(P,V,adjoint(F.U)); P
end
function numerical_rank(A::AbstractMatrix{T};rtol=eps(real(T))*max(size(A)...),atol=zero(real(T))) where {T<:GenericFloat}
    s=singular_values(A); cutoff=max(atol,rtol*maximum(s;init=zero(real(T))))
    count(>(cutoff),s)
end
function null_space(A::AbstractMatrix{T};rtol=eps(real(T))*max(size(A)...),atol=zero(real(T))) where {T<:GenericFloat}
    F=svd_bidiagonal(A;full=true); cutoff=max(atol,rtol*maximum(F.S;init=zero(real(T))))
    rank=count(>(cutoff),F.S); Matrix(adjoint(F.Vt[rank+1:end,:]))
end
left_null_space(A;kwargs...)=null_space(adjoint(A);kwargs...)
function column_space(A::AbstractMatrix{T};rtol=eps(real(T))*max(size(A)...),atol=zero(real(T))) where {T<:GenericFloat}
    F=svd_bidiagonal(A); cutoff=max(atol,rtol*maximum(F.S;init=zero(real(T))))
    F.U[:,findall(>(cutoff),F.S)]
end
row_space(A;kwargs...)=column_space(adjoint(A);kwargs...)
function condition_number(A::AbstractMatrix{T};norm=:two) where {T<:GenericFloat}
    R=real(T)
    if norm in (:two,2)
        s=singular_values(A)
        isempty(s) && return one(R)
        return iszero(s[end]) ? R(Inf) : s[1]/s[end]
    end
    matrix_norm(A,norm)*matrix_norm(native_inverse(A),norm)
end
function determinant(A::AbstractMatrix{T}) where {T<:GenericFloat}
    _square(A); F=lu_factor(A;check=false)
    F.info!=0 && return zero(T)
    value=one(T)
    for i in eachindex(F.pivots)
        value*=F.factors[i,i]
        F.pivots[i]!=i && (value=-value)
    end
    value
end
function logabsdet_factor(A::AbstractMatrix{T}) where {T<:GenericFloat}
    _square(A); F=lu_factor(A;check=false); R=real(T)
    F.info!=0 && return (-R(Inf),zero(T))
    logarithm=zero(R); phase=one(T)
    for i in eachindex(F.pivots)
        value=F.factors[i,i]; magnitude=abs(value)
        logarithm+=log(magnitude); phase*=value/magnitude
        F.pivots[i]!=i && (phase=-phase)
    end
    logarithm,phase
end

function equilibrate(A::AbstractMatrix{T};radix=false) where {T<:GenericFloat}
    Base.require_one_based_indexing(A); R=real(T); m,n=size(A); B=Matrix(A)
    rows=ones(R,m); cols=ones(R,n)
    for i in 1:m
        value=maximum(abs,view(B,i,:);init=zero(R))
        if value>0
            rows[i]=radix ? ldexp(one(R),clamp(round(Int,log2(value)),exponent(floatmin(R)),exponent(floatmax(R)))) : value
            for j in 1:n
                B[i,j]=_realdiv(B[i,j],rows[i])
            end
        end
    end
    for j in 1:n
        value=maximum(abs,view(B,:,j);init=zero(R))
        if value>0
            cols[j]=radix ? ldexp(one(R),clamp(round(Int,log2(value)),exponent(floatmin(R)),exponent(floatmax(R)))) : value
            for i in 1:m
                B[i,j]=_realdiv(B[i,j],cols[j])
            end
        end
    end
    (;matrix=B,row_scale=rows,column_scale=cols)
end
function backward_error(A::AbstractMatrix{T},x::AbstractVecOrMat{T},b::AbstractVecOrMat{T}) where {T<:GenericFloat}
    Base.require_one_based_indexing(A,x,b)
    X=ndims(x)==1 ? reshape(x,:,1) : x; B=ndims(b)==1 ? reshape(b,:,1) : b
    size(A,2)==size(X,1) && size(B)==(size(A,1),size(X,2)) || throw(DimensionMismatch("residual dimensions differ"))
    R=real(T); residual=Matrix{T}(undef,size(B)); error=zero(R)
    for j in axes(B,2), i in axes(B,1)
        value=zero(T); denominator=abs(B[i,j])
        for k in axes(A,2)
            value+=A[i,k]*X[k,j]; denominator+=abs(A[i,k])*abs(X[k,j])
        end
        r=B[i,j]-value; residual[i,j]=r
        error=max(error,!isfinite(denominator) ? R(Inf) : iszero(denominator) ? (iszero(r) ? zero(R) : R(Inf)) : abs(r)/denominator)
    end
    (;error,residual=ndims(b)==1 ? vec(residual) : residual)
end

function kron_product(A::AbstractMatrix{T},B::AbstractMatrix{T}) where {T<:GenericFloat}
    Base.require_one_based_indexing(A,B); m,n=size(A); p,q=size(B)
    mp=Base.checked_mul(m,p); nq=Base.checked_mul(n,q)
    C=Matrix{T}(undef,mp,nq)
    tot=mp*nq
    if tot >= 1_000_000 && Threads.nthreads() > 1 && !(T <: BigFloat)
        Threads.@threads for j in 1:n
            for l in 1:q
                col=(j-1)*q+l
                col_idx=(col-1)*mp+1
                B_col_offset=(l-1)*p
                @inbounds for i in 1:m
                    Aij=A[i,j]
                    @simd for k in 1:p
                        C[col_idx+k-1]=Aij*B[B_col_offset+k]
                    end
                    col_idx+=p
                end
            end
        end
    else
        idx=1
        @inbounds for j in 1:n, l in 1:q
            B_col_offset=(l-1)*p
            for i in 1:m
                Aij=A[i,j]
                @simd for k in 1:p
                    C[idx+k-1]=Aij*B[B_col_offset+k]
                end
                idx+=p
            end
        end
    end
    C
end
function khatri_rao(A::AbstractMatrix{T},B::AbstractMatrix{T}) where {T<:GenericFloat}
    Base.require_one_based_indexing(A,B)
    n=size(A,2); size(B,2)==n || throw(DimensionMismatch("Khatri-Rao column counts differ"))
    m=size(A,1); p=size(B,1)
    mp=Base.checked_mul(m,p)
    C=Matrix{T}(undef,mp,n)
    if mp*n >= 65_536 && Threads.nthreads() > 1 && !(T <: BigFloat)
        Threads.@threads for j in 1:n
            col_idx=(j-1)*mp+1
            B_col_offset=(j-1)*p
            @inbounds for i in 1:m
                Aij=A[i,j]
                @simd for k in 1:p
                    C[col_idx+k-1]=Aij*B[B_col_offset+k]
                end
                col_idx+=p
            end
        end
    else
        idx=1
        @inbounds for j in 1:n
            B_col_offset=(j-1)*p
            for i in 1:m
                Aij=A[i,j]
                @simd for k in 1:p
                    C[idx+k-1]=Aij*B[B_col_offset+k]
                end
                idx+=p
            end
        end
    end
    C
end
kron_sum(A::AbstractMatrix{T},B::AbstractMatrix{T}) where {T<:GenericFloat}=kron_product(A,_identity(T,_square(B)))+kron_product(_identity(T,_square(A)),B)
function kronecker_action(A::AbstractMatrix{T},B::AbstractMatrix{T},x::AbstractVector{T}) where {T<:GenericFloat}
    length(x)==size(A,2)*size(B,2) || throw(DimensionMismatch("Kronecker action dimensions differ"))
    X=reshape(x,size(B,2),size(A,2)); temp=Matrix{T}(undef,size(B,1),size(A,2)); Y=Matrix{T}(undef,size(B,1),size(A,1))
    gemm!(temp,B,X); gemm!(Y,temp,transpose(A)); vec(Y)
end
