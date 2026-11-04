# SPDX-License-Identifier: Apache-2.0
# Half storage is widened once to Float32/ComplexF32 before LAPACK arithmetic.
# The returned factor/eigensystem therefore retains working precision; it is
# never rounded back to half during the iteration. These are native algorithms.
const HalfFloat=Union{Float16,Complex{Float16}}
_half_work(::Type{Float16})=Float32
_half_work(::Type{Complex{Float16}})=ComplexF32
_half_matrix(A)=_half_work(eltype(A)).(A)

# Explicit forwarding methods, rather than eval-generated specializations:
# every public entry and native target is inspectable before execution.
lu_factor(A::AbstractMatrix{T};kwargs...) where {T<:HalfFloat}=lu_factor(_half_matrix(A);kwargs...)
cholesky_factor(A::AbstractMatrix{T};kwargs...) where {T<:HalfFloat}=cholesky_factor(_half_matrix(A);kwargs...)
qr_factor(A::AbstractMatrix{T};kwargs...) where {T<:HalfFloat}=qr_factor(_half_matrix(A);kwargs...)
qr_pivoted(A::AbstractMatrix{T};kwargs...) where {T<:HalfFloat}=qr_pivoted(_half_matrix(A);kwargs...)
ldlt_factor(A::AbstractMatrix{T};kwargs...) where {T<:HalfFloat}=ldlt_factor(_half_matrix(A);kwargs...)
lq_factor(A::AbstractMatrix{T};kwargs...) where {T<:HalfFloat}=lq_factor(_half_matrix(A);kwargs...)
ql_factor(A::AbstractMatrix{T};kwargs...) where {T<:HalfFloat}=ql_factor(_half_matrix(A);kwargs...)
rq_factor(A::AbstractMatrix{T};kwargs...) where {T<:HalfFloat}=rq_factor(_half_matrix(A);kwargs...)
complete_orthogonal(A::AbstractMatrix{T};kwargs...) where {T<:HalfFloat}=complete_orthogonal(_half_matrix(A);kwargs...)
hessenberg_factor(A::AbstractMatrix{T}) where {T<:HalfFloat}=hessenberg_factor(_half_matrix(A))
tridiagonal_factor(A::AbstractMatrix{T};kwargs...) where {T<:HalfFloat}=tridiagonal_factor(_half_matrix(A);kwargs...)
eigen_hermitian(A::AbstractMatrix{T};kwargs...) where {T<:HalfFloat}=eigen_hermitian(_half_matrix(A);kwargs...)
eigen_hermitian_dc(A::AbstractMatrix{T};kwargs...) where {T<:HalfFloat}=eigen_hermitian_dc(_half_matrix(A);kwargs...)
eigen_hermitian_mrrr(A::AbstractMatrix{T};kwargs...) where {T<:HalfFloat}=eigen_hermitian_mrrr(_half_matrix(A);kwargs...)
eigen_general(A::AbstractMatrix{T};kwargs...) where {T<:HalfFloat}=eigen_general(_half_matrix(A);kwargs...)
schur_factor(A::AbstractMatrix{T};kwargs...) where {T<:HalfFloat}=schur_factor(_half_matrix(A);kwargs...)
schur_multishift(A::AbstractMatrix{T};kwargs...) where {T<:HalfFloat}=schur_multishift(_half_matrix(A);kwargs...)
svd_factor(A::AbstractMatrix{T};kwargs...) where {T<:HalfFloat}=svd_factor(_half_matrix(A);kwargs...)
bidiagonal_factor(A::AbstractMatrix{T}) where {T<:HalfFloat}=bidiagonal_factor(_half_matrix(A))
svd_bidiagonal(A::AbstractMatrix{T};kwargs...) where {T<:HalfFloat}=svd_bidiagonal(_half_matrix(A);kwargs...)
svd_selected(A::AbstractMatrix{T};kwargs...) where {T<:HalfFloat}=svd_selected(_half_matrix(A);kwargs...)
svd_truncated(A::AbstractMatrix{T},rank::Integer;kwargs...) where {T<:HalfFloat}=svd_truncated(_half_matrix(A),rank;kwargs...)
svd_randomized(A::AbstractMatrix{T},rank::Integer;kwargs...) where {T<:HalfFloat}=svd_randomized(_half_matrix(A),rank;kwargs...)
qz_factor(A::AbstractMatrix{T},B::AbstractMatrix{T};kwargs...) where {T<:HalfFloat}=qz_factor(_half_matrix(A),_half_matrix(B);kwargs...)
SchurWorkspace(::Type{Float16},n::Integer;kwargs...)=SchurWorkspace(Float32,n;kwargs...)
SchurWorkspace(::Type{Complex{Float16}},n::Integer;kwargs...)=SchurWorkspace(ComplexF32,n;kwargs...)
trevc_right(A::AbstractMatrix{T};kwargs...) where {T<:HalfFloat}=trevc_right(_half_matrix(A);kwargs...)
trevc_left(A::AbstractMatrix{T};kwargs...) where {T<:HalfFloat}=trevc_left(_half_matrix(A);kwargs...)
trsna_s(A::AbstractMatrix{T}) where {T<:HalfFloat}=trsna_s(_half_matrix(A))
trsna_sep(A::AbstractMatrix{T}) where {T<:HalfFloat}=trsna_sep(_half_matrix(A))
generalized_sylvester(A::AbstractMatrix{T},D::AbstractMatrix{T},B::AbstractMatrix{T},E::AbstractMatrix{T},C::AbstractMatrix{T},F::AbstractMatrix{T}) where {T<:HalfFloat}=generalized_sylvester(_half_matrix(A),_half_matrix(D),_half_matrix(B),_half_matrix(E),_half_matrix(C),_half_matrix(F))
eigen_pencil(A::AbstractMatrix{T},B::AbstractMatrix{T};kwargs...) where {T<:HalfFloat}=eigen_pencil(_half_matrix(A),_half_matrix(B);kwargs...)
eigen_generalized(A::AbstractMatrix{T},B::AbstractMatrix{T};kwargs...) where {T<:HalfFloat}=eigen_generalized(_half_matrix(A),_half_matrix(B);kwargs...)
eigen_tridiagonal(d::AbstractVector{Float16},e::AbstractVector{Float16};kwargs...)=eigen_tridiagonal(Float32.(d),Float32.(e);kwargs...)
eigen_tridiagonal_dc(d::AbstractVector{Float16},e::AbstractVector{Float16};kwargs...)=eigen_tridiagonal_dc(Float32.(d),Float32.(e);kwargs...)
eigen_tridiagonal_mrrr(d::AbstractVector{Float16},e::AbstractVector{Float16};kwargs...)=eigen_tridiagonal_mrrr(Float32.(d),Float32.(e);kwargs...)

solve(F::LUFactor{T},b::AbstractVecOrMat{S}) where {T<:NativeFloat,S<:HalfFloat}=solve(F,T.(b))
solve(F::CompleteLUFactor{T},b::AbstractVecOrMat{S}) where {T<:NativeFloat,S<:HalfFloat}=solve(F,T.(b))
solve(F::CholeskyFactor{T},b::AbstractVecOrMat{S}) where {T<:NativeFloat,S<:HalfFloat}=solve(F,T.(b))
solve(F::QRFactor{T},b::AbstractVecOrMat{S}) where {T<:NativeFloat,S<:HalfFloat}=solve(F,T.(b))
solve(F::LDLFactor{T},b::AbstractVecOrMat{S}) where {T<:NativeFloat,S<:HalfFloat}=solve(F,T.(b))
solve(F::PivotedQRFactor{T},b::AbstractVecOrMat{S}) where {T<:NativeFloat,S<:HalfFloat}=solve(F,T.(b))
least_squares(F::QRFactor{T},b::AbstractVecOrMat{S};kwargs...) where {T<:NativeFloat,S<:HalfFloat}=least_squares(F,T.(b);kwargs...)
least_squares(F::PivotedQRFactor{T},b::AbstractVecOrMat{S};kwargs...) where {T<:NativeFloat,S<:HalfFloat}=least_squares(F,T.(b);kwargs...)
function expert_solve(A::AbstractMatrix{T},b::AbstractVecOrMat{T};kwargs...) where {T<:HalfFloat}
    W=_half_work(T); expert_solve(W.(A),W.(b);kwargs...)
end
function refine_solve(A::AbstractMatrix{T},F,b::AbstractVecOrMat{T};kwargs...) where {T<:HalfFloat}
    W=_half_work(T); refine_solve(W.(A),F,W.(b);kwargs...)
end
reciprocal_condition(A::AbstractMatrix{T};kwargs...) where {T<:HalfFloat}=reciprocal_condition(_half_matrix(A);kwargs...)
native_inverse(A::AbstractMatrix{T};kwargs...) where {T<:HalfFloat}=native_inverse(_half_matrix(A);kwargs...)
pseudoinverse(A::AbstractMatrix{T};kwargs...) where {T<:HalfFloat}=pseudoinverse(_half_matrix(A);kwargs...)
condition_number(A::AbstractMatrix{T};kwargs...) where {T<:HalfFloat}=condition_number(_half_matrix(A);kwargs...)
numerical_rank(A::AbstractMatrix{T};kwargs...) where {T<:HalfFloat}=numerical_rank(_half_matrix(A);kwargs...)
null_space(A::AbstractMatrix{T};kwargs...) where {T<:HalfFloat}=null_space(_half_matrix(A);kwargs...)
column_space(A::AbstractMatrix{T};kwargs...) where {T<:HalfFloat}=column_space(_half_matrix(A);kwargs...)
determinant(A::AbstractMatrix{T}) where {T<:HalfFloat}=determinant(_half_matrix(A))
logabsdet_factor(A::AbstractMatrix{T}) where {T<:HalfFloat}=logabsdet_factor(_half_matrix(A))

function _half_storage(P::BandedMatrix{T}) where {T<:HalfFloat}
    Q=_compact_copy(P); W=_half_work(T)
    BandedMatrix(W.(Q.data),Q.m;kl=Q.kl,ku=Q.ku,kind=Q.kind,uplo=Q.upper ? :U : :L,unit=Q.unit)
end
function _half_storage(P::PackedMatrix{T}) where {T<:HalfFloat}
    Q=_compact_copy(P); W=_half_work(T)
    PackedMatrix(W.(Q.data),Q.n;kind=Q.kind,uplo=Q.upper ? :U : :L,unit=Q.unit)
end
band_lu(A::BandedMatrix{T}) where {T<:HalfFloat}=band_lu(_half_storage(A))
band_cholesky(A::BandedMatrix{T}) where {T<:HalfFloat}=band_cholesky(_half_storage(A))
packed_cholesky(A::PackedMatrix{T}) where {T<:HalfFloat}=packed_cholesky(_half_storage(A))
aasen_factor(A::AbstractMatrix{T};kwargs...) where {T<:HalfFloat}=aasen_factor(_half_matrix(A);kwargs...)
solve(F::BandLUFactor{T},b::AbstractVecOrMat{S};kwargs...) where {T<:NativeFloat,S<:HalfFloat}=solve(F,T.(b);kwargs...)
solve(F::BandCholeskyFactor{T},b::AbstractVecOrMat{S}) where {T<:NativeFloat,S<:HalfFloat}=solve(F,T.(b))
solve(F::PackedCholeskyFactor{T},b::AbstractVecOrMat{S}) where {T<:NativeFloat,S<:HalfFloat}=solve(F,T.(b))
solve(F::AasenFactor{T},b::AbstractVecOrMat{S}) where {T<:NativeFloat,S<:HalfFloat}=solve(F,T.(b))
solve(F::CompleteOrthogonalFactor{T},b::AbstractVecOrMat{S}) where {T<:NativeFloat,S<:HalfFloat}=solve(F,T.(b))
solve_transposed(F::LUFactor{T},b::AbstractVecOrMat{S};kwargs...) where {T<:NativeFloat,S<:HalfFloat}=solve_transposed(F,T.(b);kwargs...)
solve_transposed(F::CompleteLUFactor{T},b::AbstractVecOrMat{S};kwargs...) where {T<:NativeFloat,S<:HalfFloat}=solve_transposed(F,T.(b);kwargs...)
function apply_q!(B::AbstractMatrix{S},F::QRFactor{T};kwargs...) where {T<:NativeFloat,S<:HalfFloat}
    X=T.(B); apply_q!(X,F;kwargs...); copyto!(B,X); B
end
function apply_hessenberg_q!(B::AbstractMatrix{S},F::HessenbergFactor{T};kwargs...) where {T<:NativeFloat,S<:HalfFloat}
    X=T.(B); apply_hessenberg_q!(X,F;kwargs...); copyto!(B,X); B
end
matrix_norm(A::AbstractMatrix{T},kind=:fro) where {T<:HalfFloat}=matrix_norm(_half_matrix(A),kind)
equilibrate(A::AbstractMatrix{T};kwargs...) where {T<:HalfFloat}=equilibrate(_half_matrix(A);kwargs...)
backward_error(A::AbstractMatrix{T},x::AbstractVecOrMat{T},b::AbstractVecOrMat{T}) where {T<:HalfFloat}=backward_error(_half_matrix(A),_half_work(T).(x),_half_work(T).(b))
matrix_exp(A::AbstractMatrix{T};kwargs...) where {T<:HalfFloat}=matrix_exp(_half_matrix(A);kwargs...)
matrix_sqrt(A::AbstractMatrix{T};kwargs...) where {T<:HalfFloat}=matrix_sqrt(_half_matrix(A);kwargs...)
matrix_log(A::AbstractMatrix{T};kwargs...) where {T<:HalfFloat}=matrix_log(_half_matrix(A);kwargs...)
matrix_sign(A::AbstractMatrix{T};kwargs...) where {T<:HalfFloat}=matrix_sign(_half_matrix(A);kwargs...)
matrix_sin(A::AbstractMatrix{T};kwargs...) where {T<:HalfFloat}=matrix_sin(_half_matrix(A);kwargs...)
matrix_cos(A::AbstractMatrix{T};kwargs...) where {T<:HalfFloat}=matrix_cos(_half_matrix(A);kwargs...)
matrix_power(A::AbstractMatrix{T},p::Integer;kwargs...) where {T<:HalfFloat}=matrix_power(_half_matrix(A),p;kwargs...)
matrix_power(A::AbstractMatrix{T},p::Number;kwargs...) where {T<:HalfFloat}=matrix_power(_half_matrix(A),p;kwargs...)
frechet_matrix(f,A::AbstractMatrix{T},E::AbstractMatrix{T};kwargs...) where {T<:HalfFloat}=frechet_matrix(f,_half_matrix(A),_half_matrix(E);kwargs...)

function _mixed_extended_correction(F,residual,rows,columns,::Type{T},::Type{Low}) where {T,Low}
    scaled=similar(residual)
    for j in axes(residual,2), i in axes(residual,1)
        scaled[i,j]=_realdiv(residual[i,j],rows[i])
    end
    magnitude=maximum(abs,scaled;init=zero(real(T)))
    iszero(magnitude) && return zeros(T,length(columns),size(residual,2))
    isfinite(magnitude) || throw(OverflowError("scaled refinement RHS overflow"))
    # A binary scale avoids injecting a new rounding error before an
    # ill-conditioned working-precision fallback solve.
    magnitude=ldexp(one(real(T)),exponent(magnitude))
    lowrhs=Low.(_realdiv.(scaled,magnitude)); all(isfinite,lowrhs) || throw(OverflowError("low precision RHS conversion failed"))
    result=solve(F,lowrhs); correction=Matrix{T}(undef,size(result))
    for j in axes(result,2), i in axes(result,1)
        correction[i,j]=_realdiv(T(result[i,j])*magnitude,columns[i])
    end
    correction
end
function _mixed_qr_correction(F,A,residual,rows,::Type{T},::Type{Low}) where {T,Low}
    W=_residual_worktype(T); n=size(A,2)
    gradient=Matrix{T}(undef,n,size(residual,2))
    for col in axes(residual,2), j in 1:n
        value=zero(W)
        for i in axes(A,1)
            value+=conj(W(A[i,j]))*_realdiv(W(residual[i,col]),real(W)(rows[i]))
        end
        gradient[j,col]=T(value)
    end
    magnitude=maximum(abs,gradient;init=zero(real(T)))
    iszero(magnitude) && return gradient
    isfinite(magnitude) || throw(OverflowError("least-squares refinement gradient overflow"))
    magnitude=ldexp(one(real(T)),exponent(magnitude))
    lowrhs=Low.(_realdiv.(gradient,magnitude))
    all(isfinite,lowrhs) || throw(OverflowError("low precision gradient conversion failed"))
    # Convert the high-precision normal residual, not the inconsistent RHS:
    # low-precision Q'*r has a rounding floor proportional to ||r|| even when
    # A'*r is tiny. The existing QR supplies a normal-equation preconditioner;
    # no A'*A matrix is formed or factorized.
    R=view(F.factors,1:n,1:n)
    trsm!(lowrhs,R;uplo=:U,trans=:C,threaded=false)
    trsm!(lowrhs,R;uplo=:U,threaded=false)
    T.(lowrhs).*magnitude
end
function _mixed_factor(A,kind,threaded)
    kind===:lu ? lu_factor(A;threaded) : kind===:cholesky ? cholesky_factor(A;threaded) :
    kind===:symmetric ? ldlt_factor(A;threaded) : qr_factor(A;threaded)
end
function _mixed_normal_error(A,X,B,residual)
    T=eltype(A); R=real(T); W=_residual_worktype(T)
    gradient=Matrix{W}(undef,size(A,2),size(X,2))
    for col in axes(X,2), j in axes(A,2)
        value=zero(W)
        for i in axes(A,1)
            value+=conj(W(A[i,j]))*W(residual[i,col])
        end
        gradient[j,col]=value
    end
    anorm=matrix_norm(A,:one); bound=anorm*(nrm2(B)+anorm*nrm2(X))
    isfinite(bound) || return R(Inf)
    convert(R,nrm2(gradient)/max(convert(real(W),floatmin(R)),convert(real(W),bound)))
end
function _mixed_precision_impl(A::AbstractMatrix{T},b::AbstractVecOrMat{T};maxiter=20,rtol=16eps(real(T)),
                               lowtype=nothing,fallback=true,threaded=true,factorization=:lu) where T
    Base.require_one_based_indexing(A,b); _generic_require(T)
    threaded=threaded && generic_threadsafe(T); m,n=size(A); R=real(T)
    factorization in (:lu,:cholesky,:symmetric,:qr) || throw(ArgumentError("unknown mixed factorization"))
    (factorization===:qr ? m>=n : m==n) || throw(DimensionMismatch("mixed factorization dimensions differ"))
    size(b,1)==m && maxiter>=0 && isfinite(rtol) && rtol>=0 || throw(ArgumentError("invalid mixed solve options"))
    all(isfinite,A) && all(isfinite,b) || throw(ArgumentError("finite mixed input required"))
    Low=lowtype===nothing ? (T<:NativeFloat ? (T<:Complex ? ComplexF32 : Float32) : T) : lowtype
    Low isa Type && Low<:GenericFloat && isconcretetype(Low) && (Low<:Complex)==(T<:Complex) || throw(ArgumentError("low precision type must be a concrete compatible real/complex floating type"))
    # Row/column equilibration changes neither a square LU solution nor its
    # meaning. The other families use one global positive scale to preserve
    # Hermitian structure and the Euclidean least-squares objective.
    E=if factorization===:lu
        equilibrate(A;radix=true)
    else
        scale=maximum(abs,A;init=zero(R)); scale=iszero(scale) ? one(R) : scale
        (;matrix=_realdiv.(A,scale),row_scale=fill(scale,m),column_scale=ones(R,n))
    end
    used_fallback=false
    F=try
        lowmatrix=Low.(E.matrix)
        all(isfinite,lowmatrix) || throw(OverflowError("low precision matrix conversion failed"))
        _mixed_factor(lowmatrix,factorization,threaded)
    catch error
        (error isa SingularException || error isa PosDefException || error isa OverflowError) && fallback || rethrow()
        used_fallback=true; _mixed_factor(E.matrix,factorization,threaded)
    end
    Work=used_fallback ? T : Low; B=ndims(b)==1 ? reshape(b,m,1) : b
    # A QR rank loss is diagnosed by least_squares, rather than construction.
    X=try
        _mixed_extended_correction(F,B,E.row_scale,E.column_scale,T,Work)
    catch error
        error isa SingularException && fallback && !used_fallback || rethrow()
        used_fallback=true; Work=T; F=_mixed_factor(E.matrix,factorization,threaded)
        _mixed_extended_correction(F,B,E.row_scale,E.column_scale,T,Work)
    end
    residual=Matrix{T}(undef,m,size(X,2)); error=R(Inf); iterations=0
    for iteration in 0:maxiter
        error=_refinement_residual!(residual,A,X,B); iterations=iteration
        factorization===:qr && (error=_mixed_normal_error(A,X,B,residual))
        (error<=rtol || iteration==maxiter || !isfinite(error)) && break
        correction=factorization===:qr ? _mixed_qr_correction(F,E.matrix,residual,E.row_scale,T,Work) :
                   _mixed_extended_correction(F,residual,E.row_scale,E.column_scale,T,Work)
        X .+= correction
    end
    (;x=ndims(b)==1 ? vec(X) : X,converged=error<=rtol,iterations,residual=nrm2(residual),
      backward_error=error,metric=factorization===:qr ? :normalized_normal_residual : :componentwise_backward_error,
      low_type=Low,fallback=used_fallback,factorization,
      finite=all(isfinite,X) && isfinite(error),
      reason=!all(isfinite,X) || !isfinite(error) ? :nonfinite_iteration : used_fallback ? :working_factor_fallback : error<=rtol ? :converged : :iteration_limit)
end
"""Mixed LU/Cholesky/LDL/QR with an explicitly selected low type and generic
residuals and updates. Generic inputs preserve their type by default. Reports convergence, metric, low_type and fallback provenance.
QR reports a normalized normal residual, not an original-system error bound.
"""
mixed_solve(A::AbstractMatrix{T},b::AbstractVecOrMat{T};kwargs...) where {T<:GenericFloat}=_mixed_precision_impl(A,b;kwargs...)

function mixed_solve(A::AbstractMatrix{T},b::AbstractVecOrMat{T};lowtype=nothing,kwargs...) where {T<:HalfFloat}
    W=_half_work(T); Low=lowtype===nothing ? W : lowtype
    _mixed_precision_impl(W.(A),W.(b);lowtype=Low,kwargs...)
end

"""Generic matrix exponential using a scaled Taylor series in T. The order is
chosen by convergence in the scalar provider's active precision, not a fixed
Float64 Padé truncation. Requires +,-,*,/,abs,eps,log2,ceil,ldexp and comparison.
Hardware types retain their more specific Padé implementation.
"""
function matrix_exp(A::AbstractMatrix{T};threaded=true,maxiter::Integer=10000) where {T<:GenericFloat}
    _generic_require(T); n=_square(A); _finite_scale(A); maxiter>=1 || throw(ArgumentError("positive matrix exponential iteration budget required"))
    threaded=threaded && generic_threadsafe(T); R=real(T); norm=matrix_norm(A,:one)
    isfinite(norm) || throw(OverflowError("matrix norm not representable in the working type"))
    result=_identity(T,n); n==0 && return result
    scaling=norm<=one(R)/2 ? 0 : ceil(Int,log2(norm/(one(R)/2)))
    B=Matrix(A).*ldexp(one(R),-scaling); term=copy(result); converged=false
    for k in 1:maxiter
        term=_mm(term,B;threaded)
        divisor=convert(T,k)
        for i in eachindex(term)
            term[i]/=divisor
        end
        next=result+term
        if next==result
            converged=true; break
        end
        result=next
    end
    converged || throw(ConvergenceError(:generic_matrix_exp,Int(maxiter),1:n))
    for _ in 1:scaling
        result=_mm(result,result;threaded)
    end
    result
end
