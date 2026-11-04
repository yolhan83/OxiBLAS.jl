# SPDX-License-Identifier: Apache-2.0
# Base / LinearAlgebra operations on the OxiBLAS sparse matrix formats, routed
# to OxiBLAS kernels (no SparseArrays/LinearAlgebra kernel, no external BLAS).
# Every method has an OxiBLAS sparse type in its signature (no type piracy).
# The sparse solvers (\, lu, cholesky, qr, ldlt, triangular solves) are not
# wired here yet.

# ---- LinearAlgebra.mul! → OxiBLAS.mul! ------------------------------------------
# OxiBLAS.mul! stays its own function; these methods only forward the
# LinearAlgebra calls whose operands include an OxiBLAS sparse type. The
# structured types get their own methods, as for Base.* (mul.jl).
const _MulDest=Union{StridedVecOrMat,SparseMatrixCSC,SparseVector,CSRMatrix,CSCMatrix,SparseTensor}
_LA.mul!(C::_MulDest,A::_OxiSparseOp,B::_MulRight,α::Number,β::Number)=mul!(C,A,B,α,β)
_LA.mul!(C::_MulDest,A::_MulLeft,B::_OxiSparseOp,α::Number,β::Number)=mul!(C,A,B,α,β)
_LA.mul!(C::_MulDest,A::_OxiWrapped,B::Union{_MulDenseMat,StridedVector},α::Number,β::Number)=mul!(C,A,B,α,β)
_LA.mul!(C::_MulDest,A::_MulDenseMat,B::_OxiWrapped,α::Number,β::Number)=mul!(C,A,B,α,β)
_LA.mul!(C::_MulDest,A::_OxiSparseOp,B::Union{_LA.Diagonal,_LABand,_LA.AbstractTriangular,_SymHerm},α::Number,β::Number)=mul!(C,A,B,α,β)
_LA.mul!(C::_MulDest,A::Union{_LA.Diagonal,_LABand,_LA.AbstractTriangular,_SymHerm},B::_OxiSparseOp,α::Number,β::Number)=mul!(C,A,B,α,β)
# Disambiguation for LinearAlgebra versions (1.10) that define
# mul!(C::AbstractMatrix, A::AbstractMatrix, B::<Diagonal/band>, α, β), its
# triangular form and the transposed pairs. A matrix destination with exactly
# their structured unions is more specific than both: same forwarding, chosen
# at compile time, no runtime cost; on the other versions these methods are
# simply more specific forms of the two above.
const _MulDestMat=Union{StridedMatrix,SparseMatrixCSC,CSRMatrix,CSCMatrix,SparseTensor{<:Any,2}}
const _LABandDiag=Union{_LA.Bidiagonal,_LA.Diagonal,_LA.SymTridiagonal,_LA.Tridiagonal}
_LA.mul!(C::_MulDestMat,A::_OxiSparseOp,B::_LABandDiag,α::Number,β::Number)=mul!(C,A,B,α,β)
_LA.mul!(C::_MulDestMat,A::_OxiSparseOp,B::_LA.AbstractTriangular,α::Number,β::Number)=mul!(C,A,B,α,β)
_LA.mul!(C::_MulDestMat,A::_LABandDiag,B::_OxiSparseOp,α::Number,β::Number)=mul!(C,A,B,α,β)
_LA.mul!(C::_MulDestMat,A::_LA.AbstractTriangular,B::_OxiSparseOp,α::Number,β::Number)=mul!(C,A,B,α,β)

# ---- +, - → sparse_add -------------------------------------------------------------
# Same result format rule as the products: CSC when every sparse operand is a
# plain CSCMatrix, SparseMatrixCSC when a SparseArrays operand is involved,
# CSR otherwise. Element types are promoted (format kept); transposed
# operands are transposed by the OxiBLAS kernel first.
_add_operand(T,X)=_mul_operand(T,X isa Union{Transpose,Adjoint} ? _transposed(X) : X isa SparseTensor ? CSRMatrix(X) : X)
_transposed(X::Transpose)=sparse_transpose(parent(X))
_transposed(X::Adjoint)=sparse_transpose(parent(X);conjugate=true)
function _sparse_addsub(A,B,β)
    size(A)==size(B) || throw(DimensionMismatch("dimensions must match: A has dims $(size(A)), B has dims $(size(B))"))
    # Two tensors (any order): merge of the stored patterns (sparse_base_ops.jl).
    A isa SparseTensor && B isa SparseTensor && return broadcast(β==1 ? (+) : (-),A,B)
    T=promote_type(eltype(A),eltype(B))
    R=sparse_add(_add_operand(T,A),_add_operand(T,B),one(T),convert(T,β))
    _mul_sparsearrays(A) || _mul_sparsearrays(B) ? SparseMatrixCSC(R) : _mul_csc(A,B) ? CSCMatrix(R) : R
end
const _AddLeft=Union{_OxiSparseOp,SparseTensor}
const _AddOther=Union{_OxiSparseOp,SparseTensor,_MulSAMat}
Base.:+(A::_AddLeft,B::_AddOther)=_sparse_addsub(A,B,1)
Base.:+(A::_MulSAMat,B::_AddLeft)=_sparse_addsub(A,B,1)
Base.:-(A::_AddLeft,B::_AddOther)=_sparse_addsub(A,B,-1)
Base.:-(A::_MulSAMat,B::_AddLeft)=_sparse_addsub(A,B,-1)
# Negation and division by a scalar keep the format and the pattern.
Base.:-(A::_OxiSparse)=_mul_scaled(A,-1)
Base.:-(A::SparseTensor)=(-1)*A
Base.:/(A::_OxiSparse,α::Number)=_mul_scaled(A,α,/)
Base.:/(A::SparseTensor,α::Number)=SparseTensor(A.dims,copy(A.indices),A.values./α)
Base.conj(A::_OxiSparse)=eltype(A)<:Real ? A : _mul_mapped(conj,A)

# ---- transposes → sparse_transpose ----------------------------------------------
# A CSCMatrix stays CSCMatrix; the other formats give the CSR kernel result.
_keep_csc(A,R)=A isa CSCMatrix ? CSCMatrix(R) : R
Base.copy(A::Transpose{<:Any,<:_OxiSparse})=_keep_csc(parent(A),sparse_transpose(parent(A)))
Base.copy(A::Adjoint{<:Any,<:_OxiSparse})=_keep_csc(parent(A),sparse_transpose(parent(A);conjugate=true))
Base.permutedims(A::_OxiSparse)=_keep_csc(A,sparse_transpose(A))
SparseArrays.permute(A::_OxiSparse,p::AbstractVector{<:Integer},q::AbstractVector{<:Integer})=_keep_csc(A,sparse_permute(A,p,q))

# ---- norms → sparse_norm / nrm2 --------------------------------------------------
# norm(A,p) is the entrywise p-norm (Julia semantics); opnorm(A,1) and
# opnorm(A,Inf) are the column/row sum norms. opnorm(A,2) has no OxiBLAS
# kernel yet and is left to its current path.
function _LA.norm(A::Union{_OxiSparse,SparseTensor},p::Real=2)
    vals=A isa SparseTensor ? (A.values,) : _mul_values(A)
    p==2 && return A isa SparseTensor ? nrm2(A.values) : sparse_norm(A,:fro)
    p==Inf && return maximum(v->maximum(abs,v;init=zero(real(eltype(A)))),vals)
    s=sum(v->sum(x->abs(x)^p,v;init=zero(real(eltype(A)))),vals)
    p==1 ? s : s^Base.inv(p)
end
_LA.opnorm(A::SparseTensor{<:Any,2},p::Real=2)=_LA.opnorm(CSRMatrix(A),p)
_LA.opnorm(A::_OxiSparse,p::Real)=p==1 ? sparse_norm(A,:one) : p==Inf ? sparse_norm(A,:inf) :
    throw(ArgumentError("OxiBLAS: opnorm(A,$p) of a sparse matrix is not available (p = 1 or Inf)"))

# ---- copies keep the format ---------------------------------------------------------
Base.copy(A::Union{CSRMatrix,COOMatrix,ExtendedSparse})=_mul_copy(A)
Base.similar(A::CSRMatrix,::Type{S}) where {S<:GenericFloat}=CSRMatrix(_Shared(),A.m,A.n,copy(A.rowptr),copy(A.colind),similar(A.values,S))
Base.similar(A::CSRMatrix{T}) where T=similar(A,T)
