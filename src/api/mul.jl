# SPDX-License-Identifier: Apache-2.0
# OxiBLAS's own multiplication layer: OxiBLAS.mul! and OxiBLAS.mul. These are
# new functions of this module, not exported, and unrelated to
# LinearAlgebra.mul! and Base.* (neither is extended, imported or aliased).

"""
    OxiBLAS.mul!(C, A, B, α=true, β=false)

`C ← α·A·B + β·C` with OxiBLAS kernels, for dense matrices and vectors (and
their transposes/adjoints and views), LinearAlgebra structured matrices
(Diagonal, triangular, Symmetric/Hermitian, Bidiagonal/Tridiagonal/
SymTridiagonal), SparseArrays matrices and vectors, OxiBLAS sparse formats and
2-D SparseTensors. `C` may be dense, Diagonal or sparse (SparseMatrixCSC,
SparseVector, CSRMatrix, CSCMatrix, SparseTensor); a sparse destination
receives the exact pattern of the result. `β=0` never reads `C`. Operands
whose element type differs from `C`'s are converted, keeping their structure
and storage format. Triangular/Symmetric/Hermitian wrappers of a sparse matrix
are handled as the sparse matrix they denote (never densified).

Aliasing: an operand whose storage overlaps `C` (the same array, overlapping
views, a Diagonal or structured wrapper of such data) is copied first, so the
result is always that of non-overlapping operands.

A sparse destination is filled from a temporary product: sparse when both
operands are sparse, dense when one of them is dense (that product is dense
in general anyway).
"""
function mul! end
"""
    OxiBLAS.mul(A, B)

Allocating product. The result keeps structure when it is cheap and exact:
sparse × sparse is sparse (SparseMatrixCSC/SparseVector when a SparseArrays
operand is involved, CSRMatrix otherwise), Diagonal × Diagonal is Diagonal,
Diagonal scaling of a sparse matrix stays sparse, a product of two upper
(lower) triangular matrices is UpperTriangular (LowerTriangular); anything
else is a dense Array of the promoted element type.
"""
function mul end

# Module binding only, for the structured wrapper types; no LinearAlgebra
# function is imported or extended (in particular not LinearAlgebra.mul!).
import LinearAlgebra as _LA
const _Tri=Union{_LA.UpperTriangular,_LA.LowerTriangular,_LA.UnitUpperTriangular,_LA.UnitLowerTriangular}
const _SymHerm=Union{_LA.Symmetric,_LA.Hermitian}
const _LABand=Union{_LA.Bidiagonal,_LA.Tridiagonal,_LA.SymTridiagonal}
const _SparseMat=Union{SparseMatrixCSC,_OxiSparse,SparseTensor{<:Any,2}}
const _SparseMatOp=Union{_SparseMat,Transpose{<:Any,<:Union{_SparseMat,SparseVector}},Adjoint{<:Any,<:Union{_SparseMat,SparseVector}}}
const _SparseVecOp=Union{SparseVector,SparseTensor{<:Any,1}}
const _MulOperand=Union{AbstractVector,AbstractMatrix}

# Operand kinds; the branches on them below are resolved at compile time.
_mulkind(::_SparseMatOp)=:sparse
_mulkind(::_SparseVecOp)=:sparsevec
_mulkind(::_LA.Diagonal)=:diagonal
_mulkind(::_Tri)=:triangular
_mulkind(::_SymHerm)=:symmetric
_mulkind(::_LABand)=:band
_mulkind(::BandedMatrix)=:banded_compact
_mulkind(::PackedMatrix)=:packed_compact
_mulkind(::AbstractVector)=:vector
_mulkind(::AbstractMatrix)=:dense
_mulsparse(X)=_mulkind(X) in (:sparse,:sparsevec)

mul!(C,A,B)=mul!(C,A,B,true,false)
function mul!(C::_MulOperand,A::_MulOperand,B::_MulOperand,α::Number,β::Number)
    _mul_check(C,A,B)
    T=eltype(C); T<:GenericFloat || throw(ArgumentError("unsupported OxiBLAS.mul! element type $T"))
    a,b=convert(T,α),convert(T,β)
    AA,BB=_mul_detach(C,_mul_operand(T,_mul_unwrap(A))),_mul_detach(C,_mul_operand(T,_mul_unwrap(B)))
    isempty(C) && return C
    if iszero(a) || size(AA,2)==0
        return _mul_scale!(C,b)
    end
    _mul_dispatch!(C,AA,BB,a,b)
    C
end
function _mul_check(C,A,B)
    ndims(A)<=2 && ndims(B)<=2 && ndims(C)<=2 || throw(ArgumentError("OxiBLAS.mul! takes vectors and matrices"))
    size(A,2)==size(B,1) || throw(DimensionMismatch("OxiBLAS.mul!: A has $(size(A,2)) columns, B has $(size(B,1)) rows"))
    (size(C,1)==size(A,1) && size(C,2)==size(B,2) && (ndims(B)==2 || ndims(C)==1)) ||
        throw(DimensionMismatch("OxiBLAS.mul!: output $(size(C)) differs from $(size(A,1))×$(size(B,2))"))
    Base.require_one_based_indexing(C,A,B)
    nothing
end
function _mul_scale!(C,β)
    if C isa SparseDestination
        return _sparse_axpby!(C,SparseTensor(size(C),Int[],eltype(C)[]),zero(eltype(C)),β)
    elseif C isa _LA.Diagonal
        _scale!(C.diag,β); return C
    end
    _scale!(C,β)
end

# Triangular/Symmetric/Hermitian views of sparse data become the sparse
# matrix they denote: the stored triangle is selected (and mirrored, conjugated
# for Hermitian) entry by entry, so the structure stays sparse.
_mul_unwrap(X)=X
_mul_unwrap(X::Union{_Tri,_SymHerm})=_mulsparse(parent(X)) ? _structured_sparse(X) : X
function _structured_sparse(X)
    P=_logical_csc(parent(X)); T=eltype(X); n=size(X,1)
    I=Int[]; J=Int[]; V=T[]
    tri=X isa _Tri; herm=X isa _LA.Hermitian
    upper=tri ? _tri_flags(X)[1] : X.uplo=='U'; unit=tri && _tri_flags(X)[2]
    for j in 1:P.n, p in P.colptr[j]:P.colptr[j+1]-1
        i=Int(P.rowind[p]); v=P.values[p]
        (upper ? i<=j : i>=j) || continue
        if i==j
            unit && continue
            push!(I,i); push!(J,j); push!(V,herm ? T(real(v)) : v)
        else
            push!(I,i); push!(J,j); push!(V,v)
            tri || (push!(I,j); push!(J,i); push!(V,herm ? conj(v) : v))
        end
    end
    if unit
        append!(I,1:n); append!(J,1:n); append!(V,fill(one(T),n))
    end
    SparseArrays.sparse(I,J,V,size(X,1),size(X,2))
end

# Element type conversion that keeps the operand's structure and storage
# format (never dense for sparse or structured operands).
_mul_operand(::Type{T},X) where T=eltype(X)===T ? X : _mul_convert(T,X)
_mul_convert(::Type{T},X::SparseMatrixCSC) where T=SparseMatrixCSC{T}(X)
_mul_convert(::Type{T},X::SparseVector) where T=SparseVector{T}(X)
_mul_convert(::Type{T},X::CSRMatrix) where T=CSRMatrix(_Shared(),X.m,X.n,copy(X.rowptr),copy(X.colind),T.(X.values))
_mul_convert(::Type{T},X::CSCMatrix) where T=CSCMatrix(_Shared(),X.m,X.n,copy(X.colptr),copy(X.rowind),T.(X.values))
_mul_convert(::Type{T},X::COOMatrix) where T=COOMatrix(X.m,X.n,copy(X.rows),copy(X.cols),T.(X.values))
_mul_convert(::Type{T},X::DIAMatrix) where T=DIAMatrix(X.m,X.n,copy(X.offsets),T.(X.data))
_mul_convert(::Type{T},X::ELLMatrix) where T=ELLMatrix(X.m,X.n,copy(X.indices),T.(X.values))
_mul_convert(::Type{T},X::BSRMatrix) where T=BSRMatrix(X.m,X.n,X.blockrows,X.blockcols,copy(X.rowptr),copy(X.colind),T.(X.blocks))
_mul_convert(::Type{T},X::BSCMatrix) where T=BSCMatrix(_mul_convert(T,X.transposed_blocks))
_mul_convert(::Type{T},X::HYBMatrix) where T=HYBMatrix(_mul_convert(T,X.ell),_mul_convert(T,X.coo))
_mul_convert(::Type{T},X::SELLMatrix) where T=SELLMatrix(X.m,X.n,X.sliceheight,copy(X.sliceptr),copy(X.widths),copy(X.indices),T.(X.values),copy(X.permutation))
_mul_convert(::Type{T},X::SparseTensor) where T=SparseTensor(size(X),copy(X.indices),T.(X.values))
_mul_convert(::Type{T},X::_LA.Diagonal) where T=_LA.Diagonal(_mul_operand(T,X.diag))
_mul_convert(::Type{T},X::_Tri) where T=Base.typename(typeof(X)).wrapper(_mul_operand(T,parent(X)))
_mul_convert(::Type{T},X::_LA.Symmetric) where T=_LA.Symmetric(_mul_operand(T,parent(X)),Symbol(X.uplo))
_mul_convert(::Type{T},X::_LA.Hermitian) where T=_LA.Hermitian(_mul_operand(T,parent(X)),Symbol(X.uplo))
_mul_convert(::Type{T},X::_LA.Bidiagonal) where T=_LA.Bidiagonal(T.(X.dv),T.(X.ev),Symbol(X.uplo))
_mul_convert(::Type{T},X::_LA.Tridiagonal) where T=_LA.Tridiagonal(T.(X.dl),T.(X.d),T.(X.du))
_mul_convert(::Type{T},X::_LA.SymTridiagonal) where T=_LA.SymTridiagonal(T.(X.dv),T.(X.ev))
_mul_convert(::Type{T},X::Transpose) where T=transpose(_mul_operand(T,parent(X)))
_mul_convert(::Type{T},X::Adjoint) where T=adjoint(_mul_operand(T,parent(X)))
_mul_convert(::Type{T},X::BandedMatrix) where T=BandedMatrix(Matrix{T}(X.data),X.m;kl=X.kl,ku=X.ku,kind=X.kind,uplo=X.upper ? :U : :L,unit=X.unit)
_mul_convert(::Type{T},X::PackedMatrix) where T=PackedMatrix(Vector{T}(X.data),X.n;uplo=X.upper ? :U : :L,kind=X.kind,unit=X.unit)
_mul_convert(::Type{T},X::AbstractArray) where T=T.(X)

# Storage-level aliasing. The arrays an operand reads (through views,
# transposes and structured wrappers) are compared with those C writes, using
# the rectangle test of _mightalias for matrix blocks; an overlapping operand
# is replaced by a copy. A sparse destination is only written after the full
# product is formed, so it needs no check.
_mul_storage(X::_LA.Diagonal)=_mul_storage(X.diag)
_mul_storage(X::Union{_Tri,_SymHerm})=_mul_storage(parent(X))
_mul_storage(X::_LA.Bidiagonal)=(X.dv,X.ev)
_mul_storage(X::_LA.Tridiagonal)=(X.dl,X.d,X.du)
_mul_storage(X::_LA.SymTridiagonal)=(X.dv,X.ev)
_mul_storage(X::Union{Transpose,Adjoint})=_mulsparse(X) ? _mul_storage(parent(X)) : (X,)
_mul_storage(X::SparseMatrixCSC)=(SparseArrays.nonzeros(X),)
_mul_storage(X::SparseVector)=(SparseArrays.nonzeros(X),)
_mul_storage(X::Union{CSRMatrix,CSCMatrix,COOMatrix,SparseTensor})=(X.values,)
_mul_storage(X::Union{BandedMatrix,PackedMatrix})=(X.data,)
_mul_storage(X::AbstractArray)=(X,)
function _mul_detach(C,X)
    C isa SparseDestination && return X
    out=_mul_storage(C)
    any(s->any(c->_mightalias(c,s),out),_mul_storage(X)) ? _mul_copy(X) : X
end
_mul_copy(X::_OxiSparse)=_mul_convert(eltype(X),X)   # keeps the format (copy would densify)
_mul_copy(X::Union{BandedMatrix,PackedMatrix})=_compact_copy(X)
_mul_copy(X::Union{Transpose,Adjoint})=_mulsparse(X) ? (X isa Transpose ? transpose : adjoint)(_mul_copy(parent(X))) : copy(X)
_mul_copy(X)=copy(X)

function _mul_dispatch!(C,A,B,α,β)
    ka,kb,kc=_mulkind(A),_mulkind(B),_mulkind(C)
    kc===:diagonal && !(ka===:diagonal && kb===:diagonal) && !(_mulsparse(A) && _mulsparse(B)) &&
        throw(ArgumentError("a Diagonal destination needs a Diagonal or sparse product"))
    # Sparse destinations take the exact pattern of the product.
    kc in (:sparse,:sparsevec) && return _sparse_axpby!(C,_mul_tensor(_mul_value(A,B),size(C)),α,β)
    if ka===:sparse || ka===:sparsevec
        (kb===:sparse || kb===:sparsevec) && return _mul_add_sparse!(C,_mul_sparse_product(A,B),α,β)
        kb===:diagonal && return _mul_add_sparse!(C,_mul_scale_columns(_logical_csc(A),B.diag),α,β)
        return _mul_sparse_dense!(C,A,_mul_dense_operand(B),α,β)
    end
    if kb===:sparse || kb===:sparsevec
        ka===:diagonal && return _mul_add_sparse!(C,_mul_scale_rows(_logical_csc(B),A.diag),α,β)
        return _mul_dense_sparse!(C,_mul_dense_operand(A),_logical_csc(B),α,β)
    end
    ka===:diagonal && return _mul_diag_left!(C,A.diag,B,α,β)
    kb===:diagonal && return _mul_diag_right!(C,A,B.diag,α,β)
    ka===:triangular && return _mul_triangular!(C,A,B,α,β,true)
    kb===:triangular && return _mul_triangular!(C,B,A,α,β,false)
    ka===:symmetric && return _mul_symmetric!(C,A,B,α,β,true)
    kb===:symmetric && return _mul_symmetric!(C,B,A,α,β,false)
    ka===:band && return _mul_band!(C,A,B,α,β,true)
    kb===:band && return _mul_band!(C,B,A,α,β,false)
    ka===:banded_compact && (kb===:vector || kb===:dense) && A.kind!==:triangular && return _mul_banded_compact!(C,A,B,α,β)
    ka===:packed_compact && kb===:vector && A.kind!==:triangular && return _mul_packed_compact!(C,A,B,α,β)
    _mul_dense!(C,A,B,α,β)
end
# Structured operands met in a dense role (the other side of a sparse
# product) are read through their own getindex: no densified copy.
_mul_dense_operand(X)=X

# ---- dense ------------------------------------------------------------------

function _mul_dense!(C,A,B,α,β)
    if ndims(B)==1
        ndims(A)==1 && (A=reshape(A,:,1))
        A isa Transpose{<:Any,<:StridedMatrix} && return gemv!(C,parent(A),B,α,β;trans=:T)
        A isa Adjoint{<:Any,<:StridedMatrix} && return gemv!(C,parent(A),B,α,β;trans=:C)
        return gemv!(C,A,B,α,β)
    elseif ndims(A)==1 || size(A,2)==1 && size(B,1)==1
        # Outer product α·a·bᵀ (+ β·C): rank-1 update.
        a=ndims(A)==1 ? A : view(A,:,1)
        _scale!(C,β)
        if B isa Adjoint{<:Any,<:AbstractVector}
            gerc!(C,a,parent(B),α)
        else
            ger!(C,a,B isa Transpose{<:Any,<:AbstractVector} ? parent(B) : vec(B),α)
        end
        return C
    end
    gemm!(C,A,B,α,β)
end

# ---- LinearAlgebra structured matrices ---------------------------------------

function _mul_diag_left!(C,d,B,α,β)
    C isa _LA.Diagonal && return (_axpby_diag!(C.diag,d,B.diag,α,β);C)
    # Diagonal × anything else: row scaling, B read by index.
    ndims(B)==1 && return _axpby_diag!(C,d,B,α,β)
    @inbounds for j in axes(C,2), i in axes(C,1)
        v=α*(d[i]*B[i,j])
        C[i,j]=iszero(β) ? v : v+β*C[i,j]
    end
    C
end
function _mul_diag_right!(C,A,d,α,β)
    @inbounds for j in axes(C,2), i in axes(C,1)
        v=α*(A[i,j]*d[j])
        C[i,j]=iszero(β) ? v : v+β*C[i,j]
    end
    C
end
# y ← α (d .* x) + β y without the d .* x temporary.
function _axpby_diag!(y,d,x,α,β)
    @inbounds for i in eachindex(y,d,x)
        v=α*(d[i]*x[i])
        y[i]=iszero(β) ? v : v+β*y[i]
    end
    y
end
_tri_flags(X)=(X isa Union{_LA.UpperTriangular,_LA.UnitUpperTriangular},X isa Union{_LA.UnitUpperTriangular,_LA.UnitLowerTriangular})
# T (triangular) × other (left) or other × T (right): TRMM/TRMV on a copy of
# the other operand, then combined into C (aliasing-safe).
function _mul_triangular!(C,Tr,X,α,β,left)
    upper,unit=_tri_flags(Tr); P=parent(Tr); trans=:N
    # transpose(UpperTriangular(A)) is LowerTriangular(transpose(A)): use the
    # stored triangle of A with the transposition flag.
    if P isa Union{Transpose,Adjoint}
        trans=P isa Transpose ? :T : :C; P=parent(P); upper=!upper
    end
    uplo=upper ? :U : :L
    # β = 0: C itself is the work array (operands overlapping C were
    # detached by mul!); otherwise a copy of X, combined into C.
    if iszero(β) && C isa StridedArray
        copyto!(C,X)
        if ndims(C)==1
            trmv!(C,P;uplo,trans,unit); _scale!(C,α)
        else
            trmm!(C,P,α;side=left ? :L : :R,uplo,trans,unit)
        end
        return C
    end
    W=Array{eltype(C)}(undef,size(X)); copyto!(W,X)
    if ndims(W)==1
        trmv!(W,P;uplo,trans,unit); _scale!(W,α)
    else
        trmm!(W,P,α;side=left ? :L : :R,uplo,trans,unit)
    end
    _axpby_dense!(C,W,β)
end
function _axpby_dense!(C,W,β)
    @inbounds for i in eachindex(C,W)
        C[i]=iszero(β) ? W[i] : W[i]+β*C[i]
    end
    C
end
function _mul_symmetric!(C,S,X,α,β,left)
    uplo=Symbol(S.uplo); P=parent(S); herm=S isa _LA.Hermitian
    # Sparse parents were unwrapped already; a non-strided dense parent (a
    # transposed matrix) is materialized for SYMM/HEMM.
    P isa StridedMatrix || (P=Matrix(P))
    if ndims(X)==1
        left || throw(DimensionMismatch("vector × matrix is not a matrix product"))
        return herm ? hemv!(C,P,X,α,β;uplo) : symv!(C,P,X,α,β;uplo)
    end
    herm ? hemm!(C,P,X,α,β;side=left ? :L : :R,uplo) : symm!(C,P,X,α,β;side=left ? :L : :R,uplo)
end
function _band_matrix(X)
    kl,ku=X isa _LA.Bidiagonal ? (X.uplo=='U' ? (0,1) : (1,0)) : (1,1)
    band_matrix(X;kl,ku)   # reads the band only
end
function _mul_band!(C,Bd,X,α,β,left)
    P=_band_matrix(Bd)
    if left
        ndims(X)==1 && return gbmv!(C,P,X,α,β)
        for j in axes(X,2)
            gbmv!(view(C,:,j),P,view(X,:,j),α,β)
        end
    else
        # C[i,:] = α X[i,:]·Band + β C[i,:] = (Bandᵀ X[i,:]ᵀ) row by row.
        for i in axes(X,1)
            gbmv!(view(C,i,:),P,view(X,i,:),α,β;trans=:T)
        end
    end
    C
end
function _mul_banded_compact!(C,A::BandedMatrix,B,α,β)
    if ndims(B)==1
        A.kind===:general && return gbmv!(C,A,B,α,β)
        A.kind===:symmetric && return sbmv!(C,A,B,α,β)
        A.kind===:hermitian && return hbmv!(C,A,B,α,β)
        throw(ArgumentError("unsupported compact banded matrix kind $(A.kind)"))
    end
    if A.kind===:general
        for j in axes(B,2)
            gbmv!(view(C,:,j),A,view(B,:,j),α,β)
        end
    elseif A.kind===:symmetric
        for j in axes(B,2)
            sbmv!(view(C,:,j),A,view(B,:,j),α,β)
        end
    elseif A.kind===:hermitian
        for j in axes(B,2)
            hbmv!(view(C,:,j),A,view(B,:,j),α,β)
        end
    else
        throw(ArgumentError("unsupported compact banded matrix kind $(A.kind)"))
    end
    C
end
function _mul_packed_compact!(C,A::PackedMatrix,B,α,β)
    A.kind===:symmetric && return spmv!(C,A,B,α,β;threaded=false)
    A.kind===:hermitian && return hpmv!(C,A,B,α,β;threaded=false)
    throw(ArgumentError("unsupported compact packed matrix kind $(A.kind)"))
end

# ---- sparse -----------------------------------------------------------------

# Logical compressed-column storage of a sparse operand (shared for a
# SparseMatrixCSC, a pattern transposition for transposes of CSC data).
_logical_csc(X::SparseMatrixCSC)=CSCMatrix(X)
_logical_csc(X::CSCMatrix)=X
_logical_csc(X::Union{CSRMatrix,COOMatrix,ExtendedSparse})=CSCMatrix(X)
_logical_csc(X::SparseTensor{<:Any,2})=CSCMatrix(X)
_logical_csc(X::SparseVector)=CSCMatrix(X)
_logical_csc(X::SparseTensor{<:Any,1})=CSCMatrix(SparseVector(X))
_logical_csc(X::Union{_SparseCSCOp,Transpose{<:Any,<:SparseVector},Adjoint{<:Any,<:SparseVector}})=CSCMatrix(X)
_logical_csc(X::Union{Transpose,Adjoint})=CSCMatrix(_op_csr(_sparse_parent_op(X)))
_sparse_parent_op(X::Transpose)=transpose(_logical_csc(parent(X)))
_sparse_parent_op(X::Adjoint)=adjoint(_logical_csc(parent(X)))
# CSR of the logical matrix (for the Gustavson product).
_logical_csr(X::CSRMatrix)=X
_logical_csr(X::_SparseCSCOp)=CSRMatrix(X)
_logical_csr(X)=CSRMatrix(_logical_csc(X))

function _mul_sparse_dense!(C,A,B,α,β)
    # SpMV/SpMM kernels; transposes use their trans argument, not a copy.
    op,trans=A isa Transpose ? (parent(A),:T) : A isa Adjoint ? (parent(A),:C) : (A,:N)
    K=op isa Union{SparseMatrixCSC,SparseVector,SparseTensor} ? _logical_csc(op) : op
    BB=B isa StridedArray ? B : Array(B)
    sparse_mul!(C,K,BB,α,β;trans)
end
# Dense × sparse: C[:,j] += α Σ_p A[:,row_p]·v_p over the stored entries of
# column j of B (axpy per entry), after scaling C by β.
function _mul_dense_sparse!(C,A,B::CSCMatrix,α,β)
    _scale!(C,β)
    m=size(C,1)
    @inbounds for j in 1:B.n, p in B.colptr[j]:B.colptr[j+1]-1
        k=B.rowind[p]; v=α*B.values[p]
        if ndims(C)==1   # dense A × sparse vector (j == 1)
            for i in 1:m
                C[i]=_arithmetic_madd(A[i,k],v,C[i])
            end
        else
            for i in 1:m
                C[i,j]=_arithmetic_madd(A[i,k],v,C[i,j])
            end
        end
    end
    C
end
# Sparse × sparse (Gustavson on CSR); SparseArrays pairs keep their own
# zero-copy path and result type.
function _mul_sparse_product(A,B)
    if A isa SparseMatrixCSC && B isa SparseMatrixCSC
        return sparse_matmul(A,B)
    end
    sparse_matmul(_logical_csr(A),_logical_csr(B))
end
_mul_scale_rows(X::CSCMatrix,d)=CSCMatrix(_Shared(),X.m,X.n,copy(X.colptr),copy(X.rowind),[X.values[p]*d[X.rowind[p]] for p in eachindex(X.values)])
function _mul_scale_columns(X::CSCMatrix,d)
    vals=similar(X.values)
    for j in 1:X.n, p in X.colptr[j]:X.colptr[j+1]-1
        vals[p]=X.values[p]*d[j]
    end
    CSCMatrix(_Shared(),X.m,X.n,copy(X.colptr),copy(X.rowind),vals)
end
# C (dense or Diagonal) ← α·R + β·C for a sparse R.
function _mul_add_sparse!(C,R,α,β)
    S=_as_tensor(R); _scale!(C isa _LA.Diagonal ? C.diag : C,β)
    if C isa _LA.Diagonal
        for (i,v) in zip(S.indices,S.values)
            r,c=_sparse_coords(i,(size(S,1),size(S,2))); r==c || throw(ArgumentError("product is not diagonal"))
            C.diag[r]+=α*v
        end
        return C
    end
    @inbounds for (i,v) in zip(S.indices,S.values)
        C[i]+=α*v
    end
    C
end
# The product as a SparseTensor shaped like the destination (an n×1 product
# feeds a SparseVector destination).
function _mul_tensor(R,dims)
    S=_as_tensor(R)
    size(S)==dims ? S : reshape(S,dims)
end
# Product as a value (sparse when both operands are sparse), for sparse
# destinations and for OxiBLAS.mul.
function _mul_value(A,B)
    ka,kb=_mulkind(A),_mulkind(B)
    _mulsparse(A) && _mulsparse(B) && return _mul_sparse_product(A,B)
    _mulsparse(A) && kb===:diagonal && return _mul_scale_columns(_logical_csc(A),B.diag)
    ka===:diagonal && _mulsparse(B) && return _mul_scale_rows(_logical_csc(B),A.diag)
    T=eltype(A); C=ndims(B)==1 ? zeros(T,size(A,1)) : zeros(T,size(A,1),size(B,2))
    _mul_dispatch!(C,A,B,one(T),zero(T))
end

# ---- allocating product -------------------------------------------------------

function mul(A::_MulOperand,B::_MulOperand)
    size(A,2)==size(B,1) || throw(DimensionMismatch("OxiBLAS.mul: A has $(size(A,2)) columns, B has $(size(B,1)) rows"))
    T=promote_type(eltype(A),eltype(B)); T<:GenericFloat || throw(ArgumentError("unsupported OxiBLAS.mul element type $T"))
    AA,BB=_mul_operand(T,_mul_unwrap(A)),_mul_operand(T,_mul_unwrap(B))
    ka,kb=_mulkind(AA),_mulkind(BB)
    if _mulsparse(AA) && _mulsparse(BB)
        R=_mul_sparse_product(AA,BB)
        arrays=_mul_sparsearrays(A) || _mul_sparsearrays(B)
        if kb===:sparsevec
            v=SparseVector(R isa SparseMatrixCSC ? CSCMatrix(R) : R)
            return arrays ? v : SparseTensor(v)
        end
        arrays && return R isa SparseMatrixCSC ? R : SparseMatrixCSC(R)
        return _mul_csc(A,B) ? CSCMatrix(R) : (R isa CSRMatrix ? R : CSRMatrix(R))
    end
    if (_mulsparse(AA) && kb===:diagonal) || (ka===:diagonal && _mulsparse(BB))
        R=_mul_value(AA,BB)
        (_mul_sparsearrays(A) || _mul_sparsearrays(B)) && return SparseMatrixCSC(R)
        return _mul_csc(A,B) ? (R isa CSCMatrix ? R : CSCMatrix(R)) : CSRMatrix(R)
    end
    ka===:diagonal && kb===:diagonal && return _LA.Diagonal(AA.diag .* BB.diag)
    if ka===:triangular && kb===:triangular && _tri_flags(AA)[1]==_tri_flags(BB)[1]
        C=Matrix{T}(undef,size(AA,1),size(BB,2)); mul!(C,AA,Matrix(BB))
        upper=_tri_flags(AA)[1]; unit=_tri_flags(AA)[2] && _tri_flags(BB)[2]
        return upper ? (unit ? _LA.UnitUpperTriangular(C) : _LA.UpperTriangular(C)) : (unit ? _LA.UnitLowerTriangular(C) : _LA.LowerTriangular(C))
    end
    C=ndims(BB)==1 ? Vector{T}(undef,size(AA,1)) : Matrix{T}(undef,size(AA,1),size(BB,2))
    mul!(C,AA,BB,one(T),zero(T))
end
# A product whose sparse operands are all plain CSCMatrix stays CSCMatrix
# (otherwise OxiBLAS sparse results are CSRMatrix).
_mul_csc(A,B)=_mul_isplaincsc(A) && _mul_isplaincsc(B)
_mul_isplaincsc(X)=X isa CSCMatrix || !_mulsparse(X)
_mul_sparsearrays(X)=X isa Union{SparseMatrixCSC,SparseVector}
_mul_sparsearrays(X::Union{Transpose,Adjoint,_Tri,_SymHerm})=_mul_sparsearrays(parent(X))

# ---- Base.* on OxiBLAS sparse types -------------------------------------------
# Only signatures with an OxiBLAS sparse type (possibly transposed/adjoint) as
# an operand: no type piracy. Every product of an OxiBLAS sparse operand ends
# in OxiBLAS.mul (OxiBLAS kernels). LinearAlgebra.mul! is not extended.
#
# One union per side; the LinearAlgebra structured types get their own
# methods because LinearAlgebra has specific methods for them on the other
# side (e.g. AbstractMatrix * Diagonal), which a union would make ambiguous.
# The ambiguity test in test/sparse_supertypes.jl checks the whole set.
const _OxiSparseMat=Union{_OxiSparse,SparseTensor{<:Any,2}}
const _OxiSparseOp=Union{_OxiSparseMat,Transpose{<:Any,<:_OxiSparseMat},Adjoint{<:Any,<:_OxiSparseMat}}
const _MulDenseMat=Union{StridedMatrix,Transpose{<:Any,<:StridedMatrix},Adjoint{<:Any,<:StridedMatrix}}
const _MulSAMat=Union{SparseMatrixCSC,Transpose{<:Any,<:SparseMatrixCSC},Adjoint{<:Any,<:SparseMatrixCSC}}
const _MulRight=Union{_OxiSparseOp,_MulDenseMat,_MulSAMat,StridedVector,SparseVector,SparseTensor{<:Any,1}}
const _MulLeft=Union{_MulDenseMat,_MulSAMat}
Base.:*(A::_OxiSparseOp,B::_MulRight)=mul(A,B)
Base.:*(A::_MulLeft,B::_OxiSparseOp)=mul(A,B)
# Row vectors keep LinearAlgebra's shape: x'A is an adjoint vector.
Base.:*(x::Adjoint{<:Any,<:AbstractVector},A::_OxiSparseOp)=adjoint(mul(adjoint(A),parent(x)))
Base.:*(x::Transpose{<:Any,<:AbstractVector},A::_OxiSparseOp)=transpose(mul(transpose(A),parent(x)))
# LinearAlgebra structured matrices with dense storage.
Base.:*(A::_OxiSparseOp,B::_LA.Diagonal)=mul(A,B)
Base.:*(A::_LA.Diagonal,B::_OxiSparseOp)=mul(A,B)
Base.:*(A::_OxiSparseOp,B::_LABand)=mul(A,B)
Base.:*(A::_LABand,B::_OxiSparseOp)=mul(A,B)
Base.:*(A::_OxiSparseOp,B::_LA.AbstractTriangular)=mul(A,B)
Base.:*(A::_LA.AbstractTriangular,B::_OxiSparseOp)=mul(A,B)
Base.:*(A::_OxiSparseOp,B::_SymHerm)=mul(A,B)
Base.:*(A::_SymHerm,B::_OxiSparseOp)=mul(A,B)
# Structured wrappers of an OxiBLAS sparse matrix against a dense operand.
const _OxiWrapped=Union{_LA.UpperTriangular{<:Any,<:_OxiSparseOp},_LA.LowerTriangular{<:Any,<:_OxiSparseOp},
                        _LA.UnitUpperTriangular{<:Any,<:_OxiSparseOp},_LA.UnitLowerTriangular{<:Any,<:_OxiSparseOp},
                        _LA.Symmetric{<:Any,<:_OxiSparseOp},_LA.Hermitian{<:Any,<:_OxiSparseOp}}
Base.:*(A::_OxiWrapped,B::Union{_MulDenseMat,StridedVector})=mul(A,B)
Base.:*(A::_MulDenseMat,B::_OxiWrapped)=mul(A,B)
# Scaling keeps the format and the pattern (as SparseMatrixCSC does).
Base.:*(α::Number,A::_OxiSparse)=_mul_scaled(A,α)
Base.:*(A::_OxiSparse,α::Number)=_mul_scaled(A,α)
# α*A (op=*) or A/α (op=/), entry by entry on a copy in the same format.
function _mul_scaled(A,α,op=*)
    T=promote_type(eltype(A),typeof(α))
    A isa COOMatrix && return COOMatrix(A.m,A.n,copy(A.rows),copy(A.cols),T.(op.(A.values,α)))
    B=eltype(A)===T ? _mul_copy(A) : _mul_operand(T,A)
    for v in _mul_values(B)
        v .= op.(v,α)
    end
    B
end
# f applied to the stored values (f(0)==0), same format and pattern.
function _mul_mapped(f,A)
    A isa COOMatrix && return COOMatrix(A.m,A.n,copy(A.rows),copy(A.cols),f.(A.values))
    B=_mul_copy(A)
    for v in _mul_values(B)
        v .= f.(v)
    end
    B
end
# The stored value arrays of each format (scaled in place on a copy).
_mul_values(X::Union{CSRMatrix,CSCMatrix,COOMatrix,ELLMatrix,SELLMatrix})=(X.values,)
_mul_values(X::DIAMatrix)=(X.data,)
_mul_values(X::BSRMatrix)=(X.blocks,)
_mul_values(X::BSCMatrix)=_mul_values(X.transposed_blocks)
_mul_values(X::HYBMatrix)=(_mul_values(X.ell)...,X.coo.values)
