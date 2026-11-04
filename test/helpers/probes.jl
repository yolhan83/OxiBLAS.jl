# Arrays that fail when an element that must not be read is read. Shared by
# the BLAS, tensor and audit tests (formerly defined in gemm.jl, tensors.jl
# and audit_priority.jl).
struct UnreadableMatrix{T} <: AbstractMatrix{T}
    dims::Tuple{Int,Int}
end
Base.size(A::UnreadableMatrix)=A.dims
Base.getindex(::UnreadableMatrix,i::Int,j::Int)=error("unexpected operand read")

struct TensorNoRead{T,N} <: AbstractArray{T,N}
    shape::NTuple{N,Int}
end
Base.size(A::TensorNoRead)=A.shape
Base.getindex(::TensorNoRead,I...)=error("input must not be read")

struct AuditTriangle{T} <: AbstractMatrix{T}
    data::Matrix{T}
    upper::Bool
    unit::Bool
end
Base.size(A::AuditTriangle)=size(A.data)
Base.dataids(A::AuditTriangle)=Base.dataids(A.data)
function Base.getindex(A::AuditTriangle,i::Int,j::Int)
    (A.upper ? i<=j : i>=j) || error("unstored triangle read")
    A.unit && i==j && error("implicit diagonal read")
    A.data[i,j]
end
