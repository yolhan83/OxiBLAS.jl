# SPDX-License-Identifier: Apache-2.0
# Hardware aliases NativeFloat/RealFloat remain deliberately closed in OxiBLAS.jl.
const GenericFloat=Union{AbstractFloat,Complex{<:AbstractFloat}}

"""Whether scalar arithmetic has a context safe to use in spawned kernel tasks.
Machine floats are enabled. Other providers conservatively execute kernels in
one task, preserving the caller's arithmetic context (notably BigFloat precision).
Providers may extend this OxiBLAS function explicitly; no Base method is needed.
"""
generic_threadsafe(::Type{T}) where T=T<:Union{NativeFloat,Float16,Complex{Float16}}
@inline _arithmetic_madd(a,b,c)=a*b+c
@inline _arithmetic_madd(a::T,b::T,c::T) where {T<:RealFloat}=muladd(a,b,c)
@inline _arithmetic_madd(a::T,b::T,c::T) where {T<:Union{ComplexF32,ComplexF64}}=muladd(a,b,c)
_residual_worktype(::Type{T}) where T = T===Float16 ? Float32 : T===Complex{Float16} ? ComplexF32 :
    T===Float32 ? Float64 : T===ComplexF32 ? ComplexF64 : T
function _generic_require(::Type{T}) where {T<:GenericFloat}
    isconcretetype(T) || throw(ArgumentError("a concrete floating element type is required"))
    # These are the minimal scalar operations of the generic numerical path.
    # Additional LAPACK operations (sqrt, abs, division, etc.) belong to T.
    zero(T) isa T && one(T) isa T || throw(ArgumentError("zero/one must preserve the floating element type"))
    nothing
end

function _floating_iterations(::Type{R},base::Integer) where {R<:AbstractFloat}
    R<:RealFloat && return Int(base)
    bits=try
        precision(R)
    catch error
        error isa MethodError || rethrow()
        0
    end
    max(Int(base),Base.checked_add(Base.checked_mul(2,Int(bits)),32))
end
