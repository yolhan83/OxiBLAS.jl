# SPDX-License-Identifier: Apache-2.0
"""Compatibility entry point for GEMM with caller-provided floating types.
Native Float32/64 retain specialized dispatch; other concrete AbstractFloat
and complex floating types use generic Julia arithmetic. No scalar is wrapped
or reimplemented by OxiBLAS. Float16 GEMM accumulates in Float32.
"""
function gemm_extended!(C::AbstractMatrix{T},A::AbstractMatrix{T},B::AbstractMatrix{T},alpha=one(T),beta=zero(T);kwargs...) where {T<:GenericFloat}
    gemm!(C,A,B,alpha,beta;kwargs...)
end
@inline function _compensated_sum_step(sum,correction,value)
    next=sum+value
    error=abs(sum)>=abs(value) ? (sum-next)+value : (value-next)+sum
    next,correction+error
end
@inline function _compensated_sum_step(sum::Complex,correction::Complex,value::Complex)
    re,cr=_compensated_sum_step(real(sum),real(correction),real(value))
    im,ci=_compensated_sum_step(imag(sum),imag(correction),imag(value))
    complex(re,im),complex(cr,ci)
end
"""Compensated dot product in a caller-selectable accumulation type.
Float16/32 default to Float32/64; other providers preserve their element type.
accumulator_type can explicitly select BigFloat, Double64, Float128, etc.
Products are rounded by that type; no hidden fixed extended scalar is used.
"""
function dot_extended(x::AbstractVector{T},y::AbstractVector{T};conjugate=false,accumulator_type=nothing) where {T<:GenericFloat}
    _same_length(x,y)
    W=accumulator_type===nothing ? _residual_worktype(T) : accumulator_type
    W isa Type && W<:GenericFloat || throw(ArgumentError("floating accumulation type required"))
    _generic_require(W); value=zero(W); correction=zero(W)
    for i in eachindex(x)
        a=convert(W,x[i]); b=convert(W,y[i]); term=(conjugate ? conj(a) : a)*b
        if !isfinite(term) || !isfinite(value)
            value+=term; correction=zero(W)
        else
            value,correction=_compensated_sum_step(value,correction,term)
        end
    end
    value+correction
end
