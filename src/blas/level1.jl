# SPDX-License-Identifier: Apache-2.0
@inline _abs1(x::Real)=abs(x)
@inline _abs1(x::Complex)=abs(real(x))+abs(imag(x))

function _same_length(x,y)
    Base.require_one_based_indexing(x,y)
    length(x)==length(y) || throw(DimensionMismatch("vector lengths differ"))
end

function axpy!(y::AbstractVector,x::AbstractVector,α=one(eltype(y)))
    _same_length(x,y);_check_vector_output(y)
    iszero(α) && return y
    xx=Base.mightalias(x,y) && x!==y ? copy(x) : x
    @inbounds @simd for i in eachindex(y)
        y[i]=_arithmetic_madd(α,xx[i],y[i])
    end
    y
end
scal!(x::AbstractVector,α) = (Base.require_one_based_indexing(x);_check_vector_output(x);_scale!(x,α))
function copy_vector!(y::AbstractVector,x::AbstractVector)
    _same_length(x,y);_check_vector_output(y)
    xx=Base.mightalias(x,y) ? copy(x) : x
    @inbounds for i in eachindex(y)
        y[i]=xx[i]
    end
    y
end
function swap!(x::AbstractVector,y::AbstractVector)
    _same_length(x,y);_check_vector_output(x);_check_vector_output(y)
    x===y && return (x,y)
    Base.mightalias(x,y) && throw(ArgumentError("swap requires disjoint vectors"))
    @inbounds for i in eachindex(x)
        x[i],y[i]=y[i],x[i]
    end
    (x,y)
end
function dotu(x::AbstractVector,y::AbstractVector)
    _same_length(x,y)
    s=zero(promote_type(eltype(x),eltype(y)))
    @inbounds for i in eachindex(x)
        s=_arithmetic_madd(x[i],y[i],s)
    end
    s
end
function dotc(x::AbstractVector,y::AbstractVector)
    _same_length(x,y)
    s=zero(promote_type(eltype(x),eltype(y)))
    @inbounds for i in eachindex(x)
        s=_arithmetic_madd(conj(x[i]),y[i],s)
    end
    s
end
function dot_kahan(x::AbstractVector,y::AbstractVector;conjugate=false)
    _same_length(x,y)
    s=zero(promote_type(eltype(x),eltype(y))); c=s
    @inbounds for i in eachindex(x)
        v=(conjugate ? conj(x[i]) : x[i])*y[i]-c
        t=s+v
        c=(t-s)-v
        s=t
    end
    s
end
function _pairdot(x,y,lo,hi,conjugate)
    if hi-lo<64
        s=zero(promote_type(eltype(x),eltype(y)))
        @inbounds for i in lo:hi
            s+=(conjugate ? conj(x[i]) : x[i])*y[i]
        end
        return s
    end
    mid=(lo+hi)÷2
    _pairdot(x,y,lo,mid,conjugate)+_pairdot(x,y,mid+1,hi,conjugate)
end
function dot_pairwise(x::AbstractVector,y::AbstractVector;conjugate=false)
    _same_length(x,y)
    _pairdot(x,y,1,length(x),conjugate)
end

# Sum-of-squares accumulations restart every _NRM2_BLOCK elements. A single
# running sum over N terms accumulates O(N*eps) rounding error: on the
# benchmark data this exceeded the 64eps oracle tolerance from N=262144
# (ComplexF64, ordered loop) and N=524288 (Float32/Float64, SIMD loop).
# Restarting bounds the error by O((B + N/B)*eps). B is an element count, not a
# SIMD width, and it does not depend on the target.
const _NRM2_BLOCK=1024

# Merge two scaled partial norms (scale,ssq) representing scale*sqrt(ssq).
# A zero scale is the empty partial; an infinite scale absorbs everything.
@inline function _nrm2_merge(s1::R,q1::R,s2::R,q2::R) where R
    iszero(s2) && return (s1,q1)
    iszero(s1) && return (s2,q2)
    (isinf(s1) || isinf(s2)) && return (R(Inf),one(R))
    s1>=s2 ? (s1,q1+q2*(s2/s1)^2) : (s2,q2+q1*(s1/s2)^2)
end

"""Euclidean norm with scaled sum of squares; avoids spurious overflow/underflow."""
function nrm2(x::AbstractArray{T}) where T
    R=typeof(real(zero(T)))
    total=zero(R); totalssq=one(R)
    scale=zero(R); ssq=one(R); count=0
    # Process real/imag separately: abs(complex) may overflow before scaling.
    for z in x
        for v in (real(z),imag(z))
            a=abs(v)
            isnan(a) && return a
            if !iszero(a)
                if isinf(a)
                    scale=a
                elseif scale<a
                    ssq=one(R)+ssq*(scale/a)^2
                    scale=a
                else
                    ssq+=(a/scale)^2
                end
            end
        end
        count+=1
        if count==_NRM2_BLOCK
            total,totalssq=_nrm2_merge(total,totalssq,scale,ssq)
            scale=zero(R); ssq=one(R); count=0
        end
    end
    total,totalssq=_nrm2_merge(total,totalssq,scale,ssq)
    iszero(total) ? total : total*sqrt(totalssq)
end
asum(x::AbstractVector{T}) where T = sum(_abs1,x;init=zero(real(T)))
function _iam(x,ismax)
    Base.require_one_based_indexing(x)
    isempty(x) && return 0
    best=1; val=_abs1(x[1])
    @inbounds for i in 2:length(x)
        v=_abs1(x[i])
        if (ismax ? v>val : v<val) || (isnan(v) && !isnan(val))
            best=i; val=v
        end
    end
    best
end
iamax(x::AbstractVector)=_iam(x,true)
iamin(x::AbstractVector)=_iam(x,false)

function rot!(x::AbstractVector,y::AbstractVector,c::Real,s::Number)
    _same_length(x,y);_check_vector_output(x);_check_vector_output(y)
    Base.mightalias(x,y) && throw(ArgumentError("rotation requires disjoint vectors"))
    @inbounds for i in eachindex(x)
        a,b=x[i],y[i]
        x[i]=c*a+s*b
        y[i]=c*b-conj(s)*a
    end
    (x,y)
end
"""Return (c,s,r), c*a+s*b=r and c*b-conj(s)*a=0 (real c)."""
rotg(a::T,b::T) where {T<:Number}=_rotg_generic(a,b)
function _rotg_generic(a::T,b::T) where {T<:Number}
    R=typeof(real(zero(T)))
    iszero(b) && return (one(R),zero(T),a)
    iszero(a) && return (zero(R),conj(b)/abs(b),convert(T,abs(b)))
    # hypot is scaled and handles very different magnitudes.
    h=hypot(abs(a),abs(b)); phase=a/abs(a)
    (abs(a)/h,phase*conj(b)/h,phase*h)
end

# Real ROTG follows the BLAS sign convention (the larger-magnitude input
# determines r). hypot avoids the upstream scale=abs(a)+abs(b) overflow.
function rotg(a::T,b::T) where {T<:Real}
    iszero(a) && iszero(b) && return (one(T),zero(T),zero(T))
    roe=abs(a)>abs(b) ? a : b
    r=copysign(hypot(a,b),roe)
    (a/r,b/r,r)
end

# Float32 products are formed after widening: their exact 48-bit significands
# fit in Float64, including products which overflow or underflow in Float32.
# The Float64 sum is reassociated (SIMD partial sums), as in OpenBLAS DSDOT;
# its error stays O(n eps(Float64)) sum |x||y|. As for dotu, a non-finite
# result is recomputed in order so IEEE exceptional values match the
# sequential definition.
function _wide_dot(x,y,initial)
    s=0.0
    @inbounds @simd for i in eachindex(x,y)
        s=muladd(Float64(x[i]),Float64(y[i]),s)
    end
    s+=Float64(initial)
    if !isfinite(s)
        s=Float64(initial)
        @inbounds for i in eachindex(x)
            s=_arithmetic_madd(Float64(x[i]),Float64(y[i]),s)
        end
    end
    s
end

"""Float32-input inner product, products and accumulation in Float64."""
function dsdot(x::AbstractVector{Float32},y::AbstractVector{Float32})
    _same_length(x,y)
    _wide_dot(x,y,0.0)
end

"""`sdsdot(alpha::Float32,x,y)` accumulates alpha and Float32 products in Float64, then rounds once to Float32."""
function sdsdot(alpha::Float32,x::AbstractVector{Float32},y::AbstractVector{Float32})
    _same_length(x,y)
    Float32(_wide_dot(x,y,alpha))
end

# Native strided absolute sums: expose a nonnegative reduction to LLVM without
# a hardware lane width, pointers, or changing the generic scalar provider path.
function asum(x::StridedVector{T}) where {T<:NativeFloat}
    s=zero(real(T))
    @inbounds @simd for i in eachindex(x)
        s+=_abs1(x[i])
    end
    s
end

function _dot_native_strided(x::StridedVector{T},y::StridedVector{T},::Val{C}) where {T<:NativeFloat,C}
    _same_length(x,y)
    # @simd lets LLVM pick lane count and interleave (16 partial sums on
    # NEON); manual 4-way blocks were 15-25% slower.
    s=zero(T)
    @inbounds @simd for i in eachindex(x,y)
        s=_arithmetic_madd(C ? conj(x[i]) : x[i],y[i],s)
    end
    if !isfinite(s)
        # Reproduce the ordered fallback for exceptional arithmetic, including
        # an overflow in a partial accumulator that a cancellation may avoid.
        s=zero(T)
        @inbounds for j in eachindex(x)
            s=_arithmetic_madd(C ? conj(x[j]) : x[j],y[j],s)
        end
    end
    s
end
# Complex products as four real sums (rr, ii, ri, ir). A complex accumulator
# keeps real and imaginary parts in one lane pair, which LLVM does not
# vectorize; separate real sums are deinterleaved loads and plain FMA lanes.
# The final rr∓ii, ri±ir combination has the same error bound as an ordered
# complex sum in terms of sum |x||y|.
function _dot_native_strided(x::StridedVector{T},y::StridedVector{T},::Val{C}) where {T<:Union{ComplexF32,ComplexF64},C}
    _same_length(x,y); R=real(T)
    rr=zero(R);ii=zero(R);ri=zero(R);ir=zero(R)
    @inbounds @simd for i in eachindex(x,y)
        a=x[i];b=y[i]
        rr=muladd(real(a),real(b),rr); ii=muladd(imag(a),imag(b),ii)
        ri=muladd(real(a),imag(b),ri); ir=muladd(imag(a),real(b),ir)
    end
    s=C ? complex(rr+ii,ri-ir) : complex(rr-ii,ri+ir)
    if !isfinite(s)
        # Ordered complex fallback for exceptional arithmetic, as for real types.
        s=zero(T)
        @inbounds for j in eachindex(x)
            s=_arithmetic_madd(C ? conj(x[j]) : x[j],y[j],s)
        end
    end
    s
end
dotu(x::StridedVector{T},y::StridedVector{T}) where {T<:NativeFloat}=_dot_native_strided(x,y,Val(false))

dotc(x::StridedVector{T},y::StridedVector{T}) where {T<:NativeFloat}=_dot_native_strided(x,y,Val(true))

# Float32 components squared in Float64 are exact (24-bit significands give
# 48-bit products) and stay inside the Float64 normal range (2^-298..2^256),
# so no scaling pass is needed and overflow needs more than 2^768 terms.
# Inf propagates as Inf and NaN as NaN through the sum. The Float64 sum keeps
# the error far below Float32 resolution, unlike the former Float32 running
# sum of scaled squares, which failed the 64eps oracle at N=524288.
function nrm2(x::StridedVector{T}) where {T<:Union{Float32,ComplexF32}}
    ssq=0.0
    @inbounds @simd for i in eachindex(x)
        z=x[i];ssq+=abs2(Float64(real(z)))+abs2(Float64(imag(z)))
    end
    Float32(sqrt(ssq))
end

function nrm2(x::StridedVector{Float64})
    # Maximum magnitude through the integer keys of iamax: vectorized, exact,
    # and any NaN pattern exceeds Inf, so a NaN input still returns NaN.
    key=zero(UInt64)
    @inbounds @simd for i in eachindex(x)
        key=max(key,_iamax_key(x[i]))
    end
    amax=reinterpret(Float64,key)
    (iszero(amax) || !isfinite(amax)) && return amax
    # Scale by the power of two 2^exponent(amax): multiplying by its inverse is
    # exact (scaled magnitudes lie in [0,2)), unlike a division by amax, whose
    # throughput dominated this pass. A subnormal amax keeps the division.
    e=exponent(amax)
    e>=-1022 || return _nrm2_scaled(x,amax,amax)
    _nrm2_scaled(x,ldexp(1.0,e),ldexp(1.0,-e))
end
function _nrm2_scaled(x,scale,factor)
    # Blocked pass: see _NRM2_BLOCK. Each block sum restarts at zero.
    ssq=0.0;lo=firstindex(x);n=lastindex(x)
    @inbounds while lo<=n
        hi=min(n,lo+(_NRM2_BLOCK-1));s=0.0
        if factor===scale
            @simd for i in lo:hi
                s+=abs2(x[i]/scale)
            end
        else
            @simd for i in lo:hi
                s+=abs2(x[i]*factor)
            end
        end
        ssq+=s;lo=hi+1
    end
    scale*sqrt(ssq)
end

function copy_vector!(y::Vector{T},x::Vector{T}) where {T<:NativeFloat}
    _same_length(x,y)
    xx=Base.mightalias(x,y) ? copy(x) : x
    copyto!(y,xx)
end

# Block keys compare like _abs1. Real: the bit pattern with the sign bit
# cleared by an integer AND; nonnegative floats order like unsigned integers
# and every NaN pattern exceeds Inf. On Apple cores the integer maximum runs
# at load speed while NEON fmax, or fabs feeding an integer max (FP/integer
# domain crossing), is about 1.7x slower. Complex: |re|+|im| needs a float add,
# so the float maximum (which propagates NaN) is kept.
@inline _iamax_key(z::Float32)=reinterpret(UInt32,z)&0x7fffffff
@inline _iamax_key(z::Float64)=reinterpret(UInt64,z)&0x7fffffffffffffff
@inline _iamax_key(z::Complex)=_abs1(z)
@inline _iamax_isnan(m::Unsigned,::Type{T}) where T=m>_iamax_key(T(Inf))
@inline _iamax_isnan(m::AbstractFloat,::Type)=isnan(m)
@inline function _iamax_blockmax(x,lo,hi)
    m=zero(_iamax_key(zero(eltype(x))))
    @inbounds @simd for i in lo:hi
        m=max(m,_iamax_key(x[i]))
    end
    m
end
# One pass in blocks: SIMD key maximum per block, remember the first block with
# the strict maximum, then locate it by SIMD sub-blocks (a NaN block returns its
# first NaN at once). Two full passes read the vector twice.
const _IAMAX_BLOCK=512
const _IAMAX_SUBBLOCK=32
function iamax(x::StridedVector{T}) where {T<:NativeFloat}
    Base.require_one_based_indexing(x);isempty(x) && return 0
    n=length(x);best=zero(_iamax_key(zero(T)));start=0;lo=1
    @inbounds while lo<=n
        hi=min(n,lo+(_IAMAX_BLOCK-1));m=_iamax_blockmax(x,lo,hi)
        if _iamax_isnan(m,T)
            for i in lo:hi
                isnan(_abs1(x[i])) && return i
            end
        elseif m>best || start==0
            best=m;start=lo
        end
        lo=hi+1
    end
    stop=min(n,start+(_IAMAX_BLOCK-1));lo=start
    @inbounds while lo<=stop
        hi=min(stop,lo+(_IAMAX_SUBBLOCK-1))
        if _iamax_blockmax(x,lo,hi)==best
            for i in lo:hi
                _iamax_key(x[i])==best && return i
            end
        end
        lo=hi+1
    end
    0
end



function _rotg_scaled(a::T,b::T) where {T<:ComplexF64}
    R=real(T)
    iszero(b) && return (one(R),zero(T),a)
    iszero(a) && return (zero(R),conj(b)/abs(b),T(abs(b)))
    aa=abs(a);h=hypot(aa,real(b),imag(b));phase=a/aa
    (aa/h,phase*conj(b)/h,phase*h)
end
_rotg_scaled(a::ComplexF32,b::ComplexF32)=_rotg_generic(a,b)
# Nonzero components inside these bounds have squares, and sums of four
# squares, that neither overflow nor become subnormal, so sqrt(abs2) is as
# accurate as hypot (both about one ulp). Inf, NaN and extreme magnitudes
# take the scaled path.
_rotg_safe(::Type{Float64})=(0x1p-450,0x1p450)
_rotg_safe(::Type{Float32})=(Float32(0x1p-50),Float32(0x1p50))
function rotg(a::T,b::T) where {T<:Union{ComplexF32,ComplexF64}}
    R=real(T);lo,hi=_rotg_safe(R)
    iszero(b) && return (one(R),zero(T),a)
    ar,ai,br,bi=abs(real(a)),abs(imag(a)),abs(real(b)),abs(imag(b))
    small(v)=ifelse(iszero(v),hi,v)
    (max(ar,ai,br,bi)<hi && min(small(ar),small(ai),small(br),small(bi))>lo) || return _rotg_scaled(a,b)
    sa=abs2(a);sb=abs2(b)
    iszero(sa) && (h=sqrt(sb);return (zero(R),conj(b)/h,T(h)))
    aa=sqrt(sa);h=sqrt(sa+sb);ih=Base.inv(h);phase=a*Base.inv(aa)
    (aa*ih,phase*conj(b)*ih,phase*h)
end
