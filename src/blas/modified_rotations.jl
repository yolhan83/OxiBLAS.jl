# SPDX-License-Identifier: Apache-2.0

"""
    RotmParams{T<:AbstractFloat}
    RotmParams(::Type{T}=Float64; flag=-2, h11=0, h21=0, h12=0, h22=0)
    RotmParams(p::AbstractVector)

Modified Givens parameters in BLAS order `(flag,h11,h21,h12,h22)`.
Flags -2/-1/0/1 select identity/full/unit-diagonal/implicit-off-diagonal forms.
Ignored fields need not be initialized. Compatible AbstractFloat scalars use their own arithmetic; native real floats retain SIMD.
"""
struct RotmParams{T<:AbstractFloat}
    flag::T
    h11::T
    h21::T
    h12::T
    h22::T
    function RotmParams(flag::T,h11::T,h21::T,h12::T,h22::T) where {T<:AbstractFloat}
        flag in (T(-2),-one(T),zero(T),one(T)) || throw(ArgumentError("ROTM flag must be -2, -1, 0 or 1"))
        new{T}(flag,h11,h21,h12,h22)
    end
end
RotmParams(::Type{T}=Float64;flag=-2,h11=0,h21=0,h12=0,h22=0) where {T<:AbstractFloat} =
    RotmParams(T(flag),T(h11),T(h21),T(h12),T(h22))

function RotmParams(p::AbstractVector{T}) where {T<:AbstractFloat}
    Base.require_one_based_indexing(p)
    length(p)==5 || throw(DimensionMismatch("ROTM requires five parameters"))
    f=p[1]
    f==T(-2) && return RotmParams(T)
    f==-one(T) && return RotmParams(f,p[2],p[3],p[4],p[5])
    f==zero(T) && return RotmParams(f,one(T),p[3],p[4],one(T))
    f==one(T) && return RotmParams(f,p[2],-one(T),one(T),p[5])
    throw(ArgumentError("ROTM flag must be -2, -1, 0 or 1"))
end

_views_unique(::Tuple{})=true
_views_unique(t::Tuple)=!(first(t) isa AbstractVector && !allunique(first(t))) && _views_unique(Base.tail(t))
function _check_vector_output(x)
    if x isa StridedVector
        length(x)>1 && stride(x,1)==0 && throw(ArgumentError("output entries overlap"))
    elseif x isa SubArray
        # Recursion over the index tuple, not a loop: the tuple is
        # heterogeneous (ranges and integers) and a loop variable over it
        # was boxed (one allocation per Level 1 call on a view).
        _views_unique(parentindices(x)) || throw(ArgumentError("output view contains repeated indices"))
        _check_nd_output(parent(x))
    elseif x isa Union{Base.ReshapedArray,Base.PermutedDimsArray}
        _check_nd_output(x)
    end
    nothing
end

# Four independent SIMD packets amortize loop control. Both input packets are
# loaded before either output is stored; the public entry checks disjointness.
@generated function _rotm_simd!(x::Ptr{T},y::Ptr{T},n::Int,p::RotmParams{T},::Val{W},::Val{F}) where {T<:RealFloat,W,F}
    function packet(offset)
        a,b=gensym(:x),gensym(:y)
        nx=F==-1 ? :(_vfma(h11,$a,_vmul(h12,$b))) : F==0 ? :(_vfma(h12,$b,$a)) : :(_vfma(h11,$a,$b))
        ny=F==-1 ? :(_vfma(h21,$a,_vmul(h22,$b))) : F==0 ? :(_vfma(h21,$a,$b)) : :(_vfma(h22,$b,_vmul(negative,$a)))
        quote
            $a=_vload(x+(i+$offset)*sizeof(T),Val($W))
            $b=_vload(y+(i+$offset)*sizeof(T),Val($W))
            _vstore(x+(i+$offset)*sizeof(T),$nx)
            _vstore(y+(i+$offset)*sizeof(T),$ny)
        end
    end
    unrolled=Expr(:block,(packet(j*W) for j in 0:3)...)
    tail=packet(0)
    sx=F==-1 ? :(_arithmetic_madd(p.h11,a,p.h12*b)) : F==0 ? :(_arithmetic_madd(p.h12,b,a)) : :(_arithmetic_madd(p.h11,a,b))
    sy=F==-1 ? :(_arithmetic_madd(p.h21,a,p.h22*b)) : F==0 ? :(_arithmetic_madd(p.h21,a,b)) : :(_arithmetic_madd(p.h22,b,-a))
    quote
        h11=_splat(p.h11,Val($W)); h21=_splat(p.h21,Val($W))
        h12=_splat(p.h12,Val($W)); h22=_splat(p.h22,Val($W))
        negative=_splat(-one(T),Val($W))
        i=0
        while i+$(4W)<=n
            $unrolled
            i+=$(4W)
        end
        while i+$W<=n
            $tail
            i+=$W
        end
        for j in i+1:n
            a,b=unsafe_load(x,j),unsafe_load(y,j)
            unsafe_store!(x,$sx,j); unsafe_store!(y,$sy,j)
        end
        nothing
    end
end

"""
    rotm!(x,y,p::RotmParams)  # or five-element BLAS parameter vector

Apply the modified Givens matrix to pairs `(x[i],y[i])` in native Julia.
Negative/nonunit strides are supported. Nonidentity rotations require disjoint
outputs with distinct entries. Identity never reads or writes vector entries.
Parameter vectors are read before any mutation, including when they alias x/y.
"""
function rotm!(x::AbstractVector{T},y::AbstractVector{T},p::RotmParams{T};kernel=:auto) where {T<:AbstractFloat}
    _same_length(x,y)
    kind=_compact_kind(T,kernel)
    (p.flag==T(-2) || isempty(x)) && return (x,y)
    _check_vector_output(x); _check_vector_output(y)
    Base.mightalias(x,y) && throw(ArgumentError("rotation requires disjoint vectors"))
    if T<:RealFloat && kind!==:scalar && x isa StridedVector && y isa StridedVector && stride(x,1)==stride(y,1)==1
        w=first(_shape(T,kind))
        if length(x)>=w
            flag=p.flag==-one(T) ? -1 : iszero(p.flag) ? 0 : 1
            GC.@preserve x y _rotm_simd!(pointer(x),pointer(y),length(x),p,Val(w),Val(flag))
            return (x,y)
        end
    end
    if p.flag==-one(T)
        @inbounds @simd for i in eachindex(x)
            a,b=x[i],y[i]
            x[i]=_arithmetic_madd(p.h11,a,p.h12*b)
            y[i]=_arithmetic_madd(p.h21,a,p.h22*b)
        end
    elseif iszero(p.flag)
        @inbounds @simd for i in eachindex(x)
            a,b=x[i],y[i]
            x[i]=_arithmetic_madd(p.h12,b,a)
            y[i]=_arithmetic_madd(p.h21,a,b)
        end
    else
        @inbounds @simd for i in eachindex(x)
            a,b=x[i],y[i]
            x[i]=_arithmetic_madd(p.h11,a,b)
            y[i]=_arithmetic_madd(p.h22,b,-a)
        end
    end
    (x,y)
end
rotm!(x::AbstractVector{T},y::AbstractVector{T},p::AbstractVector{T};kernel=:auto) where {T<:AbstractFloat} = rotm!(x,y,RotmParams(p);kernel)

# A floating mantissa and an unbounded-for-this-algorithm integer exponent.
# This only extends exponent range, not precision, and uses no external backend.
# No product d*x or d*x*x must overflow/underflow before the branch/ratios.
@inline _rot_scaled(x::T) where T = frexp(x)
@inline function _rot_product(a,b)
    (iszero(a[1]) || iszero(b[1])) && return (zero(a[1]),0)
    m,e=frexp(a[1]*b[1])
    (m,a[2]+b[2]+e)
end
@inline function _rot_quotient(a,b)
    iszero(a[1]) && return (zero(a[1]),0)
    m,e=frexp(a[1]/b[1])
    (m,a[2]-b[2]+e)
end
@inline _rot_shift(a,e)=(a[1],a[2]+e)
@inline _rot_value(a)=ldexp(a[1],a[2])
@inline function _rot_abs_greater(a,b)
    iszero(a[1]) && return false
    iszero(b[1]) && return true
    a[2]>b[2] || (a[2]==b[2] && abs(a[1])>abs(b[1]))
end
@inline _rot_small(a)=a[2]<-23 || (a[2]==-23 && abs(a[1])<=0.5)
@inline _rot_large(a)=a[2]>=25

function _rot_zero(::Type{T}) where T
    z=zero(T)
    (d1=z,d2=z,x1=z,params=RotmParams(-one(T),z,z,z,z))
end

"""
    rotmg(d1,d2,x1,y1) -> (d1=...,d2=...,x1=...,params=RotmParams(...))

Generate a real modified Givens transformation. Powers of 4096 normalize the
updated weights. Scaled exponent arithmetic protects intermediate products and
repeated rescaling preserves all coefficients. Inputs must be finite. Negative
d1 or an inadmissible indefinite form produces the standard zero transform.
An unrepresentable finite-type result raises OverflowError; no scaling loop
can stall on infinity. Identity keeps the inputs unchanged.
"""
function _rotmg_scaled(d1::T,d2::T,x1::T,y1::T) where {T<:AbstractFloat}
    all(isfinite,(d1,d2,x1,y1)) || throw(ArgumentError("ROTMG requires finite inputs"))
    d1<zero(T) && return _rot_zero(T)
    (iszero(d2) || iszero(y1)) && return (d1=d1,d2=d2,x1=x1,params=RotmParams(T))
    dd1,dd2,xx,yy=_rot_scaled(d1),_rot_scaled(d2),_rot_scaled(x1),_rot_scaled(y1)
    p1,p2=_rot_product(dd1,xx),_rot_product(dd2,yy)
    q1,q2=_rot_product(p1,xx),_rot_product(p2,yy)
    o=_rot_scaled(one(T)); neg=_rot_scaled(-one(T))
    if _rot_abs_greater(q1,q2)
        h21=_rot_quotient((-yy[1],yy[2]),xx)
        h12=_rot_quotient(p2,p1)
        # q2/q1 is bounded by one; unlike h12*h21 it has safe intermediates.
        u=one(T)+_rot_value(_rot_quotient(q2,q1))
        u>zero(T) || return _rot_zero(T)
        uu=_rot_scaled(u)
        dd1=_rot_quotient(dd1,uu); dd2=_rot_quotient(dd2,uu)
        xx=_rot_product(xx,uu)
        flag=zero(T); h11=o; h22=o
    else
        d2<zero(T) && return _rot_zero(T)
        h11=_rot_quotient(p1,p2); h22=_rot_quotient(xx,yy)
        u=one(T)+_rot_value(_rot_quotient(q1,q2))
        uu=_rot_scaled(u)
        dd1,dd2=_rot_quotient(dd2,uu),_rot_quotient(dd1,uu)
        xx=_rot_product(yy,uu)
        flag=one(T); h12=o; h21=neg
    end
    # Coefficients are already explicit. Never restore implicit coefficients
    # after the first rescaling (an error in the pinned Rust implementation).
    if !iszero(dd1[1])
        while _rot_small(dd1) || _rot_large(dd1)
            e=_rot_small(dd1) ? -12 : 12
            dd1=_rot_shift(dd1,-2e); xx=_rot_shift(xx,e)
            h11=_rot_shift(h11,e); h12=_rot_shift(h12,e)
            flag=-one(T)
        end
    end
    if !iszero(dd2[1])
        while _rot_small(dd2) || _rot_large(dd2)
            e=_rot_small(dd2) ? -12 : 12
            dd2=_rot_shift(dd2,-2e)
            h21=_rot_shift(h21,e); h22=_rot_shift(h22,e)
            flag=-one(T)
        end
    end
    a,b,c=_rot_value(dd1),_rot_value(dd2),_rot_value(xx)
    h11v,h21v,h12v,h22v=_rot_value(h11),_rot_value(h21),_rot_value(h12),_rot_value(h22)
    values=(a,b,c,h11v,h21v,h12v,h22v)
    scaled=(dd1,dd2,xx,h11,h21,h12,h22)
    all(isfinite,values) && all(!iszero(s[1]) ? !iszero(v) : true for (s,v) in zip(scaled,values)) ||
        throw(OverflowError("ROTMG result exceeds the exponent range of $T"))
    (d1=a,d2=b,x1=c,params=RotmParams(flag,h11v,h21v,h12v,h22v))
end

rotmg(d1::T,d2::T,x1::T,y1::T) where {T<:AbstractFloat}=_rotmg_scaled(d1,d2,x1,y1)

"""
    rotmg!(state, y1) -> RotmParams

`rotmg!(state,y1)` updates `[d1,d2,x1]` after successful generation and returns parameters.
"""
function rotmg!(state::AbstractVector{T},y1::T) where {T<:AbstractFloat}
    Base.require_one_based_indexing(state)
    length(state)==3 || throw(DimensionMismatch("ROTMG state must contain d1,d2,x1"))
    _check_vector_output(state)
    result=rotmg(state[1],state[2],state[3],y1)
    state[1]=result.d1; state[2]=result.d2; state[3]=result.x1
    result.params
end

# For dense native vectors, let LLVM vectorize the existing portable loop.
# Explicit kernel requests and all other layouts retain the original dispatch.
function rotm!(x::Vector{T},y::Vector{T},p::RotmParams{T};kernel=:auto) where {T<:RealFloat}
    invoke(rotm!,Tuple{AbstractVector{T},AbstractVector{T},RotmParams{T}},x,y,p;
           kernel=kernel===:auto ? :scalar : kernel)
end

# On native reals, direct arithmetic is safe inside this exponent envelope.
# Each input exponent is bounded by emax/8: products of three inputs, their
# ratios, an epsilon-sized u, and subsequent binary rescalings stay normal.
# Outside the envelope the original exponent-tracked implementation is used.
function _rotmg_native(d1::T,d2::T,x1::T,y1::T) where {T<:RealFloat}
    limit=ldexp(one(T),exponent(floatmax(T))÷8); low=Base.inv(limit)
    if !all(v->low<=abs(v)<=limit,(d1,d2,x1,y1))
        return _rotmg_scaled(d1,d2,x1,y1)::NamedTuple{(:d1,:d2,:x1,:params),Tuple{T,T,T,RotmParams{T}}}
    end
    d1<zero(T) && return _rot_zero(T)
    p1=d1*x1;p2=d2*y1;q1=p1*x1;q2=p2*y1
    if abs(q1)>abs(q2)
        h21=-y1/x1;h12=p2/p1;u=one(T)+q2/q1
        u>zero(T) || return _rot_zero(T)
        d1/=u;d2/=u;x1*=u
        flag=zero(T);h11=one(T);h22=one(T)
    else
        d2<zero(T) && return _rot_zero(T)
        h11=p1/p2;h22=x1/y1;u=one(T)+q1/q2
        d1,d2=d2/u,d1/u;x1=y1*u
        flag=one(T);h12=one(T);h21=-one(T)
    end
    gamma=ldexp(one(T),12);square=ldexp(one(T),24);inverse_square=ldexp(one(T),-24)
    while !iszero(d1) && (abs(d1)<=inverse_square || abs(d1)>=square)
        if abs(d1)<=inverse_square
            d1*=square;x1/=gamma;h11/=gamma;h12/=gamma
        else
            d1/=square;x1*=gamma;h11*=gamma;h12*=gamma
        end
        flag=-one(T)
    end
    while !iszero(d2) && (abs(d2)<=inverse_square || abs(d2)>=square)
        if abs(d2)<=inverse_square
            d2*=square;h21/=gamma;h22/=gamma
        else
            d2/=square;h21*=gamma;h22*=gamma
        end
        flag=-one(T)
    end
    (d1=d1,d2=d2,x1=x1,params=RotmParams(flag,h11,h21,h12,h22))
end

# Concrete public signatures avoid diagonal-type ambiguities with AbstractFloat.
for T in (Float32,Float64)
    @eval rotmg(d1::$T,d2::$T,x1::$T,y1::$T)=_rotmg_native(d1,d2,x1,y1)
end
