# SPDX-License-Identifier: Apache-2.0
# Work directly on stored columns. No logical dense matrix traversal.
@inline _compact_madd(a,b,c)=a*b+c
@inline _compact_madd(a::T,b::T,c::T) where {T<:RealFloat}=muladd(a,b,c)
@inline _compact_base(P::PackedMatrix,j)=P.upper ? j*(j-1)÷2 : (j-1)*(2P.n-j)÷2
@inline _compact_column(P::PackedMatrix,j)=(P.data,_compact_base(P,j))
@inline _compact_column(P::BandedMatrix,j)=(view(P.data,:,j),P.ku+1-j)
@inline _compact_bounds(P::PackedMatrix,j)=(P.upper ? 1 : j,P.upper ? j : P.n)
@inline _compact_bounds(P::BandedMatrix,j)=(max(1,j-P.ku),min(P.m,j+P.kl))

# Used only when compact coefficients overlap a mutable vector. Copy just
# valid entries, excluding unused band padding and implicit unit diagonals.
function _compact_copy(P::PackedMatrix{T}) where T
    Q=PackedMatrix(Vector{T}(undef,length(P.data)),P.n;uplo=P.upper ? :U : :L,kind=P.kind,unit=P.unit)
    @inbounds for j in 1:P.n
        a,base=_compact_column(P,j); lo,hi=_compact_bounds(P,j)
        for i in lo:hi
            Q.data[base+i]=P.kind===:triangular && P.unit && i==j ? one(T) : a[base+i]
        end
    end
    Q
end
function _compact_copy(P::BandedMatrix{T}) where T
    rows=P.kind===:general ? P.kl+P.ku+1 : max(P.kl,P.ku)+1
    Q=BandedMatrix(zeros(T,rows,size(P,2)),P.m;kl=P.kl,ku=P.ku,kind=P.kind,uplo=P.upper ? :U : :L,unit=P.unit)
    @inbounds for j in 1:size(P,2)
        lo,hi=_compact_bounds(P,j)
        for i in lo:hi
            Q.data[P.ku+1+i-j,j]=P.kind===:triangular && P.unit && i==j ? one(T) : P.data[P.ku+1+i-j,j]
        end
    end
    Q
end

# Real unit-stride column updates reuse the checked ISA's LLVM primitives.
# Roots of both arrays are preserved at the pointer boundary; tails are scalar.
@generated function _compact_axpy_ptr!(y::Ptr{T},a::Ptr{T},n::Int,c::T,::Val{W}) where {T<:RealFloat,W}
    function packet(offset)
        quote
            av=_vload(a+(i+$offset)*sizeof(T),Val($W))
            yv=_vload(y+(i+$offset)*sizeof(T),Val($W))
            _vstore(y+(i+$offset)*sizeof(T),_vfma(cv,av,yv))
        end
    end
    steps=Expr(:block,(packet(j*W) for j in 0:3)...)
    tail=packet(0)
    quote
        cv=_splat(c,Val($W)); i=0
        while i+$(4W)<=n
            $steps
            i+=$(4W)
        end
        while i+$W<=n
            $tail
            i+=$W
        end
        for j in i+1:n
            unsafe_store!(y,muladd(c,unsafe_load(a,j),unsafe_load(y,j)),j)
        end
        nothing
    end
end
# Keep widths in the type domain. Val(runtime_integer) would force dynamic
# dispatch and box pointers/scalars once per stored column.
@inline function _compact_vector_width(::Type{T},kind) where T
    kind===:neon && return T===Float64 ? Val(2) : Val(4)
    kind===:avx2 && return T===Float64 ? Val(4) : Val(8)
    T===Float64 ? Val(8) : Val(16)
end
@inline _compact_lanes(::Val{W}) where W=W

@inline function _compact_axpy_span!(y::AbstractVector{T},a::AbstractVector{T},iy,ia,n,c::T,kind) where T
    n<=0 && return nothing
    if T<:NativeFloat && kind!==:scalar && y isa StridedVector && a isa StridedVector && stride(y,1)==stride(a,1)==1
        width=_compact_vector_width(real(T),kind)
        if n>=(T<:Complex ? _compact_lanes(width)÷2 : _compact_lanes(width))
            GC.@preserve y a _compact_axpy_ptr!(pointer(y,iy),pointer(a,ia),n,c,width)
            return nothing
        end
    end
    @inbounds @simd for p in 0:n-1
        y[iy+p]=_compact_madd(c,a[ia+p],y[iy+p])
    end
    nothing
end

function _gbmv_direct!(y,A,x,α,rows,n)
    D=A.data;ku=A.ku;kl=A.kl
    if kl+ku<=2
        # Narrow bands by diagonals: one long SIMD loop per stored diagonal
        # instead of one 1-3 element loop per column (measured 3-5x at
        # kl=ku=1; slower for wide bands, whose rows of D are far apart).
        @inbounds for d in -kl:ku
            r=ku+1-d;lo=max(first(rows),1-d);hi=min(last(rows),n-d)
            @simd for i in lo:hi
                y[i]=muladd(D[r,i+d],α*x[i+d],y[i])
            end
        end
        return nothing
    end
    @inbounds for j in max(1,first(rows)-kl):min(n,last(rows)+ku)
        lo=max(first(rows),j-ku,1);hi=min(last(rows),j+kl,A.m)
        c=α*x[j];r=ku+1-j
        @simd for i in lo:hi
            y[i]=muladd(D[r+i,j],c,y[i])
        end
    end
    nothing
end
function gbmv!(y::AbstractVector{T},P::BandedMatrix{T},x::AbstractVector{T},α0=one(T),β0=zero(T);trans=:N,kernel=:auto,threaded=true,parallel_threshold=nothing) where {T<:GenericFloat}
    threaded=threaded && generic_threadsafe(T)
    _pkcheck(P,:general); tr=_trans_kind(trans); kind=_compact_kind(T,kernel)
    Base.require_one_based_indexing(x,y); _check_vector_output(y)
    m,n=size(P)
    length(x)==(tr==0 ? n : m) && length(y)==(tr==0 ? m : n) || throw(DimensionMismatch("GBMV dimensions differ"))
    α,β=convert(T,α0),convert(T,β0)
    (iszero(α) || m==0 || n==0) && return _scale!(y,β)
    A=Base.mightalias(P,y) ? _compact_copy(P) : P
    xx=Base.mightalias(x,y) ? copy(x) : x
    work=Float64(n)*(A.kl+A.ku+1)
    if tr==0
        _scale!(y,β)
        workers=_l2_workers(T,m,work,sizeof(T)*work,threaded,parallel_threshold)
        _l1_foreach(m,workers) do rows,_
            isempty(rows) && return nothing
            if kind!==:scalar && (T<:Union{Float32,Float64} || (T<:NativeFloat && A.kl+A.ku<=2))
                # Real columns index the band storage directly: the per-column
                # view and span kernel cost dominated short spans (measured
                # 1.1-1.3x at kl=ku=1, equal at kl=ku=32).
                _gbmv_direct!(y,A,xx,α,rows,n)
                return nothing
            end
            @inbounds for j in max(1,first(rows)-A.kl):min(n,last(rows)+A.ku)
                lo,hi=_compact_bounds(A,j); lo=max(lo,first(rows)); hi=min(hi,last(rows))
                lo>hi && continue
                a,base=_compact_column(A,j)
                _compact_axpy_span!(y,a,lo,base+lo,hi-lo+1,α*xx[j],kind)
            end
        end
    else
        workers=_l2_workers(T,n,work,sizeof(T)*work,threaded,parallel_threshold)
        _l1_foreach(n,workers) do columns,_
            @inbounds for j in columns
                lo,hi=_compact_bounds(A,j); a,base=_compact_column(A,j)
                s=_compact_fused_span!(y,a,xx,1,base+lo,lo,hi-lo+1,zero(T),kind,Val(tr==2),Val(false))
                y[j]=iszero(β) ? α*s : α*s+β*y[j]
            end
        end
    end
    y
end

function _compact_symmv!(y::AbstractVector{T},P,x::AbstractVector{T},α0,β0,::Val{H};kernel=:auto,threaded=true,parallel_threshold=nothing) where {T<:GenericFloat,H}
    threaded=threaded && generic_threadsafe(T)
    _pkcheck(P,H ? :hermitian : :symmetric)
    eltype(P)===T || throw(ArgumentError("compact storage and vector element types must agree"))
    Base.require_one_based_indexing(x,y); _check_vector_output(y)
    n=size(P,1)
    length(y)==length(x)==n || throw(DimensionMismatch("compact symmetric multiply dimensions differ"))
    kind=_compact_kind(T,kernel)
    k=P isa BandedMatrix ? min(n,max(P.kl,P.ku)) : n
    workers=_l2_workers(T,n,Float64(n)*(2k+1),_symmetric_cost(T)*(P isa BandedMatrix ? Float64(n)*(k+1) : Float64(n)*(n+1)/2),threaded,parallel_threshold)
    α,β=convert(T,α0),convert(T,β0)
    (iszero(α) || n==0) && return _scale!(y,β)
    A=Base.mightalias(P,y) ? _compact_copy(P) : P
    xx=Base.mightalias(x,y) ? copy(x) : x
    workers==1 && return _compact_symmv_into!(_scale!(y,β),A,xx,α,kind,Val(H),1:n)
    # Column ranges of the stored triangle per worker (see _symmetric_parallel!).
    # Reading A[i,j] element by element per output row was 4x slower than one
    # thread.
    _symmetric_parallel!(y,β,n,k,A.upper,workers) do t,cols
        _compact_symmv_into!(t,A,xx,α,kind,Val(H),cols)
    end
end
# Adds the stored columns cols of the compact symmetric product into t.
function _compact_symmv_into!(t,A,xx,α,kind,::Val{H},cols) where H
    T=eltype(t)
    @inbounds for j in cols
        a,base=_compact_column(A,j); lo,hi=_compact_bounds(A,j)
        lo,hi=A.upper ? (lo,j-1) : (j+1,hi)
        c=α*xx[j]
        s=_compact_fused_span!(t,a,xx,lo,base+lo,lo,hi-lo+1,c,kind,Val(H),Val(true))
        diag=H ? real(a[base+j]) : a[base+j]
        t[j]=_compact_madd(c,diag,t[j])+α*s
    end
    t
end
spmv!(y,P::PackedMatrix,x,α=one(eltype(y)),β=zero(eltype(y));kwargs...)=_compact_symmv!(y,P,x,α,β,Val(false);kwargs...)
hpmv!(y,P::PackedMatrix,x,α=one(eltype(y)),β=zero(eltype(y));kwargs...)=_compact_symmv!(y,P,x,α,β,Val(true);kwargs...)
sbmv!(y,P::BandedMatrix,x,α=one(eltype(y)),β=zero(eltype(y));kwargs...)=_compact_symmv!(y,P,x,α,β,Val(false);kwargs...)
hbmv!(y,P::BandedMatrix,x,α=one(eltype(y)),β=zero(eltype(y));kwargs...)=_compact_symmv!(y,P,x,α,β,Val(true);kwargs...)

# Narrow triangular bands (at most two off-diagonals), no transpose: by rows,
# reading A[i,j]=data[ku+1+i-j,j] directly. Upper multiplication ascends and
# lower descends, so row i only reads entries not yet overwritten; the per-
# column view/span calls dominated these 1-3 element columns. Solves keep
# the sequential recurrence.
function _compact_tri_narrow!(x,A,solve)
    n=size(A,1);D=A.data;ku=A.ku;k=A.upper ? A.ku : A.kl;unit=A.unit;M=_arithmetic_madd
    a(i,j)=@inbounds D[ku+1+i-j,j]
    if !solve
        # unit is a type parameter: a runtime select between complex values
        # packs ComplexF32 into an i64 inside the loop.
        return unit ? _compact_tri_narrow_mv!(x,A,k,Val(true)) : _compact_tri_narrow_mv!(x,A,k,Val(false))
    else
        # The last two solved entries are carried in registers, as separate
        # real and imaginary parts: a carried ComplexF32 is otherwise packed
        # into an i64 and crosses between integer and FP registers every step
        # (measured 3x); reloading x[j] after its store also stalled.
        R=real(eltype(x));p1r=p1i=p2r=p2i=zero(R);step=A.upper ? -1 : 1
        @inbounds for (t,i) in enumerate(A.upper ? (n:-1:1) : (1:n))
            z=x[i];sr=real(z);si=imag(z)
            t>1 && k>=1 && ((sr,si)=_narrow_sub(sr,si,a(i,i-step),p1r,p1i))
            t>2 && k>=2 && ((sr,si)=_narrow_sub(sr,si,a(i,i-2step),p2r,p2i))
            vr,vi=unit ? (sr,si) : _narrow_div_parts(_narrow_join(eltype(x),sr,si),a(i,i))
            x[i]=_narrow_join(eltype(x),vr,vi);p2r=p1r;p2i=p1i;p1r=vr;p1i=vi
        end
    end
    x
end
function _compact_tri_narrow_mv!(x,A,k,::Val{unit}) where unit
    n=size(A,1);D=A.data;ku=A.ku;M=_arithmetic_madd
    a(i,j)=@inbounds D[ku+1+i-j,j]
    if A.upper
        @inbounds for i in 1:n
            s=unit ? x[i] : a(i,i)*x[i]
            for d in 1:min(k,n-i)
                s=M(a(i,i+d),x[i+d],s)
            end
            x[i]=s
        end
    else
        @inbounds for i in n:-1:1
            s=unit ? x[i] : a(i,i)*x[i]
            for d in 1:min(k,i-1)
                s=M(a(i,i-d),x[i-d],s)
            end
            x[i]=s
        end
    end
    x
end
@inline _narrow_join(::Type{T},r,i) where {T<:Real}=r
@inline _narrow_join(::Type{T},r,i) where {T<:Complex}=T(r,i)
# s-a*p on separate real and imaginary parts.
@inline _narrow_sub(sr,si,a::Real,pr,pi)=(muladd(-a,pr,sr),si)
@inline _narrow_sub(sr,si,a::Complex,pr,pi)=(muladd(imag(a),pi,muladd(-real(a),pr,sr)),muladd(-imag(a),pr,muladd(-real(a),pi,si)))
# The robust complex division dominated the narrow complex solve recurrence.
# Within the bounds below, s*(conj(a)/abs2(a)) neither overflows nor loses
# precision to underflow (about two ulp); other divisors keep s/a.
# Parts are returned as two reals: a merged ComplexF32 result is packed in
# an i64 and round-trips through the stack on the recurrence's critical path.
@inline _narrow_div_parts(s::Real,a::Real)=(s/a,zero(s))
@inline function _narrow_div_parts(s::Complex{R},a::Complex{R}) where {R<:Union{Float32,Float64}}
    lo,hi=R===Float64 ? (0x1p-450,0x1p450) : (Float32(0x1p-50),Float32(0x1p50))
    m=max(abs(real(a)),abs(imag(a)))
    if lo<m<hi
        q=s*(conj(a)/abs2(a))
        return (real(q),imag(q))
    end
    q=s/a
    (real(q),imag(q))
end
function _compact_tri!(x::AbstractVector{T},P,trans,solve,kernel) where {T<:GenericFloat}
    _pkcheck(P,:triangular); tr=_trans_kind(trans); kind=_compact_kind(T,kernel)
    eltype(P)===T || throw(ArgumentError("compact storage and vector element types must agree"))
    Base.require_one_based_indexing(x); _check_vector_output(x)
    n=size(P,1); length(x)==n || throw(DimensionMismatch("compact triangular dimensions differ"))
    n==0 && return x
    A=Base.mightalias(P,x) ? _compact_copy(P) : P
    if solve && !A.unit
        @inbounds for j in 1:n
            a,base=_compact_column(A,j)
            iszero(a[base+j]) && throw(SingularException(j))
        end
    end
    if tr==0 && A isa BandedMatrix && T<:NativeFloat && kind!==:scalar && (A.upper ? A.ku : A.kl)<=2
        return _compact_tri_narrow!(x,A,solve)
    end
    # Multiplication and substitution have opposite traversal dependencies.
    forward=(A.upper == !solve) == (tr==0)
    @inbounds for j in (forward ? (1:n) : (n:-1:1))
        a,base=_compact_column(A,j); lo,hi=_compact_bounds(A,j)
        lo,hi=A.upper ? (lo,j-1) : (j+1,hi)
        if tr==0
            v=x[j]
            if solve
                v=A.unit ? v : v/a[base+j]
                x[j]=v
                _compact_axpy_span!(x,a,lo,base+lo,hi-lo+1,-v,kind)
            else
                _compact_axpy_span!(x,a,lo,base+lo,hi-lo+1,v,kind)
                x[j]=A.unit ? v : a[base+j]*v
            end
        else
            v=solve || A.unit ? x[j] : (tr==2 ? conj(a[base+j]) : a[base+j])*x[j]
            for i in lo:hi
                z=a[base+i]; z=tr==2 ? conj(z) : z
                v=_compact_madd(solve ? -z : z,x[i],v)
            end
            if solve && !A.unit
                v/=tr==2 ? conj(a[base+j]) : a[base+j]
            end
            x[j]=v
        end
    end
    x
end
tpmv!(x,P::PackedMatrix;trans=:N,kernel=:auto)=_compact_tri!(x,P,trans,false,kernel)
tpsv!(x,P::PackedMatrix;trans=:N,kernel=:auto)=_compact_tri!(x,P,trans,true,kernel)
tbmv!(x,P::BandedMatrix;trans=:N,kernel=:auto)=_compact_tri!(x,P,trans,false,kernel)
tbsv!(x,P::BandedMatrix;trans=:N,kernel=:auto)=_compact_tri!(x,P,trans,true,kernel)

function _compact_rank!(P::PackedMatrix{T},x::AbstractVector{T},y::AbstractVector{T},α,::Val{H},::Val{R};kernel=:auto) where {T<:GenericFloat,H,R}
    _pkcheck(P,H ? :hermitian : :symmetric); kind=_compact_kind(T,kernel)
    Base.require_one_based_indexing(x,y); _check_vector_output(P.data)
    n=P.n; length(x)==length(y)==n || throw(DimensionMismatch("compact rank-update dimensions differ"))
    H && !R && !isreal(α) && throw(ArgumentError("HPR alpha must be real"))
    α=convert(T,α)
    if iszero(α)
        if H
            @inbounds for j in 1:n
                d=_compact_base(P,j)+j
                P.data[d]=real(P.data[d])
            end
        end
        return P
    end
    xx=Base.mightalias(P,x) ? copy(x) : x
    yy=x===y ? xx : (Base.mightalias(P,y) ? copy(y) : y)
    @inbounds for j in 1:n
        base=_compact_base(P,j); lo,hi=_compact_bounds(P,j)
        c=α*(H ? conj(R ? yy[j] : xx[j]) : (R ? yy[j] : xx[j]))
        if R
            d=(H ? conj(α) : α)*(H ? conj(xx[j]) : xx[j])
            if T<:NativeFloat && kind!==:scalar && xx isa StridedVector && yy isa StridedVector
                # Both rank-1 terms in one pass over the packed column (two
                # separate AXPYs read and write it twice).
                D=P.data;off=base
                @simd for i in lo:hi
                    D[off+i]=_arithmetic_madd(yy[i],d,_arithmetic_madd(xx[i],c,D[off+i]))
                end
            else
                _compact_axpy_span!(P.data,xx,base+lo,lo,hi-lo+1,c,kind)
                _compact_axpy_span!(P.data,yy,base+lo,lo,hi-lo+1,d,kind)
            end
        else
            _compact_axpy_span!(P.data,xx,base+lo,lo,hi-lo+1,c,kind)
        end
        H && (P.data[base+j]=real(P.data[base+j]))
    end
    P
end
spr!(P::PackedMatrix,x,α=one(eltype(P));kwargs...)=_compact_rank!(P,x,x,α,Val(false),Val(false);kwargs...)
spr2!(P::PackedMatrix,x,y,α=one(eltype(P));kwargs...)=_compact_rank!(P,x,y,α,Val(false),Val(true);kwargs...)
hpr!(P::PackedMatrix,x,α=one(real(eltype(P)));kwargs...)=_compact_rank!(P,x,x,α,Val(true),Val(false);kwargs...)
hpr2!(P::PackedMatrix,x,y,α=one(eltype(P));kwargs...)=_compact_rank!(P,x,y,α,Val(true),Val(true);kwargs...)

function _compact_axpy_ptr!(y::Ptr{Complex{R}},a::Ptr{Complex{R}},n::Int,c::Complex{R},::Val{W}) where {R<:RealFloat,W}
    q=W÷2; i=0
    while i+q<=n
        p=i*sizeof(Complex{R})
        av=_vload(Ptr{R}(a+p),Val(W)); yv=_vload(Ptr{R}(y+p),Val(W))
        _vstore(Ptr{R}(y+p),_vadd(yv,_complex_vmul(av,c,Val(W))))
        i+=q
    end
    for j in i+1:n
        unsafe_store!(y,c*unsafe_load(a,j)+unsafe_load(y,j),j)
    end
    nothing
end

# Fuses y += c*a with dot(a,x) or dot(conj(a),x) in a single stored-column pass.
# Val(false) in Update removes all reads and writes through y for pure dots.
function _compact_reduce_ptr!(y::Ptr{T},a::Ptr{T},x::Ptr{T},n,c::T,::Val{H},::Val{W},::Val{Update}) where {T<:NativeFloat,H,W,Update}
    R=real(T); q=T<:Complex ? W÷2 : W
    acc=_splat(zero(R),Val(W)); i=0
    while i+q<=n
        off=i*sizeof(T)
        av=_vload(Ptr{R}(a+off),Val(W)); xv=_vload(Ptr{R}(x+off),Val(W))
        acc=T<:Complex ? _vadd(acc,_complex_pairmul(av,xv,Val(H))) : _vfma(av,xv,acc)
        if Update
            yv=_vload(Ptr{R}(y+off),Val(W))
            v=T<:Complex ? _vadd(yv,_complex_vmul(av,c,Val(W))) : _vfma(_splat(c,Val(W)),av,yv)
            _vstore(Ptr{R}(y+off),v)
        end
        i+=q
    end
    s=zero(T)
    if T<:Complex
        for lane in 1:2:W
            s+=complex(acc[lane].value,acc[lane+1].value)
        end
    else
        for lane in 1:W
            s+=acc[lane].value
        end
    end
    for j in i+1:n
        v=unsafe_load(a,j)
        s=_compact_madd(H ? conj(v) : v,unsafe_load(x,j),s)
        Update && unsafe_store!(y,_compact_madd(c,v,unsafe_load(y,j)),j)
    end
    s
end

@inline function _compact_fused_simd!(y,a,x,iy,ia,ix,n,c::T,::Val{H},::Val{Update}) where {T,H,Update}
    s=zero(T)
    @inbounds @simd for p in 0:n-1
        v=a[ia+p]
        s=_arithmetic_madd(H ? conj(v) : v,x[ix+p],s)
        Update && (y[iy+p]=_arithmetic_madd(v,c,y[iy+p]))
    end
    s
end
@inline function _compact_fused_simd!(y,a::StridedVector{T},x::StridedVector{T},iy,ia,ix,n,c::T,::Val{H},::Val{Update}) where {T<:Union{ComplexF32,ComplexF64},H,Update}
    R=real(T);rr=zero(R);ii=zero(R);ri=zero(R);ir=zero(R)
    @inbounds @simd for p in 0:n-1
        v=a[ia+p];b=x[ix+p]
        rr=muladd(real(v),real(b),rr);ii=muladd(imag(v),imag(b),ii)
        ri=muladd(real(v),imag(b),ri);ir=muladd(imag(v),real(b),ir)
        Update && (y[iy+p]=_arithmetic_madd(v,c,y[iy+p]))
    end
    H ? complex(rr+ii,ri-ir) : complex(rr-ii,ri+ir)
end
@inline function _compact_fused_span!(y,a,x,iy,ia,ix,n,c::T,kind,::Val{H},::Val{Update}) where {T,H,Update}
    n<=0 && return zero(T)
    if T<:NativeFloat && kind!==:scalar && a isa StridedVector{T} && x isa StridedVector{T} && (!Update || y isa StridedVector{T})
        # Plain SIMD loop: LLVM interleaves several partial sums, whereas the
        # pointer kernel below carries one accumulator vector and is bound by
        # FMA latency (measured 1.5-1.7x on spmv Float64).
        return _compact_fused_simd!(y,a,x,iy,ia,ix,n,c,Val(H),Val(Update))
    end
    if T<:NativeFloat && kind!==:scalar && a isa StridedVector{T} && x isa StridedVector{T} && stride(a,1)==stride(x,1)==1 &&
       (!Update || (y isa StridedVector{T} && stride(y,1)==1))
        width=_compact_vector_width(real(T),kind)
        if n>=(T<:Complex ? _compact_lanes(width)÷2 : _compact_lanes(width))
            GC.@preserve y a x begin
                yp=Update ? pointer(y,iy) : Ptr{T}(0)
                return _compact_reduce_ptr!(yp,pointer(a,ia),pointer(x,ix),n,c,Val(H),width,Val(Update))
            end
        end
    end
    s=zero(T)
    @inbounds for p in 0:n-1
        v=a[ia+p]
        s=_compact_madd(H ? conj(v) : v,x[ix+p],s)
        Update && (y[iy+p]=_compact_madd(c,v,y[iy+p]))
    end
    s
end

function _compact_kind(::Type{T},kernel) where T
    isconcretetype(T) || throw(ArgumentError("a concrete compact scalar type is required"))
    T<:NativeFloat && return _kernel_kind(kernel)
    kernel in (:auto,:scalar) || throw(ArgumentError("extended compact BLAS uses scalar arithmetic"))
    :scalar
end
