# SPDX-License-Identifier: Apache-2.0
# Preserve generic arithmetic and dependency-safe scalar leaves. Native leaves
# reuse each A coefficient across four RHSs or across contiguous LLVM lanes.
_trmm_leaf!(B,A,alpha,left,upper,unit;kernel=:auto,optimized=true)=_trmm_leaf_generic!(B,A,alpha,left,upper,unit)
_trsm_leaf!(B,A,left,upper,unit;kernel=:auto,optimized=true)=_trsm_leaf_generic!(B,A,left,upper,unit)

"""Reusable native TRMM/TRSM packing context for an m-by-n output. Construct
once for a fixed side/configuration; pass as `workspace` with the same config.
The scalar/generic fallback requires no packing context.
"""
function triangular_workspace(::Type{T},m::Integer,n::Integer;side=:L,config=GemmConfig(T),workers=Threads.nthreads()) where {T<:NativeFloat}
    min(m,n)>=0 && side in (:L,:R) || throw(ArgumentError("invalid triangular workspace shape/side"))
    order=side===:L ? m : n; half=cld(order,2)
    GemmWorkspace(T,side===:L ? half : m,side===:L ? n : half,half;config,workers)
end

function _tri_workspace_call(f,w,B::AbstractMatrix{T},A) where T
    w===nothing && return f()
    w isa GemmWorkspace{T} || throw(ArgumentError("triangular workspace scalar type differs"))
    lock(w.guard) do
        _workspace_operands_check(w,B,A)
        f()
    end
end

# A four-column register leaf also helps scalar CPU targets: all four updates
# reuse the same triangular coefficient. No reciprocal diagonal is introduced.
function _tri_left4!(B,A,alpha,upper,unit,solving)
    T=eltype(B); m,n=size(B); last=4*fld(n,4)
    rows=solving ? (upper ? (m:-1:1) : (1:m)) : (upper ? (1:m) : (m:-1:1))
    @inbounds for j in 1:4:last, i in rows
        if solving || unit
            a=B[i,j]; b=B[i,j+1]; c=B[i,j+2]; d=B[i,j+3]
        else
            diagonal=A[i,i]; a=diagonal*B[i,j]; b=diagonal*B[i,j+1]; c=diagonal*B[i,j+2]; d=diagonal*B[i,j+3]
        end
        for p in (upper ? (i+1:m) : (1:i-1))
            x=solving ? -A[i,p] : A[i,p]
            a=_native_madd(x,B[p,j],a); b=_native_madd(x,B[p,j+1],b)
            c=_native_madd(x,B[p,j+2],c); d=_native_madd(x,B[p,j+3],d)
        end
        if solving
            if !unit
                diagonal=A[i,i]; a/=diagonal; b/=diagonal; c/=diagonal; d/=diagonal
            end
        else
            a*=alpha; b*=alpha; c*=alpha; d*=alpha
        end
        B[i,j]=a; B[i,j+1]=b; B[i,j+2]=c; B[i,j+3]=d
    end
    if last<n
        out=view(B,:,last+1:n)
        solving ? _trsm_leaf_generic!(out,A,true,upper,unit) : _trmm_leaf_generic!(out,A,alpha,true,upper,unit)
    end
    B
end
# Left leaves of order M<=8 (the recursive leaves and small triangles), real
# types: the triangle's coefficients are hoisted out of the column loop and
# each column is fully unrolled, which LLVM vectorizes (order 8, 1024 RHS,
# Float64: TRMM 6.4 -> 2.1 us, TRSM 11 -> 3.7 us; OpenBLAS 2.8 and 12.2).
# Solves divide by the diagonal. Complex types keep _tri_left4!: the
# unrolled complex form gained little and slowed the ComplexF64 solve.
@generated function _tri_left_small!(B,A,alpha,::Val{M},::Val{U},::Val{UNIT},::Val{SOLVE}) where {M,U,UNIT,SOLVE}
    coef=Expr(:block); body=Expr(:block)
    a(r,i)=Symbol(:a_,r,:_,i); x(i)=Symbol(:x_,i)
    for i in 1:M, r in 1:M
        (U ? r<=i : r>=i) && !(r==i && UNIT) && push!(coef.args,:($(a(r,i))=A[$r,$i]))
    end
    if SOLVE
        for r in (U ? (M:-1:1) : (1:M))
            acc=:(B[$r,j])
            for i in (U ? (r+1:M) : (1:r-1))
                acc=:(muladd(-$(a(r,i)),$(x(i)),$acc))
            end
            push!(body.args,:($(x(r))=$(UNIT ? acc : :($acc/$(a(r,r))))),:(B[$r,j]=$(x(r))))
        end
    else
        for i in 1:M
            push!(body.args,:($(x(i))=alpha*B[$i,j]))
        end
        for r in 1:M
            acc=UNIT ? x(r) : :($(a(r,r))*$(x(r)))
            for i in (U ? (r+1:M) : (1:r-1))
                acc=:(muladd($(a(r,i)),$(x(i)),$acc))
            end
            push!(body.args,:(B[$r,j]=$acc))
        end
    end
    quote
        @inbounds begin
            $coef
            for j in axes(B,2)
                $body
            end
        end
        B
    end
end
# Right leaves of order N<=8, real types: the same hoisted coefficients, one
# fully unrolled row per iteration of a contiguous SIMD loop over the rows of
# B (order 8, 1024 rows, Float64: TRMM 5.8 -> 1.3 us, TRSM 4.7 -> 1.9 us;
# OpenBLAS 2.5 and 9.8). Complex types were slower this way and keep
# _tri_right_simd!.
@generated function _tri_right_small!(B,A,alpha,::Val{N},::Val{U},::Val{UNIT},::Val{SOLVE}) where {N,U,UNIT,SOLVE}
    coef=Expr(:block); body=Expr(:block)
    a(p,j)=Symbol(:a_,p,:_,j); x(p)=Symbol(:x_,p)
    for j in 1:N, p in 1:N
        (U ? p<=j : p>=j) && !(p==j && UNIT) && push!(coef.args,:($(a(p,j))=A[$p,$j]))
    end
    if SOLVE
        # X*A=B: upper ascends in j, lower descends.
        for j in (U ? (1:N) : (N:-1:1))
            acc=:(B[i,$j])
            for p in (U ? (1:j-1) : (j+1:N))
                acc=:(muladd(-$(x(p)),$(a(p,j)),$acc))
            end
            push!(body.args,:($(x(j))=$(UNIT ? acc : :($acc/$(a(j,j))))),:(B[i,$j]=$(x(j))))
        end
    else
        for p in 1:N
            push!(body.args,:($(x(p))=alpha*B[i,$p]))
        end
        for j in 1:N
            acc=UNIT ? x(j) : :($(x(j))*$(a(j,j)))
            for p in (U ? (1:j-1) : (j+1:N))
                acc=:(muladd($(x(p)),$(a(p,j)),$acc))
            end
            push!(body.args,:(B[i,$j]=$acc))
        end
    end
    quote
        @inbounds begin
            $coef
            @simd for i in axes(B,1)
                $body
            end
        end
        B
    end
end
@inline _tri_vload(p::Ptr{T},::Val{W}) where {T<:RealFloat,W}=_vload(p,Val(W))
@inline _tri_vload(p::Ptr{T},::Val{W}) where {T<:Union{ComplexF32,ComplexF64},W}=_vload(Ptr{real(T)}(p),Val(2W))
@inline _tri_vstore(p::Ptr{T},x) where {T<:RealFloat}=_vstore(p,x)
@inline _tri_vstore(p::Ptr{T},x) where {T<:Union{ComplexF32,ComplexF64}}=_vstore(Ptr{real(T)}(p),x)
@inline _tri_vmul(x,a::T,::Val{W}) where {T<:RealFloat,W}=_vmul(x,_splat(a,Val(W)))
@inline _tri_vmul(x,a::T,::Val{W}) where {T<:Union{ComplexF32,ComplexF64},W}=_complex_vmul(x,a,Val(2W))
@inline _tri_vfma(x,a::T,y,::Val{W}) where {T<:RealFloat,W}=_vfma(x,_splat(a,Val(W)),y)
@inline _tri_vfma(x,a::T,y,::Val{W}) where {T<:Union{ComplexF32,ComplexF64},W}=
    _vfma(_swap_pairs(x),_alternating(imag(a),Val(2W)),_vfma(x,_splat(real(a),Val(2W)),y))

function _tri_right_simd!(B::StridedMatrix{T},A,alpha,upper,unit,solving,::Val{W}) where {T<:NativeFloat,W}
    m,n=size(B); ld=stride(B,2); vector_end=W*fld(m,W)
    # Every raw pointer is derived inside this preservation scope. Vector tails
    # never access padding; only natural component alignment is assumed.
    GC.@preserve B begin
        base=pointer(B)
        if solving
            for j in (upper ? (1:n) : (n:-1:1))
                # Complex division keeps Julia's stable scalar division, rather
                # than multiplying by a potentially overflowing reciprocal.
                if !unit
                    diagonal=A[j,j]
                    @inbounds for i in 1:m
                        B[i,j]/=diagonal
                    end
                end
                src=base+(j-1)*ld*sizeof(T)
                for p in (upper ? (j+1:n) : (1:j-1))
                    a=-A[j,p]; dst=base+(p-1)*ld*sizeof(T)
                    for i in 0:W:vector_end-W
                        x=_tri_vload(src+i*sizeof(T),Val(W)); y=_tri_vload(dst+i*sizeof(T),Val(W))
                        _tri_vstore(dst+i*sizeof(T),_tri_vfma(x,a,y,Val(W)))
                    end
                    @inbounds for i in vector_end+1:m
                        B[i,p]=_native_madd(B[i,j],a,B[i,p])
                    end
                end
            end
        else
            for j in (upper ? (n:-1:1) : (1:n))
                dst=base+(j-1)*ld*sizeof(T)
                for i in 0:W:vector_end-W
                    value=_tri_vload(dst+i*sizeof(T),Val(W))
                    unit || (value=_tri_vmul(value,A[j,j],Val(W)))
                    for p in (upper ? (1:j-1) : (j+1:n))
                        src=base+((p-1)*ld+i)*sizeof(T)
                        value=_tri_vfma(_tri_vload(src,Val(W)),A[p,j],value,Val(W))
                    end
                    _tri_vstore(dst+i*sizeof(T),_tri_vmul(value,alpha,Val(W)))
                end
                @inbounds for i in vector_end+1:m
                    s=unit ? B[i,j] : B[i,j]*A[j,j]
                    for p in (upper ? (1:j-1) : (j+1:n))
                        s=_native_madd(B[i,p],A[p,j],s)
                    end
                    B[i,j]=alpha*s
                end
            end
        end
    end
    B
end
function _trmm_leaf!(B::StridedMatrix{T},A,alpha,left,upper,unit;kernel=:auto,optimized=true) where {T<:NativeFloat}
    (!optimized || stride(B,1)!=1) && return _trmm_leaf_generic!(B,A,alpha,left,upper,unit)
    left && T<:RealFloat && size(B,1)<=8 && return _tri_left_small!(B,A,alpha,Val(size(B,1)),Val(upper),Val(unit),Val(false))
    !left && T<:RealFloat && size(B,2)<=8 && return _tri_right_small!(B,A,alpha,Val(size(B,2)),Val(upper),Val(unit),Val(false))
    left && size(B,2)>=4 && return _tri_left4!(B,A,alpha,upper,unit,false)
    kind=_kernel_kind(kernel); width=first(_shape(T,kind))
    !left && _can_pointer(B) && kind!==:scalar && size(B,1)>=width && return _tri_right_simd!(B,A,alpha,upper,unit,false,Val(width))
    _trmm_leaf_generic!(B,A,alpha,left,upper,unit)
end
function _trsm_leaf!(B::StridedMatrix{T},A,left,upper,unit;kernel=:auto,optimized=true) where {T<:NativeFloat}
    (!optimized || stride(B,1)!=1) && return _trsm_leaf_generic!(B,A,left,upper,unit)
    left && T<:RealFloat && size(B,1)<=8 && return _tri_left_small!(B,A,one(T),Val(size(B,1)),Val(upper),Val(unit),Val(true))
    !left && T<:RealFloat && size(B,2)<=8 && return _tri_right_small!(B,A,one(T),Val(size(B,2)),Val(upper),Val(unit),Val(true))
    left && size(B,2)>=4 && return _tri_left4!(B,A,one(T),upper,unit,true)
    kind=_kernel_kind(kernel); width=first(_shape(T,kind))
    !left && _can_pointer(B) && kind!==:scalar && size(B,1)>=width && return _tri_right_simd!(B,A,one(T),upper,unit,true,Val(width))
    _trsm_leaf_generic!(B,A,left,upper,unit)
end
