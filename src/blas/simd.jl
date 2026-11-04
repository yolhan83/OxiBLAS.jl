# SPDX-License-Identifier: Apache-2.0
# LLVM vectors implement the operations performed by Rust vld1q/vfmaq/vst1q,
# _mm256_loadu/_mm256_fmadd, and their AVX-512 equivalents. Alignment is only
# sizeof(T): correctness never depends on aligned user arrays.

# A vector of at least one lane. With NTuple{W,VecElement{T}}, the empty
# tuple matched with T unbound (Aqua test_unbound_args); W = N+1 in the
# generators below, whose IR is unchanged.
const _NonEmptyVec{T,N}=Tuple{VecElement{T},Vararg{VecElement{T},N}}

@inline @generated function _vload(p::Ptr{T}, ::Val{W}) where {T<:RealFloat,W}
    ty = T === Float64 ? "double" : "float"
    ir = """
    %p = inttoptr i$(Sys.WORD_SIZE) %0 to <$W x $ty>*
    %v = load <$W x $ty>, <$W x $ty>* %p, align $(sizeof(T))
    ret <$W x $ty> %v
    """
    :(Base.llvmcall($ir, NTuple{$W,VecElement{$T}}, Tuple{Ptr{$T}}, p))
end

@inline @generated function _vadd(a::_NonEmptyVec{T,N}, b::_NonEmptyVec{T,N}) where {T<:RealFloat,N}
    W=N+1
    ty=T===Float64 ? "double" : "float"
    ir="%v = fadd <$W x $ty> %0, %1\n ret <$W x $ty> %v"
    :(Base.llvmcall($ir,NTuple{$W,VecElement{$T}},Tuple{NTuple{$W,VecElement{$T}},NTuple{$W,VecElement{$T}}},a,b))
end

@inline @generated function _swap_pairs(a::_NonEmptyVec{T,N}) where {T<:RealFloat,N}
    W=N+1
    ty=T===Float64 ? "double" : "float"
    mask=join(("i32 $(xor(i,1))" for i in 0:W-1),", ")
    ir="%v = shufflevector <$W x $ty> %0, <$W x $ty> undef, <$W x i32> <$mask>\n ret <$W x $ty> %v"
    :(Base.llvmcall($ir,NTuple{$W,VecElement{$T}},Tuple{NTuple{$W,VecElement{$T}}},a))
end

@inline _alternating(x,::Val{W}) where W = ntuple(i->VecElement(isodd(i) ? -x : x),Val(W))
@inline function _complex_vmul(a,z,::Val{W}) where W
    _vfma(_swap_pairs(a),_alternating(imag(z),Val(W)),_vmul(a,_splat(real(z),Val(W))))
end

# Interleaved complex 4M arithmetic. Unlike upstream's 3M path this avoids
# subtracting three large real matrix products to reconstruct imaginary parts.
# W counts complex elements; the LLVM vector has 2W real lanes. Each output
# vector keeps two accumulators, a*re(b) and a*im(b): the inner loop is only
# loads, splats and FMAs, and i*(a*im(b)) is formed once per tile at the store
# (pair swap and sign). Rebuilding the swapped A and the signed im(b) vector
# at every step ran ComplexF32 at 63 GF/s against 108 for OpenBLAS (M5 Max).
@generated function _simd_complex_micro!(c::Ptr{T},ldc::Int,a::Ptr{T},b::Ptr{T},k::Int,
                                        α::T,β::T,::Val{W},::Val{R},::Val{N}) where {T<:Union{ComplexF32,ComplexF64},W,R,N}
    RT=real(T); vw=2W; mr=W*R
    init,step,zero_store,beta_store=Expr(:block),Expr(:block),Expr(:block),Expr(:block)
    re(r,j)=Symbol(:zr_,r,:_,j); im(r,j)=Symbol(:zi_,r,:_,j)
    for j in 0:N-1, r in 0:R-1
        push!(init.args,:($(re(r,j))=_splat(zero($RT),Val($vw))),:($(im(r,j))=_splat(zero($RT),Val($vw))))
        off=:((($j*ldc)+$(r*W))*sizeof(T))
        val=:(_complex_vmul(_vfma(_swap_pairs($(im(r,j))),signs,$(re(r,j))),α,Val($vw)))
        push!(zero_store.args,:(_vstore(Ptr{$RT}(c+$off),$val)))
        push!(beta_store.args,:(_vstore(Ptr{$RT}(c+$off),_vadd($val,_complex_vmul(_vload(Ptr{$RT}(c+$off),Val($vw)),β,Val($vw))))))
    end
    function iter(u)
        ex=Expr(:block)
        for r in 0:R-1
            push!(ex.args,:($(Symbol(:ca_,r))=_vload(Ptr{$RT}(a+((p+$u)*$mr+$(r*W))*sizeof(T)),Val($vw))))
        end
        for j in 0:N-1
            push!(ex.args,:(z=unsafe_load(b,(p+$u)*$N+$j+1)))
            push!(ex.args,:(br=_splat(real(z),Val($vw))),:(bi=_splat(imag(z),Val($vw))))
            for r in 0:R-1
                push!(ex.args,:($(re(r,j))=_vfma($(Symbol(:ca_,r)),br,$(re(r,j)))))
                push!(ex.args,:($(im(r,j))=_vfma($(Symbol(:ca_,r)),bi,$(im(r,j)))))
            end
        end
        ex
    end
    unroll=T===ComplexF32 ? 4 : 2
    for u in 0:unroll-1
        push!(step.args,iter(u))
    end
    tail=iter(0)
    quote
        $init
        p=0
        while p+$unroll<=k
            $step
            p+=$unroll
        end
        while p<k
            $tail
            p+=1
        end
        # i*(x,y) = (-y,x): swapped pairs times (-1,+1).
        signs=_alternating(one($RT),Val($vw))
        if iszero(β)
            $zero_store
        else
            $beta_store
        end
        nothing
    end
end

@inline @generated function _vstore(p::Ptr{T}, v::NTuple{W,VecElement{T}}) where {T<:RealFloat,W}
    ty = T === Float64 ? "double" : "float"
    ir = """
    %p = inttoptr i$(Sys.WORD_SIZE) %0 to <$W x $ty>*
    store <$W x $ty> %1, <$W x $ty>* %p, align $(sizeof(T))
    ret void
    """
    :(Base.llvmcall($ir, Cvoid, Tuple{Ptr{$T},NTuple{$W,VecElement{$T}}}, p, v))
end

@inline _splat(x::T, ::Val{W}) where {T,W} = ntuple(_ -> VecElement(x), Val(W))

@inline @generated function _vfma(a::_NonEmptyVec{T,N}, b::_NonEmptyVec{T,N}, c::_NonEmptyVec{T,N}) where {T<:RealFloat,N}
    W=N+1
    ty = T === Float64 ? "double" : "float"
    intr = "llvm.fma.v$(W)f$(8sizeof(T))"
    vt = "<$W x $ty>"
    ir = """
    declare $vt @$intr($vt, $vt, $vt)
    define $vt @entry($vt %a, $vt %b, $vt %c) alwaysinline {
    %v = call $vt @$intr($vt %a, $vt %b, $vt %c)
    ret $vt %v
    }
    """
    :(Base.llvmcall(($ir, "entry"), NTuple{$W,VecElement{$T}},
      Tuple{NTuple{$W,VecElement{$T}},NTuple{$W,VecElement{$T}},NTuple{$W,VecElement{$T}}}, a,b,c))
end

@inline @generated function _vmul(a::_NonEmptyVec{T,N}, b::_NonEmptyVec{T,N}) where {T<:RealFloat,N}
    W=N+1
    ty = T === Float64 ? "double" : "float"
    ir = "%v = fmul <$W x $ty> %0, %1\n ret <$W x $ty> %v"
    :(Base.llvmcall($ir, NTuple{$W,VecElement{$T}},
      Tuple{NTuple{$W,VecElement{$T}},NTuple{$W,VecElement{$T}}}, a,b))
end

# Fixed register tiles, four-way K unrolling, one specialization per ISA/type.
# No dimensions enter the type domain; no fast-math flags or reassociation.
@generated function _simd_micro!(c::Ptr{T}, ldc::Int, a::Ptr{T}, b::Ptr{T}, k::Int,
                               α::T, β::T, ::Val{W}, ::Val{R}, ::Val{N}) where {T<:RealFloat,W,R,N}
    mr = W*R
    init, step, store0, storeβ = Expr(:block), Expr(:block), Expr(:block), Expr(:block)
    acc(r,j) = Symbol(:acc_,r,:_,j)
    for j in 0:N-1, r in 0:R-1
        push!(init.args, :($(acc(r,j)) = _splat(zero(T), Val($W))))
        off = :((($j*ldc)+$(r*W))*sizeof(T))
        push!(store0.args, :(_vstore(c + $off, _vmul(av, $(acc(r,j))))))
        push!(storeβ.args, :(_vstore(c + $off, _vfma(bv, _vload(c + $off, Val($W)), _vmul(av, $(acc(r,j)))))))
    end
    function iter(offset)
        ex = Expr(:block)
        for r in 0:R-1
            push!(ex.args, :($(Symbol(:a_,r)) = _vload(a + ((p+$offset)*$mr+$(r*W))*sizeof(T), Val($W))))
        end
        for j in 0:N-1
            push!(ex.args, :(bb = _splat(unsafe_load(b, (p+$offset)*$N+$j+1), Val($W))))
            for r in 0:R-1
                push!(ex.args, :($(acc(r,j)) = _vfma($(Symbol(:a_,r)), bb, $(acc(r,j)))))
            end
        end
        ex
    end
    for u in 0:3
        push!(step.args, iter(u))
    end
    tail = iter(0)
    quote
        $init
        p = 0
        while p+4 <= k
            $step
            p += 4
        end
        while p < k
            $tail
            p += 1
        end
        av = _splat(α, Val($W))
        if iszero(β)
            $store0
        else
            bv = _splat(β, Val($W))
            $storeβ
        end
        nothing
    end
end

# Multiply two vectors of interleaved complex numbers. Width counts real lanes.
@inline _conjugate_signs(::Type{T},::Val{W},::Val{C}) where {T,W,C} =
    ntuple(i->VecElement(C && iseven(i) ? -one(T) : one(T)),Val(W))
@inline @generated function _complex_pairmul(a::_NonEmptyVec{T,N},b::_NonEmptyVec{T,N},::Val{C}) where {T<:RealFloat,N,C}
    W=N+1
    ty=T===Float64 ? "double" : "float"
    re=join(("i32 $(2*(i÷2))" for i in 0:W-1),", ")
    im=join(("i32 $(2*(i÷2)+1)" for i in 0:W-1),", ")
    # Conjugation is applied to a, never to x in the compact Hermitian dot.
    # Separate llvmcalls avoid aggregate-return ABI dependencies.
    rr="%r = shufflevector <$W x $ty> %0, <$W x $ty> undef, <$W x i32> <$re>\n ret <$W x $ty> %r"
    ii="%i = shufflevector <$W x $ty> %0, <$W x $ty> undef, <$W x i32> <$im>\n ret <$W x $ty> %i"
    quote
        aa=_vmul(a,_conjugate_signs(T,Val($W),Val($C)))
        br=Base.llvmcall($rr,NTuple{$W,VecElement{$T}},Tuple{NTuple{$W,VecElement{$T}}},b)
        bi=Base.llvmcall($ii,NTuple{$W,VecElement{$T}},Tuple{NTuple{$W,VecElement{$T}}},b)
        signs2=_alternating(one(T),Val($W))
        _vfma(_swap_pairs(aa),_vmul(bi,signs2),_vmul(aa,br))
    end
end
