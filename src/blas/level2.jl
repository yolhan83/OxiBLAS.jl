# SPDX-License-Identifier: Apache-2.0
# BLAS 2 kernels stream the matrix once and are memory bound like Level 1, so
# native types share its sizing (level1_parallel.jl): no task below
# _L1_MINBYTES of traffic, then one worker per _L1_GRAIN bytes, the caller
# taking the first range. `bytes` is equivalent Level 1 traffic: the matrix
# for GEMV/GBMV, more for the slower symmetric kernels (_symmetric_cost).
# Spawning min(n,nthreads) tasks from 512x512 on was slower than one thread
# up to about 2048x2048 (Apple M5 Max, 4-18 threads). A GEMV grain of 8 MiB
# from 2 MiB on was neutral (geometric mean 1.00 over 77 cases, +-50% wake-up
# noise either way). An explicit parallel_threshold (matrix elements) keeps
# the previous rule, as do extended types, whose arithmetic rather than
# traffic dominates.
function _l2_workers(::Type{T},n,work,bytes,threaded,threshold) where T
    threshold===nothing || threshold>=0 || throw(ArgumentError("negative parallel threshold"))
    threshold===nothing && T<:NativeFloat && return _l1_workers(n,unsafe_trunc(Int,min(bytes,0x1p62)),threaded,nothing)
    _operation_workers(n,work,threaded,something(threshold,262144))
end
# Equivalent traffic per stored entry of the symmetric/Hermitian kernels
# (Level 1 sizing): the fused axpy+dot runs about 60 GB/s for real types and
# 30 GB/s for complex, against about 125 GB/s for Level 1, and keeps scaling
# to 18 workers (complex 4096: 6-7x).
_symmetric_cost(::Type{T}) where T=(T<:Complex ? 4 : 2)*sizeof(T)
function gemv!(y::AbstractVector{T},A::AbstractMatrix{T},x::AbstractVector{T},α=one(T),β=zero(T);trans=:N,threaded=true,parallel_threshold=nothing) where {T<:GenericFloat}
    threaded=threaded && generic_threadsafe(T)
    AA=_op(A,trans)
    Base.require_one_based_indexing(y,AA,x)
    size(AA)==(length(y),length(x)) || throw(DimensionMismatch("GEMV dimensions differ"))
    _check_vector_output(y)
    work=Float64(length(y))*length(x)
    workers=_l2_workers(T,length(y),work,sizeof(T)*work,threaded,parallel_threshold)
    α,β=convert(T,α),convert(T,β)
    (iszero(α) || isempty(x)) && return _scale!(y,β)
    aa=Base.mightalias(y,AA) ? copy(AA) : AA
    xx=Base.mightalias(y,x) ? copy(x) : x
    _scale!(y,β)
    if aa isa Union{Transpose,Adjoint} && _column_preferred(parent(aa))
        _l1_foreach(length(y),workers) do rows,_
            @inbounds for i in rows
                value=zero(T)
                for j in eachindex(xx)
                    value=_arithmetic_madd(aa[i,j],xx[j],value)
                end
                y[i]+=α*value
            end
        end
        return y
    end
    _l1_foreach(length(y),workers) do rows,_
        _gemv_columns!(y,aa,xx,α,rows)
    end
    y
end
# y[rows]+=α*A[rows,:]*x by columns. Four columns per pass over y[rows] cut
# its load/store traffic by four, as in _trsv_columns!; complex products use
# the fused complex muladd.
function _gemv_columns!(y,A,x,α,rows)
    n=length(x);j=1;M=_arithmetic_madd
    @inbounds while j+3<=n
        v1=α*x[j];v2=α*x[j+1];v3=α*x[j+2];v4=α*x[j+3]
        @simd for i in rows
            y[i]=M(A[i,j+3],v4,M(A[i,j+2],v3,M(A[i,j+1],v2,M(A[i,j],v1,y[i]))))
        end
        j+=4
    end
    @inbounds while j<=n
        v=α*x[j]
        @simd for i in rows
            y[i]=M(A[i,j],v,y[i])
        end
        j+=1
    end
    y
end

function _uplo(uplo)
    uplo in (:U,'U') && return true
    uplo in (:L,'L') && return false
    throw(ArgumentError("uplo must be U or L"))
end
function _square(A)
    Base.require_one_based_indexing(A)
    size(A,1)==size(A,2) || throw(DimensionMismatch("matrix must be square"))
    size(A,1)
end
@inline function _symget(A,i,j,upper,hermitian)
    if i==j && hermitian
        return real(A[i,i])
    end
    stored=upper ? i<=j : i>=j
    stored ? A[i,j] : (hermitian ? conj(A[j,i]) : A[j,i])
end

# Prefer logical columns only when their physical strides justify it. Unknown
# layouts and transposed row-contiguous operands retain the row fallback.
_column_preferred(A)=A isa StridedMatrix && abs(stride(A,1))<=abs(stride(A,2))
function _symmetric_snapshot(A,upper,hermitian)
    n=size(A,1);S=Matrix{eltype(A)}(undef,n,n)
    for j in 1:n,i in (upper ? (1:j) : (j:n))
        S[i,j]=hermitian && i==j ? real(A[i,j]) : A[i,j]
    end
    S
end
# One stored column j of a symmetric/Hermitian matrix: y[rows]+=A[rows,j]*v and
# the mirrored dot sum (conj(A[rows,j]) or A[rows,j]) . x[rows]. x, y and A are
# distinct here (aliases are copied by _symv!), so the loop is SIMD-safe.
@inline function _symv_column!(y,A,x,v,j,rows,hermitian)
    s=zero(eltype(y))
    @inbounds @simd for i in rows
        a=A[i,j]
        y[i]=_arithmetic_madd(a,v,y[i])
        s=_arithmetic_madd(hermitian ? conj(a) : a,x[i],s)
    end
    s
end
# Complex: a complex accumulator does not vectorize; four real sums do (as in
# the complex dotu/dotc kernels).
@inline function _symv_column!(y::StridedVector{T},A::StridedMatrix{T},x::StridedVector{T},v::T,j,rows,hermitian) where {T<:Union{ComplexF32,ComplexF64}}
    R=real(T);rr=zero(R);ii=zero(R);ri=zero(R);ir=zero(R)
    @inbounds @simd for i in rows
        a=A[i,j];b=x[i]
        y[i]=_arithmetic_madd(a,v,y[i])
        rr=muladd(real(a),real(b),rr); ii=muladd(imag(a),imag(b),ii)
        ri=muladd(real(a),imag(b),ri); ir=muladd(imag(a),real(b),ir)
    end
    hermitian ? complex(rr+ii,ri-ir) : complex(rr-ii,ri+ir)
end
# Real types: four stored columns per pass share the loads and stores of y
# and x (measured 1.2-1.6x over one column). The 4x4 diagonal block is
# applied separately. Returns the first column of cols left for the
# one-column loop.
function _symv_real4!(y,A,x,alpha,upper,cols)
    n=length(y);j=first(cols);M=_arithmetic_madd
    @inbounds while j+3<=last(cols)
        v1=alpha*x[j];v2=alpha*x[j+1];v3=alpha*x[j+2];v4=alpha*x[j+3]
        s1=s2=s3=s4=zero(eltype(y))
        @simd for i in (upper ? (1:j-1) : (j+4:n))
            a1=A[i,j];a2=A[i,j+1];a3=A[i,j+2];a4=A[i,j+3];xi=x[i]
            y[i]=M(a1,v1,M(a2,v2,M(a3,v3,M(a4,v4,y[i]))))
            s1=M(a1,xi,s1);s2=M(a2,xi,s2);s3=M(a3,xi,s3);s4=M(a4,xi,s4)
        end
        y[j]+=alpha*s1;y[j+1]+=alpha*s2;y[j+2]+=alpha*s3;y[j+3]+=alpha*s4
        for jj in j:j+3, ii in (upper ? (j:jj) : (jj:j+3))
            a=A[ii,jj]
            if ii==jj
                y[ii]+=alpha*(a*x[ii])
            else
                y[ii]+=alpha*(a*x[jj]);y[jj]+=alpha*(a*x[ii])
            end
        end
        j+=4
    end
    j
end
# Adds the stored columns cols of the symmetric/Hermitian product into t.
function _symv_columns_into!(t,A,x,alpha,upper,hermitian,cols)
    n=length(t);j0=first(cols)
    if t isa StridedVector{<:Union{Float32,Float64}} && A isa StridedMatrix{eltype(t)} && x isa StridedVector{eltype(t)}
        j0=_symv_real4!(t,A,x,alpha,upper,cols)
    end
    for j in j0:last(cols)
        v=alpha*x[j]
        s=_symv_column!(t,A,x,v,j,upper ? (1:j-1) : (j+1:n),hermitian)
        diagonal=hermitian ? real(A[j,j]) : A[j,j]
        t[j]+=alpha*(diagonal*x[j]+s)
    end
    t
end
function _symv_columns!(y,A,x,alpha,beta,upper,hermitian,workers)
    n=length(y)
    workers==1 && return _symv_columns_into!(_scale!(y,beta),A,x,alpha,upper,hermitian,1:n)
    _symmetric_parallel!(y,beta,n,n,upper,workers) do t,cols
        _symv_columns_into!(t,A,x,alpha,upper,hermitian,cols)
    end
end
# Parallel symmetric/Hermitian multiply (dense, packed, banded). Workers own
# column ranges of the stored triangle with equal work (column j holds j
# entries, upper, or n-j+1, lower, up to the half-bandwidth k) and run the
# sequential fused kernel on them: the caller into y, the others into private
# vectors added to y at the end, as OpenBLAS does. The triangle is read once;
# owning output rows instead reads it twice (axpy part plus mirrored dot) and
# was 1.6-2x slower at n=4096 on 12-18 threads, where traffic is the limit.
function _symmetric_parallel!(f,y,beta,n,k,upper,workers)
    T=eltype(y)
    _scale!(y,beta)
    bound(w)=k<n ? fld(w*n,workers) : upper ? round(Int,n*sqrt(w/workers)) : n-round(Int,n*sqrt(1-w/workers))
    touched(cols)=upper ? (max(1,first(cols)-k):last(cols)) : (first(cols):min(n,last(cols)+k))
    partial=Matrix{T}(undef,n,workers-1)
    _l1_foreach(workers,workers) do _,w
        cols=bound(w-1)+1:bound(w)
        isempty(cols) && return nothing
        w==1 && return (f(y,cols);nothing)
        t=view(partial,:,w-1)
        @inbounds for i in touched(cols); t[i]=zero(T); end
        f(t,cols)
        nothing
    end
    @inbounds for w in 2:workers
        cols=bound(w-1)+1:bound(w)
        isempty(cols) && continue
        t=view(partial,:,w-1)
        @simd for i in touched(cols); y[i]+=t[i]; end
    end
    y
end
function _symv!(y,A,x,α,β,uplo,hermitian;threaded=true,parallel_threshold=nothing)
    threaded=threaded && generic_threadsafe(eltype(y))
    n=_square(A); upper=_uplo(uplo)
    length(y)==length(x)==n || throw(DimensionMismatch("symmetric GEMV dimensions differ"))
    Base.require_one_based_indexing(x,y)
    _check_vector_output(y)
    workers=_l2_workers(eltype(y),n,Float64(n)*n,_symmetric_cost(eltype(y))*Float64(n)*(n+1)/2,threaded,parallel_threshold)
    α,β=convert(eltype(y),α),convert(eltype(y),β)
    (iszero(α) || n==0) && return _scale!(y,β)
    AA=Base.mightalias(A,y) ? _symmetric_snapshot(A,upper,hermitian) : A
    xx=Base.mightalias(x,y) ? copy(x) : x
    _column_preferred(AA) && return _symv_columns!(y,AA,xx,α,β,upper,hermitian,workers)
    _foreach_range(n,workers) do rows,_
        @inbounds for i in rows
            s=zero(eltype(y))
            for j in 1:n
                s+=_symget(AA,i,j,upper,hermitian)*xx[j]
            end
            y[i]=iszero(β) ? α*s : α*s+β*y[i]
        end
    end
    y
end

symv!(y,A,x,α=one(eltype(y)),β=zero(eltype(y));uplo=:U,kwargs...)=_symv!(y,A,x,α,β,uplo,false;kwargs...)
hemv!(y,A,x,α=one(eltype(y)),β=zero(eltype(y));uplo=:U,kwargs...)=_symv!(y,A,x,α,β,uplo,true;kwargs...)

function _ger!(A,x,y,α,conjugate)
    Base.require_one_based_indexing(A,x,y)
    size(A)==(length(x),length(y)) || throw(DimensionMismatch("GER dimensions differ"))
    _check_output_layout(A)
    iszero(α) && return A
    xx=Base.mightalias(A,x) ? copy(x) : x
    yy=Base.mightalias(A,y) ? copy(y) : y
    # One coefficient per column, then a contiguous fused AXPY down the column.
    @inbounds for j in eachindex(yy)
        c=α*(conjugate ? conj(yy[j]) : yy[j])
        @simd for i in eachindex(xx)
            A[i,j]=_arithmetic_madd(xx[i],c,A[i,j])
        end
    end
    A
end
ger!(A,x,y,α=one(eltype(A)))=_ger!(A,x,y,α,false)
gerc!(A,x,y,α=one(eltype(A)))=_ger!(A,x,y,α,true)

function _rank12!(A,x,y,α,uplo,hermitian,rank2)
    n=_square(A); upper=_uplo(uplo)
    _check_output_layout(A)
    Base.require_one_based_indexing(x,y)
    length(x)==length(y)==n || throw(DimensionMismatch("rank update dimensions differ"))
    hermitian && !rank2 && !isreal(α) && throw(ArgumentError("HER alpha must be real"))
    if iszero(α)
        if hermitian
            for i in 1:n
                A[i,i]=real(A[i,i])
            end
        end
        return A
    end
    xx=Base.mightalias(A,x) ? copy(x) : x
    yy=Base.mightalias(A,y) ? copy(y) : y
    # Column form: per stored column j the coefficients are formed once and
    # the update is a contiguous fused AXPY (two for rank 2). A Hermitian
    # diagonal keeps real(old)+real(update), i.e. the real part of the sum.
    @inbounds for j in 1:n
        rows=upper ? (1:j) : (j:n)
        if rank2
            c1=hermitian ? α*conj(yy[j]) : α*yy[j]
            c2=hermitian ? conj(α)*conj(xx[j]) : α*xx[j]
            @simd for i in rows
                A[i,j]=_arithmetic_madd(yy[i],c2,_arithmetic_madd(xx[i],c1,A[i,j]))
            end
        else
            c=α*(hermitian ? conj(xx[j]) : xx[j])
            @simd for i in rows
                A[i,j]=_arithmetic_madd(xx[i],c,A[i,j])
            end
        end
        hermitian && (A[j,j]=real(A[j,j]))
    end
    A
end
syr!(A,x,α=one(eltype(A));uplo=:U)=_rank12!(A,x,x,α,uplo,false,false)
syr2!(A,x,y,α=one(eltype(A));uplo=:U)=_rank12!(A,x,y,α,uplo,false,true)
her!(A,x,α=one(real(eltype(A)));uplo=:U)=_rank12!(A,x,x,α,uplo,true,false)
her2!(A,x,y,α=one(eltype(A));uplo=:U)=_rank12!(A,x,y,α,uplo,true,true)

function _tri_setup(A,trans,uplo,unit)
    n=_square(A); upper=_uplo(uplo)
    AA=_op(A,trans)
    upper=trans in (:N,'N',:n,'n') ? upper : !upper
    AA,n,upper
end

# Column form: upper columns advance, lower columns retreat, so x[j] still
# holds its original value when consumed. Four columns are fused into one
# pass over the already-final part of x (as _trsv_columns!), then the 4x4
# diagonal block is applied in the order that reads only original entries.
function _trmv_columns!(x,A,n,upper,unit)
    M=_arithmetic_madd
    if upper
        j=1
        @inbounds while j+3<=n
            v1=x[j];v2=x[j+1];v3=x[j+2];v4=x[j+3]
            @simd for i in 1:j-1
                x[i]=M(A[i,j+3],v4,M(A[i,j+2],v3,M(A[i,j+1],v2,M(A[i,j],v1,x[i]))))
            end
            for ii in j:j+3
                s=unit ? x[ii] : A[ii,ii]*x[ii]
                for jj in ii+1:j+3
                    s=M(A[ii,jj],x[jj],s)
                end
                x[ii]=s
            end
            j+=4
        end
        @inbounds for jj in j:n
            value=x[jj]
            x[jj]=unit ? value : A[jj,jj]*value
            @simd for i in 1:jj-1
                x[i]=M(A[i,jj],value,x[i])
            end
        end
    else
        j=n
        @inbounds while j>=4
            v1=x[j-3];v2=x[j-2];v3=x[j-1];v4=x[j]
            @simd for i in j+1:n
                x[i]=M(A[i,j-3],v1,M(A[i,j-2],v2,M(A[i,j-1],v3,M(A[i,j],v4,x[i]))))
            end
            for ii in j:-1:j-3
                s=unit ? x[ii] : A[ii,ii]*x[ii]
                for jj in j-3:ii-1
                    s=M(A[ii,jj],x[jj],s)
                end
                x[ii]=s
            end
            j-=4
        end
        @inbounds for jj in j:-1:1
            value=x[jj]
            x[jj]=unit ? value : A[jj,jj]*value
            @simd for i in jj+1:n
                x[i]=M(A[i,jj],value,x[i])
            end
        end
    end
    x
end
function trmv!(x,A;uplo=:U,trans=:N,unit=false)
    AA,n,upper=_tri_setup(A,trans,uplo,unit)
    Base.require_one_based_indexing(x);_check_vector_output(x)
    length(x)==n || throw(DimensionMismatch("TRMV dimensions differ"))
    aa=Base.mightalias(A,x) ? _tri_snapshot(AA,upper,unit) : AA
    if _column_preferred(aa)
        _trmv_columns!(x,aa,n,upper,unit)
    else
        old=copy(x)
        @inbounds for i in 1:n
            s=zero(eltype(x))
            for j in (upper ? (i:n) : (1:i))
                s+=(unit && i==j ? one(eltype(x)) : aa[i,j])*old[j]
            end
            x[i]=s
        end
    end
    x
end

# SingularException/PosDefException: LinearAlgebra's (imported in OxiBLAS.jl).

# Column-oriented substitution (as reference xTRSV for column-major storage):
# solved entries update the rest with contiguous AXPYs, never striding across
# a row. Four solved columns are fused into one pass over x, which cuts the
# load/store traffic on x by four (measured 1.2-1.6x; eight columns spill
# registers for complex types). Each 4x4 diagonal block is solved first.
@inline function _trsv_diagonal!(x,A,j,unit,upper,lo,hi)
    @inbounds for jj in (upper ? (hi:-1:lo) : (lo:hi))
        value=unit ? x[jj] : x[jj]/A[jj,jj]
        x[jj]=value
        for ii in (upper ? (lo:jj-1) : (jj+1:hi))
            x[ii]=_arithmetic_madd(A[ii,jj],-value,x[ii])
        end
    end
end
function _trsv_columns!(x,A,n,upper,unit)
    if upper
        j=n
        @inbounds while j>=4
            _trsv_diagonal!(x,A,j,unit,true,j-3,j)
            c1=-x[j-3];c2=-x[j-2];c3=-x[j-1];c4=-x[j]
            @simd for i in 1:j-4
                x[i]=_arithmetic_madd(A[i,j-3],c1,_arithmetic_madd(A[i,j-2],c2,
                     _arithmetic_madd(A[i,j-1],c3,_arithmetic_madd(A[i,j],c4,x[i]))))
            end
            j-=4
        end
        j>=1 && _trsv_diagonal!(x,A,j,unit,true,1,j)
    else
        j=1
        @inbounds while j+3<=n
            _trsv_diagonal!(x,A,j,unit,false,j,j+3)
            c1=-x[j];c2=-x[j+1];c3=-x[j+2];c4=-x[j+3]
            @simd for i in j+4:n
                x[i]=_arithmetic_madd(A[i,j+3],c4,_arithmetic_madd(A[i,j+2],c3,
                     _arithmetic_madd(A[i,j+1],c2,_arithmetic_madd(A[i,j],c1,x[i]))))
            end
            j+=4
        end
        j<=n && _trsv_diagonal!(x,A,j,unit,false,j,n)
    end
    x
end
function trsv!(x,A;uplo=:U,trans=:N,unit=false)
    AA,n,upper=_tri_setup(A,trans,uplo,unit)
    Base.require_one_based_indexing(x);_check_vector_output(x)
    length(x)==n || throw(DimensionMismatch("TRSV dimensions differ"))
    aa=Base.mightalias(A,x) ? _tri_snapshot(AA,upper,unit) : AA
    if !unit
        for i in 1:n
            iszero(aa[i,i]) && throw(SingularException(i))
        end
    end
    _column_preferred(aa) && return _trsv_columns!(x,aa,n,upper,unit)
    # Complex transpose/adjoint of a contiguous matrix: row i is column i of
    # the parent, reduced by the Level 1 complex dot (four real sums); the
    # complex accumulator below does not vectorize (3-4x behind OpenBLAS).
    if aa isa Union{Transpose,Adjoint} && eltype(x)<:Union{ComplexF32,ComplexF64} &&
       x isa StridedVector{eltype(x)} && parent(aa) isa StridedMatrix{eltype(x)} && stride(parent(aa),1)==1
        P=parent(aa); conjugate=aa isa Adjoint
        @inbounds for i in (upper ? (n:-1:1) : (1:n))
            r=upper ? (i+1:n) : (1:i-1)
            s=isempty(r) ? zero(eltype(x)) : (conjugate ? dotc(view(P,r,i),view(x,r)) : dotu(view(P,r,i),view(x,r)))
            s=x[i]-s
            x[i]=unit ? s : s/aa[i,i]
        end
        return x
    end
    # Row form for transposed/adjoint or row-contiguous operands: each row is
    # a contiguous dot product, reduced with SIMD partial sums.
    @inbounds for i in (upper ? (n:-1:1) : (1:n))
        s=zero(eltype(x))
        @simd for j in (upper ? (i+1:n) : (1:i-1))
            s=_arithmetic_madd(aa[i,j],x[j],s)
        end
        s=x[i]-s
        x[i]=unit ? s : s/aa[i,i]
    end
    x
end
