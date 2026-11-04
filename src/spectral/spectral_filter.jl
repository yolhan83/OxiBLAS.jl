# SPDX-License-Identifier: Apache-2.0
"""Gershgorin spectral enclosure estimate for a finite Hermitian matrix.
Computed in working precision with a rounding margin, not directed-rounding
interval arithmetic. An exactly diagonal matrix retains its exact extrema.
"""
function hermitian_spectral_bounds(A::AbstractMatrix{T};check_hermitian=true) where {T<:NativeFloat}
    n=_square(A); C=CSRMatrix(A); R=real(T)
    all(isfinite,C.values) || throw(ArgumentError("finite Hermitian data required"))
    check_hermitian && !_block_is_hermitian(C) && throw(ArgumentError("Hermitian operator required"))
    n==0 && return (zero(R),zero(R))
    low=R(Inf); high=-R(Inf); offdiagonal=false
    for i in 1:n
        center=zero(R); radius=zero(R)
        for p in C.rowptr[i]:C.rowptr[i+1]-1
            j=C.colind[p]; value=C.values[p]
            if j==i
                center=real(value)
            else
                radius+=abs(value); offdiagonal|=!iszero(value)
            end
        end
        low=min(low,center-radius); high=max(high,center+radius)
    end
    isfinite(low) && isfinite(high) || throw(ArgumentError("spectral bounds overflow working precision"))
    if offdiagonal
        margin=8eps(R)*max(abs(low),abs(high))*max(1,n)
        low-=margin; high+=margin
    end
    isfinite(low) && isfinite(high) || throw(ArgumentError("spectral margin overflows; supply scaled data"))
    low,high
end

"""Owned Chebyshev window filter of a Hermitian matrix, with Jackson damping
by default. The sparse matrix and its scaled copy are retained. Spectral bounds
provided by the caller assert an enclosure; they are not independently certified.
"""
struct ChebyshevFilter{T,R}
    matrix::CSRMatrix{T,Int}
    scaled::CSRMatrix{T,Int}
    coefficients::Vector{R}
    interval::Tuple{R,R}
    bounds::Tuple{R,R}
    scale::R
    center::R
    radius::R
    damping::Symbol
    bounds_kind::Symbol
end

function polynomial_filter(A::AbstractMatrix{T},low::Real,high::Real;degree::Integer=20,
                           bounds=nothing,damping=:jackson,check_hermitian=true) where {T<:NativeFloat}
    _square(A); R=real(T); lo,hi=R(low),R(high)
    isfinite(lo) && isfinite(hi) && lo<hi || throw(ArgumentError("finite increasing filter interval required"))
    degree>=0 && damping in (:jackson,:none) || throw(ArgumentError("invalid filter degree/damping"))
    C=CSRMatrix(COOMatrix(CSRMatrix(A)))
    all(isfinite,C.values) || throw(ArgumentError("finite filter matrix required"))
    check_hermitian && !_block_is_hermitian(C) && throw(ArgumentError("Hermitian filter matrix required"))
    bounds===nothing || length(bounds)==2 || throw(ArgumentError("spectral bounds must have two endpoints"))
    lower,upper=bounds===nothing ? hermitian_spectral_bounds(C;check_hermitian=false) : R.(bounds)
    isfinite(lower) && isfinite(upper) && lower<=upper || throw(ArgumentError("finite ordered spectral bounds required"))
    kind=bounds===nothing ? :gershgorin_estimate : :caller_assertion
    # One assignment: scale is captured by `mapped` below.
    bound=max(abs(lower),abs(upper)); scale=iszero(bound) ? one(R) : bound
    l,u=lower/scale,upper/scale; center=l/2+u/2; radius=(u-l)/2
    coefficients=zeros(R,Base.checked_add(Int(degree),1))
    if lower==upper
        coefficients[1]=lo<=lower<=hi ? one(R) : zero(R)
    elseif hi<lower || lo>upper
        # Disjoint window: zero polynomial, not a clamped endpoint window.
    elseif lo<=lower && upper<=hi
        coefficients[1]=one(R)
    else
        radius>0 || throw(ArgumentError("spectral range not representable after scaling"))
        mapped(x)=clamp(((clamp(x,lower,upper)/scale)-center)/radius,-one(R),one(R))
        theta_hi=acos(Float64(mapped(lo))); theta_lo=acos(Float64(mapped(hi)))
        moments=Float64(Base.checked_add(Int(degree),2)); angle=pi/moments; cotangent=cos(angle)/sin(angle)
        for k in 0:Int(degree)
            coefficient=k==0 ? (theta_hi-theta_lo)/pi : 2*(sin(k*theta_hi)-sin(k*theta_lo))/(k*pi)
            damping_factor=damping===:none ? 1.0 : ((moments-k)*cos(k*angle)+sin(k*angle)*cotangent)/moments
            coefficients[k+1]=R(damping_factor*coefficient)
        end
    end
    # Trim exact zeros, so a constant/zero window never applies the operator.
    last=findlast(!iszero,coefficients); resize!(coefficients,last===nothing ? 1 : last)
    scaled=CSRMatrix(size(C,1),size(C,2),C.rowptr,C.colind,C.values./scale)
    ChebyshevFilter(C,scaled,coefficients,(lo,hi),(lower,upper),scale,center,radius,damping,kind)
end

"""Evaluate the scalar polynomial response (no interval membership guarantee)."""
function filter_response(P::ChebyshevFilter{T,R},x::Real) where {T,R}
    value=R(x); isfinite(value) || throw(ArgumentError("finite response argument required"))
    length(P.coefficients)==1 && return P.coefficients[1]
    z=(value/P.scale-P.center)/P.radius; previous=one(R); current=z
    result=P.coefficients[1]+P.coefficients[2]*current
    for k in 3:length(P.coefficients)
        following=2z*current-previous; result+=P.coefficients[k]*following
        previous,current=current,following
    end
    isfinite(result) || throw(ArgumentError("filter response overflow"))
    result
end

"""Reusable four-buffer Chebyshev recurrence workspace. Sharing serializes
calls; metadata and native GEMM/SpMM packing may still allocate.
"""
mutable struct FilterWorkspace{T}
    buffers::NTuple{4,Matrix{T}}
    rows::Int
    columns::Int
    guard::ReentrantLock
    active::Bool
end
function FilterWorkspace(P::ChebyshevFilter{T};columns::Integer=1) where T
    columns>=0 || throw(ArgumentError("nonnegative workspace column capacity required"))
    n=size(P.matrix,1)
    FilterWorkspace(ntuple(_->Matrix{T}(undef,n,columns),4),n,Int(columns),ReentrantLock(),false)
end

"""Apply the window polynomial to vectors or column blocks.
The result is complete before aliased output is changed. alpha=0 avoids reading
input values; beta=0 avoids reading previous output values. Plan/workspace
storage aliases are rejected. Nonfinite input/recurrence values are rejected.
"""
function apply_filter!(Y::AbstractVecOrMat{T},P::ChebyshevFilter{T},X::AbstractVecOrMat{T};
                        α=one(T),β=zero(T),workspace=nothing,threaded=true) where T
    Base.require_one_based_indexing(Y,X)
    size(Y)==size(X) && size(X,1)==size(P.matrix,1) || throw(DimensionMismatch("filter dimensions differ"))
    ndims(Y)==1 ? _check_vector_output(Y) : _check_output_layout(Y)
    any(A->Base.mightalias(Y,A),(P.matrix,P.scaled,P.coefficients)) && throw(ArgumentError("filter output aliases plan storage"))
    alpha,beta=convert(T,α),convert(T,β)
    isfinite(alpha) && isfinite(beta) || throw(ArgumentError("finite filter scalars required"))
    iszero(alpha) && workspace===nothing && return _scale!(Y,beta)
    columns=ndims(X)==1 ? 1 : size(X,2)
    W=workspace===nothing ? FilterWorkspace(P;columns) : workspace
    W isa FilterWorkspace{T} || throw(ArgumentError("filter workspace scalar type differs"))
    lock(W.guard) do
        W.active && throw(ArgumentError("recursive filter workspace use is unsupported"))
        W.rows==size(X,1) || throw(DimensionMismatch("filter workspace row count differs"))
        any(B->Base.mightalias(X,B) || Base.mightalias(Y,B),W.buffers) && throw(ArgumentError("filter operand aliases workspace storage"))
        iszero(alpha) && return _scale!(Y,beta)
        all(isfinite,X) || throw(ArgumentError("finite filter input required"))
        if columns>W.columns
            W.buffers=ntuple(_->Matrix{T}(undef,W.rows,columns),4); W.columns=columns
        end
        W.active=true
        try
            previous,current,following,result=map(B->view(B,:,1:columns),W.buffers)
            copyto!(previous,reshape(X,size(X,1),columns))
            for i in eachindex(result,previous)
                result[i]=P.coefficients[1]*previous[i]
            end
            degree=length(P.coefficients)-1
            for k in 1:degree
                source=k==1 ? previous : current
                _block_apply!(following,P.scaled,source;threaded)
                for i in eachindex(following,source)
                    mapped=(following[i]-P.center*source[i])/P.radius
                    following[i]=k==1 ? mapped : 2mapped-previous[i]
                end
                all(isfinite,following) || throw(ArgumentError("nonfinite filter recurrence; check spectral bounds and scaling"))
                for i in eachindex(result,following)
                    result[i]+=P.coefficients[k+1]*following[i]
                end
                if k==1
                    current,following=following,current
                else
                    previous,current,following=current,following,previous
                end
            end
            all(isfinite,result) || throw(ArgumentError("nonfinite filter result"))
            target=reshape(Y,size(Y,1),columns)
            for i in eachindex(target,result)
                target[i]=iszero(beta) ? alpha*result[i] : alpha*result[i]+beta*target[i]
            end
            Y
        finally
            W.active=false
        end
    end
end
