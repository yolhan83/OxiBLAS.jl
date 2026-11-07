# SPDX-License-Identifier: Apache-2.0
# One-sided Jacobi, adapting upstream svd/bidiagonal.rs and svd/complex.rs.
# Works on A's columns directly, never on a formed A'*A. No external SVD.

struct SVDFactor{T,R}
    U::Union{Nothing,Matrix{T}}
    S::Vector{R}
    Vt::Union{Nothing,Matrix{T}}
    sweeps::Int
    # Inner constructor: replaces the default outer one, whose T was unbound
    # when U and Vt are both nothing (svd_truncated with vectors=false failed
    # with UndefVarError).
    SVDFactor{T,R}(U,S,Vt,sweeps) where {T,R}=new{T,R}(U,S,Vt,sweeps)
end
SVDFactor(U::Matrix{T},S::Vector{R},Vt::Union{Nothing,Matrix{T}},sweeps::Int) where {T,R}=SVDFactor{T,R}(U,S,Vt,sweeps)
SVDFactor(::Nothing,S::Vector{R},Vt::Matrix{T},sweeps::Int) where {T,R}=SVDFactor{T,R}(nothing,S,Vt,sweeps)
# No vector element type to infer without vectors: T = R, as for EigenFactor.
SVDFactor(::Nothing,S::Vector{R},::Nothing,sweeps::Int) where R=SVDFactor{R,R}(nothing,S,nothing,sweeps)

function _svd_rotate_columns!(B,i,j,c,s)
    B===nothing && return nothing
    @inbounds @simd for row in axes(B,1)
        x,y=B[row,i],B[row,j]
        B[row,i]=c*x-conj(s)*y
        B[row,j]=s*x+c*y
    end
    nothing
end

# Native behavior stays unchanged. For arbitrary-precision providers, the
# exponent floor can be millions of bits below the working precision: a
# correlated roundoff column must not wait for exponent underflow to deflate.
_svd_deflation_floor(::Type{R},nx,ny,m) where R = R<:RealFloat ? floatmin(R)*max(1,m) :
    max(floatmin(R),eps(R)*max(nx,ny))*max(1,m)

function _jacobi_svd!(B,V,rtol,maxsweeps)
    T=eltype(B); R=real(T); m,n=size(B); sweeps=0
    while true
        rotated=false
        @inbounds for j in 2:n, i in 1:j-1
            nx,ny,ρ=_svd_dense_pair_stats(B,i,j)
            (iszero(nx) || iszero(ny)) && continue
            off=abs(ρ)
            off<=rtol && continue
            sweeps>=maxsweeps && throw(ConvergenceError(:jacobi_svd,sweeps,i:j))
            # Rank-deficient sweeps can leave correlated columns pinned at
            # the subnormal rounding floor. Do not normalize these into
            # duplicate U vectors. Orthogonal tiny input columns are retained.
            if min(nx,ny)<=_svd_deflation_floor(R,nx,ny,m)
                col=nx<=ny ? i : j
                for row in 1:m
                    B[row,col]=zero(T)
                end
                rotated=true
                continue
            end
            # Scale the local 2x2 problem by the larger column norm. Avoid
            # squared tiny norms and an overflowing Jacobi tangent parameter.
            ratio=min(nx,ny)/max(nx,ny)
            g=ratio*off
            iszero(g) && continue  # rotation below representable dynamic range
            a,b=nx>=ny ? (one(R),ratio*ratio) : (ratio*ratio,one(R))
            delta=(b-a)/2
            den=delta+copysign(hypot(delta,g),delta)
            t=g/den; c=Base.inv(hypot(one(R),t))
            s=(c*t)*_realdiv(ρ,off)
            _svd_rotate_columns!(B,i,j,c,s)
            _svd_rotate_columns!(V,i,j,c,s)
            rotated=true
        end
        !rotated && return sweeps
        sweeps+=1
    end
end

function _complete_svd_column!(U,col,scratch)
    T=eltype(U); R=real(T); m=size(U,1)
    for candidate in 1:m
        fill!(scratch,zero(T)); scratch[candidate]=one(T)
        # Two passes keep null-space vectors orthogonal near rank deficiency.
        @inbounds for pass in 1:2, j in 1:col-1
            s=zero(T)
            for row in 1:m
                s+=conj(U[row,j])*scratch[row]
            end
            for row in 1:m
                scratch[row]-=s*U[row,j]
            end
        end
        norm=nrm2(scratch)
        if norm>sqrt(eps(R))
            @inbounds for row in 1:m
                U[row,col]=_realdiv(scratch[row],norm)
            end
            return nothing
        end
    end
    throw(ConvergenceError(:svd_nullspace,0,col:col))
end

"""
    svd_factor(A;vectors=true,maxsweeps=100,rtol=8eps(real(eltype(A)))*sqrt(min(size(A)...)))

Economy SVD A=U*Diagonal(S)*Vt, sorted descending, for real/complex rectangular
matrices, including empty and rank-deficient inputs. Native one-sided cyclic
Jacobi; never forms the full Gram matrix. Wide matrices are handled through
their adjoints. Column norms and normalized correlations protect graded inputs.
Correlated columns at the working type's underflow floor are deflated to zero;
orthogonal tiny columns are retained.
vectors=false returns U=Vt=nothing. Nonconvergence throws ConvergenceError.
`full=true` completes the null spaces. `algorithm=:parallel_jacobi` selects
disjoint column rounds; `:golub_kahan` and `:divide_conquer` use the bidiagonal
entry point.
"""
function svd_factor(A::AbstractMatrix{T};vectors=true,maxsweeps=100,algorithm=:auto,full=false,threaded=true,
                    rtol=8eps(real(T))*sqrt(real(T)(max(1,min(size(A)...)))),kwargs...) where {T<:GenericFloat}
    threaded=threaded && generic_threadsafe(T)
    if algorithm===:auto
        cfg=get_spectral_config(T)
        m,n=size(A); k=min(m,n)
        cutoff=cfg!==nothing ? cfg.svd_bidiag_threshold : ((T<:Complex) ? 4 : 8)
        if cfg!==nothing && cfg.svd_algorithm in (:golub_kahan,:divide_conquer) && k>=cutoff
            return svd_bidiagonal(A;algorithm=cfg.svd_algorithm,vectors,full,threaded,rtol,kwargs...)
        elseif T<:NativeFloat && maxsweeps>0 && k>=cutoff
            return svd_bidiagonal(A;algorithm=:golub_kahan,vectors,full,threaded,rtol,kwargs...)
        else
            algorithm=:jacobi
        end
    end
    algorithm in (:golub_kahan,:divide_conquer) && return svd_bidiagonal(A;algorithm,vectors,full,threaded,rtol,kwargs...)
    algorithm in (:jacobi,:parallel_jacobi,:block_jacobi,:parallel_block_jacobi) || throw(ArgumentError("unknown SVD algorithm"))
    isempty(kwargs) || throw(ArgumentError("unknown Jacobi SVD keyword"))
    Base.require_one_based_indexing(A)
    maxsweeps isa Integer && maxsweeps>=0 || throw(ArgumentError("maxsweeps must be a nonnegative integer"))
    isfinite(rtol) && zero(rtol)<rtol<one(rtol) || throw(ArgumentError("rtol must lie strictly between 0 and 1"))
    m,n=size(A); R=real(T)
    if m<n
        F=svd_factor(adjoint(A);vectors,maxsweeps,rtol,algorithm,full,threaded)
        return SVDFactor{T,R}(F.Vt===nothing ? nothing : Matrix(adjoint(F.Vt)),F.S,
                             F.U===nothing ? nothing : Matrix(adjoint(F.U)),F.sweeps)
    end
    if T<:NativeFloat && n>=32 && m>=2n && maxsweeps>=4
        Q,small,scale=_svd_dense_tall_qr(A,true,threaded)
        if _svd_dense_qr_eligible(Q)
            G=svd_factor(small;vectors,maxsweeps,rtol,algorithm,full=false,threaded)
            return _svd_dense_qr_result(Q,G,scale,m,n,full,threaded)
        end
    end
    B,scale=_normalized_copy(A)
    V=vectors ? _identity(T,n) : nothing
    parallel_algorithm=algorithm in (:parallel_jacobi,:parallel_block_jacobi)
    blocks=algorithm in (:block_jacobi,:parallel_block_jacobi) || T<:Complex ||
           (parallel_algorithm && threaded && Threads.nthreads()>1)
    if T<:NativeFloat && blocks && n>=128 && maxsweeps>=4
        sweeps=_svd_dense_block_jacobi!(B,V,rtol,maxsweeps,threaded && parallel_algorithm)
    else
        sweeps=parallel_algorithm ? _jacobi_parallel_svd!(B,V,rtol,maxsweeps,threaded) : _jacobi_svd!(B,V,rtol,maxsweeps)
    end
    norms=[nrm2(view(B,:,j)) for j in 1:n]
    perm=sortperm(norms;rev=true)
    sigma=norms[perm].*scale
    if !vectors
        return SVDFactor{T,R}(nothing,sigma,nothing,sweeps)
    end
    U=Matrix{T}(undef,m,n); scratch=Vector{T}(undef,m)
    @inbounds for j in 1:n
        index=perm[j]; norm=norms[index]
        if iszero(norm)
            _complete_svd_column!(U,j,scratch)
        else
            for row in 1:m
                U[row,j]=_realdiv(B[row,index],norm)
            end
        end
    end
    Vt=Matrix{T}(undef,n,n)
    @inbounds for j in 1:n, i in 1:n
        Vt[i,j]=conj(V[j,perm[i]])
    end
    result=SVDFactor{T,R}(U,sigma,Vt,sweeps)
    full ? _full_svd(result,m,n) : result
end

singular_values(A::AbstractMatrix;kwargs...)=svd_factor(A;vectors=false,kwargs...).S
