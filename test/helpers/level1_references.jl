# External calls are independent references used by tests/benchmarks only.
for (T,mg,rm) in ((Float32,:srotmg_,:srotm_),(Float64,:drotmg_,:drotm_))
    @eval function reference_rotmg(d1::$T,d2::$T,x1::$T,y1::$T)
        a,b,c=Ref(d1),Ref(d2),Ref(x1); p=zeros($T,5)
        ccall((LA.BLAS.@blasfunc($mg),LA.BLAS.libblastrampoline),Cvoid,
              (Ref{$T},Ref{$T},Ref{$T},Ref{$T},Ptr{$T}),a,b,c,y1,p)
        (d1=a[],d2=b[],x1=c[],params=RotmParams(p))
    end
    @eval function reference_rotm!(x::Vector{$T},y::Vector{$T},p::Vector{$T})
        ccall((LA.BLAS.@blasfunc($rm),LA.BLAS.libblastrampoline),Cvoid,
              (Ref{LA.BlasInt},Ptr{$T},Ref{LA.BlasInt},Ptr{$T},Ref{LA.BlasInt},Ptr{$T}),length(x),x,1,y,1,p)
        (x,y)
    end
end
function reference_dsdot(x::Vector{Float32},y::Vector{Float32})
    ccall((LA.BLAS.@blasfunc(dsdot_),LA.BLAS.libblastrampoline),Float64,
          (Ref{LA.BlasInt},Ptr{Float32},Ref{LA.BlasInt},Ptr{Float32},Ref{LA.BlasInt}),length(x),x,1,y,1)
end
function reference_sdsdot(alpha::Float32,x::Vector{Float32},y::Vector{Float32})
    ccall((LA.BLAS.@blasfunc(sdsdot_),LA.BLAS.libblastrampoline),Float32,
          (Ref{LA.BlasInt},Ref{Float32},Ptr{Float32},Ref{LA.BlasInt},Ptr{Float32},Ref{LA.BlasInt}),length(x),alpha,x,1,y,1)
end

function rotm_matrix(p)
    T=typeof(p.flag)
    p.flag==T(-2) && return Matrix{T}(LA.I,2,2)
    iszero(p.flag) && return T[1 p.h12; p.h21 1]
    p.flag==one(T) && return T[p.h11 1; -1 p.h22]
    T[p.h11 p.h12; p.h21 p.h22]
end

# Fortran negative increments start at the last logical element of the base
# span, so pass its lowest address rather than Julia's first logical pointer.
function reference_span(x)
    step=stride(x,1); p=pointer(x)
    step<0 && (p+=(length(x)-1)*step*sizeof(eltype(x)))
    p,step
end
for (T,rm) in ((Float32,:srotm_),(Float64,:drotm_))
    @eval function reference_rotm!(x::StridedVector{$T},y::StridedVector{$T},p::Vector{$T})
        GC.@preserve x y p begin
            xp,ix=reference_span(x); yp,iy=reference_span(y)
            ccall((LA.BLAS.@blasfunc($rm),LA.BLAS.libblastrampoline),Cvoid,
                  (Ref{LA.BlasInt},Ptr{$T},Ref{LA.BlasInt},Ptr{$T},Ref{LA.BlasInt},Ptr{$T}),length(x),xp,ix,yp,iy,p)
        end
        (x,y)
    end
end
function reference_dsdot(x::StridedVector{Float32},y::StridedVector{Float32})
    GC.@preserve x y begin
        xp,ix=reference_span(x); yp,iy=reference_span(y)
        ccall((LA.BLAS.@blasfunc(dsdot_),LA.BLAS.libblastrampoline),Float64,
              (Ref{LA.BlasInt},Ptr{Float32},Ref{LA.BlasInt},Ptr{Float32},Ref{LA.BlasInt}),length(x),xp,ix,yp,iy)
    end
end
function reference_sdsdot(alpha::Float32,x::StridedVector{Float32},y::StridedVector{Float32})
    GC.@preserve x y begin
        xp,ix=reference_span(x); yp,iy=reference_span(y)
        ccall((LA.BLAS.@blasfunc(sdsdot_),LA.BLAS.libblastrampoline),Float32,
              (Ref{LA.BlasInt},Ref{Float32},Ptr{Float32},Ref{LA.BlasInt},Ptr{Float32},Ref{LA.BlasInt}),length(x),alpha,xp,ix,yp,iy)
    end
end
