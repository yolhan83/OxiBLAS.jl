# External reference only: never included by src/. Called only after the full
# worker has verified the requested backend's GESVDX symbol. No work at include.
# ABI/workspaces: Netlib DGESVDX/ZGESVDX, linked in LOT_SVD_PARTIELLE.md.
for (T,R,prefix) in ((Float32,Float32,"s"),(Float64,Float64,"d"),(ComplexF32,Float32,"c"),(ComplexF64,Float64,"z"))
    symbol=Symbol(prefix,"gesvdx_")
    if T<:Real
        @eval function _reference_gesvdx_call!(A::Matrix{$T},job,range,lo,hi,il,iu,ns,S,U,Vt,work,lwork,rwork,iwork,info)
            ccall((LA.BLAS.@blasfunc($symbol),LA.BLAS.libblastrampoline),Cvoid,
                (Ref{UInt8},Ref{UInt8},Ref{UInt8},Ref{LA.BlasInt},Ref{LA.BlasInt},Ptr{$T},Ref{LA.BlasInt},Ref{$R},Ref{$R},Ref{LA.BlasInt},Ref{LA.BlasInt},Ref{LA.BlasInt},Ptr{$R},Ptr{$T},Ref{LA.BlasInt},Ptr{$T},Ref{LA.BlasInt},Ptr{$T},Ref{LA.BlasInt},Ptr{LA.BlasInt},Ref{LA.BlasInt},Clong,Clong,Clong),
                job,job,range,size(A,1),size(A,2),A,max(1,size(A,1)),lo,hi,il,iu,ns,S,U,max(1,size(U,1)),Vt,max(1,size(Vt,1)),work,lwork,iwork,info,1,1,1)
        end
    else
        @eval function _reference_gesvdx_call!(A::Matrix{$T},job,range,lo,hi,il,iu,ns,S,U,Vt,work,lwork,rwork,iwork,info)
            ccall((LA.BLAS.@blasfunc($symbol),LA.BLAS.libblastrampoline),Cvoid,
                (Ref{UInt8},Ref{UInt8},Ref{UInt8},Ref{LA.BlasInt},Ref{LA.BlasInt},Ptr{$T},Ref{LA.BlasInt},Ref{$R},Ref{$R},Ref{LA.BlasInt},Ref{LA.BlasInt},Ref{LA.BlasInt},Ptr{$R},Ptr{$T},Ref{LA.BlasInt},Ptr{$T},Ref{LA.BlasInt},Ptr{$T},Ref{LA.BlasInt},Ptr{$R},Ptr{LA.BlasInt},Ref{LA.BlasInt},Clong,Clong,Clong),
                job,job,range,size(A,1),size(A,2),A,max(1,size(A,1)),lo,hi,il,iu,ns,S,U,max(1,size(U,1)),Vt,max(1,size(Vt,1)),work,lwork,rwork,iwork,info,1,1,1)
        end
    end
end
function reference_selected_svd(A::AbstractMatrix{T};indices=nothing,interval=nothing,vectors=true) where {T<:OxiBLAS.NativeFloat}
    m,n=size(A); r=min(m,n); R=real(T)
    indices===nothing && (indices=1:r)
    il=isempty(indices) ? 1 : first(indices); iu=isempty(indices) ? 0 : last(indices)
    range=interval===nothing ? 'I' : 'V'; lo,hi=interval===nothing ? (zero(R),zero(R)) : interval
    capacity=interval===nothing ? length(indices) : r
    (r==0 || capacity==0) && return (;S=R[],U=vectors ? zeros(T,m,0) : nothing,Vt=vectors ? zeros(T,0,n) : nothing)
    job=vectors ? 'V' : 'N'; a=Matrix(A); S=zeros(R,r)
    U=zeros(T,vectors ? m : 1,vectors ? capacity : 1); Vt=zeros(T,vectors ? max(1,capacity) : 1,vectors ? n : 1)
    ns=Ref{LA.BlasInt}(0); info=Ref{LA.BlasInt}(0); work=zeros(T,1)
    # Netlib's published complex bound is 17*r^2; keep additional linear
    # scratch to cover the bidiagonal driver's documented 2*r^2+14*r path.
    rwork=zeros(R,T<:Complex ? max(1,17r*r+32r) : 1); iwork=zeros(LA.BlasInt,max(1,12r))
    _reference_gesvdx_call!(a,job,range,R(lo),R(hi),il,iu,ns,S,U,Vt,work,-1,rwork,iwork,info)
    info[]==0 || error("GESVDX workspace query failed: INFO=$(info[])")
    lwork=max(1,ceil(Int,real(work[1]))); work=Vector{T}(undef,lwork); copyto!(a,A)
    _reference_gesvdx_call!(a,job,range,R(lo),R(hi),il,iu,ns,S,U,Vt,work,lwork,rwork,iwork,info)
    info[]==0 || error("GESVDX failed: INFO=$(info[])")
    0<=ns[]<=capacity || error("GESVDX returned an invalid selected count")
    q=Int(ns[])
    (;S=S[1:q],U=vectors ? U[:,1:q] : nothing,Vt=vectors ? Vt[1:q,:] : nothing)
end
