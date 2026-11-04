# Direct external BLAS references, restricted to tests/benchmarks.
for (T,prefix) in ((Float32,"s"),(Float64,"d"),(ComplexF32,"c"),(ComplexF64,"z"))
    for (op,band) in ((:tpmv,false),(:tpsv,false),(:tbmv,true),(:tbsv,true))
        name=Symbol(prefix,op,"_"); fun=Symbol(:reference_,op,:!)
        if band
            @eval function $fun(x::Vector{$T},P,trans)
                u=P.upper ? 'U' : 'L'; d=P.unit ? 'U' : 'N'
                ccall((LA.BLAS.@blasfunc($name),LA.BLAS.libblastrampoline),Cvoid,
                      (Ref{UInt8},Ref{UInt8},Ref{UInt8},Ref{LA.BlasInt},Ref{LA.BlasInt},Ptr{$T},Ref{LA.BlasInt},Ptr{$T},Ref{LA.BlasInt},Clong,Clong,Clong),
                      u,trans,d,length(x),max(P.kl,P.ku),P.data,max(1,stride(P.data,2)),x,1,1,1,1)
                x
            end
        else
            @eval function $fun(x::Vector{$T},P,trans)
                u=P.upper ? 'U' : 'L'; d=P.unit ? 'U' : 'N'
                ccall((LA.BLAS.@blasfunc($name),LA.BLAS.libblastrampoline),Cvoid,
                      (Ref{UInt8},Ref{UInt8},Ref{UInt8},Ref{LA.BlasInt},Ptr{$T},Ptr{$T},Ref{LA.BlasInt},Clong,Clong,Clong),
                      u,trans,d,length(x),P.data,x,1,1,1,1)
                x
            end
        end
    end
    for rank2 in (false,true)
        herm=T<:Complex; op=herm ? (rank2 ? :hpr2 : :hpr) : (rank2 ? :spr2 : :spr)
        name=Symbol(prefix,op,"_"); fun=Symbol(:reference_,op,:!)
        R=herm && !rank2 ? real(T) : T
        if rank2
            @eval function $fun(P,x::Vector{$T},y::Vector{$T},α::$R)
                ccall((LA.BLAS.@blasfunc($name),LA.BLAS.libblastrampoline),Cvoid,
                      (Ref{UInt8},Ref{LA.BlasInt},Ref{$R},Ptr{$T},Ref{LA.BlasInt},Ptr{$T},Ref{LA.BlasInt},Ptr{$T},Clong),
                      P.upper ? 'U' : 'L',P.n,α,x,1,y,1,P.data,1)
                P
            end
        else
            @eval function $fun(P,x::Vector{$T},α::$R)
                ccall((LA.BLAS.@blasfunc($name),LA.BLAS.libblastrampoline),Cvoid,
                      (Ref{UInt8},Ref{LA.BlasInt},Ref{$R},Ptr{$T},Ref{LA.BlasInt},Ptr{$T},Clong),
                      P.upper ? 'U' : 'L',P.n,α,x,1,P.data,1)
                P
            end
        end
    end
end

# Dense rank-2 updates absent from LinearAlgebra.BLAS: xSYR2 (real), xHER2 (complex).
for (T,name) in ((Float32,:ssyr2_),(Float64,:dsyr2_),(ComplexF32,:cher2_),(ComplexF64,:zher2_))
    fun=T<:Complex ? :reference_her2! : :reference_syr2!
    @eval function $fun(uplo::Char,α::$T,x::Vector{$T},y::Vector{$T},A::Matrix{$T})
        n=LA.checksquare(A)
        ccall((LA.BLAS.@blasfunc($name),LA.BLAS.libblastrampoline),Cvoid,
              (Ref{UInt8},Ref{LA.BlasInt},Ref{$T},Ptr{$T},Ref{LA.BlasInt},Ptr{$T},Ref{LA.BlasInt},Ptr{$T},Ref{LA.BlasInt},Clong),
              uplo,n,α,x,1,y,1,A,max(1,stride(A,2)),1)
        A
    end
end
function reference_compact_mv!(op,y,P,x,α,β,trans)
    u=P.upper ? 'U' : 'L'
    op===:gbmv && return LA.BLAS.gbmv!(trans,size(P,1),P.kl,P.ku,α,P.data,x,β,y)
    op===:sbmv && return LA.BLAS.sbmv!(u,max(P.kl,P.ku),α,P.data,x,β,y)
    op===:hbmv && return LA.BLAS.hbmv!(u,max(P.kl,P.ku),α,P.data,x,β,y)
    op===:spmv && return LA.BLAS.spmv!(u,α,P.data,x,β,y)
    LA.BLAS.hpmv!(u,α,P.data,x,β,y)
end
