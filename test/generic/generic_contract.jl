# Test-only, single-word fixture. It adds no precision and is not exported by
# the library. Base methods below belong to this test-owned type.
struct NonFMAFloat <: AbstractFloat
    value::Float64
end
NonFMAFloat(x::NonFMAFloat)=x
Base.zero(::Type{NonFMAFloat})=NonFMAFloat(0.0)
Base.one(::Type{NonFMAFloat})=NonFMAFloat(1.0)
Base.convert(::Type{NonFMAFloat},x::NonFMAFloat)=x
Base.convert(::Type{NonFMAFloat},x::Real)=NonFMAFloat(Float64(x))
Base.Float64(x::NonFMAFloat)=x.value
Base.:+(a::NonFMAFloat,b::NonFMAFloat)=NonFMAFloat(a.value+b.value)
Base.:-(a::NonFMAFloat,b::NonFMAFloat)=NonFMAFloat(a.value-b.value)
Base.:*(a::NonFMAFloat,b::NonFMAFloat)=NonFMAFloat(a.value*b.value)
Base.:/(a::NonFMAFloat,b::NonFMAFloat)=NonFMAFloat(a.value/b.value)
Base.iszero(x::NonFMAFloat)=iszero(x.value)
Base.isfinite(x::NonFMAFloat)=isfinite(x.value)
Base.:(==)(a::NonFMAFloat,b::NonFMAFloat)=a.value==b.value
Base.muladd(::NonFMAFloat,::NonFMAFloat,::NonFMAFloat)=error("generic GEMM must not impose muladd")
Base.fma(::NonFMAFloat,::NonFMAFloat,::NonFMAFloat)=error("generic GEMM must not impose fma")
@testset "Generic interface, dispatch and native allocation regression" begin
    F=NonFMAFloat; A=F.([1.0 2;3 4]); B=F.([2.0 0;0 3]); C=zeros(F,2,2)
    gemm!(C,A,B)
    @test Float64.(C)==[2.0 6;6 12]
    # More specific hardware methods must win, retaining native packed contexts.
    for T in TYPES
        matrix=zeros(T,64,64)
        method=which(gemm!,Tuple{typeof(matrix),typeof(matrix),typeof(matrix),T,T})
        @test basename(String(method.file))=="gemm.jl"
        A=fill(T(.125),64,64); B=fill(T(.25),64,64); C=similar(A)
        workspace=GemmWorkspace(T,64,64,64;workers=1)
        config=GemmConfig(T;small_threshold=0)
        gemm!(C,A,B;workspace,config,threaded=false)
        gemm!(C,A,B;workspace,config,threaded=false)
        bytes=@allocated gemm!(C,A,B;workspace,config,threaded=false)
        @test bytes<sizeof(T)*length(C) # No repeated full matrix/packing allocation.
        @test all(==(T(2)),C)
    end
    for T in (BigFloat,Complex{BigFloat})
        method=which(gemm!,Tuple{Matrix{T},Matrix{T},Matrix{T},T,T})
        @test basename(String(method.file))=="extended_blas.jl"
    end
    ambiguities=Test.detect_ambiguities(OxiBLAS;recursive=true)
    # Capture generic/native intersections specifically; unrelated historical
    # ambiguity reports remain available in the diagnostic script.
    for pair in ambiguities
        files=basename.(String.([pair[1].file,pair[2].file]))
        @test !("extended_blas.jl" in files || "extended_lapack.jl" in files)
    end
end
