# Compatibility contract only: numeric providers own their arithmetic.
# EXT_TYPES, extref, exttol: test/helpers/extended.jl
@testset "Generic floating architecture and provider precision" begin
    @test !isdefined(OxiBLAS,:DoubleDouble)
    project=read(joinpath(@__DIR__,"..","..","Project.toml"),String)
    @test !occursin("DoubleFloats",project) && !occursin("Quadmath",project)
    setprecision(BigFloat,512) do
        for T in EXT_TYPES
            R=real(T)
            @test zero(T) isa T && one(T) isa T
            @test convert(T,1) isa T
            @test isfinite(one(T)) && eps(R)>zero(R)
            @test promote_type(T,T)===T
            promoted=promote(one(T),one(Float64))
            @test typeof(promoted[1])===typeof(promoted[2])
            @test promoted[1]==one(promoted[1])
            if !(R in (Float16,Float32,Float64))
                @test promote_type(T,Float64)===T
            end
            if !(T<:OxiBLAS.NativeFloat)
                @test GemmWorkspace(T,3,3,3) isa GenericWorkspace
            else
                @test GemmWorkspace(T,3,3,3) isa GemmWorkspace
            end
            if !(R in (Float16,Float32,Float64))
                bits=precision(R); delta=ldexp(one(R),-max(60,bits÷2))
                value=one(R)+delta
                @test value!=one(R)
                @test Float64(value)==1.0 # The low information must survive OxiBLAS.
                A=reshape(T[value],1,1); B=ones(T,1,1); C=zeros(T,1,1)
                gemm!(C,A,B)
                @test eltype(C)===T && real(C[1])==value
                result=solve(lu_factor(A),T[value])
                @test eltype(result)===T && result[1]==one(T)
                @test dot_extended(T[value],ones(T,1))==T(value)
                if T<:Complex
                    z=complex(value,delta); A[1]=z
                    gemm!(C,A,B)
                    @test C[1]==z && imag(C[1])==delta
                end
            end
        end
        @test dot_extended([1e16,1.0,-1e16],ones(3))==1
        value=dot_extended([1.0,2,3],ones(3);accumulator_type=BigFloat)
        @test value isa BigFloat && value==6
        @test !generic_threadsafe(BigFloat) && !generic_threadsafe(Complex{BigFloat})
    end
end
