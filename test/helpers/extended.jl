# Extended-precision references (formerly in extended_scalar.jl and
# extended_blas.jl): element types, BigFloat references and tolerances.
const EXT_TYPES=Tuple(vcat(DataType[Float16,Float32,Float64,Complex{Float16},ComplexF32,ComplexF64,BigFloat,Complex{BigFloat}],optional_float_types()))
extref(x::BigFloat)=x
extref(x::AbstractFloat)=BigFloat(x)
extref(x::Real)=BigFloat(x)
extref(x::Complex)=complex(extref(real(x)),extref(imag(x)))
extref(A::AbstractArray)=extref.(A)
function exttol(::Type{T}) where T
    R=real(T)
    R===Float16 && return big".02"
    R===Float32 && return big"2e-5"
    R===Float64 && return big"3e-12"
    BigFloat(eps(R))*10000
end
function check_extended_array(actual,expected,T;factor=1)
    @test size(actual)==size(expected)
    @test maximum(abs,extref(actual)-expected;init=big"0")<=factor*exttol(T)*max(1,maximum(abs,expected;init=big"0"))
end
