# Non-regression for the seven methods with unbound type parameters reported
# by Aqua (test_unbound_args): the SIMD helpers of simd.jl and the
# EigenFactor/SVDFactor constructors. Can be run standalone or from runtests.jl.
using Test, Random, OxiBLAS
import LinearAlgebra as LA
const OUB=OxiBLAS

@testset "unbound type parameters: regression" begin

@testset "SIMD helpers $T, $W lanes" for T in (Float32,Float64), W in (1,2,4,8,16)
    rng=Xoshiro(W); a=randn(rng,T,W); b=randn(rng,T,W); c=randn(rng,T,W)
    va,vb,vc=(ntuple(i->VecElement(x[i]),W) for x in (a,b,c))
    val(v)=[e.value for e in v]
    @test val(OUB._vadd(va,vb))==a.+b
    @test val(OUB._vmul(va,vb))==a.*b
    @test val(OUB._vfma(va,vb,vc))==fma.(a,b,c)
    if iseven(W)
        @test val(OUB._swap_pairs(va))==[a[xor(i-1,1)+1] for i in 1:W]
        # (re,im) pairs: a·b, and conj(a)·b with C=true
        z(x)=[complex(x[2k-1],x[2k]) for k in 1:W÷2]
        flat(v)=collect(Iterators.flatten((real(x),imag(x)) for x in v))
        @test val(OUB._complex_pairmul(va,vb,Val(false))) ≈ flat(z(a).*z(b))
        @test val(OUB._complex_pairmul(va,vb,Val(true))) ≈ flat(conj.(z(a)).*z(b))
    end
end

@testset "SIMD helpers reject empty vectors" begin
    @test_throws MethodError OUB._vadd((),())
    @test_throws MethodError OUB._vmul((),())
    @test_throws MethodError OUB._vfma((),(),())
    @test_throws MethodError OUB._swap_pairs(())
    @test_throws MethodError OUB._complex_pairmul((),(),Val(false))
    # mismatched lengths are still rejected
    @test_throws MethodError OUB._vadd((VecElement(1.0),),(VecElement(1.0),VecElement(2.0)))
end

@testset "EigenFactor constructors" begin
    E=OUB.EigenFactor([1.0,2.0],randn(ComplexF64,2,2),3)
    @test E isa OUB.EigenFactor{ComplexF64,Float64} && E.iterations==3
    @test OUB.EigenFactor([1f0],nothing,0) isa OUB.EigenFactor{Float32,Float32}
    @test OUB.EigenFactor{ComplexF32,Float32}([1f0],nothing,0).vectors===nothing
end

@testset "SVDFactor constructors" begin
    U=randn(ComplexF64,3,2); Vt=randn(ComplexF64,2,2); S=[2.0,1.0]
    @test OUB.SVDFactor(U,S,Vt,1) isa OUB.SVDFactor{ComplexF64,Float64}
    @test OUB.SVDFactor(U,S,nothing,1) isa OUB.SVDFactor{ComplexF64,Float64}
    @test OUB.SVDFactor(nothing,S,Vt,1) isa OUB.SVDFactor{ComplexF64,Float64}
    @test OUB.SVDFactor(nothing,S,nothing,1) isa OUB.SVDFactor{Float64,Float64}
    @test OUB.SVDFactor{ComplexF64,Float64}(nothing,S,nothing,0).U===nothing
    @test_throws MethodError OUB.SVDFactor(randn(3,2),S,randn(ComplexF64,2,2),1)  # mixed element types
end

@testset "truncated SVD without vectors (UndefVarError before)" for T in (Float32,Float64,ComplexF32,ComplexF64)
    A=randn(Xoshiro(3),T,8,5); s=OUB.singular_values(A)
    for algorithm in (:golub_kahan,:divide_conquer)
        F=OUB.svd_truncated(A,3;algorithm,vectors=false)
        @test F.U===nothing && F.Vt===nothing
        @test F.S ≈ s[1:3] rtol=(real(T)===Float32 ? 1f-4 : 1e-10)
    end
end

end
