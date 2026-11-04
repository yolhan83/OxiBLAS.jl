using Test, Random, OxiBLAS
import LinearAlgebra as LA
Random.seed!(72)
@testset "Portable native BLAS1 reductions" begin
 for T in (Float32,Float64,ComplexF32,ComplexF64),n in (0,1,3,7,65,512,4096)
  R=real(T);a=randn(T,2n);b=randn(T,2n)
  for (x,y) in ((view(a,1:2:2n),view(b,1:2:2n)),(reverse(a[1:n]),reverse(b[1:n])))
   exact=sum(abs(BigFloat(real(z))) for z in x;init=big"0")+sum(abs(BigFloat(imag(z))) for z in x;init=big"0")
   @test abs(BigFloat(asum(x))-exact)<=max(1,exact)*32eps(R)*max(1,n)
   @test nrm2(x)≈LA.norm(x) rtol=8sqrt(eps(R)) atol=8eps(R)
   for (f,conjleft) in ((dotu,false),(dotc,true))
    high(z)=complex(BigFloat(real(z)),BigFloat(imag(z)))
    exact=sum((conjleft ? conj(high(x[i])) : high(x[i]))*high(y[i]) for i in eachindex(x);init=complex(big"0"))
    bound=sum(abs(high(x[i]))*abs(high(y[i])) for i in eachindex(x);init=big"0")
    @test abs(high(f(x,y))-exact)<=32eps(R)*max(1,n)*max(1,bound)
   end
  end
 end
 for T in (Float32,Float64,ComplexF32,ComplexF64)
  R=real(T)
  @test nrm2(T[floatmax(R)/4,floatmax(R)/4])≈hypot(floatmax(R)/4,floatmax(R)/4)
  @test nrm2(T[floatmin(R),floatmin(R)])>0
  @test isinf(nrm2(T[Inf,Inf]))
  @test isnan(nrm2(T[Inf,NaN]))
  @test isnan(asum(T[Inf,NaN]))
  @test isinf(asum(T[Inf,1]))
 end
end

# Large vectors: running sums of squares over N terms accumulate O(N eps) error
# (benchmark nrm2 guard failures at N=524288). Independent BigFloat oracle.
@testset "nrm2 large vectors vs BigFloat oracle" begin
 oracle(x)=sqrt(sum(abs2(BigFloat(real(z)))+abs2(BigFloat(imag(z))) for z in x;init=big"0"))
 relerr(r,e)=iszero(e) ? abs(BigFloat(r)) : abs(BigFloat(r)-e)/e
 pattern(::Type{T},n) where T=(R=real(T);v=T[R(mod(17i+3,61)-30)/32 for i in 1:n];
     T<:Complex && (v.+=T[complex(0,R(mod(11i+1,53)-26)/32) for i in 1:n]);v)
 for T in (Float32,Float64,ComplexF32,ComplexF64)
  R=real(T);bound=64eps(R)
  inputs=Any[]
  for n in (524288,1048576)
   push!(inputs,("pattern N=$n",pattern(T,n)),("randn N=$n",randn(T,n)))
  end
  big_e,small_e=R===Float32 ? (100,-100) : (600,-600)
  push!(inputs,("pattern*2^$big_e",pattern(T,262144).*ldexp(one(R),big_e)),
                ("pattern*2^$small_e",pattern(T,262144).*ldexp(one(R),small_e)),
                ("strided pattern",view(pattern(T,1048576),1:2:1048576)))
  for n in (1023,1024,1025,2049,4097)
   push!(inputs,("randn N=$n",randn(T,n)))
  end
  for (label,x) in inputs
   e=oracle(x)
   @testset "$T $label" begin
    @test relerr(nrm2(x),e)<=bound
    @test relerr(nrm2_parallel(x;parallel_threshold=1),e)<=bound
   end
  end
  # Exceptional values beyond the first accumulation block.
  x=randn(T,5000);x[3000]=R(Inf);@test isinf(nrm2(x));@test isinf(nrm2_parallel(x;parallel_threshold=1))
  x=randn(T,5000);x[3000]=R(NaN);@test isnan(nrm2(x));@test isnan(nrm2_parallel(x;parallel_threshold=1))
  x=randn(T,5000);x[10]=R(Inf);x[3000]=R(NaN);@test isnan(nrm2(x))
  @test nrm2(zeros(T,3000))==0
 end
end

# Forced parallel paths (parallel_threshold=1 gives at least two workers when
# threads allow) against the sequential kernels, including tiny lengths.
@testset "BLAS1 parallel variants vs sequential kernels" begin
 for T in (Float32,Float64,ComplexF32,ComplexF64),n in (0,1,2,3,17,1000,4099)
  R=real(T);x=randn(T,n);y=randn(T,n);a=T(0.75);k=max(1,n)*16eps(R)
  scale=max(1,sum(abs,x;init=zero(R)))
  @test isapprox(dot_parallel(x,y;parallel_threshold=1),dotu(x,y);atol=k*scale)
  @test isapprox(dot_parallel(x,y;conjugate=true,parallel_threshold=1),dotc(x,y);atol=k*scale)
  @test isapprox(asum_parallel(x;parallel_threshold=1),asum(x);atol=k*scale)
  @test isapprox(nrm2_parallel(x;parallel_threshold=1),nrm2(x);rtol=16eps(R))
  # muladd may or may not fuse per element: compare within a few eps.
  near(u,w)=maximum(abs,u.-w;init=zero(R))<=4eps(R)*max(1,maximum(abs,w;init=zero(R)))
  @test near(axpy_parallel!(copy(y),x,a;parallel_threshold=1),axpy!(copy(y),x,a))
  @test scal_parallel!(copy(x),a;parallel_threshold=1)==scal!(copy(x),a)
  z=copy(x);w=copy(x);@test near(axpy_parallel!(z,z,a;parallel_threshold=1),axpy!(w,w,a))
  v=view(repeat(x,inner=2),1:2:2n)
  @test isapprox(asum_parallel(v;parallel_threshold=1),asum(x);atol=k*scale)
  @test isapprox(nrm2_parallel(v;parallel_threshold=1),nrm2(x);rtol=16eps(R),atol=floatmin(R))
  @test asum_parallel(x;threaded=false,parallel_threshold=1)==asum(x)
 end
end

if !isdefined(@__MODULE__,:TYPES)
    @eval const TYPES=(Float32,Float64,ComplexF32,ComplexF64)
    @eval tol(::Type{T}) where T=real(T)===Float32 ? 2f-4 : 3e-12
end
legacy=read(joinpath(@__DIR__,"blas.jl"),String)
marker="@testset \"Level 2\""
@assert occursin(marker,legacy)
include_string(@__MODULE__,first(split(legacy,marker;limit=2)),joinpath(@__DIR__,"blas.jl"))
@testset "BLAS1 exceptional values and overlap" begin
 for T in TYPES
    R=real(T)
    @test iamax(T[1,NaN,NaN])==2
    @test iamax(T[Inf,1,NaN])==3
    @test iamax(T[2,-2,1])==1
    x=T[1,2,3];@test copy_vector!(x,x)==T[1,2,3]
    a=T[1,2,3,4];before=copy(a)
    copy_vector!(view(a,2:4),view(a,1:3));@test a[2:4]==before[1:3]
    for alpha in (T(.625),T<:Complex ? T(.4+.3im) : T(.4))
        x=T[1,2,3];y=T[4,5,6];old=copy(y)
        axpy!(y,x,alpha);@test y≈old+alpha*x
    end
    x=T[floatmax(R),-floatmax(R),floatmax(R),-floatmax(R),floatmax(R)]
    if T<:Real
        @test dotu(x,ones(T,5))==floatmax(R)
    end
 end
end
@testset "Bounded ROTMG vs exponent-tracked fallback" begin
 for T in (Float32,Float64),sample in 1:1000
    e=T===Float32 ? 14 : 126
    args=ntuple(i->ldexp(T(rand()+.5),rand(-e:e))*(i==1 || rand(Bool) ? one(T) : -one(T)),4)
    fast=rotmg(args...)
    robust=OxiBLAS._rotmg_scaled(args...)
    @test fast.params.flag==robust.params.flag
    for field in (:d1,:d2,:x1)
        @test getproperty(fast,field)≈getproperty(robust,field) rtol=64eps(T)
    end
    for field in (:h11,:h21,:h12,:h22)
        @test getproperty(fast.params,field)≈getproperty(robust.params,field) rtol=64eps(T)
    end
 end
end
