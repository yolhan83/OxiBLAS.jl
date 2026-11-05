# references: test/helpers/level1_references.jl (testsetup.jl)
struct NoReadVector{T} <: AbstractVector{T}
    n::Int
end
Base.size(x::NoReadVector)=(x.n,)
Base.getindex(::NoReadVector,i::Int)=error("unexpected vector read")
Base.setindex!(::NoReadVector,v,i::Int)=error("unexpected vector write")
struct SelectedParameters{T} <: AbstractVector{T}
    data::Vector{T}
    readable::Vector{Int}
end
Base.size(x::SelectedParameters)=size(x.data)
Base.getindex(x::SelectedParameters,i::Int)=i in x.readable ? x.data[i] : error("unused rotation parameter read")

@testset "Extended Float32 dot products" begin
    for n in (0,1,7,65,257,4097)
        x=randn(Float32,n); y=randn(Float32,n); alpha=0.7f0
        exact=setprecision(BigFloat,256) do
            sum(BigFloat(x[i])*BigFloat(y[i]) for i in eachindex(x);init=BigFloat(0))
        end
        ordered=0.0; biased=Float64(alpha)
        for i in eachindex(x)
            product=Float64(x[i])*Float64(y[i])
            ordered+=product; biased+=product
        end
        @test dsdot(x,y) isa Float64
        @test sdsdot(alpha,x,y) isa Float32
        # Reassociated Float64 accumulation (as OpenBLAS): agreement with the
        # ordered sum within the Float64 summation bound, not bitwise.
        bound=8eps(Float64)*max(1,n)*max(1,sum(abs.(Float64.(x).*Float64.(y))))
        @test abs(dsdot(x,y)-ordered)<=bound
        @test abs(Float64(sdsdot(alpha,x,y))-biased)<=bound+eps(Float32)*abs(biased)
        @test abs(BigFloat(dsdot(x,y))-exact)<=bound
        # Some OpenBLAS x86 kernels round the products to Float32 before the
        # Float64 accumulation: the reference itself carries an eps(Float32)
        # error per product. Exactness is checked against BigFloat above.
        refbound=bound+2eps(Float32)*sum(abs.(Float64.(x).*Float64.(y)))
        @test abs(dsdot(x,y)-reference_dsdot(x,y))<=refbound
        @test sdsdot(alpha,x,y)≈reference_sdsdot(alpha,x,y) atol=16eps(Float32)*max(1,abs(Float64(biased)))
        backingx=randn(Float32,2n+7); backingy=randn(Float32,3n+5)
        xv=view(backingx,2n+2:-2:4); yv=view(backingy,2:3:3n-1)
        oldx,oldy=copy(backingx),copy(backingy)
        @test length(xv)==length(yv)==n
        # Strided views and contiguous copies may use different partial sums.
        vbound=8eps(Float64)*max(1,n)*max(1,sum(abs.(Float64.(xv).*Float64.(yv))))
        @test abs(dsdot(xv,yv)-dsdot(collect(xv),collect(yv)))<=vbound
        @test abs(Float64(sdsdot(alpha,xv,yv))-Float64(sdsdot(alpha,collect(xv),collect(yv))))<=vbound+2eps(Float32)*abs(Float64(sdsdot(alpha,xv,yv)))
        @test backingx==oldx && backingy==oldy
    end
    x=Float32[2^24,1,-2^24]; y=ones(Float32,3)
    @test dsdot(x,y)==1.0
    @test sdsdot(0f0,x,y)==1f0
    for v in (floatmax(Float32),floatmin(Float32),nextfloat(0f0))
        @test dsdot(Float32[v],Float32[v])==Float64(v)^2
    end
    @test dsdot(Float32[floatmax(Float32),floatmax(Float32)],Float32[floatmax(Float32),-floatmax(Float32)])==0.0
    @test isinf(sdsdot(0f0,Float32[floatmax(Float32)],Float32[floatmax(Float32)]))
    @test sdsdot(3f0,Float32[],Float32[])==3f0
    @test isnan(dsdot(Float32[Inf,Inf],Float32[1,-1]))
    @test isnan(sdsdot(NaN32,Float32[],Float32[]))
    @test_throws DimensionMismatch dsdot(NoReadVector{Float32}(3),NoReadVector{Float32}(4))
    @test_throws DimensionMismatch sdsdot(0f0,NoReadVector{Float32}(3),NoReadVector{Float32}(4))
    @test_throws MethodError dsdot(ones(Float64,2),ones(Float64,2))
end

@testset "Modified Givens application" begin
    for T in (Float32,Float64), flag in (-2,-1,0,1), n in (0,1,7,65,257)
        coeff=T[flag,0.7,-0.3,0.4,1.2]; p=RotmParams(coeff)
        H=rotm_matrix(p); x=randn(T,n); y=randn(T,n); oldx=copy(x); oldy=copy(y)
        expected=H*vcat(transpose(x),transpose(y))
        xx,yy=rotm!(x,y,p)
        @test xx===x && yy===y
        @test x≈expected[1,:] rtol=8eps(T) atol=8eps(T)
        @test y≈expected[2,:] rtol=8eps(T) atol=8eps(T)
        bx,by=copy(oldx),copy(oldy); reference_rotm!(bx,by,coeff)
        @test x≈bx rtol=8eps(T) atol=8eps(T)
        @test y≈by rtol=8eps(T) atol=8eps(T)
        backingx=fill(T(99),2n+5); backingy=fill(T(77),3n+5)
        xv=view(backingx,2n+2:-2:4); yv=view(backingy,2:3:3n-1)
        xv.=oldx; yv.=oldy
        beforex=copy(backingx); beforey=copy(backingy)
        rotm!(xv,yv,coeff)
        @test xv≈x rtol=8eps(T) atol=8eps(T)
        @test yv≈y rtol=8eps(T) atol=8eps(T)
        @test all(backingx[i]==beforex[i] for i in eachindex(backingx) if !(i in 2n+2:-2:4))
        @test all(backingy[i]==beforey[i] for i in eachindex(backingy) if !(i in 2:3:3n-1))
    end
    for T in (Float32,Float64)
        for (flag,reads) in ((-2,[1]),(0,[1,3,4]),(1,[1,2,5]))
            p=SelectedParameters(T[flag,0.7,-0.3,0.4,1.2],reads)
            x=ones(T,9); y=ones(T,9)
            @test rotm!(x,y,p)==rotm!(ones(T,9),ones(T,9),RotmParams(p))
        end
        @test rotm!(NoReadVector{T}(9),NoReadVector{T}(9),RotmParams(T))[1] isa NoReadVector
        @test rotm!(ones(T,9),ones(T,9),RotmParams(T;flag=0,h11=NaN,h22=NaN,h21=0,h12=0))[1]==ones(T,9)
        @test rotm!(ones(T,9),ones(T,9),RotmParams(T;flag=1,h12=NaN,h21=NaN,h11=1,h22=1))==(fill(T(2),9),zeros(T,9))
        storage=T[0,0,-0.3,0.4,0,6,7,8,9]; x=view(storage,1:7); y=ones(T,7); p=view(storage,1:5)
        expected=rotm_matrix(RotmParams(p))*vcat(transpose(copy(x)),transpose(copy(y)))
        rotm!(x,y,p)
        @test x≈expected[1,:] rtol=8eps(T) atol=8eps(T)
        @test y≈expected[2,:] rtol=8eps(T) atol=8eps(T)
        a=ones(T,10); old=copy(a)
        @test_throws ArgumentError rotm!(view(a,1:9),view(a,2:10),RotmParams(T;flag=0))
        @test a==old
        @test_throws ArgumentError rotm!(view(a,[1,1,2]),ones(T,3),RotmParams(T;flag=0))
        @test_throws ArgumentError RotmParams(T;flag=2)
        @test_throws ArgumentError RotmParams(T;flag=NaN)
        @test_throws DimensionMismatch RotmParams(ones(T,4))
        @test_throws DimensionMismatch rotm!(NoReadVector{T}(3),NoReadVector{T}(4),RotmParams(T))
    end
end

function verify_rotmg(d1,d2,x,y,F)
    T=typeof(d1); H=rotm_matrix(F.params)
    setprecision(BigFloat,512) do
        h=BigFloat.(H); a,b,c,z=BigFloat(d1),BigFloat(d2),BigFloat(x),BigFloat(y)
        aa,bb,xx=BigFloat(F.d1),BigFloat(F.d2),BigFloat(F.x1)
        bound=BigFloat(256eps(T))
        if F.params.flag!=T(-2)
            @test abs(h[2,1]*c+h[2,2]*z)<=bound*max(abs(h[2,1]*c),abs(h[2,2]*z))
            @test abs(h[1,1]*c+h[1,2]*z-xx)<=bound*max(abs(h[1,1]*c),abs(h[1,2]*z),abs(xx))
        end
        @test abs(aa*h[1,1]^2+bb*h[2,1]^2-a)<=bound*abs(a)
        @test abs(aa*h[1,2]^2+bb*h[2,2]^2-b)<=bound*abs(b)
        @test abs(aa*h[1,1]*h[1,2]+bb*h[2,1]*h[2,2])<=bound*sqrt(abs(a*b))
    end
end

@testset "Modified Givens generation and independent invariants" begin
    for T in (Float32,Float64)
        for _ in 1:200
            d1=exp2(T(16rand()-8)); d2=exp2(T(16rand()-8)); x=T(randn()); y=T(randn())
            F=rotmg(d1,d2,x,y); R=reference_rotmg(d1,d2,x,y)
            @test F.d1≈R.d1 rtol=32eps(T)
            @test F.d2≈R.d2 rtol=32eps(T)
            @test F.x1≈R.x1 rtol=32eps(T)
            @test rotm_matrix(F.params)≈rotm_matrix(R.params) rtol=32eps(T)
            verify_rotmg(d1,d2,x,y,F)
            state=T[d1,d2,x]; params=rotmg!(state,y)
            @test state==T[F.d1,F.d2,F.x1]
            @test params==F.params
        end
        exponents=T===Float32 ? (-149,-100,-48,-24,0,24,48,100,127) : (-1074,-1000,-500,-48,-24,0,24,48,500,1000,1023)
        accepted=0; total=0
        for e1 in exponents, e2 in exponents, (x,y) in ((T(3),T(4)),(T(-4),T(3)),(T(0.5),T(-2)))
            d1=ldexp(one(T),e1); d2=ldexp(one(T),e2); total+=1
            F=try rotmg(d1,d2,x,y) catch err
                @test err isa OverflowError
                continue
            end
            accepted+=1
            @test all(isfinite,(F.d1,F.d2,F.x1))
            @test F.d1==0 || 2.0^-24<F.d1<2.0^24
            @test F.d2==0 || 2.0^-24<abs(F.d2)<2.0^24
            verify_rotmg(d1,d2,x,y,F)
        end
        @test accepted>0.7total
        for e in (-48,0,48), r in (T(-0.1),T(-0.5))
            d1=ldexp(one(T),e); d2=r*d1
            F=rotmg(d1,d2,T(4),T(1))
            verify_rotmg(d1,d2,T(4),T(1),F)
        end
        for (d1,d2,x,y) in ((T(-1),T(1),T(3),T(4)),(T(1),T(-9),T(1),T(2)),(T(1),T(-1),T(1),T(1)))
            F=rotmg(d1,d2,x,y)
            @test F.d1==F.d2==F.x1==0
            @test rotm_matrix(F.params)==zeros(T,2,2)
        end
        for (d1,d2,x,y) in ((T(1),T(0),T(3),T(4)),(T(1),T(9),T(3),T(0)))
            F=rotmg(d1,d2,x,y)
            @test (F.d1,F.d2,F.x1)==(d1,d2,x)
            @test F.params.flag==T(-2)
        end
        F=rotmg(T(0),T(1),T(3),T(4)); verify_rotmg(T(0),T(1),T(3),T(4),F)
        for values in ((T(Inf),T(1),T(1),T(1)),(T(1),T(NaN),T(1),T(1)),(T(1),T(1),T(1),T(Inf)))
            @test_throws ArgumentError rotmg(values...)
        end
        state=T[floatmax(T),floatmax(T),floatmax(T)]; old=copy(state)
        @test_throws OverflowError rotmg!(state,floatmax(T))
        @test state==old
        @test_throws OverflowError rotmg(nextfloat(zero(T)),nextfloat(zero(T)),nextfloat(zero(T)),nextfloat(zero(T)))
        @test_throws DimensionMismatch rotmg!(ones(T,4),one(T))
        @test_throws ArgumentError rotmg!(view(ones(T,3),[1,1,2]),one(T))
    end
end

@testset "Modified rotations SIMD dispatch, tails and canaries" begin
    for T in (Float32,Float64), kind in (:scalar,:neon,:avx2,:avx512)
        available=try OxiBLAS._kernel_kind(kind); true catch err; err isa ArgumentError ? false : rethrow() end
        if !available
            x=ones(T,33); y=ones(T,33)
            @test_throws ArgumentError rotm!(x,y,RotmParams(T;flag=0);kernel=kind)
            @test x==y==ones(T,33)
            continue
        end
        w=kernel_info(T;kernel=kind).lanes
        for n in unique((0,1,w-1,w,w+1,4w-1,4w,4w+1,65)), flag in (-1,0,1)
            p=RotmParams(T;flag,h11=.7,h21=-.3,h12=.4,h22=1.2)
            a=randn(T,n); b=randn(T,n); x=copy(a); y=copy(b)
            rotm!(x,y,p;kernel=:scalar)
            storex=fill(T(99),n+7); storey=fill(T(77),n+7)
            xv=view(storex,3:n+2); yv=view(storey,4:n+3)
            xv.=a; yv.=b
            GC.gc()
            rotm!(xv,yv,p;kernel=kind)
            @test xv≈x rtol=8eps(T) atol=8eps(T)
            @test yv≈y rtol=8eps(T) atol=8eps(T)
            @test all(storex[i]==99 for i in eachindex(storex) if !(i in 3:n+2))
            @test all(storey[i]==77 for i in eachindex(storey) if !(i in 4:n+3))
        end
    end
end
