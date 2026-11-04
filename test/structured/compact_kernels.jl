# A guarded buffer makes reads of padding/unit diagonals fail immediately.
struct CompactGuardVector{T} <: AbstractVector{T}
    data::Vector{T}
    allowed::Vector{Bool}
end
Base.size(A::CompactGuardVector)=size(A.data)
Base.getindex(A::CompactGuardVector,i::Int)=A.allowed[i] ? A.data[i] : error("unused compact entry read")
Base.dataids(A::CompactGuardVector)=Base.dataids(A.data)
struct CompactGuardMatrix{T} <: AbstractMatrix{T}
    data::Matrix{T}
    allowed::Matrix{Bool}
end
Base.size(A::CompactGuardMatrix)=size(A.data)
Base.getindex(A::CompactGuardMatrix,i::Int,j::Int)=A.allowed[i,j] ? A.data[i,j] : error("unused band entry read")
Base.dataids(A::CompactGuardMatrix)=Base.dataids(A.data)

struct CompactNoReadVector{T} <: AbstractVector{T}
    n::Int
end
Base.size(x::CompactNoReadVector)=(x.n,)
Base.getindex(::CompactNoReadVector,i::Int)=error("unexpected compact vector read")
struct CompactNoReadMatrix{T} <: AbstractMatrix{T}
    dims::Tuple{Int,Int}
end
Base.size(A::CompactNoReadMatrix)=A.dims
Base.getindex(::CompactNoReadMatrix,i::Int,j::Int)=error("unexpected compact matrix read")
struct NoReadOutputVector{T} <: AbstractVector{T}
    data::Vector{T}
    written::Vector{Bool}
end
NoReadOutputVector(data::Vector{T}) where T=NoReadOutputVector(data,fill(false,length(data)))
Base.size(x::NoReadOutputVector)=size(x.data)
Base.getindex(x::NoReadOutputVector,i::Int)=x.written[i] ? x.data[i] : error("old output read")
Base.setindex!(x::NoReadOutputVector,v,i::Int)=(x.data[i]=v; x.written[i]=true; v)

function guarded_band(P)
    allowed=fill(false,size(P.data))
    for j in 1:size(P,2), i in max(1,j-P.ku):min(size(P,1),j+P.kl)
        P.kind===:triangular && P.unit && i==j && continue
        allowed[P.ku+1+i-j,j]=true
    end
    BandedMatrix(CompactGuardMatrix(copy(P.data),allowed),size(P,1);kl=P.kl,ku=P.ku,kind=P.kind,uplo=P.upper ? :U : :L,unit=P.unit)
end
function guarded_packed(P)
    allowed=fill(true,length(P.data))
    if P.kind===:triangular && P.unit
        for i in 1:P.n
            allowed[OxiBLAS._pindex(P,i,i)]=false
        end
    end
    PackedMatrix(CompactGuardVector(copy(P.data),allowed),P.n;kind=P.kind,uplo=P.upper ? :U : :L,unit=P.unit)
end

@testset "Direct general band traversal" begin
    for T in TYPES, (m,n) in ((0,0),(0,7),(7,0),(1,1),(17,35),(35,17),(65,67)), (kl,ku) in ((0,0),(1,3),(5,2),(80,0)), tr in (:N,:T,:C)
        A=randn(T,m,n); P=band_matrix(A;kl,ku); D=Matrix(P)
        G=guarded_band(P); aa=op(D,tr)
        x=randn(T,size(aa,2)); y=randn(T,size(aa,1)); old=copy(y)
        α=T<:Real ? T(.7) : T(.7+.3im); β=T(-.2)
        @test gbmv!(y,G,x,α,β;trans=tr)≈α*aa*x+β*old rtol=5tol(T) atol=5tol(T)
        bx=randn(T,2length(x)+5); by=fill(T(99),2length(y)+7)
        xv=view(bx,2length(x)+2:-2:4); yv=view(by,3:2:2length(y)+1)
        xv.=x; yv.=old
        gbmv!(yv,P,xv,α,β;trans=tr)
        @test yv≈y rtol=5tol(T) atol=5tol(T)
        @test all(by[i]==99 for i in eachindex(by) if !(i in 3:2:2length(y)+1))
    end
end

@testset "Direct symmetric packed and band traversal" begin
    for T in TYPES, n in (0,1,17,35,65), uplo in (:U,:L), herm in (false,true), format in (:packed,:banded)
        A=randn(T,n,n)
        if herm && T<:Complex
            for i in 1:n; A[i,i]=complex(real(A[i,i]),real(T)(NaN)); end
        end
        kind=herm ? :hermitian : :symmetric
        P=format===:packed ? pack_matrix(A;uplo,kind) : band_matrix(A;kl=4,ku=4,uplo,kind)
        if herm && T<:Complex
            for i in 1:n
                if format===:packed
                    d=OxiBLAS._pindex(P,i,i); P.data[d]=complex(real(P.data[d]),real(T)(NaN))
                else
                    P.data[P.ku+1,i]=complex(real(P.data[P.ku+1,i]),real(T)(NaN))
                end
            end
        end
        D=Matrix(P); guard=format===:packed ? guarded_packed(P) : guarded_band(P)
        f=format===:packed ? (herm ? hpmv! : spmv!) : (herm ? hbmv! : sbmv!)
        x=randn(T,n); old=randn(T,n); y=copy(old)
        α=T<:Real ? T(.7) : T(.7+.3im); β=T(-.2)
        @test f(y,guard,x,α,β)≈α*D*x+β*old rtol=5tol(T) atol=5tol(T)
        z=copy(x); f(z,P,z,α,β)
        @test z≈α*D*x+β*x rtol=5tol(T) atol=5tol(T)
    end
end

@testset "Compact triangular dependencies, units and layouts" begin
    for T in TYPES, n in (0,1,17,35,65), uplo in (:U,:L), unit in (false,true), tr in (:N,:T,:C), format in (:packed,:banded)
        A=randn(T,n,n)/max(1,n); A[LA.diagind(A)].+=T(2)
        P=format===:packed ? pack_matrix(A;uplo,kind=:triangular,unit) : band_matrix(A;kl=4,ku=4,uplo,kind=:triangular,unit)
        D=Matrix(P); guard=format===:packed ? guarded_packed(P) : guarded_band(P)
        mv,sv=format===:packed ? (tpmv!,tpsv!) : (tbmv!,tbsv!)
        old=randn(T,n); x=copy(old); expected=op(D,tr)*old
        @test mv(x,guard;trans=tr)===x
        @test x≈expected rtol=5tol(T) atol=5tol(T)
        x=copy(old); sv(x,guard;trans=tr)
        @test op(D,tr)*x≈old rtol=5tol(T) atol=5tol(T)
        backing=fill(T(99),2n+7); v=view(backing,2n+2:-2:4); v.=old
        mv(v,P;trans=tr,kernel=:scalar)
        @test v≈expected rtol=5tol(T) atol=5tol(T)
        sv(v,P;trans=tr)
        @test v≈old rtol=5tol(T) atol=5tol(T)
        @test all(backing[i]==99 for i in eachindex(backing) if !(i in 2n+2:-2:4))
    end
end

@testset "Compact rank updates and overlapping inputs" begin
    for T in TYPES, n in (0,1,17,35), uplo in (:U,:L), herm in (false,true), rank2 in (false,true)
        A=randn(T,n,n); x=randn(T,n); y=randn(T,n)
        if herm && T<:Complex
            for i in 1:n; A[i,i]=complex(real(A[i,i]),real(T)(NaN)); end
        end
        P=pack_matrix(A;uplo,kind=herm ? :hermitian : :symmetric)
        if herm && T<:Complex
            for i in 1:n
                d=OxiBLAS._pindex(P,i,i); P.data[d]=complex(real(P.data[d]),real(T)(NaN))
            end
        end
        old=Matrix(P)
        α=herm && !rank2 ? real(T)(.7) : T<:Real ? T(.7) : T(.7+.3im)
        rhs(z)=herm ? adjoint(z) : transpose(z)
        expected=old+(rank2 ? α*x*rhs(y)+(herm ? conj(α) : α)*y*rhs(x) : α*x*rhs(x))
        f=herm ? (rank2 ? hpr2! : hpr!) : (rank2 ? spr2! : spr!)
        rank2 ? f(P,x,y,α) : f(P,x,α)
        @test Matrix(P)≈expected rtol=5tol(T) atol=5tol(T)
        if n>0
            # x/y overlap the matrix storage: snapshot both before any update.
            xx=view(P.data,1:n); yy=view(P.data,length(P.data)-n+1:length(P.data))
            a,b=copy(xx),copy(yy); old=Matrix(P)
            expected=old+(rank2 ? α*a*rhs(b)+(herm ? conj(α) : α)*b*rhs(a) : α*a*rhs(a))
            rank2 ? f(P,xx,yy,α) : f(P,xx,α)
            @test Matrix(P)≈expected rtol=5tol(T) atol=5tol(T)
        end
    end
end

@testset "Compact storage aliases, no-read and failure guarantees" begin
    for T in TYPES
        n=17; α=T(.7); β=T(-.2)
        # Matrix storage and output intersect for general band multiply.
        P=band_matrix(randn(T,n,n);kl=2,ku=3); D=Matrix(P); y=view(P.data,1,:); old=copy(y); x=randn(T,n)
        gbmv!(y,P,x,α,β)
        @test y≈α*D*x+β*old rtol=5tol(T) atol=5tol(T)
        for format in (:packed,:banded), uplo in (:U,:L), tr in (:N,:T,:C)
            A=randn(T,n,n)/n; A[LA.diagind(A)].+=T(2)
            P=format===:packed ? pack_matrix(A;uplo,kind=:triangular) : band_matrix(A;kl=3,ku=3,uplo,kind=:triangular)
            D=Matrix(P); x=format===:packed ? view(P.data,1:n) : view(P.data,1,:); old=copy(x)
            mv=format===:packed ? tpmv! : tbmv!
            mv(x,P;trans=tr)
            @test x≈op(D,tr)*old rtol=5tol(T) atol=5tol(T)
        end
        # β=0 output is write-only; α=0 avoids both matrix and input reads.
        for f in (spmv!,hpmv!,sbmv!,hbmv!)
            herm=f in (hpmv!,hbmv!); kind=herm ? :hermitian : :symmetric
            P=f in (spmv!,hpmv!) ? pack_matrix(randn(T,n,n);kind) : band_matrix(randn(T,n,n);kl=2,ku=2,kind)
            out=NoReadOutputVector(fill(T(NaN),n)); x=randn(T,n)
            f(out,P,x)
            @test out.data≈Matrix(P)*x rtol=5tol(T) atol=5tol(T)
        end
        P=BandedMatrix(CompactNoReadMatrix{T}((6,n)),n;kl=2,ku=3)
        @test gbmv!(fill(T(NaN),n),P,CompactNoReadVector{T}(n),zero(T),zero(T))==zeros(T,n)
        for format in (:packed,:banded)
            P=format===:packed ? pack_matrix(Matrix{T}(LA.I,n,n);kind=:triangular) : band_matrix(Matrix{T}(LA.I,n,n);kl=2,ku=2,kind=:triangular)
            if format===:packed; P.data[OxiBLAS._pindex(P,9,9)]=0; else P.data[P.ku+1,9]=0; end
            x=ones(T,n); old=copy(x)
            sv=format===:packed ? tpsv! : tbsv!
            @test_throws OxiBLAS.SingularException sv(x,P)
            @test x==old
        end
    end
end

@testset "Compact quick returns and noncontiguous coefficient buffers" begin
    for T in TYPES, kind in (:symmetric,:hermitian)
        n=17; f=kind===:symmetric ? spmv! : hpmv!
        P=PackedMatrix(CompactNoReadVector{T}(n*(n+1)÷2),n;kind)
        @test f(fill(T(NaN),n),P,CompactNoReadVector{T}(n),zero(T),zero(T))==zeros(T,n)
        B=BandedMatrix(CompactNoReadMatrix{T}((5,n)),n;kl=2,ku=2,kind)
        bf=kind===:symmetric ? sbmv! : hbmv!
        @test bf(fill(T(NaN),n),B,CompactNoReadVector{T}(n),zero(T),zero(T))==zeros(T,n)
        Q=pack_matrix(randn(T,n,n);kind)
        a=copy(Q.data); backing=fill(T(99),2length(a)+5); values=view(backing,2length(a)+2:-2:4); values.=a
        R=PackedMatrix(values,n;kind)
        x=randn(T,n); old=Matrix(R)
        rf=kind===:symmetric ? spr! : hpr!
        rf(R,x,real(T)(.7))
        expected=old+T(.7)*x*(kind===:symmetric ? transpose(x) : adjoint(x))
        @test Matrix(R)≈expected rtol=5tol(T) atol=5tol(T)
        @test all(backing[i]==99 for i in eachindex(backing) if !(i in 2length(a)+2:-2:4))
        if kind===:symmetric
            @test spr!(P,CompactNoReadVector{T}(n),zero(T))===P
            @test spr2!(P,CompactNoReadVector{T}(n),CompactNoReadVector{T}(n),zero(T))===P
        else
            @test hpr!(Q,CompactNoReadVector{T}(n),zero(T))===Q
            @test hpr2!(Q,CompactNoReadVector{T}(n),CompactNoReadVector{T}(n),zero(T))===Q
        end
    end
    for T in TYPES, trans in (:N,:T,:C)
        n=17; P=band_matrix(randn(T,n,n);kl=2,ku=3)
        backing=fill(T(99),2size(P.data,1)+3,2n+3)
        data=view(backing,2:2:2size(P.data,1),2n:-2:2); data.=P.data
        Q=BandedMatrix(data,n;kl=2,ku=3)
        x=randn(T,n); y=zeros(T,n)
        @test gbmv!(y,Q,x;trans)≈op(Matrix(P),trans)*x rtol=5tol(T) atol=5tol(T)
    end
    for T in TYPES, format in (:packed,:banded), uplo in (:U,:L)
        n=9; A=randn(T,n,n); A[LA.diagind(A)].+=T(2)
        # Packing also ignores implicit diagonals of a dense triangular source.
        guard=CompactGuardMatrix(A,[i!=j for i in 1:n,j in 1:n])
        Q=format===:packed ? pack_matrix(guard;uplo,kind=:triangular,unit=true) : band_matrix(guard;kl=2,ku=2,uplo,kind=:triangular,unit=true)
        @test all(Q[i,i]==one(T) for i in 1:n)
        F=format===:packed ? pack_matrix(A;uplo,kind=:triangular) : band_matrix(A;kl=2,ku=2,uplo,kind=:triangular)
        # Strongly varying diagonal scales; use BigFloat solve as reference.
        for i in 1:n
            scale=T(exp2(-(real(T)===Float32 ? 12 : 36)*(i-1)/(n-1)))
            if format===:packed; F.data[OxiBLAS._pindex(F,i,i)]=scale; else F.data[F.ku+1,i]=scale; end
        end
        b=randn(T,n); D=Matrix(F)
        ref=setprecision(BigFloat,256) do
            R=T<:Real ? BigFloat : Complex{BigFloat}
            R.(D)\R.(b)
        end
        x=copy(b); (format===:packed ? tpsv! : tbsv!)(x,F)
        @test LA.norm(x-T.(ref))<=20tol(T)*max(1,LA.norm(T.(ref)))
        @test_throws ArgumentError (format===:packed ? tpmv! : tbmv!)(view(ones(T,n),[1,1,2,3,4,5,6,7,8]),F)
    end
end

@testset "Structural band zeros are not multiplied" begin
    for T in (Float32,Float64)
        n=17; x=ones(T,n); x[9]=T(Inf)
        P=BandedMatrix(ones(T,1,n),n)
        y=zeros(T,n); gbmv!(y,P,x)
        @test isequal(y,x)
        for unit in (false,true)
            Q=BandedMatrix(ones(T,1,n),n;kind=:triangular,unit)
            @test isequal(tbmv!(copy(x),Q),x)
            @test isequal(tbsv!(copy(x),Q),x)
        end
    end
end

@testset "Compact triangular extreme diagonal scales" begin
    for T in TYPES, format in (:packed,:banded), trans in (:N,:T,:C)
        R=real(T); d=R[floatmax(R)/16,floatmin(R),nextfloat(zero(R)),one(R),eps(R)]
        A=zeros(T,5,5)
        for i in 1:5; A[i,i]=d[i]; end
        P=format===:packed ? pack_matrix(A;kind=:triangular) : band_matrix(A;kind=:triangular)
        mv,sv=format===:packed ? (tpmv!,tpsv!) : (tbmv!,tbsv!)
        x=ones(T,5); mv(x,P;trans)
        @test x==T.(d)
        sv(x,P;trans)
        @test x==ones(T,5)
    end
end

@testset "Compact element types are checked before mutation" begin
    x=ones(Float64,3); old=copy(x)
    P=pack_matrix(ones(Float32,3,3);kind=:triangular)
    @test_throws ArgumentError tpsv!(x,P)
    @test x==old
    S=pack_matrix(ones(Float32,3,3))
    @test_throws ArgumentError spmv!(x,S,old)
    @test x==old
end
