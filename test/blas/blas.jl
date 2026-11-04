@testset "Level 1" begin
    for T in TYPES, n in (0,1,7,101)
        x=randn(T,n); y=randn(T,n); old=copy(y); α=T(0.7)
        @test dotu(x,y)≈sum(x.*y) atol=tol(T)
        @test dotc(x,y)≈LA.dot(x,y) atol=tol(T)
        @test dot_kahan(x,y;conjugate=true)≈LA.dot(x,y) atol=tol(T)
        @test dot_pairwise(x,y;conjugate=true)≈LA.dot(x,y) atol=tol(T)
        @test nrm2(x)≈LA.norm(x) atol=tol(T)
        @test asum(x)≈sum(abs.(real.(x))+abs.(imag.(x))) atol=tol(T)
        @test axpy!(y,x,α)≈old+α*x
        @test scal!(y,α)≈α*(old+α*x)
        @test copy_vector!(y,x)==x
        if n>0
            @test iamax(x)==argmax(abs.(real.(x))+abs.(imag.(x)))
            @test iamin(x)==argmin(abs.(real.(x))+abs.(imag.(x)))
        else
            @test iamax(x)==iamin(x)==0
        end
        y=randn(T,n); xx=copy(x); yy=copy(y)
        swap!(x,y)
        @test x==yy && y==xx
        c,s,r=rotg(T(3),T(4))
        @test c*T(3)+s*T(4)≈r
        @test c*T(4)-conj(s)*T(3)≈0 atol=tol(T)
        rot!(x,y,c,s)
        @test x≈c*yy+s*xx
        @test y≈c*xx-conj(s)*yy
    end
    for T in (Float32,Float64)
        @test isfinite(nrm2(T[floatmax(T)/4,floatmax(T)/4]))
        @test nrm2(T[floatmin(T),floatmin(T)])>0
        @test isinf(nrm2(T[Inf,Inf]))
        @test isnan(nrm2(T[NaN,Inf]))
    end
    for (a,b) in ((3+4im,2-7im),(0im,3+4im),(3+4im,0im))
        c,s,r=rotg(ComplexF64(a),ComplexF64(b))
        @test c*a+s*b≈r
        @test c*b-conj(s)*a≈0 atol=1e-14
        @test c*c+abs2(s)≈1
    end
    x=collect(1.0:10); old=copy(x)
    axpy!(view(x,2:10),view(x,1:9),2)
    @test x[2:10]==old[2:10]+2old[1:9]
    @test_throws ArgumentError swap!(view(x,1:9),view(x,2:10))
    @test_throws DimensionMismatch dotu(ones(2),ones(3))
    for (a,b) in ((-3.0,4.0),(3.0,-4.0),(-4.0,3.0),(0.0,-1.0))
        c,s,r=rotg(a,b)
        @test c*a+s*b≈r
        @test c*b-s*a≈0 atol=1e-14
        @test signbit(r)==signbit(abs(a)>abs(b) ? a : b)
    end
end

@testset "Level 2" begin
    for T in TYPES
        A=randn(T,11,7); x=randn(T,7); y=randn(T,11); old=copy(y)
        @test gemv!(y,A,x,T(0.7),T(0.4))≈T(0.7)*A*x+T(0.4)*old rtol=tol(T)
        for tr in (:N,:T,:C)
            AA=op(A,tr); x=randn(T,size(AA,2)); y=fill(T(NaN),size(AA,1))
            @test gemv!(y,A,x;trans=tr)≈AA*x rtol=tol(T)
        end
        A=randn(T,9,9); x=randn(T,9)
        expected=A*x; gemv!(x,A,x)
        @test x≈expected rtol=tol(T)
        for upper in (true,false)
            uplo=upper ? :U : :L
            A=randn(T,9,9); x=randn(T,9); y=randn(T,9); old=copy(y)
            S=LA.Symmetric(A,upper ? :U : :L); H=LA.Hermitian(A,upper ? :U : :L)
            @test symv!(y,A,x,T(0.3),T(0.4);uplo)≈T(0.3)*S*x+T(0.4)*old rtol=tol(T)
            y=copy(old)
            @test hemv!(y,A,x,T(0.3),T(0.4);uplo)≈T(0.3)*H*x+T(0.4)*old rtol=tol(T)
            for herm in (false,true), rank2 in (false,true)
                X=copy(A); α=herm && !rank2 ? real(T)(0.4) : T(0.4)
                expected=Matrix(herm ? H : S)
                v=herm ? x*x' : x*transpose(x)
                rank2 && (v=herm ? x*old'+old*x' : x*transpose(old)+old*transpose(x))
                expected+=α*v
                f=herm ? (rank2 ? her2! : her!) : (rank2 ? syr2! : syr!)
                rank2 ? f(X,x,old,α;uplo) : f(X,x,α;uplo)
                @test Matrix(herm ? LA.Hermitian(X,uplo) : LA.Symmetric(X,uplo))≈expected rtol=tol(T)
            end
            for tr in (:N,:T,:C), unit in (false,true)
                AA=copy(A)
                AA[LA.diagind(AA)].+=T(8)
                tri=upper ? (unit ? LA.UnitUpperTriangular(AA) : LA.UpperTriangular(AA)) : (unit ? LA.UnitLowerTriangular(AA) : LA.LowerTriangular(AA))
                v=randn(T,9); old=copy(v)
                @test trmv!(v,AA;uplo,trans=tr,unit)≈op(tri,tr)*old rtol=tol(T)
                @test trsv!(v,AA;uplo,trans=tr,unit)≈old rtol=tol(T) atol=tol(T)
            end
        end
        A=randn(T,11,7); old=copy(A); x=randn(T,11); y=randn(T,7)
        @test ger!(A,x,y)≈old+x*transpose(y) rtol=tol(T)
        A=copy(old)
        @test gerc!(A,x,y)≈old+x*y' rtol=tol(T)
    end
    @test_throws OxiBLAS.SingularException trsv!(ones(3),zeros(3,3))
end

@testset "Level 3" begin
    for T in TYPES, side in (:L,:R), uplo in (:U,:L)
        A=randn(T,9,9); A[LA.diagind(A)].+=T(8)
        B=randn(T,side===:L ? (9,7) : (7,9)); C=randn(T,size(B)); old=copy(C)
        for herm in (false,true)
            S=herm ? LA.Hermitian(A,uplo) : LA.Symmetric(A,uplo)
            expected=side===:L ? S*B : B*S
            f=herm ? hemm! : symm!
            C=copy(old)
            @test f(C,A,B,T(0.7),T(0.2);side,uplo)≈T(0.7)*expected+T(0.2)*old rtol=tol(T)
        end
        for tr in (:N,:T,:C), unit in (true,false)
            tri=uplo===:U ? (unit ? LA.UnitUpperTriangular(A) : LA.UpperTriangular(A)) : (unit ? LA.UnitLowerTriangular(A) : LA.LowerTriangular(A))
            S=op(tri,tr); X=copy(B)
            @test trmm!(X,A,T(0.7);side,uplo,trans=tr,unit)≈T(0.7)*(side===:L ? S*B : B*S) rtol=tol(T)
            trsm!(X,A,inv(T(0.7));side,uplo,trans=tr,unit)
            @test X≈B rtol=tol(T) atol=tol(T)
        end
    end
    for T in TYPES, uplo in (:U,:L), herm in (false,true), rank2 in (false,true), trans in (:N,:T)
        tr=herm && trans===:T ? :C : trans
        A=randn(T,trans===:N ? (9,7) : (7,9)); B=randn(T,size(A)); C=randn(T,9,9)
        aa,bb=op(A,tr),op(B,tr)
        old=Matrix(herm ? LA.Hermitian(C,uplo) : LA.Symmetric(C,uplo))
        product=aa*(herm ? aa' : transpose(aa))
        rank2 && (product=aa*(herm ? bb' : transpose(bb))+bb*(herm ? aa' : transpose(aa)))
        expected=T(0.4)*product+T(0.2)*old
        f=herm ? (rank2 ? her2k! : herk!) : (rank2 ? syr2k! : syrk!)
        rank2 ? f(C,A,B,T(0.4),T(0.2);uplo,trans=tr) : f(C,A,T(0.4),T(0.2);uplo,trans=tr)
        @test Matrix(herm ? LA.Hermitian(C,uplo) : LA.Symmetric(C,uplo))≈expected rtol=tol(T) atol=tol(T)
    end
    # α=0 does not inspect triangular input or NaN output.
    @test trsm!(fill(NaN,3,2),UnreadableMatrix{Float64}((3,3)),0.0)==zeros(3,2)
    @test_throws DimensionMismatch trsm!(ones(2,3),ones(4,4))
end
