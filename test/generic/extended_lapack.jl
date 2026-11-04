@testset "Extended LU/Cholesky/QR/LDL and compact solves" begin
    setprecision(BigFloat,512) do
        for T in EXT_TYPES
            for n in (0,1,5)
                A=T.([i==j ? 6.0+i/4 : (i+2j)%5/16 for i in 1:n,j in 1:n])
                T<:Complex && (A .+= T(im/16)); saved=copy(A); b=T.([i/8+j for i in 1:n,j in 1:2])
                for pivot in (:partial,:complete,:rook), blocksize in (1,3)
                    F=lu_factor(A;pivot,blocksize)
                    q=pivot===:partial ? collect(1:n) : F.q
                    check_extended_array(F.L*F.U,extref(A)[F.p,q],T;factor=100)
                    X=solve(F,b); check_extended_array(extref(A)*extref(X),extref(b),T;factor=100)
                    @test A==saved
                    if T<:Union{Float16,Complex{Float16}}
                        @test eltype(F.factors)==(T===Float16 ? Float32 : ComplexF32)
                    else
                        @test eltype(F.factors)===T
                    end
                    for trans in (:N,:T,:C)
                        X=solve_transposed(F,b;trans)
                        check_extended_array(op(extref(A),trans)*extref(X),extref(b),T;factor=100)
                    end
                    @test abs(extref(determinant(A))-LA.det(extref(A)))<=100exttol(T)*max(1,abs(LA.det(extref(A))))
                    if n>0
                        logvalue,phase=logabsdet_factor(A); detref=LA.det(extref(A))
                        @test abs(extref(logvalue)-log(abs(detref)))<100exttol(T)
                        @test abs(extref(phase)-detref/abs(detref))<100exttol(T)
                    end
                end
                H=T.([i==j ? 8.0+i : 1/8 for i in 1:n,j in 1:n])
                if T<:Complex
                    for i in 2:n
                        H[i,i-1]+=T(im/16);H[i-1,i]=conj(H[i,i-1])
                    end
                end
                for blocksize in (1,3)
                    F=cholesky_factor(H;blocksize)
                    check_extended_array(F.L*F.L',extref(H),T;factor=100)
                    X=solve(F,b); check_extended_array(extref(H)*extref(X),extref(b),T;factor=100)
                end
                if n>0
                    for hermitian in (false,true), uplo in (:L,:U)
                        S=copy(H); hermitian || (S=Matrix(transpose(S))+S)
                        for i in 1:n
                            S[i,i]=T(isodd(i) ? 4 : -4)
                        end
                        F=ldlt_factor(S;hermitian,uplo)
                        check_extended_array(F.L*F.D*(hermitian ? F.L' : transpose(F.L)),extref(S)[F.p,F.p],T;factor=100)
                        check_extended_array(extref(S)*extref(solve(F,b)),extref(b),T;factor=100)
                    end
                end
            end
            n=6; A=T.([i==j ? 5.0 : abs(i-j)==1 ? .125 : 0.0 for i in 1:n,j in 1:n]); b=ones(T,n)
            F=band_lu(band_matrix(A;kl=1,ku=1))
            check_extended_array(extref(A)*extref(solve(F,b)),extref(b),T;factor=100)
            for trans in (:T,:C)
                check_extended_array(op(extref(A),trans)*extref(solve(F,b;trans)),extref(b),T;factor=100)
            end
            for make in (() ->band_cholesky(band_matrix(A;kl=1,ku=1,kind=:hermitian)),
                         () ->packed_cholesky(pack_matrix(A;kind=:hermitian)),()->aasen_factor(A))
                F=make(); check_extended_array(extref(A)*extref(solve(F,b)),extref(b),T;factor=200)
            end
            for (m,n) in ((0,0),(1,1),(7,4),(4,7))
                A=T.([i==j ? 3.0 : ((i+2j)%5)/16 for i in 1:m,j in 1:n])
                T<:Complex && (A .+= T(im/32))
                for pivot in (false,true), blocksize in (1,3)
                    F=qr_factor(A;pivot,blocksize); target=pivot ? A[:,F.p] : A
                    check_extended_array(F.Q*F.R,extref(target),T;factor=200)
                    Q=extref(F.Q); check_extended_array(Q'*Q,Matrix{BigFloat}(LA.I,m,m),T;factor=200)
                    if m>=n
                        rhs=A*ones(T,n); x=least_squares(F,rhs)
                        check_extended_array(extref(A)*extref(x),extref(rhs),T;factor=200)
                    end
                end
                for make in (lq_factor,ql_factor,rq_factor)
                    F=make(A); actual=F.kind===:LQ ? F.L*F.Q : F.kind===:RQ ? F.R*F.Q : F.Q*F.L
                    check_extended_array(actual,extref(A),T;factor=200)
                end
            end
            A=T.([3.0 .125 0;.25 4 .125;0 .25 5]); b=ones(T,3)
            X=native_inverse(A); check_extended_array(extref(A)*extref(X),Matrix{BigFloat}(LA.I,3,3),T;factor=100)
            result=expert_solve(A,b)
            check_extended_array(extref(A)*extref(result.x),extref(b),T;factor=200)
            @test result.converged
            result=refine_solve(A,lu_factor(A),b)
            @test result.converged && result.backward_error>=0
            @test reciprocal_condition(A)>0 && reciprocal_condition(A)<=1
            @test matrix_norm(A,:one)>0
            E=equilibrate(A;radix=true)
            check_extended_array(extref(E.row_scale).*extref(E.matrix).*transpose(extref(E.column_scale)),extref(A),T;factor=100)
            F=complete_orthogonal(A); check_extended_array(extref(A)*extref(solve(F,b)),extref(b),T;factor=200)
            GC.gc(); F=lu_factor(A); tasks=[Threads.@spawn solve(F,b) for _ in 1:3]
            @test all(x->maximum(abs,extref(A)*extref(x)-extref(b))<=200exttol(T),fetch.(tasks))
            @test_throws SingularException lu_factor(zeros(T,3,3))
            @test_throws PosDefException cholesky_factor(T.([-1.0 0;0 1]))
            @test_throws DimensionMismatch solve(F,ones(T,4))
            @test_throws ArgumentError lu_factor(A;blocksize=0)
        end
        for T in filter(T->!(T<:OxiBLAS.NativeFloat) && !(T<:OxiBLAS.HalfFloat),EXT_TYPES)
            A=T.([3.0 .25 0;.125 4 .25;0 .125 5]); b=ones(T,3)
            for lowtype in (T<:Complex ? (ComplexF32,ComplexF64) : (Float32,Float64))
                result=mixed_solve(A,b;lowtype,maxiter=30)
                @test result.converged && result.low_type===lowtype
                check_extended_array(extref(A)*extref(result.x),extref(b),T;factor=100)
            end
            @test mixed_solve(A,b).low_type===T
            @test !mixed_solve(A,b;lowtype=T<:Complex ? ComplexF32 : Float32,maxiter=0,rtol=0).converged
            @test_throws ArgumentError mixed_solve(A,b;lowtype=Int)
        end
        # Low precision loses a pivot; extended fallback and provenance required.
        D=BigFloat; delta=ldexp(one(D),-160)
        A=D[1 1;1 1+delta]; b=D[2,2+delta]
        result=mixed_solve(A,b;lowtype=Float64,fallback=true)
        @test result.fallback && result.reason===:working_factor_fallback
        @test result.converged
        check_extended_array(result.x,BigFloat[1,1],D;factor=100)
        @test_throws SingularException mixed_solve(A,b;lowtype=Float64,fallback=false)
        @test isempty(mixed_solve(zeros(D,0,0),D[]).x)
    end
end
@testset "Mixed factorization families and least-squares metric" begin
    setprecision(BigFloat,512) do
        for T in EXT_TYPES
            H=T.([4.0 .25 0;.25 5 .125;0 .125 6]); b=ones(T,3)
            for factorization in (:lu,:cholesky,:symmetric,:qr)
                result=mixed_solve(H,b;factorization,maxiter=30)
                @test result.converged && result.finite && result.factorization===factorization
                @test result.metric===(factorization===:qr ? :normalized_normal_residual : :componentwise_backward_error)
                check_extended_array(extref(H)*extref(result.x),extref(b),T;factor=200)
            end
            A=T.([1.0 0;0 1;1 1;2 -1]); b=T.([1.0,2,4,0])
            result=mixed_solve(A,b;factorization=:qr,maxiter=30)
            @test result.converged && result.metric===:normalized_normal_residual
            residual=extref(b)-extref(A)*extref(result.x)
            @test maximum(abs,extref(A)'*residual)<200exttol(T)
            @test maximum(abs,residual)>big".01" # QR does not claim r=0.
            @test_throws ArgumentError mixed_solve(H,ones(T,3);factorization=:bad)
            @test_throws DimensionMismatch mixed_solve(A,b;factorization=:lu)
        end
    end
end
