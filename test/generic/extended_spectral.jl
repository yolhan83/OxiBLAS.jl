@testset "Extended spectral, generalized and matrix functions" begin
    setprecision(BigFloat,512) do
        for T in EXT_TYPES
            R=real(T)
            for n in (0,1,5)
                H=zeros(T,n,n)
                for i in 1:n
                    H[i,i]=T(4)
                    if i<n
                        H[i+1,i]=T<:Complex ? T((1+im)/8) : T(.125)
                        H[i,i+1]=conj(H[i+1,i])
                    end
                end
                magnitude=T<:Complex ? sqrt(big"2")/8 : big".125"
                expected=sort([big"4"+2magnitude*cos(BigFloat(k)*big(pi)/BigFloat(n+1)) for k in 1:n])
                for make in (eigen_hermitian,eigen_hermitian_dc,eigen_hermitian_mrrr)
                    E=make(H)
                    check_extended_array(E.values,expected,T;factor=100)
                    V=extref(E.vectors)
                    check_extended_array(extref(H)*V,V*LA.Diagonal(extref(E.values)),T;factor=200)
                    check_extended_array(V'*V,Matrix{BigFloat}(LA.I,n,n),T;factor=200)
                    valuesonly=make(H;vectors=false)
                    @test valuesonly.vectors===nothing
                    check_extended_array(valuesonly.values,expected,T;factor=100)
                end
                F=tridiagonal_factor(H)
                Q=extref(tridiagonal_q(F)); tri=Matrix(LA.Tridiagonal(extref(F.offdiagonal),extref(F.diagonal),extref(F.offdiagonal)))
                check_extended_array(Q*tri*Q',extref(H),T;factor=200)
                d=R.(fill(4,n)); e=R.(fill(.125,max(0,n-1)))
                for make in (eigen_tridiagonal,eigen_tridiagonal_dc,eigen_tridiagonal_mrrr)
                    E=make(d,e); V=extref(E.vectors)
                    tri=Matrix(LA.Tridiagonal(extref(e),extref(d),extref(e)))
                    check_extended_array(tri*V,V*LA.Diagonal(extref(E.values)),T;factor=200)
                end
                B=T.(2Matrix{Float64}(LA.I,n,n)); E=eigen_generalized(H,B)
                V=extref(E.vectors)
                check_extended_array(extref(H)*V,extref(B)*V*LA.Diagonal(extref(E.values)),T;factor=200)
                check_extended_array(V'*extref(B)*V,Matrix{BigFloat}(LA.I,n,n),T;factor=200)
            end
            for (m,n) in ((0,0),(1,1),(6,4),(4,6))
                A=T.([i==j ? i+2.0 : (i+j)%4/16 for i in 1:m,j in 1:n])
                T<:Complex && (A .+= T(im/32))
                for algorithm in (:jacobi,:parallel_jacobi,:golub_kahan,:divide_conquer), full in (false,true)
                    F=svd_factor(A;algorithm,full)
                    q=min(m,n); U=extref(F.U[:,1:q]); Vt=extref(F.Vt[1:q,:])
                    check_extended_array(U*LA.Diagonal(extref(F.S))*Vt,extref(A),T;factor=500)
                    check_extended_array(U'*U,Matrix{BigFloat}(LA.I,q,q),T;factor=500)
                    check_extended_array(Vt*Vt',Matrix{BigFloat}(LA.I,q,q),T;factor=500)
                    @test issorted(F.S;rev=true) && all(>=(0),F.S)
                    valuesonly=svd_factor(A;algorithm,vectors=false)
                    @test valuesonly.U===nothing && valuesonly.Vt===nothing
                    check_extended_array(valuesonly.S,extref(F.S),T;factor=500)
                end
                for make in (svd_truncated,svd_randomized)
                    q=min(m,n); F=make(A,q)
                    check_extended_array(extref(F.U)*LA.Diagonal(extref(F.S))*extref(F.Vt),extref(A),T;factor=1000)
                end
                selected=svd_selected(A;indices=1:min(2,min(m,n)))
                @test length(selected.S)==min(2,min(m,n))
                @test numerical_rank(A)==min(m,n)
                P=pseudoinverse(A)
                check_extended_array(extref(A)*extref(P)*extref(A),extref(A),T;factor=1000)
                N=null_space(A)
                check_extended_array(extref(A)*extref(N),zeros(BigFloat,m,size(N,2)),T;factor=1000)
            end
            A=T.([1.0 .5 0;.125 3 .25;0 .125 5])
            T<:Complex && (A[1,3]=T(im/8))
            F=hessenberg_factor(A)
            check_extended_array(extref(F.Q)*extref(F.H)*extref(F.Q)',extref(A),T;factor=500)
            for make in (schur_factor,schur_multishift)
                F=make(A); Z=extref(F.Z)
                check_extended_array(Z*extref(F.T)*Z',extref(A),T;factor=1000)
                check_extended_array(Z'*Z,Matrix{BigFloat}(LA.I,3,3),T;factor=1000)
            end
            E=eigen_general(A); V=extref(E.vectors)
            check_extended_array(extref(A)*V,V*LA.Diagonal(extref(E.values)),T;factor=1500)
            B=T.([2.0 0 0;0 3 0;0 0 4]); F=qz_factor(A,B)
            check_extended_array(extref(F.Q)*extref(F.S)*extref(F.Z)',extref(A),T;factor=1500)
            check_extended_array(extref(F.Q)*extref(F.T)*extref(F.Z)',extref(B),T;factor=1500)
            # Matrix functions have analytic diagonal references, using scalar
            # MPFR only in the oracle. Generic exp uses the active scalar precision.
            A=T.([.125 0 0;0 .25 0;0 0 .5]); target=extref(A)
            for (f,scalar) in ((matrix_exp,exp),(matrix_log,log),(matrix_sqrt,sqrt),(matrix_sin,sin),(matrix_cos,cos))
                actual=f(A); expected=zeros(eltype(extref(actual)),3,3)
                for i in 1:3
                    expected[i,i]=scalar(target[i,i])
                end
                check_extended_array(actual,expected,T;factor=2000)
            end
            check_extended_array(matrix_power(A,-2),LA.Diagonal([target[i,i]^(-2) for i in 1:3]),T;factor=100)
            check_extended_array(matrix_sign(A),Matrix{BigFloat}(LA.I,3,3),T;factor=1000)
            E=T.(Matrix{Float64}(LA.I,3,3)); L=frechet_exp(A,E)
            check_extended_array(L.derivative,extref(matrix_exp(A)),T;factor=2000)
            @test condition_number(A)>0
            C=column_space(A); @test size(C,2)==3
            # Collect while independent spectral calls own their work arrays.
            GC.gc(); tasks=[Threads.@spawn svd_factor(A) for _ in 1:2]
            @test all(F->length(F.S)==3,fetch.(tasks))
            @test_throws DimensionMismatch eigen_generalized(A,zeros(T,2,2))
            @test_throws ArgumentError svd_factor(A;maxsweeps=-1)
            @test_throws ArgumentError eigen_hermitian(A;maxiter=-1)
        end
    end
end
@testset "Extended applications, reorder and expert entry points" begin
    setprecision(BigFloat,512) do
        for T in EXT_TYPES
            A=T.([1.0 .25 0;0 3 .125;0 0 5]); b=ones(T,3); R=real(T)
            QF=qr_factor(A); expected=extref(QF.Q)
            HF=hessenberg_factor(A); Hexpected=extref(HF.Q)
            for trans in (:N,:T,:C), side in (:L,:R)
                input=T.(Matrix{Float64}(LA.I,3,3)); actual=copy(input)
                apply_q!(actual,QF;side,trans)
                check_extended_array(actual,op(expected,trans),T;factor=500)
                actual=copy(input); apply_hessenberg_q!(actual,HF;side,trans)
                check_extended_array(actual,op(Hexpected,trans),T;factor=500)
            end
            H=T.([4.0 .25 0;.25 5 .125;0 .125 6])
            for make in (expert_cholesky,expert_symmetric)
                result=make(H,b); @test result.converged
                check_extended_array(extref(H)*extref(result.x),extref(b),T;factor=200)
            end
            error=backward_error(H,ones(T,3),H*ones(T,3))
            @test error.error<=R(T<:OxiBLAS.HalfFloat ? .01 : 1e-10) # Native and generic paths both finite.
            @test all(isfinite,error.residual)
            for shape in ((5,3),(3,5))
                matrix=T.([i==j ? i+2.0 : (i+2j)%4/16 for i in 1:shape[1],j in 1:shape[2]])
                F=bidiagonal_factor(matrix)
                check_extended_array(extref(F.U)*extref(F.B)*extref(F.V)',extref(matrix),T;factor=1000)
                values=singular_values(matrix)
                check_extended_array(values,extref(svd_factor(matrix).S),T;factor=1000)
            end
            C=Complex{R}; F=schur_factor(C.(A)); reordered=schur_reorder(F,lambda->real(lambda)>3.5)
            @test real(reordered.values[1])>3.5
            check_extended_array(extref(reordered.Z)*extref(reordered.T)*extref(reordered.Z)',extref(A),T;factor=1000)
            B=T.(2Matrix{Float64}(LA.I,3,3)); E=eigen_pencil(A,B)
            check_extended_array(extref(A)*extref(E.vectors),extref(B)*extref(E.vectors)*LA.Diagonal(extref(E.values)),T;factor=1500)
            @test eigen_pencil(A,B;vectors=false).vectors===nothing
            E=eigen_general(A;left=true,balance=true)
            check_extended_array(extref(A)'*extref(E.left_vectors),extref(E.left_vectors)*LA.Diagonal(conj.(extref(E.values))),T;factor=1500)
            A=T.([.125 0 0;0 .25 0;0 0 .5]); direction=T.(Matrix{Float64}(LA.I,3,3))
            result=frechet_log(A,direction)
            check_extended_array(result.derivative,LA.Diagonal([inv(extref(A[i,i])) for i in 1:3]),T;factor=2000)
            result=frechet_sqrt(A,direction)
            check_extended_array(result.derivative,LA.Diagonal([inv(2sqrt(extref(A[i,i]))) for i in 1:3]),T;factor=2000)
        end
        # Elementary functions belong to the providers; matrix functions above
        # test their use in OxiBLAS rather than duplicating provider math tests.
    end
end
