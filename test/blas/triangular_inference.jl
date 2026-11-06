using LinearAlgebra
using Test

@testset "Triangular Type Inference and Non-mutation Regression" begin
    @testset "Public API type inference (Float64 and ComplexF64)" begin
        for T in (Float64, ComplexF64)
            A = randn(T, 17, 17)
            for i in 1:17; A[i, i] += T(5); end
            B_L = randn(T, 17, 13)
            B_R = randn(T, 13, 17)
            alpha = T(1.5)

            for side in (:L, :R), uplo in (:U, :L), trans in (:N, :T, :C), unit in (false, true)
                B = copy(side === :L ? B_L : B_R)
                # Test trmm!
                @test @inferred(OxiBLAS.trmm!(copy(B), A, alpha; side=side, uplo=uplo, trans=trans, unit=unit, blocksize=8, threaded=false)) isa Matrix{T}
                # Test trsm!
                @test @inferred(OxiBLAS.trsm!(copy(B), A, alpha; side=side, uplo=uplo, trans=trans, unit=unit, blocksize=8, threaded=false)) isa Matrix{T}

                # With explicit workspace
                w = GemmWorkspace(T, 17, 17, 17)
                @test @inferred(OxiBLAS.trmm!(copy(B), A, alpha; side=side, uplo=uplo, trans=trans, unit=unit, blocksize=8, threaded=false, workspace=w)) isa Matrix{T}
                @test @inferred(OxiBLAS.trsm!(copy(B), A, alpha; side=side, uplo=uplo, trans=trans, unit=unit, blocksize=8, threaded=false, workspace=w)) isa Matrix{T}
            end

            # Aliasing
            A_alias1 = copy(A)
            @test @inferred(OxiBLAS.trmm!(A_alias1, A_alias1, alpha; side=:L, uplo=:U, trans=:N, unit=false, blocksize=8, threaded=false)) isa Matrix{T}
            A_alias2 = copy(A)
            @test @inferred(OxiBLAS.trsm!(A_alias2, A_alias2, alpha; side=:L, uplo=:U, trans=:N, unit=false, blocksize=8, threaded=false)) isa Matrix{T}
        end
    end

    @testset "Internal helper type stability (ComplexF64 monomorphic pipeline)" begin
        T = ComplexF64
        A = randn(T, 16, 16)
        for i in 1:16; A[i, i] += T(5); end
        B = randn(T, 16, 8)
        alpha = T(1.0)
        config = GemmConfig(T; mc=16, kc=16, nc=16, parallel_threshold=0)
        w = GemmWorkspace(T, 16, 16, 16)

        # _trmm_trans! and _trsm_trans!
        for trans_code in (Val{0}(), Val{1}(), Val{2}())
            for upper in (true, false), unit in (false, true)
                @test @inferred(OxiBLAS._trmm_trans!(trans_code, copy(B), A, alpha, true, upper, unit, 8, :auto, :auto, false, nothing, config)) isa Matrix{T}
                @test @inferred(OxiBLAS._trsm_trans!(trans_code, copy(B), A, alpha, true, upper, unit, 8, :auto, :auto, false, nothing, config)) isa Matrix{T}
                A_c = copy(A)
                @test @inferred(OxiBLAS._trmm_trans!(trans_code, A_c, A_c, alpha, true, upper, unit, 8, :auto, :auto, false, nothing, config)) isa Matrix{T}
            end
        end

        # _trmm_compute! and _trsm_compute!
        for upper in (true, false), unit in (false, true)
            @test @inferred(OxiBLAS._trmm_compute!(copy(B), A, alpha, true, upper, unit, 8, :auto, :auto, false, nothing, config)) isa Matrix{T}
            @test @inferred(OxiBLAS._trsm_compute!(copy(B), A, alpha, true, upper, unit, 8, :auto, :auto, false, nothing, config)) isa Matrix{T}
            @test @inferred(OxiBLAS._trmm_compute!(copy(B), A, alpha, true, upper, unit, 8, :auto, :auto, false, w, config)) isa Matrix{T}
            @test @inferred(OxiBLAS._trsm_compute!(copy(B), A, alpha, true, upper, unit, 8, :auto, :auto, false, w, config)) isa Matrix{T}
        end

        # Decoupled left/right recursive functions
        opt_l = OxiBLAS._tri_options(T, B, 16, true, :auto, :auto, false, w, config)
        B_right = randn(T, 8, 16)
        opt_r = OxiBLAS._tri_options(T, B_right, 16, false, :auto, :auto, false, w, config)

        for upper in (true, false), unit in (false, true)
            @test @inferred(OxiBLAS._trmm_blocks_left!(copy(B), A, alpha, upper, unit, 8, opt_l)) isa Matrix{T}
            @test @inferred(OxiBLAS._trsm_blocks_left!(copy(B), A, upper, unit, 8, opt_l)) isa Matrix{T}
            @test @inferred(OxiBLAS._trmm_blocks_right!(copy(B_right), A, alpha, upper, unit, 8, opt_r)) isa Matrix{T}
            @test @inferred(OxiBLAS._trsm_blocks_right!(copy(B_right), A, upper, unit, 8, opt_r)) isa Matrix{T}
        end

        # _tri_pooled type stability
        res_pool = @inferred Union{Nothing, GemmWorkspace{T}} OxiBLAS._tri_pooled(pool_w -> pool_w, T, B, 16, true, :auto, false, nothing, config)
        @test res_pool === nothing || res_pool isa GemmWorkspace{T}

        res_ws = @inferred GemmWorkspace{T} OxiBLAS._tri_pooled(passed_w -> passed_w, T, B, 16, true, :auto, false, w, config)
        @test res_ws === w
    end

    @testset "TRSM Singular Non-Mutation" begin
        for T in (Float32, Float64, ComplexF32, ComplexF64)
            for side in (:L, :R), uplo in (:U, :L), trans in (:N, :T, :C)
                A = randn(T, 10, 10)
                A[3, 3] = zero(T) # Force singularity on diagonal
                B = side === :L ? randn(T, 10, 6) : randn(T, 6, 10)
                B_orig = copy(B)

                @test_throws OxiBLAS.SingularException OxiBLAS.trsm!(B, A, one(T); side=side, uplo=uplo, trans=trans, unit=false)
                # Guarantee: B MUST NOT be mutated before or when exception is thrown
                @test B == B_orig

                # Also verify aliased case
                A_alias = copy(A)
                @test_throws OxiBLAS.SingularException OxiBLAS.trsm!(A_alias, A_alias, one(T); side=side, uplo=uplo, trans=trans, unit=false)
                @test A_alias == A
            end
        end
    end
end
