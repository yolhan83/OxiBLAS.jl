using LinearAlgebra

@testset "TRMM and GEMM Edge Workspace Cache" begin
    @testset "Edge buffer reuse without reallocation" begin
        for T in (Float32, Float64, ComplexF32, ComplexF64)
            w = GemmWorkspace(T, 128, 128, 128)
            @test isempty(w.edge_tiles)

            # Direct check on _workspace_edge_tiles!
            tiles1 = OxiBLAS._workspace_edge_tiles!(w, 4, 16, 8)
            @test length(tiles1) == 4
            @test all(size(t) == (16, 8) for t in tiles1)
            ids1 = [objectid(t) for t in tiles1]
            ptrs1 = [pointer(t) for t in tiles1]

            # Second call with same dimensions: must reuse exact same matrices
            tiles2 = OxiBLAS._workspace_edge_tiles!(w, 4, 16, 8)
            @test tiles2 === tiles1
            @test [objectid(t) for t in tiles2] == ids1
            @test [pointer(t) for t in tiles2] == ptrs1

            # Requesting more workers expands the vector but preserves earlier ones
            tiles3 = OxiBLAS._workspace_edge_tiles!(w, 8, 16, 8)
            @test length(tiles3) == 8
            @test [objectid(tiles3[i]) for i in 1:4] == ids1

            # Requesting different dimensions resizes only what is needed
            tiles4 = OxiBLAS._workspace_edge_tiles!(w, 2, 8, 4)
            @test size(tiles4[1]) == (8, 4)
            @test size(tiles4[2]) == (8, 4)
        end
    end

    @testset "Edge cache in TRMM and GEMM execution" begin
        for T in (Float32, Float64, ComplexF32, ComplexF64)
            w = GemmWorkspace(T, 128, 128, 128)
            A = randn(T, 67, 67)
            B = randn(T, 67, 23)
            C = zeros(T, 67, 23)

            # Run gemm! with explicit workspace
            gemm!(C, A, B; workspace=w, algorithm=:simd)
            @test !isempty(w.edge_tiles)
            tile_obj = objectid(w.edge_tiles[1])
            tile_ptr = pointer(w.edge_tiles[1])

            # Successive calls must reuse the exact same edge tile
            for _ in 1:3
                gemm!(C, A, B; workspace=w, algorithm=:simd)
                @test objectid(w.edge_tiles[1]) == tile_obj
                @test pointer(w.edge_tiles[1]) == tile_ptr
            end
        end
    end

    @testset "Non-multiple micro-kernel dimensions (SIMD edge tiles)" begin
        # Test shapes that exercise partial SIMD edge microkernels
        odd_shapes = [(17, 13), (33, 19), (51, 27), (65, 35), (97, 43)]
        for T in (Float32, Float64, ComplexF32, ComplexF64)
            for (N, RHS) in odd_shapes
                for side in (:L, :R), uplo in (:U, :L), unit in (false, true)
                    # Prepare matrices
                    A = randn(T, N, N)
                    for i in 1:N; A[i, i] += 2; end
                    B = side == :L ? randn(T, N, RHS) : randn(T, RHS, N)
                    B_orig = copy(B)

                    # LinearAlgebra reference
                    A_ref = copy(A)
                    if uplo == :U
                        A_ref = triu(A_ref)
                    else
                        A_ref = tril(A_ref)
                    end
                    if unit
                        for i in 1:N; A_ref[i, i] = 1; end
                    end

                    expected = side == :L ? A_ref * B_orig : B_orig * A_ref

                    # Test with implicit (pooled) workspace
                    B_test1 = copy(B_orig)
                    OxiBLAS.trmm!(B_test1, A; side=side, uplo=uplo, unit=unit, algorithm=:auto)
                    @test isapprox(B_test1, expected; rtol=tol(T), atol=tol(T)*max(1, N))

                    # Test with explicit workspace
                    w = GemmWorkspace(T, max(N, RHS), max(N, RHS), max(N, RHS))
                    B_test2 = copy(B_orig)
                    OxiBLAS.trmm!(B_test2, A; side=side, uplo=uplo, unit=unit, workspace=w, algorithm=:auto)
                    @test isapprox(B_test2, expected; rtol=tol(T), atol=tol(T)*max(1, N))
                end
            end
        end
    end

    @testset "Independence of w.edge_tiles and w.tiles (SYRK / HERK isolation)" begin
        for T in (Float64, ComplexF64)
            w = GemmWorkspace(T, 128, 128, 128)
            N = 64
            K = 32

            # SYRK / HERK uses w.tiles
            C_syrk = zeros(T, N, N)
            A_syrk = randn(T, N, K)
            if T <: Complex
                OxiBLAS.herk!(C_syrk, A_syrk; uplo=:U, workspace=w)
            else
                OxiBLAS.syrk!(C_syrk, A_syrk; uplo=:U, workspace=w)
            end

            # TRMM / GEMM uses w.edge_tiles
            B_trmm = randn(T, N, 17)
            A_trmm = randn(T, N, N)
            OxiBLAS.trmm!(B_trmm, A_trmm; side=:L, uplo=:U, workspace=w)

            # Verify that edge_tiles and tiles are independent storages
            @test !isempty(w.edge_tiles)
            for et in w.edge_tiles, t in w.tiles
                @test pointer(et) != pointer(t)
                @test !Base.mightalias(et, t)
            end

            # Interleaved executions
            for _ in 1:3
                if T <: Complex
                    OxiBLAS.herk!(C_syrk, A_syrk; uplo=:U, workspace=w)
                else
                    OxiBLAS.syrk!(C_syrk, A_syrk; uplo=:U, workspace=w)
                end
                OxiBLAS.trmm!(B_trmm, A_trmm; side=:L, uplo=:U, workspace=w)
            end
        end
    end

    @testset "Aliasing checks on edge tiles" begin
        for T in (Float64, ComplexF64)
            w = GemmWorkspace(T, 64, 64, 64)
            OxiBLAS._workspace_edge_tiles!(w, 1, 16, 8)
            alias_tile = w.edge_tiles[1]

            # Operands aliasing owned edge tile must throw ArgumentError
            other = randn(T, 16, 8)
            @test_throws ArgumentError OxiBLAS._workspace_operands_check(w, alias_tile, other)
            @test_throws ArgumentError OxiBLAS._workspace_operands_check(w, other, alias_tile)
        end
    end

    @testset "Robustness to NaN when beta=0" begin
        for T in (Float32, Float64, ComplexF32, ComplexF64)
            # gemm! with beta=0 and C filled with NaN
            C = fill(T(NaN), 23, 19)
            A = randn(T, 23, 15)
            B = randn(T, 15, 19)
            gemm!(C, A, B, one(T), zero(T); algorithm=:simd)
            @test all(!isnan, C)
            @test isapprox(C, A * B; rtol=tol(T), atol=tol(T)*15)

            # partial edge GEMM with beta=0
            C = fill(T(NaN), 35, 27)
            A = randn(T, 35, 13)
            B = randn(T, 13, 27)
            gemm!(C, A, B, one(T), zero(T); algorithm=:simd)
            @test all(!isnan, C)
            @test isapprox(C, A * B; rtol=tol(T), atol=tol(T)*13)
        end
    end

    @testset "Multi-threaded shared workspace safety" begin
        for T in (Float32, Float64)
            w = GemmWorkspace(T, 128, 128, 128)
            tasks = []
            results = Vector{Matrix{T}}(undef, 8)
            expected = Vector{Matrix{T}}(undef, 8)

            for i in 1:8
                A = randn(T, 45, 45)
                B = randn(T, 45, 19)
                expected[i] = triu(A) * B
                let i=i, A=A, B=copy(B)
                    t = Threads.@spawn begin
                        OxiBLAS.trmm!(B, A; side=:L, uplo=:U, workspace=w)
                        results[i] = B
                    end
                    push!(tasks, t)
                end
            end

            for t in tasks
                wait(t)
            end

            for i in 1:8
                @test isapprox(results[i], expected[i]; rtol=tol(T), atol=tol(T)*45)
            end
        end
    end
end
