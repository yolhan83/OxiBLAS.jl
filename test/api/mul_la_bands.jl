# Tests ciblés pour la Mission 2 : direct loops pour Bidiagonal, Tridiagonal, SymTridiagonal
using Test
import LinearAlgebra as LA
using LinearAlgebra: norm, Bidiagonal, Tridiagonal, SymTridiagonal, Diagonal, transpose, adjoint
using OxiBLAS

_mrand(::Type{T}, dims...) where T = T <: Complex ? T.(randn(real(T), dims...), randn(real(T), dims...)) : T.(randn(T, dims...))
_mtol(::Type{T}) where T = T <: Union{BigFloat, Complex{BigFloat}} ? 1e-25 : (T <: Float16 ? 0.05 : 1000 * eps(real(T)))
_mclose(x, y, T) = norm(Array(x) - Array(y)) <= _mtol(T) * max(1.0, norm(Array(y)))

@testset "mul_la_bands" begin
    # ---------------------------------------------------------
    # 1. Base test suite: Float32, Float64, ComplexF32, ComplexF64
    # ---------------------------------------------------------
    types = (Float32, Float64, ComplexF32, ComplexF64)
    sizes = (0, 1, 2, 3, 7, 256)
    alphas_betas = [
        (1.0, 0.0),
        (0.0, 0.0),
        (0.0, 1.0),
        (1.0, 1.0),
        (1.7, -0.4)
    ]

    for T in types
        @testset "Standard Type $T" begin
            tol = _mtol(T)

            for n in sizes
                dv = _mrand(T, n)
                ev = n > 1 ? _mrand(T, n - 1) : T[]
                dl = n > 1 ? _mrand(T, n - 1) : T[]
                du = n > 1 ? _mrand(T, n - 1) : T[]

                bands = Any[
                    Bidiagonal(copy(dv), copy(ev), :U),
                    Bidiagonal(copy(dv), copy(ev), :L),
                    Tridiagonal(copy(dl), copy(dv), copy(du)),
                    SymTridiagonal(copy(dv), copy(ev))
                ]

                for B in bands
                    # 1. Left product with Vector: B * v
                    v = _mrand(T, n)
                    ref_mat = Array(B)

                    for (a_val, b_val) in alphas_betas
                        α = T(a_val)
                        β = T(b_val)

                        # B * v (in-place)
                        C0 = _mrand(T, n)
                        C = copy(C0)
                        ref_v = α * (ref_mat * v) + β * C0
                        res = OxiBLAS.mul!(C, B, v, α, β)
                        @test res === C
                        @test eltype(C) === T
                        @test _mclose(C, ref_v, T)

                        # B * v (allocating mul)
                        if isone(α) && iszero(β)
                            C_alloc = OxiBLAS.mul(B, v)
                            @test eltype(C_alloc) === T
                            @test _mclose(C_alloc, ref_mat * v, T)
                        end
                    end

                    # 2. Left product with Matrix: B * X (rectangular n x k)
                    for k in (1, 4)
                        X = _mrand(T, n, k)
                        for (a_val, b_val) in alphas_betas
                            α = T(a_val)
                            β = T(b_val)

                            C0 = _mrand(T, n, k)
                            C = copy(C0)
                            ref_X = α * (ref_mat * X) + β * C0
                            res = OxiBLAS.mul!(C, B, X, α, β)
                            @test res === C
                            @test eltype(C) === T
                            @test _mclose(C, ref_X, T)

                            if isone(α) && iszero(β)
                                C_alloc = OxiBLAS.mul(B, X)
                                @test eltype(C_alloc) === T
                                @test _mclose(C_alloc, ref_mat * X, T)
                            end
                        end
                    end

                    # 3. Right product with Matrix: X * B (rectangular m x n)
                    for m in (1, 5)
                        X = _mrand(T, m, n)
                        for (a_val, b_val) in alphas_betas
                            α = T(a_val)
                            β = T(b_val)

                            C0 = _mrand(T, m, n)
                            C = copy(C0)
                            ref_X = α * (X * ref_mat) + β * C0
                            res = OxiBLAS.mul!(C, X, B, α, β)
                            @test res === C
                            @test eltype(C) === T
                            @test _mclose(C, ref_X, T)

                            if isone(α) && iszero(β)
                                C_alloc = OxiBLAS.mul(X, B)
                                @test eltype(C_alloc) === T
                                @test _mclose(C_alloc, X * ref_mat, T)
                            end
                        end
                    end
                end

                # Aliasing tests
                if n >= 2
                    for B in bands
                        # Aliasing C with right vector operand
                        C_vec = _mrand(T, n)
                        ref_vec = Array(B) * copy(C_vec)
                        OxiBLAS.mul!(C_vec, B, C_vec)
                        @test _mclose(C_vec, ref_vec, T)

                        # Aliasing C with right matrix operand (square n x n)
                        C_mat = _mrand(T, n, n)
                        ref_mat = Array(B) * copy(C_mat)
                        OxiBLAS.mul!(C_mat, B, C_mat)
                        @test _mclose(C_mat, ref_mat, T)

                        # Aliasing C with left matrix operand
                        C_mat2 = _mrand(T, n, n)
                        ref_mat2 = copy(C_mat2) * Array(B)
                        OxiBLAS.mul!(C_mat2, C_mat2, B)
                        @test _mclose(C_mat2, ref_mat2, T)
                    end
                end
            end

            # Dimension mismatch errors
            n = 5
            B = Bidiagonal(_mrand(T, n), _mrand(T, n-1), :U)
            @test_throws DimensionMismatch OxiBLAS.mul!(zeros(T, n+1), B, zeros(T, n))
            @test_throws DimensionMismatch OxiBLAS.mul!(zeros(T, n), B, zeros(T, n+1))
            @test_throws DimensionMismatch OxiBLAS.mul!(zeros(T, n, 3), B, zeros(T, n+1, 3))
            @test_throws DimensionMismatch OxiBLAS.mul!(zeros(T, 4, n), zeros(T, 4, n+1), B)
        end
    end

    # ---------------------------------------------------------
    # A. Types numériques (Float16, BigFloat, Complex{BigFloat}, Types mixtes)
    # ---------------------------------------------------------
    @testset "A. Types numériques - Float16" begin
        T = Float16
        n = 4
        # Explicit Float16 coefficients (no Float64 literals)
        α1, β0 = Float16(1.0), Float16(0.0)
        α_nz, β_nz = Float16(1.7), Float16(-0.4)

        dv = Float16[1, 2, 3, 4]
        ev = Float16[1, 1, 1]
        dl = Float16[1, 1, 1]
        du = Float16[1, 1, 1]
        bands = [
            Bidiagonal(copy(dv), copy(ev), :U),
            Bidiagonal(copy(dv), copy(ev), :L),
            Tridiagonal(copy(dl), copy(dv), copy(du)),
            SymTridiagonal(copy(dv), copy(ev))
        ]
        x = Float16[1, 2, 3, 4]
        X = Float16[1 2; 3 4; 5 6; 7 8]
        X_left = Float16[1 2 3 4; 5 6 7 8]

        for B in bands
            ref_mat = Array(B)

            # Vector left: B * x
            c1 = zeros(Float16, n)
            OxiBLAS.mul!(c1, B, x, α1, β0)
            @test eltype(c1) === Float16
            @test _mclose(c1, ref_mat * x, Float16)

            c2 = zeros(Float16, n)
            OxiBLAS.mul!(c2, B, x, α_nz, β_nz)
            @test eltype(c2) === Float16
            @test _mclose(c2, α_nz * (ref_mat * x), Float16)

            alloc_v = OxiBLAS.mul(B, x)
            @test eltype(alloc_v) === Float16
            @test _mclose(alloc_v, ref_mat * x, Float16)

            # Matrix left: B * X
            C1 = zeros(Float16, n, 2)
            OxiBLAS.mul!(C1, B, X, α1, β0)
            @test eltype(C1) === Float16
            @test _mclose(C1, ref_mat * X, Float16)

            C2 = zeros(Float16, n, 2)
            OxiBLAS.mul!(C2, B, X, α_nz, β_nz)
            @test eltype(C2) === Float16
            @test _mclose(C2, α_nz * (ref_mat * X), Float16)

            alloc_m = OxiBLAS.mul(B, X)
            @test eltype(alloc_m) === Float16
            @test _mclose(alloc_m, ref_mat * X, Float16)

            # Matrix right: X_left * B
            Cr1 = zeros(Float16, 2, n)
            OxiBLAS.mul!(Cr1, X_left, B, α1, β0)
            @test eltype(Cr1) === Float16
            @test _mclose(Cr1, X_left * ref_mat, Float16)

            Cr2 = zeros(Float16, 2, n)
            OxiBLAS.mul!(Cr2, X_left, B, α_nz, β_nz)
            @test eltype(Cr2) === Float16
            @test _mclose(Cr2, α_nz * (X_left * ref_mat), Float16)

            alloc_r = OxiBLAS.mul(X_left, B)
            @test eltype(alloc_r) === Float16
            @test _mclose(alloc_r, X_left * ref_mat, Float16)
        end
    end

    @testset "A. Types numériques - BigFloat et Complex{BigFloat}" begin
        setprecision(BigFloat, 128) do
            for T in (BigFloat, Complex{BigFloat})
                n = 4
                α1, β0 = T(1.0), T(0.0)
                α_nz, β_nz = T(1.7), T(-0.4)

                dv = _mrand(T, n)
                ev = _mrand(T, n - 1)
                dl = _mrand(T, n - 1)
                du = _mrand(T, n - 1)

                bands = [
                    Bidiagonal(copy(dv), copy(ev), :U),
                    Bidiagonal(copy(dv), copy(ev), :L),
                    Tridiagonal(copy(dl), copy(dv), copy(du)),
                    SymTridiagonal(copy(dv), copy(ev))
                ]

                x = _mrand(T, n)
                X = _mrand(T, n, 2)
                X_left = _mrand(T, 2, n)

                for B in bands
                    ref_mat = Array(B)

                    # Vector left
                    c1 = zeros(T, n)
                    OxiBLAS.mul!(c1, B, x, α1, β0)
                    @test eltype(c1) === T
                    @test _mclose(c1, ref_mat * x, T)

                    c2 = _mrand(T, n)
                    ref_c2 = α_nz * (ref_mat * x) + β_nz * c2
                    OxiBLAS.mul!(c2, B, x, α_nz, β_nz)
                    @test eltype(c2) === T
                    @test _mclose(c2, ref_c2, T)

                    alloc_v = OxiBLAS.mul(B, x)
                    @test eltype(alloc_v) === T
                    @test _mclose(alloc_v, ref_mat * x, T)

                    # Matrix left
                    C1 = zeros(T, n, 2)
                    OxiBLAS.mul!(C1, B, X, α1, β0)
                    @test eltype(C1) === T
                    @test _mclose(C1, ref_mat * X, T)

                    alloc_m = OxiBLAS.mul(B, X)
                    @test eltype(alloc_m) === T
                    @test _mclose(alloc_m, ref_mat * X, T)

                    # Matrix right
                    Cr1 = zeros(T, 2, n)
                    OxiBLAS.mul!(Cr1, X_left, B, α1, β0)
                    @test eltype(Cr1) === T
                    @test _mclose(Cr1, X_left * ref_mat, T)

                    alloc_r = OxiBLAS.mul(X_left, B)
                    @test eltype(alloc_r) === T
                    @test _mclose(alloc_r, X_left * ref_mat, T)
                end
            end
        end
    end

    @testset "A. Types numériques - Types mixtes" begin
        n = 4
        dv = Float32[1, 2, 3, 4]
        ev = Float32[1, 1, 1]
        dl = Float32[1, 1, 1]
        du = Float32[1, 1, 1]

        bands = [
            Bidiagonal(copy(dv), copy(ev), :U),
            Bidiagonal(copy(dv), copy(ev), :L),
            Tridiagonal(copy(dl), copy(dv), copy(du)),
            SymTridiagonal(copy(dv), copy(ev))
        ]

        x64 = Float64[1, 2, 3, 4]
        X64 = Float64[1 2; 3 4; 5 6; 7 8]
        X64_left = Float64[1 2 3 4; 5 6 7 8]

        for B in bands
            ref_mat = Array(B)

            # Destination ComplexF64 with Float32 band and Float64 operand
            C_c64 = zeros(ComplexF64, n, 2)
            α = ComplexF64(1.5, 0.5)
            β = ComplexF64(-0.5, 0.2)
            ref_c64 = α * (ref_mat * X64)
            OxiBLAS.mul!(C_c64, B, X64, α, β)
            @test eltype(C_c64) === ComplexF64
            @test _mclose(C_c64, ref_c64, ComplexF64)

            # Vector mixed
            c_c64 = zeros(ComplexF64, n)
            OxiBLAS.mul!(c_c64, B, x64, α, β)
            @test eltype(c_c64) === ComplexF64
            @test _mclose(c_c64, α * (ref_mat * x64), ComplexF64)

            # Right product mixed
            Cr_c64 = zeros(ComplexF64, 2, n)
            OxiBLAS.mul!(Cr_c64, X64_left, B, α, β)
            @test eltype(Cr_c64) === ComplexF64
            @test _mclose(Cr_c64, α * (X64_left * ref_mat), ComplexF64)

            # Allocating mul promotes Float32 and Float64 to Float64
            res_alloc = OxiBLAS.mul(B, x64)
            @test eltype(res_alloc) === Float64
            @test _mclose(res_alloc, ref_mat * x64, Float64)

            res_alloc_m = OxiBLAS.mul(B, X64)
            @test eltype(res_alloc_m) === Float64
            @test _mclose(res_alloc_m, ref_mat * X64, Float64)
        end
    end

    # ---------------------------------------------------------
    # B. Vues non contiguës et Transpose / Adjoint
    # ---------------------------------------------------------
    @testset "B. Vues non contiguës et Transpose/Adjoint" begin
        n = 4
        T = Float64
        dv = _mrand(T, n)
        ev = _mrand(T, n - 1)
        dl = _mrand(T, n - 1)
        du = _mrand(T, n - 1)

        bands = [
            Bidiagonal(copy(dv), copy(ev), :U),
            Bidiagonal(copy(dv), copy(ev), :L),
            Tridiagonal(copy(dl), copy(dv), copy(du)),
            SymTridiagonal(copy(dv), copy(ev))
        ]

        # 1. Row stride > 1
        X_storage = randn(T, 2n, 3)
        X_view = view(X_storage, 1:2:2n-1, :) # size (4, 3) with row stride 2
        C_storage = zeros(T, 2n, 3)
        C_view = view(C_storage, 1:2:2n-1, :) # size (4, 3) with row stride 2

        v_storage = randn(T, 2n)
        v_view = view(v_storage, 1:2:2n-1)   # size 4 with stride 2
        c_storage = zeros(T, 2n)
        c_view = view(c_storage, 1:2:2n-1)   # size 4 with stride 2

        for B in bands
            ref_mat = Array(B)

            # Non-contiguous vector left product
            fill!(c_storage, 0.0)
            OxiBLAS.mul!(c_view, B, v_view)
            @test _mclose(c_view, ref_mat * v_view, T)

            # Non-contiguous matrix left product
            fill!(C_storage, 0.0)
            OxiBLAS.mul!(C_view, B, X_view)
            @test _mclose(C_view, ref_mat * X_view, T)

            # Non-contiguous matrix right product
            Y_storage = randn(T, 3, 2n)
            Y_view = view(Y_storage, :, 1:2:2n-1) # size (3, 4) with column stride 2
            CY_storage = zeros(T, 3, 2n)
            CY_view = view(CY_storage, :, 1:2:2n-1) # size (3, 4) with column stride 2
            OxiBLAS.mul!(CY_view, Y_view, B)
            @test _mclose(CY_view, Y_view * ref_mat, T)
        end

        # 2. Transpose and Adjoint with Complex numbers
        CT = ComplexF64
        c_dv = _mrand(CT, n)
        c_ev = _mrand(CT, n - 1)
        c_dl = _mrand(CT, n - 1)
        c_du = _mrand(CT, n - 1)

        c_bands = [
            Bidiagonal(copy(c_dv), copy(c_ev), :U),
            Bidiagonal(copy(c_dv), copy(c_ev), :L),
            Tridiagonal(copy(c_dl), copy(c_dv), copy(c_du)),
            SymTridiagonal(copy(c_dv), copy(c_ev))
        ]

        X_c = _mrand(CT, n, 2)
        X_left_c = _mrand(CT, 2, n)

        for B in c_bands
            ref_b = Array(B)

            # transpose(B) * X
            C_t = zeros(CT, n, 2)
            OxiBLAS.mul!(C_t, transpose(B), X_c)
            @test _mclose(C_t, transpose(ref_b) * X_c, CT)

            # adjoint(B) * X
            C_a = zeros(CT, n, 2)
            OxiBLAS.mul!(C_a, adjoint(B), X_c)
            @test _mclose(C_a, adjoint(ref_b) * X_c, CT)

            # X_left * transpose(B)
            Cr_t = zeros(CT, 2, n)
            OxiBLAS.mul!(Cr_t, X_left_c, transpose(B))
            @test _mclose(Cr_t, X_left_c * transpose(ref_b), CT)

            # X_left * adjoint(B)
            Cr_a = zeros(CT, 2, n)
            OxiBLAS.mul!(Cr_a, X_left_c, adjoint(B))
            @test _mclose(Cr_a, X_left_c * adjoint(ref_b), CT)
        end
    end

    # ---------------------------------------------------------
    # C. Aliasing avec les diagonales (dv, ev, dl, du)
    # ---------------------------------------------------------
    @testset "C. Aliasing avec les diagonales" begin
        n = 4
        T = Float64
        α = 1.7
        β = -0.4
        x = randn(T, n)

        # 1. Bidiagonal(:U) - C aliases dv
        dv = randn(T, n)
        ev = randn(T, n - 1)
        Bu = Bidiagonal(dv, ev, :U)
        ref_u = Array(copy(Bu)) * copy(x)
        OxiBLAS.mul!(dv, Bu, x)
        @test _mclose(dv, ref_u, T)

        # with α, β
        dv2 = randn(T, n)
        ev2 = randn(T, n - 1)
        Bu2 = Bidiagonal(dv2, ev2, :U)
        ref_u2 = α * (Array(copy(Bu2)) * copy(x)) + β * copy(dv2)
        OxiBLAS.mul!(dv2, Bu2, x, α, β)
        @test _mclose(dv2, ref_u2, T)

        # 2. Bidiagonal(:L) - C aliases dv
        dv_l = randn(T, n)
        ev_l = randn(T, n - 1)
        Bl = Bidiagonal(dv_l, ev_l, :L)
        ref_l = Array(copy(Bl)) * copy(x)
        OxiBLAS.mul!(dv_l, Bl, x)
        @test _mclose(dv_l, ref_l, T)

        # 3. Tridiagonal - C aliases d
        d = randn(T, n)
        dl = randn(T, n - 1)
        du = randn(T, n - 1)
        Tri = Tridiagonal(dl, d, du)
        ref_tri = Array(copy(Tri)) * copy(x)
        OxiBLAS.mul!(d, Tri, x)
        @test _mclose(d, ref_tri, T)

        # with α, β
        d2 = randn(T, n)
        dl2 = randn(T, n - 1)
        du2 = randn(T, n - 1)
        Tri2 = Tridiagonal(dl2, d2, du2)
        ref_tri2 = α * (Array(copy(Tri2)) * copy(x)) + β * copy(d2)
        OxiBLAS.mul!(d2, Tri2, x, α, β)
        @test _mclose(d2, ref_tri2, T)

        # 4. SymTridiagonal - C aliases dv
        s_dv = randn(T, n)
        s_ev = randn(T, n - 1)
        Sym = SymTridiagonal(s_dv, s_ev)
        ref_sym = Array(copy(Sym)) * copy(x)
        OxiBLAS.mul!(s_dv, Sym, x)
        @test _mclose(s_dv, ref_sym, T)
    end

    # ---------------------------------------------------------
    # D. Deux opérandes structurés
    # ---------------------------------------------------------
    @testset "D. Deux opérandes structurés" begin
        n = 4
        T = Float64
        B1 = Bidiagonal(randn(T, n), randn(T, n - 1), :U)
        B2 = Bidiagonal(randn(T, n), randn(T, n - 1), :L)
        T1 = Tridiagonal(randn(T, n - 1), randn(T, n), randn(T, n - 1))
        D = Diagonal(randn(T, n))
        S = SymTridiagonal(randn(T, n), randn(T, n - 1))

        structured_pairs = [
            ("Bidiagonal x Bidiagonal", B1, B2),
            ("Bidiagonal x Tridiagonal", B1, T1),
            ("Tridiagonal x Bidiagonal", T1, B1),
            ("Bidiagonal x Diagonal", B1, D),
            ("SymTridiagonal x Bidiagonal", S, B1)
        ]

        for (label, P1, P2) in structured_pairs
            ref = Array(P1) * Array(P2)

            # α=1, β=0 with NaN destination (tests non-reading of C)
            C = fill(NaN, n, n)
            res = OxiBLAS.mul!(C, P1, P2)
            @test res === C
            @test all(!isnan, C)
            @test _mclose(C, ref, T)

            # α≠1, β≠0
            α = 1.7
            β = -0.4
            C0 = randn(T, n, n)
            C_inplace = copy(C0)
            ref_inplace = α * ref + β * C0
            OxiBLAS.mul!(C_inplace, P1, P2, α, β)
            @test _mclose(C_inplace, ref_inplace, T)

            # Allocating mul
            Alloc = OxiBLAS.mul(P1, P2)
            @test eltype(Alloc) === T
            @test _mclose(Alloc, ref, T)
        end
    end
end
