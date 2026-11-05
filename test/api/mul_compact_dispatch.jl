# Tests ciblés pour la Mission 1 : dispatch des matrices compactes OxiBLAS
using Test
import LinearAlgebra as LA
using LinearAlgebra: norm, transpose, adjoint
using OxiBLAS

_mrand(::Type{T}, dims...) where T = T <: Complex ? T.(randn(dims...), randn(dims...)) : T.(randn(dims...))
_mtol(::Type{T}) where T = 1000 * eps(real(T))
_mclose(x, y, T) = norm(Array(x) - Array(y)) <= _mtol(T) * max(1.0, norm(Array(y)))

@testset "mul_compact_dispatch" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        tol = _mtol(T)
        n = 6
        k_cols = 3
        rnd_vec(len) = _mrand(T, len)
        rnd_mat(r, c) = _mrand(T, r, c)

        function check_compact(A, B, α=T(1.5), β=T(-0.5))
            m_out = size(A, 1)
            ref_mat = Array(A)
            ref = α * (ref_mat * Array(B))

            # mul! with α, β
            C0 = ndims(B) == 1 ? rnd_vec(m_out) : rnd_mat(m_out, size(B, 2))
            C = copy(C0)
            ref_inplace = ref + β * Array(C0)
            res = OxiBLAS.mul!(C, A, B, α, β)
            @test res === C
            @test _mclose(C, ref_inplace, T)

            # mul! with α=1, β=0
            C = similar(C0)
            OxiBLAS.mul!(C, A, B, one(T), zero(T))
            @test _mclose(C, ref_mat * Array(B), T)

            # mul! with β=0 and destination filled with NaN
            C = fill(T(NaN), size(C0))
            OxiBLAS.mul!(C, A, B, α, zero(T))
            @test all(!isnan, C)
            @test _mclose(C, ref, T)

            # mul! with α=0 (scales C only)
            C = copy(C0)
            OxiBLAS.mul!(C, A, B, zero(T), β)
            @test _mclose(C, β * Array(C0), T)

            # allocating OxiBLAS.mul
            AllocRes = OxiBLAS.mul(A, B)
            @test eltype(AllocRes) === T
            @test _mclose(AllocRes, ref_mat * Array(B), T)
        end

        # Matrices compactes testées
        # 1. Banded general
        data_g = rnd_mat(4, n) # kl=2, ku=1 -> kl+ku+1 = 4
        Bg = BandedMatrix(data_g, n; kl=2, ku=1, kind=:general)

        # 2. Banded symmetric (:U et :L)
        data_s = rnd_mat(3, n)
        Bsu = BandedMatrix(data_s, n; ku=2, kind=:symmetric, uplo=:U)
        Bsl = BandedMatrix(data_s, n; kl=2, kind=:symmetric, uplo=:L)

        # 3. Banded hermitian (:U et :L)
        data_h = rnd_mat(3, n)
        Bhu = BandedMatrix(data_h, n; ku=2, kind=:hermitian, uplo=:U)
        Bhl = BandedMatrix(data_h, n; kl=2, kind=:hermitian, uplo=:L)

        # 4. Packed symmetric (:U et :L)
        p_len = n*(n+1)÷2
        Psu = PackedMatrix(rnd_vec(p_len), n; kind=:symmetric, uplo=:U)
        Psl = PackedMatrix(rnd_vec(p_len), n; kind=:symmetric, uplo=:L)

        # 5. Packed hermitian (:U et :L)
        Phu = PackedMatrix(rnd_vec(p_len), n; kind=:hermitian, uplo=:U)
        Phl = PackedMatrix(rnd_vec(p_len), n; kind=:hermitian, uplo=:L)

        # Test des 5 chemins vectoriels (BandedMatrix et PackedMatrix)
        for A in [Bg, Bsu, Bsl, Bhu, Bhl, Psu, Psl, Phu, Phl]
            x = rnd_vec(n)
            check_compact(A, x)
        end

        # Test des chemins matriciels multi-colonnes pour BandedMatrix
        for A in [Bg, Bsu, Bsl, Bhu, Bhl]
            Y = rnd_mat(n, k_cols)
            check_compact(A, Y)
        end

        # Test PackedMatrix * Matrix (qui retombe dans _mul_dense!)
        for P in [Psu, Psl, Phu, Phl]
            Y = rnd_mat(n, k_cols)
            check_compact(P, Y)
        end

        # Dimensions N=1 et cas vide
        Bg1 = BandedMatrix(rnd_mat(1, 1), 1; kl=0, ku=0, kind=:general)
        Bs1 = BandedMatrix(rnd_mat(1, 1), 1; ku=0, kind=:symmetric, uplo=:U)
        Bh1 = BandedMatrix(rnd_mat(1, 1), 1; ku=0, kind=:hermitian, uplo=:U)
        Ps1 = PackedMatrix(rnd_vec(1), 1; kind=:symmetric, uplo=:U)
        Ph1 = PackedMatrix(rnd_vec(1), 1; kind=:hermitian, uplo=:U)
        for A in [Bg1, Bs1, Bh1, Ps1, Ph1]
            check_compact(A, rnd_vec(1))
            check_compact(A, rnd_mat(1, 2))
        end

        # Cas vide (0x0 et rectangulaire 0x3)
        Bg0 = BandedMatrix(zeros(T, 1, 0), 0; kl=0, ku=0, kind=:general)
        y0 = zeros(T, 0); x0 = zeros(T, 0)
        @test OxiBLAS.mul!(y0, Bg0, x0) === y0
        @test isempty(OxiBLAS.mul(Bg0, x0))
        Bg_0x3 = BandedMatrix(zeros(T, 2, 3), 0; kl=0, ku=1, kind=:general)
        y_0 = zeros(T, 0); x_3 = rnd_vec(3)
        @test OxiBLAS.mul!(y_0, Bg_0x3, x_3) === y_0

        # BandedMatrix rectangulaire avec kl != ku
        m_rec, n_rec = 8, 5
        data_rec1 = rnd_mat(5, n_rec)
        B_tall = BandedMatrix(data_rec1, m_rec; kl=3, ku=1, kind=:general)
        check_compact(B_tall, rnd_vec(n_rec))
        check_compact(B_tall, rnd_mat(n_rec, 4))

        m_w, n_w = 4, 7
        data_rec2 = rnd_mat(4, n_w)
        B_wide = BandedMatrix(data_rec2, m_w; kl=1, ku=2, kind=:general)
        check_compact(B_wide, rnd_vec(n_w))
        check_compact(B_wide, rnd_mat(n_w, 2))

        # Partage de stockage / Aliasing
        # a) x et y partagent le même stockage (y === x)
        for A in [Bg, Bsu, Bhu, Psu, Phu]
            x_shared = rnd_vec(n)
            ref_x = 2.0 * (Array(A) * copy(x_shared))
            OxiBLAS.mul!(x_shared, A, x_shared, T(2), T(0))
            @test _mclose(x_shared, ref_x, T)
        end

        # b) Destination chevauche les coefficients compacts
        data_alias = copy(Bg.data)
        A_alias = BandedMatrix(data_alias, n; kl=Bg.kl, ku=Bg.ku, kind=:general)
        out_row = view(data_alias, 1, 1:n)
        x_in = rnd_vec(n)
        expected_out = Array(A_alias) * x_in
        OxiBLAS.mul!(out_row, A_alias, x_in)
        @test _mclose(out_row, expected_out, T)

        # Dimensions incompatibles et conservation des types
        @test_throws DimensionMismatch OxiBLAS.mul!(zeros(T, n+1), Bg, rnd_vec(n))
        @test_throws DimensionMismatch OxiBLAS.mul!(zeros(T, n), Bg, rnd_vec(n+1))
        @test_throws DimensionMismatch OxiBLAS.mul!(zeros(T, n, 2), Bg, rnd_mat(n, 3))
        if T === Float32
            B_f64 = rnd_vec(n) .|> Float64
            res_conv = OxiBLAS.mul(Bg, B_f64)
            @test eltype(res_conv) === Float64
            @test _mclose(res_conv, Array(Bg) * B_f64, Float64)
        end
    end

    # Distinction explicite symétrique vs hermitienne pour types complexes
    for CT in (ComplexF32, ComplexF64)
        n = 5
        data_c = _mrand(CT, 3, n)
        p_c = _mrand(CT, n*(n+1)÷2)
        for i in eachindex(data_c); data_c[i] += CT(0, 1.5); end
        for i in eachindex(p_c); p_c[i] += CT(0, 1.5); end

        B_sym = BandedMatrix(data_c, n; ku=2, kind=:symmetric, uplo=:U)
        B_herm = BandedMatrix(data_c, n; ku=2, kind=:hermitian, uplo=:U)
        P_sym = PackedMatrix(p_c, n; kind=:symmetric, uplo=:U)
        P_herm = PackedMatrix(p_c, n; kind=:hermitian, uplo=:U)

        @test Array(B_sym) != Array(B_herm)
        @test Array(P_sym) != Array(P_herm)
        @test Array(B_sym) == transpose(Array(B_sym))
        @test Array(B_herm) == adjoint(Array(B_herm))
        @test Array(P_sym) == transpose(Array(P_sym))
        @test Array(P_herm) == adjoint(Array(P_herm))

        x = _mrand(CT, n)
        y_sym = OxiBLAS.mul(B_sym, x)
        y_herm = OxiBLAS.mul(B_herm, x)
        @test !isapprox(y_sym, y_herm)
        @test _mclose(y_sym, Array(B_sym)*x, CT)
        @test _mclose(y_herm, Array(B_herm)*x, CT)

        yp_sym = OxiBLAS.mul(P_sym, x)
        yp_herm = OxiBLAS.mul(P_herm, x)
        @test !isapprox(yp_sym, yp_herm)
        @test _mclose(yp_sym, Array(P_sym)*x, CT)
        @test _mclose(yp_herm, Array(P_herm)*x, CT)
    end

    # Tests de non-régression
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        n = 4
        # 1. BandedMatrix et PackedMatrix triangulaires (non optimisées)
        data_tri = _mrand(T, 3, n)
        B_tri_u = BandedMatrix(data_tri, n; ku=2, kind=:triangular, uplo=:U)
        B_tri_l = BandedMatrix(data_tri, n; kl=2, kind=:triangular, uplo=:L)
        P_tri_u = PackedMatrix(_mrand(T, n*(n+1)÷2), n; kind=:triangular, uplo=:U)
        P_tri_l = PackedMatrix(_mrand(T, n*(n+1)÷2), n; kind=:triangular, uplo=:L)
        x = _mrand(T, n)
        for Tri in [B_tri_u, B_tri_l, P_tri_u, P_tri_l]
            y = zeros(T, n)
            OxiBLAS.mul!(y, Tri, x)
            @test _mclose(y, Array(Tri)*x, T)
            Y = zeros(T, n, 2)
            M = _mrand(T, n, 2)
            OxiBLAS.mul!(Y, Tri, M)
            @test _mclose(Y, Array(Tri)*M, T)
        end

        # 2. Matrices compactes en opérande droit (non optimisées)
        X_dense = _mrand(T, 3, n)
        Bg = BandedMatrix(_mrand(T, 3, n), n; kl=1, ku=1, kind=:general)
        Psu = PackedMatrix(_mrand(T, n*(n+1)÷2), n; kind=:symmetric, uplo=:U)
        for P in [Bg, Psu]
            C_right = zeros(T, 3, n)
            OxiBLAS.mul!(C_right, X_dense, P)
            @test _mclose(C_right, X_dense * Array(P), T)
        end

        # 3. Chemin dense classique
        D1 = _mrand(T, 5, 4); D2 = _mrand(T, 4, 3)
        Cd = zeros(T, 5, 3)
        OxiBLAS.mul!(Cd, D1, D2)
        @test _mclose(Cd, D1 * D2, T)
    end
end
