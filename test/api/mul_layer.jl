# OxiBLAS.mul! / OxiBLAS.mul: an OxiBLAS-only multiplication layer.
import LinearAlgebra as LA
import SparseArrays as SA
using SparseArrays: SparseMatrixCSC, SparseVector, sparse, sparsevec

_mrand(::Type{T},dims...) where T=T<:Complex ? T.(randn(dims...),randn(dims...)) : T.(randn(dims...))
function _msparse(::Type{T},m,n,density=0.3) where T
    S=SA.sprand(m,n,density)
    SparseMatrixCSC{T,Int}(m,n,copy(S.colptr),copy(S.rowval),_mrand(T,length(S.nzval)))
end
_mtol(::Type{T}) where T=T<:BigFloat || T<:Complex{BigFloat} ? 1e-30 : 1000*eps(real(T))
_mclose(x,y,T)=LA.norm(Array(x)-Array(y))<=_mtol(T)*max(1,LA.norm(Array(y)))

# Does a signature mention a type defined by OxiBLAS (through unions, type
# parameters and UnionAll bounds)?
function _mentions_oxiblas(T,seen=Base.IdSet{Any}())
    T in seen && return false; push!(seen,T)
    T isa UnionAll && return _mentions_oxiblas(T.body,seen) || _mentions_oxiblas(T.var.ub,seen)
    T isa Union && return _mentions_oxiblas(T.a,seen) || _mentions_oxiblas(T.b,seen)
    T isa TypeVar && return _mentions_oxiblas(T.ub,seen)
    T isa DataType || return false
    parentmodule(T)===OxiBLAS && return true
    any(p->_mentions_oxiblas(p,seen),T.parameters)
end

@testset "OxiBLAS.mul! ownership" begin
    @test isdefined(OxiBLAS,:mul!) && isdefined(OxiBLAS,:mul)
    @test parentmodule(OxiBLAS.mul!)===OxiBLAS && parentmodule(OxiBLAS.mul)===OxiBLAS
    @test OxiBLAS.mul! !== LA.mul! && OxiBLAS.mul !== Base.:*
    @test :mul! ∉ names(OxiBLAS) && :mul ∉ names(OxiBLAS)
    # OxiBLAS methods of LinearAlgebra.mul! and Base.* all name an OxiBLAS type
    # in their signature (no type piracy); OxiBLAS.mul! stays a separate function.
    @test any(m->m.module===OxiBLAS,methods(LA.mul!))
    @test all(m->m.module!==OxiBLAS || _mentions_oxiblas(m.sig),methods(LA.mul!))
    @test all(m->m.module!==OxiBLAS || _mentions_oxiblas(m.sig),methods(Base.:*))
end

@testset "OxiBLAS.mul! combinations $T" for T in (Float32,Float64,ComplexF32,ComplexF64)
    m,k,n=7,5,6
    α,β=T(2)/3,T(-1)/4
    D=_mrand(T,m,k); E=_mrand(T,k,n); x=_mrand(T,k)
    S=_msparse(T,m,k); R=_msparse(T,k,n); sx=SA.sprand(T<:Complex ? ComplexF64 : Float64,k,0.6)
    sx=SparseVector{T,Int}(k,sx.nzind,T.(sx.nzval))
    Sq=_msparse(T,k,k); Dq=_mrand(T,k,k)
    Hm=_mrand(T,k,k); Hm=Hm+Hm'
    lefts=Any[D,transpose(_mrand(T,k,m)),adjoint(_mrand(T,k,m)),view(_mrand(T,m+2,k),2:m+1,:),
              S,transpose(_msparse(T,k,m)),adjoint(_msparse(T,k,m)),
              CSRMatrix(S),CSCMatrix(S),COOMatrix(S),SparseTensor(S)]
    rights=Any[E,transpose(_mrand(T,n,k)),R,adjoint(_msparse(T,n,k)),CSRMatrix(R),CSCMatrix(R),
               LA.Diagonal(_mrand(T,k))]
    square=Any[LA.Diagonal(_mrand(T,k)),LA.UpperTriangular(Dq),LA.LowerTriangular(Dq),
               LA.UnitUpperTriangular(Dq),LA.UnitLowerTriangular(Dq),transpose(LA.UpperTriangular(Dq)),
               LA.Symmetric(Dq),LA.Symmetric(Dq,:L),LA.Hermitian(Hm),LA.Hermitian(Hm,:L),
               LA.Bidiagonal(_mrand(T,k),_mrand(T,k-1),:U),LA.Bidiagonal(_mrand(T,k),_mrand(T,k-1),:L),
               LA.Tridiagonal(_mrand(T,k-1),_mrand(T,k),_mrand(T,k-1)),
               LA.SymTridiagonal(_mrand(T,k),_mrand(T,k-1)),Sq,CSRMatrix(Sq)]
    check(C0,A,B)=begin
        ref=α*(Array(A)*Array(B))+β*Array(C0)
        C=copy(C0); r=OxiBLAS.mul!(C,A,B,α,β)
        r===C && _mclose(C,ref,T)
    end
    for A in lefts, B in rights
        @test check(_mrand(T,m,size(B,2)),A,B)
        @test check(_mrand(T,m),A,x) && check(_mrand(T,m),A,sx)
    end
    for A in square, B in Any[E,_msparse(T,k,n)]
        @test check(_mrand(T,k,n),A,B)
        @test check(_mrand(T,n,k),transpose(B),A)
    end
    for A in square
        @test check(_mrand(T,k),A,x)
    end
    # Three-argument form, β = 0 never reads C (NaN not propagated).
    C=fill(T(NaN),m,n); OxiBLAS.mul!(C,D,E); @test _mclose(C,D*E,T)
    C=fill(T(NaN),m,n); OxiBLAS.mul!(C,S,R); @test _mclose(C,Array(S)*Array(R),T)
    C=fill(T(NaN),m,n); OxiBLAS.mul!(C,D,E,α,zero(T)); @test _mclose(C,α*D*E,T)
    # α = 0 only scales.
    C0=_mrand(T,m,n); C=copy(C0); OxiBLAS.mul!(C,D,E,zero(T),β); @test C≈β*C0
    # Outer products.
    a=_mrand(T,m); b=_mrand(T,n)
    @test check(_mrand(T,m,n),a,transpose(b)) && check(_mrand(T,m,n),a,adjoint(b))
    @test check(_mrand(T,m,n),reshape(a,m,1),reshape(b,1,n))
end

@testset "OxiBLAS.mul! sparse destinations $T" for T in (Float64,ComplexF64)
    m,k,n=8,6,7; α,β=T(3)/2,T(1)/2
    S=_msparse(T,m,k); R=_msparse(T,k,n); C0=_msparse(T,m,n); D=_mrand(T,m,k)
    ref=α*(Array(S)*Array(R))+β*Array(C0)
    for C in (copy(C0),CSRMatrix(copy(C0)),CSCMatrix(copy(C0)),SparseTensor(copy(C0)))
        r=OxiBLAS.mul!(C,S,R,α,β)
        @test r===C && _mclose(C,ref,T)
    end
    C=copy(C0); OxiBLAS.mul!(C,S,R); @test _mclose(C,Array(S)*Array(R),T)
    # dense × sparse into a sparse destination: exact pattern of the result.
    C=copy(C0); OxiBLAS.mul!(C,D,R,α,β); @test _mclose(C,α*D*Array(R)+β*Array(C0),T)
    # sparse matrix × sparse vector into a SparseVector.
    v=sparsevec([1,3,5],T[1,2,3],k); y0=sparsevec([2,4],T[1,-1],m)
    y=copy(y0); OxiBLAS.mul!(y,S,v,α,β); @test y isa SparseVector && _mclose(y,α*Array(S)*Array(v)+β*Array(y0),T)
    # Diagonal destination of a Diagonal product.
    d1,d2,d0=_mrand(T,k),_mrand(T,k),_mrand(T,k)
    Dd=LA.Diagonal(copy(d0)); OxiBLAS.mul!(Dd,LA.Diagonal(d1),LA.Diagonal(d2),α,β)
    @test Dd.diag≈α*d1.*d2+β*d0
    @test_throws ArgumentError OxiBLAS.mul!(LA.Diagonal(copy(d0)),_mrand(T,k,k),_mrand(T,k,k))
end

@testset "OxiBLAS.mul! aliasing and errors" begin
    T=Float64; A=randn(6,6); B=randn(6,6)
    C=copy(A); OxiBLAS.mul!(C,C,B); @test C≈A*B
    C=copy(B); OxiBLAS.mul!(C,A,C,2.0,1.0); @test C≈2A*B+B
    C=copy(B); OxiBLAS.mul!(C,LA.UpperTriangular(A),C); @test C≈LA.UpperTriangular(A)*B
    C=copy(B); OxiBLAS.mul!(C,sparse(A),C); @test C≈A*B
    C=copy(A); OxiBLAS.mul!(C,C,sparse(B)); @test C≈A*B
    x=randn(6); y=copy(x); OxiBLAS.mul!(y,A,y); @test y≈A*x
    @test_throws DimensionMismatch OxiBLAS.mul!(zeros(6,5),A,B)
    @test_throws DimensionMismatch OxiBLAS.mul!(zeros(6,6),A,randn(5,6))
    @test_throws DimensionMismatch OxiBLAS.mul!(zeros(5),A,randn(6))
    # Mixed element types: operands are converted to the destination's type.
    C=zeros(ComplexF64,6,6); OxiBLAS.mul!(C,Float32.(A),sparse(B)); @test C≈Float32.(A)*B
    C=zeros(6,6); OxiBLAS.mul!(C,LA.Symmetric(Float32.(A)),B); @test C≈LA.Symmetric(Float32.(A))*B atol=1e-5
    # Empty and zero-inner dimensions.
    @test OxiBLAS.mul!(zeros(0,3),zeros(0,2),zeros(2,3))==zeros(0,3)
    C=ones(3,3); OxiBLAS.mul!(C,zeros(3,0),zeros(0,3),1.0,2.0); @test C==fill(2.0,3,3)
end

@testset "OxiBLAS.mul! generic $T" for T in (BigFloat,Complex{BigFloat})
    setprecision(BigFloat,128) do
        A=_mrand(T,5,4); B=_mrand(T,4,3); S=_msparse(T,5,4); x=_mrand(T,4); α,β=T(2),T(1)/3
        C0=_mrand(T,5,3)
        for (L,Rm) in ((A,B),(S,B),(A,_msparse(T,4,3)),(S,_msparse(T,4,3)),(LA.Diagonal(_mrand(T,5)),_mrand(T,5,3)))
            C=copy(size(L,1)==5 ? C0 : _mrand(T,size(L,1),3)); Cref=α*(Array(L)*Array(Rm))+β*C
            OxiBLAS.mul!(C,L,Rm,α,β); @test _mclose(C,Cref,T)
        end
        y=zeros(T,5); OxiBLAS.mul!(y,S,x); @test _mclose(y,Array(S)*x,T)
        H=_mrand(T,4,4); H=H+H'
        C=zeros(T,4,3); OxiBLAS.mul!(C,LA.Hermitian(H),B); @test _mclose(C,H*B,T)
        C=zeros(T,4,3); OxiBLAS.mul!(C,LA.LowerTriangular(H),B); @test _mclose(C,LA.LowerTriangular(H)*B,T)
        C=zeros(T,4,3); OxiBLAS.mul!(C,LA.SymTridiagonal(real.(_mrand(T,4)),real.(_mrand(T,3))),B)
    end
end

@testset "OxiBLAS.mul result types" begin
    A=randn(5,4); B=randn(4,3); S=sparse(A.*(rand(5,4).<0.4)); R=sparse(B.*(rand(4,3).<0.4))
    @test OxiBLAS.mul(A,B) isa Matrix{Float64} && OxiBLAS.mul(A,B)≈A*B
    @test OxiBLAS.mul(A,randn(4)) isa Vector{Float64}
    P=OxiBLAS.mul(S,R); @test P isa SparseMatrixCSC{Float64} && P≈S*R
    P=OxiBLAS.mul(CSRMatrix(S),CSCMatrix(R)); @test P isa CSRMatrix{Float64} && Array(P)≈Array(S)*Array(R)
    P=OxiBLAS.mul(CSRMatrix(S),R); @test P isa SparseMatrixCSC && P≈S*R
    P=OxiBLAS.mul(transpose(S),S); @test P isa SparseMatrixCSC && P≈transpose(S)*S
    P=OxiBLAS.mul(SparseTensor(S),SparseTensor(R)); @test P isa CSRMatrix && Array(P)≈Array(S)*Array(R)
    v=sparsevec([1,4],[2.0,-1.0],4)
    P=OxiBLAS.mul(S,v); @test P isa SparseVector && P≈S*v
    P=OxiBLAS.mul(S,B); @test P isa Matrix && P≈S*B
    P=OxiBLAS.mul(A,R); @test P isa Matrix && P≈A*R
    d=randn(5); e=randn(4)
    @test OxiBLAS.mul(LA.Diagonal(d),LA.Diagonal(d)) isa LA.Diagonal
    P=OxiBLAS.mul(LA.Diagonal(d),S); @test P isa SparseMatrixCSC && P≈LA.Diagonal(d)*S && SA.nnz(P)==SA.nnz(S)
    P=OxiBLAS.mul(S,LA.Diagonal(e)); @test P isa SparseMatrixCSC && P≈S*LA.Diagonal(e)
    P=OxiBLAS.mul(LA.Diagonal(d),CSRMatrix(S)); @test P isa CSRMatrix && Array(P)≈LA.Diagonal(d)*Array(S)
    U=randn(4,4)
    P=OxiBLAS.mul(LA.UpperTriangular(U),LA.UpperTriangular(U)); @test P isa LA.UpperTriangular && P≈LA.UpperTriangular(U)^2
    P=OxiBLAS.mul(LA.UnitLowerTriangular(U),LA.UnitLowerTriangular(U)); @test P isa LA.UnitLowerTriangular && P≈LA.UnitLowerTriangular(U)^2
    P=OxiBLAS.mul(LA.UpperTriangular(U),LA.LowerTriangular(U)); @test P isa Matrix && P≈LA.UpperTriangular(U)*LA.LowerTriangular(U)
    # Promotion keeps the structure.
    P=OxiBLAS.mul(Float32.(S),ComplexF64.(R)); @test P isa SparseMatrixCSC{ComplexF64} && P≈Float32.(S)*R
    P=OxiBLAS.mul(big.(A),B); @test P isa Matrix{BigFloat} && P≈A*B
    @test_throws DimensionMismatch OxiBLAS.mul(A,A)
end

@testset "OxiBLAS.mul sparse-backed structured wrappers $T" for T in (Float64,ComplexF64)
    n=7; S=_msparse(T,n,n,0.4); X=_mrand(T,n,3)
    for W in (LA.UpperTriangular,LA.LowerTriangular,LA.UnitUpperTriangular,LA.UnitLowerTriangular)
        A=W(S); ref=W(Array(S))
        @test _mclose(OxiBLAS.mul!(zeros(T,n,3),A,X),ref*X,T)
        P=OxiBLAS.mul(A,_msparse(T,n,n)); @test P isa SparseMatrixCSC
        P=OxiBLAS.mul(W(CSRMatrix(S)),CSRMatrix(S)); @test P isa CSRMatrix && _mclose(P,ref*Array(S),T)
    end
    for (W,u) in ((LA.Symmetric,:U),(LA.Symmetric,:L),(LA.Hermitian,:U),(LA.Hermitian,:L))
        A=W(S,u); ref=W(Array(S),u)
        @test _mclose(OxiBLAS.mul!(zeros(T,n,3),A,X),ref*X,T)
        @test _mclose(OxiBLAS.mul!(zeros(T,3,n),transpose(X),A),transpose(X)*ref,T)
        @test _mclose(OxiBLAS.mul!(zeros(T,n),A,X[:,1]),ref*X[:,1],T)
        P=OxiBLAS.mul(A,S); @test P isa SparseMatrixCSC && _mclose(P,ref*Array(S),T)
        P=OxiBLAS.mul(S,A); @test P isa SparseMatrixCSC && _mclose(P,Array(S)*ref,T)
        # No densification: the mirrored matrix has at most twice nnz(S).
        @test SA.nnz(OxiBLAS._structured_sparse(A))<=2SA.nnz(S)
    end
end

@testset "OxiBLAS.mul! element conversion keeps the format" begin
    S=_msparse(Float32,6,6); x=randn(6)
    for A in (CSRMatrix(S),CSCMatrix(S),COOMatrix(S),DIAMatrix(CSRMatrix(S)),ELLMatrix(CSRMatrix(S)),
              BSRMatrix(CSRMatrix(S);blocksize=(2,2)),SELLMatrix(CSRMatrix(S)),HYBMatrix(CSRMatrix(S)))
        B=OxiBLAS._mul_operand(Float64,A)
        @test Base.typename(typeof(B))===Base.typename(typeof(A)) && eltype(B)===Float64
        @test Array(B)≈Array(A)
        @test OxiBLAS.mul!(zeros(6),A,x)≈Array(S)*x
    end
    @test OxiBLAS._mul_operand(ComplexF64,LA.Symmetric(CSRMatrix(S))) isa LA.Symmetric{ComplexF64,<:CSRMatrix}
end

@testset "OxiBLAS.mul! overlapping storage" begin
    X=randn(12,12); n=5
    # C and an operand are offset views of the same array, on every path.
    function check(f,ref)
        Y=copy(X); C=view(Y,1:n,1:n); A=view(Y,2:n+1,2:n+1)
        expected=ref(copy(A)); OxiBLAS.mul!(C,f(A)...,2.0,0.5) ; C≈2expected+0.5*X[1:n,1:n]
    end
    B=randn(n,n); d=randn(n)
    @test check(A->(A,B),A->A*B)
    @test check(A->(B,A),A->B*A)
    @test check(A->(transpose(A),B),A->transpose(A)*B)
    @test check(A->(LA.Diagonal(d),A),A->LA.Diagonal(d)*A)
    @test check(A->(A,LA.Diagonal(d)),A->A*LA.Diagonal(d))
    @test check(A->(LA.UpperTriangular(A),B),A->LA.UpperTriangular(A)*B)
    @test check(A->(LA.Symmetric(A),B),A->LA.Symmetric(A)*B)
    Bb=LA.Bidiagonal(randn(n),randn(n-1),:U)
    @test check(A->(Bb,A),A->Bb*A) && check(A->(A,Bb),A->A*Bb)
    Bd=LA.Tridiagonal(randn(n-1),randn(n),randn(n-1))
    @test check(A->(Bd,A),A->Bd*A) && check(A->(A,Bd),A->A*Bd)
    Sp=_msparse(Float64,n,n)
    @test check(A->(Sp,A),A->Sp*A) && check(A->(A,Sp),A->A*Sp)
    # Diagonal destination sharing the diagonal of an operand.
    v=randn(n); ref=v.*v; Dd=LA.Diagonal(v); OxiBLAS.mul!(Dd,Dd,Dd); @test Dd.diag≈ref
    w=randn(2n); Dc=LA.Diagonal(view(w,1:n)); Da=LA.Diagonal(view(w,2:n+1)); e=w[2:n+1].*3
    OxiBLAS.mul!(Dc,Da,LA.Diagonal(fill(3.0,n))); @test Dc.diag≈e
    # Vector overlap: y and x are offset views of one array.
    z=randn(2n); y=view(z,1:n); x=view(z,2:n+1); M=randn(n,n); ref=M*z[2:n+1]
    OxiBLAS.mul!(y,M,x); @test y≈ref
    z=randn(2n); y=view(z,1:n); x=view(z,2:n+1); ref=LA.Diagonal(d)*z[2:n+1]
    OxiBLAS.mul!(y,LA.Diagonal(d),x); @test y≈ref
end

@testset "OxiBLAS.mul! sparse destination from a dense operand" begin
    S=_msparse(Float64,6,5); D=randn(5,4); C0=_msparse(Float64,6,4)
    C=copy(C0); OxiBLAS.mul!(C,S,D,2.0,1.0); @test C≈2*(Array(S)*D)+Array(C0)
    C=CSRMatrix(copy(C0)); OxiBLAS.mul!(C,randn(6,5)*0,sparse(D)); @test SA.nnz(C)<=SA.nnz(C0)
end

@testset "OxiBLAS.mul! compact storages (Mission 1)" begin
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

        # Matrices for each of the 5 optimized paths:
        # 1. Banded general
        data_g = rnd_mat(4, n) # kl=2, ku=1 -> kl+ku+1 = 4
        Bg = BandedMatrix(data_g, n; kl=2, ku=1, kind=:general)

        # 2. Banded symmetric (:U and :L)
        data_s = rnd_mat(3, n)
        Bsu = BandedMatrix(data_s, n; ku=2, kind=:symmetric, uplo=:U)
        Bsl = BandedMatrix(data_s, n; kl=2, kind=:symmetric, uplo=:L)

        # 3. Banded hermitian (:U and :L)
        data_h = rnd_mat(3, n)
        Bhu = BandedMatrix(data_h, n; ku=2, kind=:hermitian, uplo=:U)
        Bhl = BandedMatrix(data_h, n; kl=2, kind=:hermitian, uplo=:L)

        # 4. Packed symmetric (:U and :L)
        p_len = n*(n+1)÷2
        Psu = PackedMatrix(rnd_vec(p_len), n; kind=:symmetric, uplo=:U)
        Psl = PackedMatrix(rnd_vec(p_len), n; kind=:symmetric, uplo=:L)

        # 5. Packed hermitian (:U and :L)
        Phu = PackedMatrix(rnd_vec(p_len), n; kind=:hermitian, uplo=:U)
        Phl = PackedMatrix(rnd_vec(p_len), n; kind=:hermitian, uplo=:L)

        compact_matrices = [Bg, Bsu, Bsl, Bhu, Bhl, Psu, Psl, Phu, Phl]

        for A in compact_matrices
            # Vector operand
            x = rnd_vec(n)
            check_compact(A, x)
            # Matrix operand (multiple columns)
            Y = rnd_mat(n, k_cols)
            check_compact(A, Y)
        end

        # 2. Small dimensions: N=1 and empty case
        # N = 1
        Bg1 = BandedMatrix(rnd_mat(1, 1), 1; kl=0, ku=0, kind=:general)
        Bs1 = BandedMatrix(rnd_mat(1, 1), 1; ku=0, kind=:symmetric, uplo=:U)
        Bh1 = BandedMatrix(rnd_mat(1, 1), 1; ku=0, kind=:hermitian, uplo=:U)
        Ps1 = PackedMatrix(rnd_vec(1), 1; kind=:symmetric, uplo=:U)
        Ph1 = PackedMatrix(rnd_vec(1), 1; kind=:hermitian, uplo=:U)
        for A in [Bg1, Bs1, Bh1, Ps1, Ph1]
            check_compact(A, rnd_vec(1))
            check_compact(A, rnd_mat(1, 2))
        end
        # Empty case (0x0)
        Bg0 = BandedMatrix(zeros(T, 1, 0), 0; kl=0, ku=0, kind=:general)
        y0 = zeros(T, 0); x0 = zeros(T, 0)
        @test OxiBLAS.mul!(y0, Bg0, x0) === y0
        @test isempty(OxiBLAS.mul(Bg0, x0))
        # Empty case rectangular (0x3)
        Bg_0x3 = BandedMatrix(zeros(T, 2, 3), 0; kl=0, ku=1, kind=:general)
        y_0 = zeros(T, 0); x_3 = rnd_vec(3)
        @test OxiBLAS.mul!(y_0, Bg_0x3, x_3) === y_0

        # 3. Rectangular BandedMatrix with kl != ku
        # Tall: m > n, kl != ku
        m_rec, n_rec = 8, 5
        data_rec1 = rnd_mat(5, n_rec) # kl=3, ku=1 -> kl+ku+1 = 5
        B_tall = BandedMatrix(data_rec1, m_rec; kl=3, ku=1, kind=:general)
        check_compact(B_tall, rnd_vec(n_rec))
        check_compact(B_tall, rnd_mat(n_rec, 4))
        # Wide: m < n, kl != ku
        m_w, n_w = 4, 7
        data_rec2 = rnd_mat(4, n_w) # kl=1, ku=2 -> kl+ku+1 = 4
        B_wide = BandedMatrix(data_rec2, m_w; kl=1, ku=2, kind=:general)
        check_compact(B_wide, rnd_vec(n_w))
        check_compact(B_wide, rnd_mat(n_w, 2))

        # 8. Storage sharing / aliasing
        # a) x and y share storage (in-place vector update: y === x)
        for A in [Bg, Bsu, Bhu, Psu, Phu]
            x_shared = rnd_vec(n)
            ref_x = 2.0 * (Array(A) * copy(x_shared))
            OxiBLAS.mul!(x_shared, A, x_shared, T(2), T(0))
            @test _mclose(x_shared, ref_x, T)
        end
        # b) Output overlaps with compact coefficients
        data_alias = copy(Bg.data)
        A_alias = BandedMatrix(data_alias, n; kl=Bg.kl, ku=Bg.ku, kind=:general)
        out_row = view(data_alias, 1, 1:n)
        x_in = rnd_vec(n)
        expected_out = Array(A_alias) * x_in
        OxiBLAS.mul!(out_row, A_alias, x_in)
        @test _mclose(out_row, expected_out, T)

        # 10. Incompatible dimensions and result types
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

    # 9. Complex types: explicitly distinguish symmetric from hermitian
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

    # Non-regression: Unoptimized paths continue to function as before
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        n = 4
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
        X_dense = _mrand(T, 3, n)
        Bg = BandedMatrix(_mrand(T, 3, n), n; kl=1, ku=1, kind=:general)
        Psu = PackedMatrix(_mrand(T, n*(n+1)÷2), n; kind=:symmetric, uplo=:U)
        for P in [Bg, Psu]
            C_right = zeros(T, 3, n)
            OxiBLAS.mul!(C_right, X_dense, P)
            @test _mclose(C_right, X_dense * Array(P), T)
        end
    end
end
