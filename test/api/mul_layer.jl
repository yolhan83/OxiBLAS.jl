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
