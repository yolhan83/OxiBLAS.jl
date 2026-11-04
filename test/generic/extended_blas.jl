function extcontext(::Type{T},m,n,k;blocksize=32,workers=1) where T
    T<:OxiBLAS.NativeFloat ? GemmWorkspace(T,m,n,k;workers) : GenericWorkspace(T;blocksize,workers)
end
struct ExtendedNoReadMatrix{T} <: AbstractMatrix{T}
    dimensions::Tuple{Int,Int}
end
Base.size(A::ExtendedNoReadMatrix)=A.dimensions
Base.getindex(::ExtendedNoReadMatrix,i::Int,j::Int)=error("extended operand must not be read")
# check_extended_array: test/helpers/extended.jl
@testset "Extended BLAS 1-3, storage and contexts" begin
    setprecision(BigFloat,512) do
        for T in EXT_TYPES
            tolerance=exttol(T); R=real(T)
            x=T.([1.0,-2,.125,4]); y=T.([.5,3,-4,2]); oldx=copy(x); oldy=copy(y)
            alpha=T(.375)
            axpy!(y,x,alpha); check_extended_array(y,extref(oldy)+extref(alpha)*extref(oldx),T)
            scal!(x,T(-2)); check_extended_array(x,-2extref(oldx),T)
            copy_vector!(y,oldy); @test y==oldy
            @test extref(dotu(oldx,oldy))≈sum(extref(oldx).*extref(oldy)) rtol=tolerance
            @test extref(dotc(oldx,oldy))≈sum(conj.(extref(oldx)).*extref(oldy)) rtol=tolerance
            @test extref(dot_kahan(oldx,oldy))≈sum(extref(oldx).*extref(oldy)) rtol=tolerance
            @test extref(dot_pairwise(oldx,oldy))≈sum(extref(oldx).*extref(oldy)) rtol=tolerance
            @test extref(nrm2(oldx))≈sqrt(sum(abs2,extref(oldx))) rtol=tolerance
            @test iamax(oldx)==4 && iamin(oldx)==3
            @test extref(asum(oldx))==sum(abs.(real.(extref(oldx)))+abs.(imag.(extref(oldx))))
            @test dot_parallel(oldx,oldy)==dotu(oldx,oldy)
            @test nrm2_parallel(oldx)≈nrm2(oldx)
            @test asum_parallel(oldx)==asum(oldx)
            px=copy(oldx);py=copy(oldy);axpy_parallel!(py,px,alpha)
            check_extended_array(py,extref(oldy)+extref(alpha)*extref(oldx),T)
            scal_parallel!(px,T(2));check_extended_array(px,2extref(oldx),T)
            swap!(x,y); @test x==oldy
            c=R(.6); s=T(.8); rx=copy(oldx); ry=copy(oldy)
            rot!(rx,ry,c,s)
            check_extended_array(rx,extref(c)*extref(oldx)+extref(s)*extref(oldy),T)
            check_extended_array(ry,extref(c)*extref(oldy)-conj(extref(s))*extref(oldx),T)
            if T<:Real
                p=RotmParams(T;flag=-1,h11=2,h12=.25,h21=-.5,h22=1)
                rx=copy(oldx); ry=copy(oldy); rotm!(rx,ry,p;kernel=:scalar)
                check_extended_array(rx,2extref(oldx)+big".25"*extref(oldy),T)
                check_extended_array(ry,-big".5"*extref(oldx)+extref(oldy),T)
                modified=rotmg(T(2),T(3),T(4),T(.5))
                @test all(isfinite,(modified.d1,modified.d2,modified.x1))
            end
            for (m,k,n) in ((0,3,2),(3,0,4),(1,1,1),(7,5,9)), ta in (:N,:T,:C), tb in (:N,:T,:C)
                a=reshape([i%7/8-0.25 for i in 1:m*k],m,k)
                b=reshape([i%5/4-.5 for i in 1:k*n],k,n)
                A=T.(ta===:N ? a : transpose(a)); B=T.(tb===:N ? b : transpose(b))
                T<:Complex && (A .+= T(im/8); B .-= T(im/4))
                C=fill(T(.25),m,n); before=copy(C); workspace=extcontext(T,m,n,k;blocksize=3,workers=2)
                alpha=T(.5); beta=T(-.25)
                expected=extref(alpha)*(op(extref(A),ta)*op(extref(B),tb))+extref(beta)*extref(before)
                gemm!(C,A,B,alpha,beta;transA=ta,transB=tb,workspace)
                check_extended_array(C,expected,T;factor=8)
                gemm_extended!(C,A,B,alpha,zero(T);transA=ta,transB=tb,workspace)
                check_extended_array(C,extref(alpha)*(op(extref(A),ta)*op(extref(B),tb)),T;factor=8)
            end
            A=T.([2.0 .5 -.25;0 3 .125;0 0 4]); B=T.([1.0 2;3 4;5 6])
            context=GemmWorkspace(T,3,3,3;workers=2)
            @test (T<:OxiBLAS.NativeFloat ? context isa GemmWorkspace : context isa GenericWorkspace)
            C=fill(T(NaN),3,2); gemm!(C,A,B;workspace=context)
            check_extended_array(C,extref(A)*extref(B),T;factor=8)
            original=copy(A); gemm!(A,A,Matrix{T}(LA.I,3,3);workspace=context)
            @test A==original
            C=fill(T(NaN),3,2)
            gemm!(C,ExtendedNoReadMatrix{T}((3,3)),ExtendedNoReadMatrix{T}((3,2)),zero(T),zero(T))
            @test all(iszero,C)
            for operation in (trmm!,trsm!)
                out=fill(T(NaN),3,2)
                operation(out,ExtendedNoReadMatrix{T}((3,3)),zero(T))
                @test all(iszero,out)
            end
            out=fill(T(NaN),3,3)
            syrk!(out,ExtendedNoReadMatrix{T}((3,2)),zero(T),zero(T);uplo=:U)
            @test all(iszero(out[i,j]) for j in 1:3 for i in 1:j)
            @test isnan(out[3,1])
            if !(T<:OxiBLAS.NativeFloat)
                @test_throws ArgumentError gemm!(C,A,B;kernel=:avx2)
                @test_throws ArgumentError gemm!(C,A,B;workspace=GemmWorkspace(Float64,3,2,3))
            end
            if context isa GenericWorkspace && eltype(context.buffers[1])===T
                @test_throws ArgumentError gemm!(view(context.buffers[1],1:3,1:2),A,B;workspace=context)
            end
            parent=fill(zero(T),8,6); output=view(parent,7:-2:1,6:-2:2)
            aa=T.(reshape(collect(1:8),4,2)./8); bb=T.(reshape(collect(1:6),2,3)./4)
            gemm!(output,aa,bb); check_extended_array(output,extref(aa)*extref(bb),T;factor=8)
            # Triangular operations retain side/transpose/unit contracts.
            for side in (:L,:R), uplo in (:L,:U), trans in (:N,:T,:C), unit in (false,true)
                tri=T.([2.0 .25 -.5;.125 3 .5;-.25 .375 4])
                effective=uplo===:L ? Matrix(LA.LowerTriangular(extref(tri))) : Matrix(LA.UpperTriangular(extref(tri)))
                unit && (effective[LA.diagind(effective)].=1)
                for i in 1:3,j in 1:3
                    (uplo===:L ? i<j : i>j) && (tri[i,j]=T(NaN))
                    unit && i==j && (tri[i,j]=T(NaN))
                end
                rhs=side===:L ? T.([1.0 2;3 4;5 6]) : T.([1.0 2 3;4 5 6])
                expected=side===:L ? op(effective,trans)\extref(rhs) : extref(rhs)/op(effective,trans)
                got=copy(rhs); trsm!(got,tri;side,uplo,trans,unit); check_extended_array(got,expected,T;factor=20)
                got=copy(rhs); trmm!(got,tri;side,uplo,trans,unit)
                expected=side===:L ? op(effective,trans)*extref(rhs) : extref(rhs)*op(effective,trans)
                check_extended_array(got,expected,T;factor=20)
            end
            H=T.([4.0 .5 -.25;.5 3 .125;-.25 .125 2]); rhs=T.([1.0,2,3])
            for uplo in (:L,:U)
                got=zeros(T,3); symv!(got,H,rhs;uplo); check_extended_array(got,extref(H)*extref(rhs),T;factor=8)
                hemv!(got,H,rhs;uplo); check_extended_array(got,extref(H)*extref(rhs),T;factor=8)
                P=pack_matrix(H;uplo); spmv!(got,P,rhs;kernel=:scalar)
                check_extended_array(got,extref(H)*extref(rhs),T;factor=8)
                P=pack_matrix(H;uplo,kind=:hermitian); hpmv!(got,P,rhs;kernel=:scalar)
                check_extended_array(got,extref(H)*extref(rhs),T;factor=8)
                P=pack_matrix(A;uplo,kind=:triangular); v=copy(rhs); tpmv!(v,P;kernel=:scalar)
                @test all(isfinite,v); tpsv!(v,P;kernel=:scalar); check_extended_array(v,extref(rhs),T;factor=20)
            end
            for uplo in (:L,:U), hermitian in (false,true), rank2 in (false,true)
                aa=T.([1.0 2;3 .5;-.25 4]); bb=reverse(aa;dims=1)
                T<:Complex && (aa .+= T(im/8);bb .-= T(im/4))
                C=fill(T(NaN),3,3)
                if hermitian
                    rank2 ? her2k!(C,aa,bb,T(.25),zero(R);uplo) : herk!(C,aa,R(.25),zero(R);uplo)
                else
                    rank2 ? syr2k!(C,aa,bb,T(.25),zero(T);uplo) : syrk!(C,aa,T(.25),zero(T);uplo)
                end
                X,Y=extref(aa),extref(bb); rhs=hermitian ? adjoint : transpose
                expected=big".25"*(rank2 ? X*rhs(Y)+Y*rhs(X) : X*rhs(X))
                for j in 1:3, i in 1:3
                    if uplo===:L ? i>=j : i<=j
                        @test abs(extref(C[i,j])-expected[i,j])<=20tolerance*max(1,abs(expected[i,j]))
                    else
                        @test isnan(C[i,j])
                    end
                end
            end
            A=T.([1.0 2;3 4]); B=T.([2.0 .5;-.25 3]); C=fill(T(NaN),2,2)
            gemmt!(C,A,B;uplo=:U)
            for j in 1:2,i in 1:j
                @test abs(extref(C[i,j])-(extref(A)*extref(B))[i,j])<=10tolerance
            end
            @test isnan(C[2,1])
            Cs=[zeros(T,2,2) for _ in 1:4]; As=[copy(A) for _ in 1:4]; Bs=[copy(B) for _ in 1:4]
            gemm_batched!(Cs,As,Bs;workspace=extcontext(T,2,2,2;workers=2),batch_parallel_threshold=0)
            @test all(C->maximum(abs,extref(C)-extref(A)*extref(B))<=10tolerance,Cs)
            cube=zeros(T,2,2,3); aa=repeat(reshape(A,2,2,1),1,1,3); bb=repeat(reshape(B,2,2,1),1,1,3)
            gemm_strided_batched!(cube,aa,bb;batch_parallel_threshold=0)
            check_extended_array(cube[:,:,2],extref(A)*extref(B),T;factor=10)
            GC.gc(); shared=extcontext(T,2,2,2;workers=2)
            tasks=[Threads.@spawn begin out=zeros(T,2,2);gemm!(out,A,B;workspace=shared);out end for _ in 1:3]
            @test all(C->maximum(abs,extref(C)-extref(A)*extref(B))<=10tolerance,fetch.(tasks))
        end
    end
end
@testset "Extended BLAS complete dense/compact operation families" begin
    setprecision(BigFloat,512) do
        for T in EXT_TYPES
            R=real(T); x=T.([1.0,.5,-.25]); y=T.([.125,-2,3]); A=T.([4.0 .5 -.25;.5 3 .125;-.25 .125 2])
            T<:Complex && (x .+= T(im/8); y .-= T(im/4))
            c,s,r=rotg(x[1],y[1])
            @test abs(extref(c*x[1]+s*y[1]-r))<20exttol(T)
            @test abs(extref(c*y[1]-conj(s)*x[1]))<20exttol(T)
            for trans in (:N,:T,:C)
                v=zeros(T,3); gemv!(v,A,x;trans)
                check_extended_array(v,op(extref(A),trans)*extref(x),T;factor=20)
                P=band_matrix(A;kl=2,ku=2); gbmv!(v,P,x;trans,kernel=:scalar)
                check_extended_array(v,op(extref(A),trans)*extref(x),T;factor=20)
                tri=T.([2.0 .5 -.25;0 3 .125;0 0 4]); v=copy(x)
                trmv!(v,tri;trans); trsv!(v,tri;trans); check_extended_array(v,extref(x),T;factor=20)
                P=band_matrix(tri;kl=0,ku=2,kind=:triangular); v=copy(x)
                tbmv!(v,P;trans,kernel=:scalar); tbsv!(v,P;trans,kernel=:scalar)
                check_extended_array(v,extref(x),T;factor=20)
            end
            for conjugated in (false,true)
                C=zeros(T,3,3); conjugated ? gerc!(C,x,y,T(.25)) : ger!(C,x,y,T(.25))
                expected=big".25"*extref(x)*(conjugated ? extref(y)' : transpose(extref(y)))
                check_extended_array(C,expected,T;factor=20)
            end
            for hermitian in (false,true), rank2 in (false,true), upper in (false,true)
                uplo=upper ? :U : :L; C=zeros(T,3,3); alpha=hermitian && !rank2 ? R(.25) : T(.25)
                if hermitian
                    rank2 ? her2!(C,x,y,alpha;uplo) : her!(C,x,alpha;uplo)
                else
                    rank2 ? syr2!(C,x,y,alpha;uplo) : syr!(C,x,alpha;uplo)
                end
                X,Y=extref(x),extref(y); rhs=hermitian ? adjoint : transpose
                expected=big".25"*(rank2 ? X*rhs(Y)+Y*rhs(X) : X*rhs(X))
                P=pack_matrix(zeros(T,3,3);uplo,kind=hermitian ? :hermitian : :symmetric)
                if hermitian
                    rank2 ? hpr2!(P,x,y,alpha;kernel=:scalar) : hpr!(P,x,alpha;kernel=:scalar)
                else
                    rank2 ? spr2!(P,x,y,alpha;kernel=:scalar) : spr!(P,x,alpha;kernel=:scalar)
                end
                check_extended_array(Matrix(P),expected,T;factor=30)
                for j in 1:3,i in (upper ? (1:j) : (j:3))
                    @test abs(extref(C[i,j])-expected[i,j])<=30exttol(T)
                end
            end
            for hermitian in (false,true), side in (:L,:R)
                v=zeros(T,3); P=band_matrix(A;kl=2,ku=2,kind=hermitian ? :hermitian : :symmetric)
                hermitian ? hbmv!(v,P,x;kernel=:scalar) : sbmv!(v,P,x;kernel=:scalar)
                check_extended_array(v,extref(A)*extref(x),T;factor=20)
                B=T.([1.0 2 3;4 5 6;7 8 9]); C=zeros(T,3,3)
                hermitian ? hemm!(C,A,B;side) : symm!(C,A,B;side)
                expected=side===:L ? extref(A)*extref(B) : extref(B)*extref(A)
                check_extended_array(C,expected,T;factor=30)
            end
            @test_throws DimensionMismatch gemv!(zeros(T,2),A,x)
            @test_throws DimensionMismatch gemm!(zeros(T,2,2),A,A)
            @test_throws ArgumentError swap!(view(x,1:2),view(x,2:3))
            @test_throws ArgumentError GenericWorkspace(T;workers=0)
        end
        x=[1e16,1.0,-1e16]; y=ones(3)
        @test dot_extended(x,y)==1
        @test extref(dot_extended(ComplexF64.(x),ComplexF64.(y);conjugate=true))==1
    end
end
