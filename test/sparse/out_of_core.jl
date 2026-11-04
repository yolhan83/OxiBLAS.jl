# All filesystem work here is confined to owned mktempdir trees when authorized.
@testset "Native disk tiles: LU and Cholesky" begin
    mktempdir() do parent
        sentinel=joinpath(parent,"keep.txt"); write(sentinel,"caller-owned")
        for T in TYPES, n in (0,1,7), blocksize in (1,2,3,16), kind in (:lu,:cholesky)
            seed=randn(T,n,n); seed[abs.(seed).<real(T)(.8)].=zero(T)
            A=kind===:lu ? seed+T(8)*LA.I : seed*seed'+T(4)*LA.I
            if kind===:lu && n>1
                A=A[collect(n:-1:1),:] # Forces swaps spanning previous L tiles.
            end
            C=CSRMatrix(A); before=copy(C.values)
            make=kind===:lu ? out_of_core_lu : out_of_core_cholesky
            F=make(C;blocksize,directory=parent,ordering=collect(n:-1:1))
            cache=F.store.directory
            try
                @test F isa OutOfCoreFactor
                @test size(F)==(n,n)
                @test isdir(cache) && dirname(cache)==parent
                @test C.values==before
                usage=out_of_core_memory_usage(F)
                @test usage.payload_bytes<=usage.budget_bytes
                @test usage.payload_bytes==sizeof(T)*(length(F.panel)+sum(length,F.buffers)+sum(length(B.a)+length(B.b) for B in F.workspace.buffers))
                @test !usage.includes_input_output && !usage.includes_runtime_overhead
                @test usage.metadata_bytes>0
                for trans in (:N,:T,:C)
                    b=randn(T,n,3); original=copy(b)
                    x=solve(F,view(b,n:-1:1,3:-1:1);trans)
                    @test op(A,trans)*x≈b[n:-1:1,3:-1:1] rtol=100tol(T) atol=100tol(T)
                    @test op(A,trans)*solve(F,view(b,:,1);trans)≈b[:,1] rtol=100tol(T) atol=100tol(T)
                    @test x≈op(A,trans)\b[n:-1:1,3:-1:1] rtol=100tol(T) atol=100tol(T)
                    @test b==original
                end
                @test size(solve(F,zeros(T,n,0)))==(n,0)
                b=ones(T,n); GC.gc()
                tasks=[Threads.@spawn solve(F,b) for _ in 1:3]
                GC.gc()
                @test all(x->isapprox(A*x,b;rtol=100tol(T),atol=100tol(T)),fetch.(tasks))
                @test_throws DimensionMismatch solve(F,ones(T,n+1))
                @test_throws ArgumentError solve(F,b;trans=:bad)
                n>0 && @test_throws ArgumentError solve(F,view(F.panel,1:n))
            finally
                close(F)
            end
            @test !ispath(cache)
            @test close(F)===nothing
            @test_throws ArgumentError solve(F,ones(T,n))
            @test read(sentinel,String)=="caller-owned"
        end
        # Delayed cross-tile pivot: k=3 swaps row 3 with row 6, whose first
        # two L entries differ. This detects forgetting to swap previous L.
        A=[10.0 0 0 0 0 0; 0 10 0 0 0 0;
           1 2 0 1 0 0; 3 4 1 0 1 0;
           5 6 0 0 0 1; 7 8 9 0 0 1]
        F=out_of_core_lu(CSRMatrix(A);blocksize=2,directory=parent)
        try
            @test F.p[3]==6
            for tr in (:N,:T,:C)
                b=collect(1.0:6.0)
                @test op(A,tr)*solve(F,b;trans=tr)≈b
            end
        finally
            close(F)
        end
        # Cholesky reads exactly one triangle, ignoring poisoned storage.
        H=ComplexF64[5 1-im 0;1+im 6 2im;0 -2im 7]
        for uplo in (:L,:U)
            stored=copy(H)
            for i in 1:3, j in 1:3
                (uplo===:L ? i<j : i>j) && (stored[i,j]=ComplexF64(NaN))
            end
            F=out_of_core_cholesky(CSRMatrix(stored);uplo,blocksize=2,directory=parent)
            try
                @test H*solve(F,ones(ComplexF64,3))≈ones(3)
            finally
                close(F)
            end
        end
        # The explicit budget covers numeric payload, and shrinks the tile.
        n=19; C=CSRMatrix(Matrix{Float64}(LA.I,n,n))
        budget=OxiBLAS._ooc_payload(Float64,n,2,1)
        F=out_of_core_lu(C;blocksize=8,memory_budget=budget,directory=parent)
        try
            @test F.store.blocksize==2
            @test out_of_core_memory_usage(F).payload_bytes<=budget
            @test solve(F,ones(n))==ones(n)
        finally
            close(F)
        end
        before=sort(readdir(parent))
        @test_throws ArgumentError out_of_core_lu(C;memory_budget=0,directory=parent)
        @test_throws ArgumentError out_of_core_lu(C;blocksize=0,directory=parent)
        @test_throws SingularException out_of_core_lu(CSRMatrix(zeros(3,3));directory=parent)
        @test_throws PosDefException out_of_core_cholesky(CSRMatrix([-1.0 0;0 1]);directory=parent)
        @test_throws ArgumentError out_of_core_lu(CSRMatrix([Inf;;]);directory=parent)
        @test_throws ArgumentError out_of_core_lu(C;ordering=[1,1],directory=parent)
        @test_throws DimensionMismatch out_of_core_lu(CSRMatrix(ones(2,3));directory=parent)
        @test sort(readdir(parent))==before # Failed construction removes owned files.
        @test_throws Exception out_of_core_lu(C;directory=joinpath(parent,"absent"))
        # Independent factors own different directories and can close separately.
        F=out_of_core_lu(C;directory=parent); G=out_of_core_lu(C;directory=parent)
        try
            @test F.store.directory!=G.store.directory
            close(F); @test solve(G,ones(n))==ones(n)
        finally
            close(F); close(G)
        end
        # File guards: size, magic, byte order, scalar tag, shape, checksum,
        # and deletion of a tile marked present must all fail explicitly.
        for damage in (:truncate,:magic,:endian,:tag,:rows,:columns,:payload,:missing)
            F=out_of_core_lu(CSRMatrix([3.0 1;1 4]);blocksize=2,directory=parent)
            try
                path=OxiBLAS._ooc_path(F.store,1,1)
                if damage===:missing
                    rm(path)
                elseif damage===:truncate
                    open(path,"r+") do io
                        truncate(io,38)
                    end
                else
                    offset=damage===:magic ? 0 : damage===:endian ? 8 : damage===:tag ? 12 : damage===:rows ? 13 : damage===:columns ? 21 : 37
                    open(path,"r+") do io
                        seek(io,offset); old=read(io,UInt8); seek(io,offset); write(io,old ⊻ UInt8(0xff))
                    end
                end
                if damage===:missing
                    @test_throws Exception solve(F,ones(2))
                else
                    @test_throws ArgumentError solve(F,ones(2))
                end
            finally
                close(F)
            end
        end
        # Inner parallel kernels share only the guarded, preallocated workspace.
        A=Matrix(LA.Tridiagonal(fill(-1.0,32),fill(4.0,33),fill(-1.0,32)))
        for make in (out_of_core_lu,out_of_core_cholesky)
            F=make(CSRMatrix(A);blocksize=17,threaded=true,directory=parent)
            try
                @test A*solve(F,ones(33))≈ones(33)
            finally
                close(F)
            end
        end
    end
end
