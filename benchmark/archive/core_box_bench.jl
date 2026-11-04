# Core.Box campaign: one call per site reported by Test.detect_closure_boxes
# (reports/CORE_BOX_INVENTORY_20261104.md), measured before and after the
# fixes. Steady state: minimum time over several calls and minimum bytes
# allocated per call (@allocated), after one warm-up call.
#
# umf_lu.jl:38 is not measured: that engine is kept unwired on purpose.
#
# Run (4 threads, so that the parallel paths and their closures are taken):
#   julia -t4 --project=. benchmark/archive/core_box_bench.jl OUT.tsv
using OxiBLAS, SparseArrays, Random
import LinearAlgebra as LA
LA.BLAS.set_num_threads(1)
const O=OxiBLAS
const OUT=isempty(ARGS) ? "core_box_bench.tsv" : ARGS[1]

function measure(f;seconds=0.3,maxruns=50)
    f(); bytes=typemax(Int); best=Inf; runs=0; t0=time()
    while runs<3 || (time()-t0<seconds && runs<maxruns)
        b=@allocated f(); bytes=min(bytes,b)
        t=@elapsed f(); best=min(best,t); runs+=1
    end
    best,bytes
end
lap(k)=(e=ones(k); T=spdiagm(-1=>-e[2:end],0=>2e,1=>-e[2:end]); I=sparse(1.0LA.I,k,k); kron(I,T)+kron(T,I))
rng=Xoshiro(1)
A192=randn(rng,192,192); B192=randn(rng,192,192); C192=zeros(192,192)
A512=randn(rng,512,512); B512=randn(rng,512,512); C512=zeros(512,512)
I_(n)=Matrix(n*1.0LA.I,n,n)
U512=LA.triu(randn(rng,512,512))+I_(512)
A256=randn(rng,256,256); B256=randn(rng,256,256); C256=zeros(256,256)
A16=randn(rng,Float16,64,64); C16=zeros(Float16,64,64)
S=lap(40); Scsr=CSRMatrix(S); Sp=lap(30)
spd(n)=(X=randn(rng,n,n); X*X'+n*I_(n))
X3=SparseTensor((30,30,30),[(rand(rng,1:30),rand(rng,1:30),rand(rng,1:30)) for _ in 1:2000],randn(rng,2000))
Y3=SparseTensor((30,30,5),[(rand(rng,1:30),rand(rng,1:30),rand(rng,1:5)) for _ in 1:400],randn(rng,400))
U3=randn(rng,30,8)
Ds=[randn(rng,32,32) for _ in 1:16]; Es=[randn(rng,32,32) for _ in 1:16]; Fs=[zeros(32,32) for _ in 1:16]
d=randn(rng,300); e=randn(rng,299)
Aq=randn(rng,40,40); Bq=randn(rng,40,40)
Azq=randn(rng,ComplexF64,30,30); Bzq=randn(rng,ComplexF64,30,30)
Pspd=spd(40)
Ar=randn(rng,300,120)
Dense3=randn(rng,24,24,24); Dm=randn(rng,24,24)
Aspd=spd(100)

cases=[
 # site, label, call
 ("gemm.jl:537/104","gemm! 192, 1 thread",()->O.gemm!(C192,A192,B192;threaded=false)),
 ("gemm.jl:471","gemm! 512, 4 threads",()->O.gemm!(C512,A512,B512)),
 ("gemm.jl:537","gemm! 192, algorithm=:blocked",()->O.gemm!(C192,A192,B192;algorithm=:blocked,threaded=false)),
 ("level3.jl:232","trsm! 512x512, 4 threads",()->O.trsm!(copy(B512),U512)),
 ("level3.jl:324","syrk! 512, 4 threads",()->O.syrk!(C512,A512)),
 ("level3.jl:350","syrk! 192, 1 thread",()->O.syrk!(C192,A192;threaded=false)),
 ("extended_blas.jl:114","syrk! Float16 64",()->O.syrk!(C16,A16)),
 ("gemm_variants.jl:24","gemm_cache_oblivious! 256",()->O.gemm_cache_oblivious!(C256,A256,B256)),
 ("gemm_variants.jl:114","gemm_strassen! 256",()->O.gemm_strassen!(C256,A256,B256)),
 ("batched.jl:4","gemm_batched! 16x32²",()->O.gemm_batched!(Fs,Ds,Es)),
 ("sparse_tensors.jl:197","findnz(SparseTensor 30³, 2000 nnz)",()->SparseArrays.findnz(X3)),
 ("sparse_tensors.jl:384","einsum sparse³ × dense",()->einsum("abc,cz->abz",X3,U3)),
 ("sparse_tensors.jl:629","einsum sparse³ × sparse³",()->einsum("abc,bcy->ay",X3,Y3)),
 ("tensors.jl:323","einsum dense, memory-bounded",()->einsum("abc,cd->abd",Dense3,Dm;memory_limit=10^6)),
 ("tensors.jl:373","plan_einsum dense",()->plan_einsum("abc,cd->abd",Dense3,Dm)),
 ("tensor_planning.jl:97","plan_einsum optimize=:beam, 4 operands",()->plan_einsum("ab,bc,cd,de->ae",A192,B192,A192,B192;optimize=:beam)),
 ("sparse_base_ops.jl:166","broadcast CSR .* 2 .+ CSR",()->Scsr .* 2.0 .+ Scsr),
 ("sparse_base_ops.jl:483","hcat(CSR, CSR)",()->hcat(Scsr,Scsr)),
 ("advanced_factorizations.jl:52","lu_factor pivot=:rook 192",()->O.lu_factor(A192;pivot=:rook)),
 ("supernodal_lu.jl:133","multifrontal_lu laplace 40²",()->O.multifrontal_lu(Scsr)),
 ("sparse_indefinite.jl:104","ldl_inertia",let F=O.sparse_ldlt(Scsr); ()->O.ldl_inertia(F) end),
 ("svd_advanced.jl:284","svd_randomized 300x120 rank 5",()->O.svd_randomized(Ar,5)),
 ("svd_selected.jl:211","svd_selected 300x120 top 3",()->O.svd_selected(Ar;indices=1:3)),
 ("mrrr.jl:197","eigen_tridiagonal_mrrr 300",()->O.eigen_tridiagonal_mrrr(d,e)),
 ("matrix_functions.jl:62","matrix_log SPD 40",()->O.matrix_log(Pspd)),
 ("qz_real.jl:7","qz_factor real 40",()->O.qz_factor(Aq,Bq)),
 ("schur_blocks.jl:50","eigen_pencil complex 30 (vectors)",()->O.eigen_pencil(Azq,Bzq)),
 ("sparse_spectral.jl:167","lobpcg laplace 40², nev=2",()->O.lobpcg(Scsr;nev=2)),
 ("sparse_preconditioners.jl:205","amg laplace 40²",()->O.amg(Scsr)),
 ("spectral_filter.jl:50","polynomial_filter SPD 100",()->O.polynomial_filter(Aspd,0.0,50.0)),
 ("xtrace_optimized.jl:5","trace_xtrace SPD 100",()->O.trace_xtrace(Aspd)),
 ("out_of_core.jl:80","out_of_core_cholesky laplace 30²",()->O.out_of_core_cholesky(CSRMatrix(Sp))),
 ("bidiagonal.jl:51","bidiagonal_factor(A).B",let F=O.bidiagonal_factor(A192); ()->F.B end),
 ("graph_structure.jl:19","nested_dissection laplace 40²",()->O.nested_dissection(Scsr)),
 ("multilevel_partition.jl:114/183","multilevel_partition laplace 40², k=4",()->O.multilevel_partition(Scsr,4)),
 ("multilevel_partition.jl:239/245","multilevel_nested_dissection laplace 40²",()->O.multilevel_nested_dissection(Scsr)),
]

open(OUT,"w") do io
    println(io,"site\tcase\tmin_us\tbytes")
    for (site,label,f) in cases
        line=try
            t,b=measure(f)
            "$site\t$label\t$(round(1e6t;sigdigits=4))\t$b"
        catch err
            "$site\t$label\tERROR\t$(first(sprint(showerror,err),120))"
        end
        println(io,line); flush(io); println(line); flush(stdout)
    end
end
