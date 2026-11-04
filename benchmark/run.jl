# High-level benchmark suite: the operations users call, against the matching
# reference (OpenBLAS or Accelerate through LinearAlgebra, SparseArrays'
# UMFPACK/CHOLMOD, TensorOperations).
#
#   julia --project=benchmark -e 'using Pkg; Pkg.instantiate()'
#   julia --project=benchmark --threads=1 benchmark/run.jl              # every case
#   julia --project=benchmark --threads=1 benchmark/run.jl gemm lu      # names starting with gemm or lu
#   OXIBLAS_REFERENCE=accelerate julia --project=benchmark benchmark/run.jl   # macOS
#
# Method:
#   - same problem, same data and same thread count for both sides (the
#     reference BLAS gets Threads.nthreads() threads);
#   - each result is checked against the reference before timing;
#   - allocating and in-place forms are separate cases (column `form`);
#   - one warm-up call compiles; times are medians of warm samples (Chairmarks);
#   - only the selected cases are compiled: each case is its own function,
#     called through invokelatest.
# Output: benchmark/results/<stamp>.csv and .md (time, bytes, ratio).
using OxiBLAS, Chairmarks, SparseArrays, Random, Printf, Dates
import LinearAlgebra as LA
const REFERENCE=get(ENV,"OXIBLAS_REFERENCE","openblas")
if REFERENCE=="accelerate"
    Sys.isapple() || error("Accelerate is only available on macOS")
    @eval using AppleAccelerate
end
using TensorOperations
LA.BLAS.set_num_threads(Threads.nthreads())
const SECONDS=parse(Float64,get(ENV,"OXIBLAS_BENCH_SECONDS","0.5"))
const O=OxiBLAS

struct Case
    name::String       # unique, used for selection
    family::String
    form::String       # "allocating" or "in-place"
    run::Function      # () -> (oxiblas closure, reference closure, check)
end
relerr(x,y)=LA.norm(x-y)/max(LA.norm(y),floatmin(real(eltype(y))))
tolfor(T)=real(T)===Float32 ? 1f-3 : 1e-8
rng()=Xoshiro(20261104)
lap2d(k;c=0.0)=(e=ones(k); T=spdiagm(-1=>-e[2:end],0=>2e,1=>-e[2:end]); C=spdiagm(-1=>-e[2:end],1=>e[2:end]);
                I=sparse(1.0LA.I,k,k); kron(I,T)+kron(T,I)+c*(kron(I,C)+kron(C,I)))
spd(r,T,n)=(X=randn(r,T,n,n); LA.Hermitian(X*X'+n*LA.I))

# Each builder returns (oxi, ref, check): oxi and ref are zero-argument
# closures over the same data; check(xo, xr) validates the two results.
approx(T)=(a,b)->relerr(a,b)<=tolfor(T)*100
gemm_case(T,n)=() -> begin
    r=rng(); A=randn(r,T,n,n); B=randn(r,T,n,n); C=zeros(T,n,n); D=zeros(T,n,n)
    w=GemmWorkspace(T,n,n,n)
    (()->gemm!(C,A,B;workspace=w), ()->LA.mul!(D,A,B), approx(T))
end
mul_sparse_dense() = () -> begin
    S=lap2d(100;c=0.3); A=CSRMatrix(S); X=randn(rng(),size(S,2),16); Y=zeros(size(S,1),16); Z=similar(Y)
    (()->O.mul!(Y,A,X), ()->LA.mul!(Z,S,X), approx(Float64))
end
mul_sparse_sparse() = () -> begin
    S=lap2d(100;c=0.3); A=CSRMatrix(S)
    (()->O.mul(A,A), ()->S*S, (a,b)->relerr(Matrix(SparseMatrixCSC(a)),Matrix(b))<=1e-12)
end
syrk_case(n) = () -> begin
    A=randn(rng(),n,n); C=zeros(n,n); D=zeros(n,n)
    (()->syrk!(C,A), ()->LA.BLAS.syrk!('U','N',1.0,A,0.0,D), (a,b)->relerr(LA.triu(a),LA.triu(b))<=1e-10)
end
trsm_case(n) = () -> begin
    r=rng(); U=LA.triu(randn(r,n,n))+n*LA.I; B=randn(r,n,n); X=copy(B); Y=copy(B)
    (()->(copyto!(X,B); trsm!(X,U)), ()->(copyto!(Y,B); LA.BLAS.trsm!('L','U','N','N',1.0,U,Y)), approx(Float64))
end
lu_case(T,n) = () -> begin
    A=randn(rng(),T,n,n)+n*LA.I; b=randn(rng(),T,n)
    (()->O.lu(A), ()->LA.lu(A), (Fo,Fr)->relerr(Fo\b,Fr\b)<=tolfor(T)*100)
end
cholesky_case(T,n) = () -> begin
    H=spd(rng(),T,n); b=randn(rng(),T,n)
    (()->O.cholesky(H), ()->LA.cholesky(H), (Fo,Fr)->relerr(Fo\b,Fr\b)<=tolfor(T)*100)
end
qr_case(T,m,n) = () -> begin
    A=randn(rng(),T,m,n); b=randn(rng(),T,m)
    (()->O.qr(A), ()->LA.qr(A), (Fo,Fr)->relerr(Fo\b,Fr\b)<=tolfor(T)*100)
end
svd_case(T,n;algorithm) = () -> begin
    A=randn(rng(),T,n,n)
    (()->svd_factor(A;algorithm,threaded=Threads.nthreads()>1), ()->LA.svd(A),
     (Fo,Fr)->relerr(Fo.S,Fr.S)<=tolfor(T)*100)
end
eigen_case(T,n) = () -> begin
    H=spd(rng(),T,n)
    (()->O.eigen(H), ()->LA.eigen(H), (Eo,Er)->relerr(Eo.values,Er.values)<=tolfor(T)*100)
end
dense_solve_alloc(n) = () -> begin
    A=randn(rng(),n,n)+n*LA.I; B=randn(rng(),n,8)
    (()->O.ldiv(A,B), ()->A\B, approx(Float64))
end
dense_solve_inplace(n) = () -> begin
    A=randn(rng(),n,n)+n*LA.I; B=randn(rng(),n,8); Fo=O.lu(A); Fr=LA.lu(A); Y=similar(B); Z=similar(B)
    (()->O.ldiv!(Y,Fo,B), ()->LA.ldiv!(Z,Fr,B), approx(Float64))
end
sparse_backslash(;c) = () -> begin
    S=lap2d(100;c); A=CSRMatrix(S); b=randn(rng(),size(S,1))
    (()->A\b, ()->S\b, approx(Float64))
end
sparse_lu() = () -> begin
    S=lap2d(100;c=0.3); A=CSRMatrix(S); b=randn(rng(),size(S,1))
    (()->O.lu(A), ()->LA.lu(S), (Fo,Fr)->relerr(Fo\b,Fr\b)<=1e-8)
end
sparse_cholesky() = () -> begin
    S=lap2d(100); A=CSRMatrix(S); b=randn(rng(),size(S,1))
    (()->O.cholesky(LA.Hermitian(A)), ()->LA.cholesky(LA.Hermitian(S)), (Fo,Fr)->relerr(Fo\b,Fr\b)<=1e-8)
end
sparse_ldiv_inplace() = () -> begin
    S=lap2d(100;c=0.3); A=CSRMatrix(S); b=randn(rng(),size(S,1)); Fo=O.lu(A); Fr=LA.lu(S); y=similar(b); z=similar(b)
    (()->O.ldiv!(y,Fo,b), ()->LA.ldiv!(z,Fr,b), approx(Float64))
end
einsum_alloc() = () -> begin
    r=rng(); X=randn(r,48,48,48); Y=randn(r,48,48)
    (()->einsum("abc,cd->abd",X,Y), ()->(@tensor Z[a,b,d]:=X[a,b,c]*Y[c,d]), approx(Float64))
end
einsum_inplace() = () -> begin
    r=rng(); X=randn(r,48,48,48); Y=randn(r,48,48); C=zeros(48,48,48); D=similar(C)
    P=plan_einsum("abc,cd->abd",X,Y); ws=EinsumWorkspace(Float64)
    (()->einsum!(C,P,X,Y;workspace=ws), ()->(@tensor D[a,b,d]=X[a,b,c]*Y[c,d]), approx(Float64))
end
einsum_contract4() = () -> begin
    r=rng(); X=randn(r,16,16,16,16); Y=randn(r,16,16,16,16)
    (()->einsum("abcd,ebfd->aecf",X,Y), ()->(@tensor Z[a,e,c,f]:=X[a,b,c,d]*Y[e,b,f,d]), approx(Float64))
end
tridiagonal_solve(n) = () -> begin
    r=rng(); Tm=LA.Tridiagonal(randn(r,n-1),randn(r,n).+4,randn(r,n-1)); b=randn(r,n)
    (()->O.lu(Tm)\b, ()->LA.lu(Tm)\b, approx(Float64))
end
packed_spmv(n) = () -> begin
    r=rng(); H=spd(r,Float64,n); P=pack_matrix(Matrix(H);uplo=:U,kind=:symmetric)
    AP=[H[i,j] for j in 1:n for i in 1:j]; x=randn(r,n); y=zeros(n); z=zeros(n)
    (()->spmv!(y,P,x), ()->LA.BLAS.spmv!('U',1.0,AP,x,0.0,z), approx(Float64))
end

const CASES=[
    Case("gemm_f64_512","GEMM","in-place",gemm_case(Float64,512)),
    Case("gemm_f32_512","GEMM","in-place",gemm_case(Float32,512)),
    Case("gemm_c64_256","GEMM","in-place",gemm_case(ComplexF64,256)),
    Case("mul_sparse_dense","multiplication creuse","in-place",mul_sparse_dense()),
    Case("mul_sparse_sparse","multiplication creuse","allocating",mul_sparse_sparse()),
    Case("syrk_f64_512","BLAS 3","in-place",syrk_case(512)),
    Case("trsm_f64_512","BLAS 3","in-place",trsm_case(512)),
    Case("lu_f64_512","factorisations denses","allocating",lu_case(Float64,512)),
    Case("lu_c64_256","factorisations denses","allocating",lu_case(ComplexF64,256)),
    Case("cholesky_f64_512","factorisations denses","allocating",cholesky_case(Float64,512)),
    Case("qr_f64_512x256","factorisations denses","allocating",qr_case(Float64,512,256)),
    Case("svd_jacobi_f64_192","SVD et spectres","allocating",svd_case(Float64,192;algorithm=:jacobi)),
    Case("svd_golubkahan_f64_192","SVD et spectres","allocating",svd_case(Float64,192;algorithm=:golub_kahan)),
    Case("eigen_hermitian_f64_256","SVD et spectres","allocating",eigen_case(Float64,256)),
    Case("solve_dense_ldiv","résolution","allocating",dense_solve_alloc(512)),
    Case("solve_dense_ldiv!","résolution","in-place",dense_solve_inplace(512)),
    Case("solve_sparse_backslash_spd","résolution creuse","allocating",sparse_backslash(;c=0.0)),
    Case("solve_sparse_backslash_lu","résolution creuse","allocating",sparse_backslash(;c=0.3)),
    Case("sparse_lu","factorisations creuses","allocating",sparse_lu()),
    Case("sparse_cholesky","factorisations creuses","allocating",sparse_cholesky()),
    Case("sparse_ldiv!","résolution creuse","in-place",sparse_ldiv_inplace()),
    Case("einsum_abc_cd","tenseurs","allocating",einsum_alloc()),
    Case("einsum!_abc_cd","tenseurs","in-place",einsum_inplace()),
    Case("einsum_abcd_ebfd","tenseurs","allocating",einsum_contract4()),
    Case("tridiagonal_solve","matrices structurées","allocating",tridiagonal_solve(10_000)),
    Case("packed_spmv","matrices structurées","in-place",packed_spmv(512)),
]

function median_sample(b)
    ts=sort([s.time for s in b.samples]); q=length(ts)
    (isodd(q) ? ts[(q+1)÷2] : (ts[q÷2]+ts[q÷2+1])/2), minimum(s.bytes for s in b.samples)
end
function run_case(c::Case)
    oxi,ref,check=c.run()
    xo=oxi(); xr=ref()            # warm-up (compilation) and results
    check(xo,xr) || error("$(c.name): OxiBLAS result differs from the reference")
    bo=@be oxi() seconds=SECONDS
    br=@be ref() seconds=SECONDS
    (to,ao),(tr,ar)=median_sample(bo),median_sample(br)
    (;name=c.name,family=c.family,form=c.form,oxiblas_s=to,reference_s=tr,ratio=to/tr,oxiblas_bytes=ao,reference_bytes=ar)
end

selected=isempty(ARGS) ? CASES : filter(c->any(a->startswith(c.name,a),ARGS),CASES)
isempty(selected) && error("no case matches $(ARGS); names: $(join((c.name for c in CASES),", "))")
outdir=joinpath(@__DIR__,"results"); mkpath(outdir)
stamp=Dates.format(now(),"yyyymmdd_HHMMSS")
header="Julia $(VERSION), $(Threads.nthreads()) thread(s), reference $(REFERENCE) ($(LA.BLAS.get_config().loaded_libs[1].libname |> basename)), $(Sys.CPU_NAME)"
println(header)
rows=[]
for c in selected
    row=Base.invokelatest(run_case,c); push!(rows,row)
    @printf("%-28s %-12s oxiblas %10.3f ms  ref %10.3f ms  ratio %6.2f  bytes %d / %d\n",
            row.name,row.form,1e3row.oxiblas_s,1e3row.reference_s,row.ratio,row.oxiblas_bytes,row.reference_bytes)
end
open(joinpath(outdir,"$stamp.csv"),"w") do io
    println(io,"name,family,form,oxiblas_ms,reference_ms,ratio,oxiblas_bytes,reference_bytes")
    for r in rows
        println(io,join((r.name,r.family,r.form,round(1e3r.oxiblas_s;sigdigits=5),round(1e3r.reference_s;sigdigits=5),
                         round(r.ratio;digits=3),r.oxiblas_bytes,r.reference_bytes),','))
    end
end
open(joinpath(outdir,"$stamp.md"),"w") do io
    println(io,"# Banc haut niveau, $stamp\n\n$header\n")
    println(io,"| Cas | Famille | Forme | OxiBLAS (ms) | Référence (ms) | OxiBLAS / référence | Octets OxiBLAS | Octets référence |")
    println(io,"|---|---|---|---|---|---|---|---|")
    for r in rows
        @printf(io,"| `%s` | %s | %s | %.3f | %.3f | %.2f | %d | %d |\n",r.name,r.family,r.form,1e3r.oxiblas_s,1e3r.reference_s,r.ratio,r.oxiblas_bytes,r.reference_bytes)
    end
end
println("Results: ",joinpath(outdir,"$stamp.csv"))
