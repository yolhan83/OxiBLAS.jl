# Sparse and sparse-tensor benchmark worker, one process per thread count
# (driven by `OXIBLAS_TARGET=sparse python3 benchmark/archive/level1_targeted.py`).
# OxiBLAS and its reference (SparseArrays; TensorOperations for dense tensor
# references when installed) are measured in the same process on the same
# data. Inputs are built and converted outside the timed region; the guarded
# first call excludes compilation; samples adapt to the call duration
# (same contract as full_worker.jl). Every recorded measurement is checked
# against its reference (or, without one, against an independent probe).
# No dense equivalent of a large sparse operand is ever formed.
using OxiBLAS, SparseArrays, LinearAlgebra, Random
const O=OxiBLAS
const SP_THREADS=Threads.nthreads()
const SP_SECONDS=parse(Float64,get(ENV,"OXIBLAS_SP_SECONDS","0.5"))
const SP_SAMPLES=parse(Int,get(ENV,"OXIBLAS_SP_SAMPLES","7"))
_sp_list(name)=filter(!isempty,strip.(split(get(ENV,name,""),',')))
const SP_SELECT=Symbol.(_sp_list("OXIBLAS_SP_SELECT"))       # operations
const SP_CASES=String.(_sp_list("OXIBLAS_SP_CASES"))          # data sets
const SP_TYPES=String.(_sp_list("OXIBLAS_SP_TYPES"))
const SP_DENSE_LIMIT=parse(Int,get(ENV,"OXIBLAS_SP_DENSE_LIMIT_MIB","64"))*2^20  # largest dense reference/operand
const HAS_TO=try
    @eval import TensorOperations
    true
catch
    false
end
# BLAS behind the SuiteSparse references: OpenBLAS (default) or Accelerate.
# Loading AppleAccelerate forwards LinearAlgebra's BLAS/LAPACK to Accelerate
# and also provides Accelerate's native sparse solvers (libSparse). OxiBLAS
# never calls an external BLAS, so its timings do not depend on this choice.
const SP_BLAS=get(ENV,"OXIBLAS_SP_BLAS","openblas")
const HAS_AA=SP_BLAS=="accelerate" && Sys.isapple() && try
    @eval import AppleAccelerate
    true
catch
    false
end
SP_BLAS=="accelerate" && !HAS_AA && error("OXIBLAS_SP_BLAS=accelerate but AppleAccelerate could not be loaded")
# Forward every BLAS/LAPACK symbol to Accelerate only (no OpenBLAS fallback).
HAS_AA && AppleAccelerate.load_accelerate(;clear=true,load_ilp64=true)
const REF_LABEL=SP_BLAS=="accelerate" ? "suitesparse+accelerate" : "suitesparse+openblas"
const SP_REF_ONLY=get(ENV,"OXIBLAS_SP_REF_ONLY","0")=="1"        # skip OxiBLAS rows (measured in the other run)
const SP_DIRECT_ONLY=get(ENV,"OXIBLAS_SP_DIRECT_ONLY","0")=="1"  # only the direct-solver families
LinearAlgebra.BLAS.set_num_threads(SP_THREADS)
Random.seed!(0x5ba75e)

# ---- measurement ------------------------------------------------------------

function sp_sample(f::F,setup::S) where {F,S}
    setup(); first=@timed f()
    compile=hasproperty(first,:compile_time) ? first.compile_time : 0.0
    t1=first.time-compile
    t1>=SP_SECONDS && return (;median=t1,minimum=t1,bytes=first.bytes,allocs=Base.gc_alloc_count(first.gcstats),samples=1,value=first.value)
    t1<SP_SECONDS/SP_SAMPLES && (setup();f())
    times=Float64[];bytes=0;allocs=0;start=time();value=first.value
    while length(times)<SP_SAMPLES && (isempty(times) || time()-start<SP_SECONDS)
        setup(); s=@timed f()
        push!(times,s.time);bytes=s.bytes;allocs=Base.gc_alloc_count(s.gcstats);value=s.value
    end
    sort!(times); m=length(times)
    (;median=isodd(m) ? times[(m+1)÷2] : (times[m÷2]+times[m÷2+1])/2,minimum=times[1],bytes,allocs,samples=m,value)
end
_json(x::AbstractString)="\""*replace(x,"\\"=>"\\\\","\""=>"\\\"")*"\""
_json(x::Symbol)=_json(String(x))
_json(x::Bool)=x ? "true" : "false"
_json(x::Integer)=string(x)
_json(x::AbstractFloat)=isfinite(x) ? string(Float64(x)) : "null"
_json(x::Nothing)="null"
_json(x::Union{AbstractVector,Tuple})="["*join(_json.(x),",")*"]"
_json(x::AbstractDict)="{"*join((_json(string(k))*":"*_json(v) for (k,v) in x),",")*"}"
_json(x::NamedTuple)=_json(Dict(string(k)=>v for (k,v) in pairs(x)))
_json(x)=_json(string(x))
function emit(row)
    println("@@SPARSE_ROW\t",_json(row)); flush(stdout)
end
_nnz(A::SparseMatrixCSC)=nnz(A)
_nnz(A::SparseVector)=nnz(A)
_nnz(A::SparseTensor)=nnz(A)
_nnz(A::Union{O.CSRMatrix,O.CSCMatrix,O.COOMatrix})=length(A.values)
_nnz(A::AbstractArray)=count(!iszero,A)
_nnz(A)=0
_elements(A)=A isa AbstractArray ? prod(BigInt.(size(A));init=BigInt(1)) : BigInt(0)
_density(A)=_elements(A)==0 ? 0.0 : Float64(_nnz(A)/_elements(A))
_isdense(A)=A isa Array || A isa SubArray
const FAILURES=Ref(0)

# One measured implementation of one operation. `check(value)` validates the
# result (relative error against the reference); a failed check is recorded
# as ERROR and the timing is discarded.
function measure(meta,backend,phase,f,setup=()->nothing;check=nothing,output_of=identity,extra=Dict{String,Any}())
    SP_REF_ONLY && backend=="oxiblas" && return nothing
    row=Dict{String,Any}(string(k)=>v for (k,v) in pairs(meta))
    row["backend"]=backend; row["phase"]=phase; row["threads"]=SP_THREADS
    try
        r=sp_sample(f,setup)
        out=output_of(r.value)
        if check!==nothing && !check(r.value)
            FAILURES[]+=1; row["status"]="ERROR"; row["note"]="correctness check failed"; emit(row); return nothing
        end
        merge!(row,Dict("status"=>"OK","median"=>r.median,"minimum"=>r.minimum,"bytes"=>r.bytes,"allocs"=>r.allocs,"samples"=>r.samples))
        out===nothing || (row["nnz_out"]=_nnz(out); row["density_out"]=_density(out); row["output_kind"]=_isdense(out) ? "dense" : string(nameof(typeof(out))))
        merge!(row,extra); emit(row); return r
    catch err
        FAILURES[]+=1
        row["status"]=err isa OutOfMemoryError ? "OOM" : "ERROR"; row["note"]=first(sprint(showerror,err),300)
        emit(row); return nothing
    end
end
_unavailable(meta,backend,reason)=emit(merge(Dict{String,Any}(string(k)=>v for (k,v) in pairs(meta)),Dict("backend"=>backend,"phase"=>"kernel","threads"=>SP_THREADS,"status"=>"N/A","note"=>reason)))

# Relative errors without densifying sparse results.
_tosparse(A::SparseMatrixCSC)=A
_tosparse(A::SparseTensor{T,2}) where T=SparseMatrixCSC(A)
_tosparse(A::Union{O.CSRMatrix,O.CSCMatrix,O.COOMatrix})=SparseMatrixCSC(A)
_tosparse(A)=A
function _relerr(a,b)
    a=_tosparse(a); b=_tosparse(b)
    if a isa SparseTensor || b isa SparseTensor
        x=a isa SparseTensor ? a : SparseTensor(a); y=b isa SparseTensor ? b : SparseTensor(b)
        d=Dict{Int,Any}(); for (i,v) in zip(x.indices,x.values); d[i]=v; end
        for (i,v) in zip(y.indices,y.values); d[i]=get(d,i,zero(v))-v; end
        num=sqrt(sum(abs2,values(d);init=0.0)); den=sqrt(sum(abs2,y.values;init=0.0))
        return num/max(den,floatmin(Float64))
    end
    size(a)==size(b) || return Inf
    Float64(norm(a-b)/max(norm(b),floatmin(real(eltype(b)))))
end
_tol(T)=1000*eps(real(T))
_ok(T,ref)=v->_relerr(v,ref)<=_tol(T)

# ---- data sets (never dense) -------------------------------------------------

_val(::Type{T},rng) where T=T<:Complex ? T(randn(rng)+im*randn(rng)) : T(randn(rng))
function _from_triplets(::Type{T},I,J,m,n,rng;values=nothing) where T
    V=values===nothing ? T[_val(T,rng) for _ in I] : T.(values)
    sparse(I,J,V,m,n)
end
function ds_diagonal(T,n,rng); _from_triplets(T,1:n,1:n,n,n,rng); end
function ds_laplace1d(T,n,rng)
    I=vcat(1:n,1:n-1,2:n); J=vcat(1:n,2:n,1:n-1); V=vcat(fill(2.0,n),fill(-1.0,n-1),fill(-1.0,n-1))
    _from_triplets(T,I,J,n,n,rng;values=V)
end
function ds_banded(T,n,b,rng)
    I=Int[];J=Int[]
    for j in 1:n, i in max(1,j-b):min(n,j+b); push!(I,i);push!(J,j); end
    S=_from_triplets(T,I,J,n,n,rng); S+spdiagm(0=>fill(T(4b+2),n))
end
function ds_lower(T,n,density,rng)
    S=_random_pattern(T,n,n,density,rng); L=tril(S,-1)+spdiagm(0=>fill(T(1+n*density),n)); L
end
function ds_laplace2d(T,k,rng)
    L=ds_laplace1d(Float64,k,rng); Id=sparse(1.0I,k,k); T.(kron(Id,L)+kron(L,Id))
end
function ds_laplace3d(T,k,rng)
    L=ds_laplace1d(Float64,k,rng); Id=sparse(1.0I,k,k)
    T.(kron(Id,kron(Id,L))+kron(Id,kron(L,Id))+kron(L,kron(Id,Id)))
end
function ds_blockdiag(T,n,bs,rng)
    I=Int[];J=Int[]
    for b in 0:bs:n-1, j in b+1:min(n,b+bs), i in b+1:min(n,b+bs); push!(I,i);push!(J,j); end
    _from_triplets(T,I,J,n,n,rng)
end
function ds_blocksparse(T,n,bs,blockdensity,rng)
    nb=cld(n,bs); I=Int[];J=Int[]
    for bj in 1:nb, bi in 1:nb
        (bi==bj || rand(rng)<blockdensity) || continue
        for j in (bj-1)*bs+1:min(n,bj*bs), i in (bi-1)*bs+1:min(n,bi*bs); push!(I,i);push!(J,j); end
    end
    _from_triplets(T,I,J,n,n,rng)
end
# Irregular graph: Zipf-like degrees (heavy hubs) on a symmetric pattern.
function ds_powerlaw(T,n,avgdeg,rng)
    I=Int[];J=Int[]; w=[1/(i^0.8) for i in 1:n]; c=cumsum(w); total=c[end]
    for _ in 1:n*avgdeg÷2
        a=rand(rng,1:n); b=min(n,searchsortedfirst(c,rand(rng)*total))
        a==b && continue; push!(I,a,b); push!(J,b,a)
    end
    S=_from_triplets(T,I,J,n,n,rng); S+spdiagm(0=>fill(T(avgdeg),n))
end
function _random_pattern(T,m,n,density,rng)
    target=round(Int,density*m*n); I=rand(rng,1:m,target); J=rand(rng,1:n,target)
    _from_triplets(T,I,J,m,n,rng)
end
ds_random(T,m,n,density,rng)=_random_pattern(T,m,n,density,rng)
function ds_hypersparse(T,n,k,rng)
    I=rand(rng,1:n,k); J=rand(rng,1:n,k); _from_triplets(T,I,J,n,n,rng)
end
# Unsymmetric: 2-D convection-diffusion (upwind convection on a Laplacian).
function ds_convdiff2d(T,k,rng)
    L=ds_laplace1d(Float64,k,rng); Id=sparse(1.0I,k,k); D=spdiagm(0=>ones(k),-1=>-ones(k-1))
    T.(kron(Id,L)+kron(L,Id)+0.7*(kron(Id,D)+kron(D,Id)))
end
# Symmetric indefinite: shifted 2-D Laplacian (Helmholtz), shift inside the spectrum.
ds_helmholtz2d(T,k,rng)=(A=ds_laplace2d(Float64,k,rng); T.(A-2.0*sparse(1.0I,size(A)...)))
# Tall least-squares matrix: random sparse rows over an identity block.
function ds_tall(T,m,n,density,rng)
    R=_random_pattern(T,m,n,density,rng); R+sparse(1:n,1:n,fill(T(4),n),m,n)
end
function ds_tensor(T,dims,k,rng)
    coords=[ntuple(d->rand(rng,1:dims[d]),length(dims)) for _ in 1:k]
    SparseTensor(dims,coords,T[_val(T,rng) for _ in 1:k])
end

const ALL4=("Float32","Float64","ComplexF32","ComplexF64")
_type(s)=s=="Float32" ? Float32 : s=="Float64" ? Float64 : s=="ComplexF32" ? ComplexF32 : s=="ComplexF64" ? ComplexF64 : BigFloat
# Representative cases (no Cartesian product): every structure in Float64,
# a subset in the other native types, small cases in BigFloat. `ops` selects
# the families measured on each.
const MATRIX_CASES=[
    (name="diagonal_1e4",types=("Float64",),build=(T,r)->ds_diagonal(T,10^4,r),ops=(:core,)),
    (name="diagonal_1e6",types=("Float64",),build=(T,r)->ds_diagonal(T,10^6,r),ops=(:core,)),
    (name="laplace1d_1e4",types=("Float64",),build=(T,r)->ds_laplace1d(T,10^4,r),ops=(:core,:spd)),
    (name="laplace1d_1e6",types=("Float64",),build=(T,r)->ds_laplace1d(T,10^6,r),ops=(:core,)),
    (name="banded16_1e4",types=("Float64","ComplexF64"),build=(T,r)->ds_banded(T,10^4,16,r),ops=(:core,:unsym)),
    (name="banded16_1e5",types=("Float64",),build=(T,r)->ds_banded(T,10^5,16,r),ops=(:core,)),
    (name="lower_1e4_d1e-3",types=("Float64",),build=(T,r)->ds_lower(T,10^4,1e-3,r),ops=(:core,:lower)),
    (name="laplace2d_100",types=ALL4,build=(T,r)->ds_laplace2d(T,100,r),ops=(:core,:spd,:lower,:einsum)),
    (name="laplace2d_1000",types=("Float64",),build=(T,r)->ds_laplace2d(T,1000,r),ops=(:core,)),
    (name="laplace3d_22",types=("Float64","ComplexF32"),build=(T,r)->ds_laplace3d(T,22,r),ops=(:core,:spd)),
    (name="laplace2d_30",types=ALL4,build=(T,r)->ds_laplace2d(T,30,r),ops=(:spd,)),
    (name="convdiff2d_30",types=ALL4,build=(T,r)->ds_convdiff2d(T,30,r),ops=(:unsym,)),
    (name="convdiff2d_100",types=("Float64","ComplexF64"),build=(T,r)->ds_convdiff2d(T,100,r),ops=(:unsym,)),
    (name="helmholtz2d_30",types=("Float64","ComplexF64"),build=(T,r)->ds_helmholtz2d(T,30,r),ops=(:indef,)),
    (name="helmholtz2d_70",types=("Float64",),build=(T,r)->ds_helmholtz2d(T,70,r),ops=(:indef,)),
    (name="tall_4000x1000",types=("Float64","ComplexF64","Float32"),build=(T,r)->ds_tall(T,4000,1000,2e-3,r),ops=(:tall,)),
    (name="laplace3d_100",types=("Float64",),build=(T,r)->ds_laplace3d(T,100,r),ops=(:core,)),
    (name="blockdiag32_8192",types=("Float64",),build=(T,r)->ds_blockdiag(T,8192,32,r),ops=(:core,)),
    (name="blocksparse8_16384",types=("Float64","Float32"),build=(T,r)->ds_blocksparse(T,16384,8,1e-3,r),ops=(:core,)),
    (name="powerlaw_1e5",types=ALL4,build=(T,r)->ds_powerlaw(T,10^5,8,r),ops=(:core,:einsum)),
    (name="random_4096_d1e-3",types=("Float64",),build=(T,r)->ds_random(T,4096,4096,1e-3,r),ops=(:core,)),
    (name="random_4096_d1e-2",types=ALL4,build=(T,r)->ds_random(T,4096,4096,1e-2,r),ops=(:core,:einsum)),
    (name="random_4096_d1e-1",types=("Float64",),build=(T,r)->ds_random(T,4096,4096,1e-1,r),ops=(:core,)),
    (name="random_1e5_d1e-4",types=("Float64",),build=(T,r)->ds_random(T,10^5,10^5,1e-4,r),ops=(:core,)),
    (name="rect_1e5x1e3_d1e-3",types=("Float64","ComplexF64"),build=(T,r)->ds_random(T,10^5,10^3,1e-3,r),ops=(:core,)),
    (name="rect_1e3x1e5_d1e-3",types=("Float64",),build=(T,r)->ds_random(T,10^3,10^5,1e-3,r),ops=(:core,)),
    (name="hypersparse_1e6_k1e3",types=("Float64",),build=(T,r)->ds_hypersparse(T,10^6,10^3,r),ops=(:core,:einsum)),
    (name="hypersparse_1e6_k1e5",types=("Float64",),build=(T,r)->ds_hypersparse(T,10^6,10^5,r),ops=(:core,)),
    (name="laplace2d_20_big",types=("BigFloat",),build=(T,r)->ds_laplace2d(T,20,r),ops=(:core,:einsum)),
    (name="random_400_d1e-2_big",types=("BigFloat",),build=(T,r)->ds_random(T,400,400,1e-2,r),ops=(:core,)),
]
# Scaling series: fixed nnz with growing dimension (cost must follow nnz, not
# m*n), and fixed dimension with growing nnz.
const SCALING_CASES=vcat(
    [(name="scale_n$(n)_k1e4",n=n,k=10^4) for n in (10^3,10^4,10^5,10^6)],
    [(name="scale_n1e5_k$(k)",n=10^5,k=k) for k in (10^3,10^4,10^5,10^6)])

# ---- operation families -------------------------------------------------------

# Operations with a parallel OxiBLAS path (measured at every thread count).
const PARALLEL_OPS=(:spmv,:spmv_t,:spmm,:spmm_ext,:einsum_sd,:einsum_ds,:mf_cholesky,:mf_lu,:lu_blocked)
_selected(op)=(isempty(SP_SELECT) || op in SP_SELECT) && (SP_THREADS==1 || op in PARALLEL_OPS)
# Families entered only when one of their operations is selected: compiling
# bench_direct alone costs about 170 s per (case, type).
const DIRECT_OPS=(:cholesky,:incomplete,:ldlt,:lu,:lu_blocked,:lu_static,:mf_cholesky,:mf_lu,:ordering,:qr,:trisolve)
const CORE_OPS=(:add,:convert,:norm,:permute,:spgemm,:spmm,:spmm_ext,:spmv,:spmv_t,:transpose)
_any_selected(ops)=any(_selected,ops)

function bench_core(case,T,S,rng)
    m,n=size(S); base=(case=case,T=string(T),m=m,n=n,nnz_in=nnz(S),density_in=_density(S))
    x=T[_val(T,rng) for _ in 1:n]; xt=T[_val(T,rng) for _ in 1:m]; y=zeros(T,m); yt=zeros(T,n)
    C=O.CSRMatrix(S); W=O.CSCMatrix(S)
    if _selected(:spmv)
        ref=S*x; meta=merge(base,(op="spmv",))
        measure(merge(meta,(format="SparseMatrixCSC",)),"sparsearrays","kernel",()->mul!(y,S,x))
        measure(merge(meta,(format="CSR",)),"oxiblas","kernel",()->O.sparse_mul!(y,C,x);check=_ok(T,ref))
        measure(merge(meta,(format="CSC",)),"oxiblas","kernel",()->O.sparse_mul!(y,W,x);check=_ok(T,ref))
        measure(merge(meta,(format="SparseMatrixCSC (wrapped)",)),"oxiblas","total",()->O.sparse_mul!(y,S,x);check=_ok(T,ref))
    end
    if _selected(:spmv_t)
        ref=transpose(S)*xt; meta=merge(base,(op="spmv_t",))
        measure(merge(meta,(format="SparseMatrixCSC",)),"sparsearrays","kernel",()->mul!(yt,transpose(S),xt))
        measure(merge(meta,(format="CSR",)),"oxiblas","kernel",()->O.sparse_mul!(yt,C,xt;trans=:T);check=_ok(T,ref))
        measure(merge(meta,(format="CSC",)),"oxiblas","kernel",()->O.sparse_mul!(yt,W,xt;trans=:T);check=_ok(T,ref))
    end
    k=16
    if _selected(:spmm) && Float64(n)*k*sizeof(T)<=SP_DENSE_LIMIT && Float64(m)*k*sizeof(T)<=SP_DENSE_LIMIT
        X=randn(rng,real(T),n,k).+zero(T); Y=zeros(T,m,k); ref=S*X; meta=merge(base,(op="spmm",k=k))
        measure(merge(meta,(format="SparseMatrixCSC",)),"sparsearrays","kernel",()->mul!(Y,S,X))
        measure(merge(meta,(format="CSR",)),"oxiblas","kernel",()->O.sparse_mul!(Y,C,X);check=_ok(T,ref))
        measure(merge(meta,(format="SparseMatrixCSC (wrapped)",)),"oxiblas","total",()->O.sparse_mul!(Y,S,X);check=_ok(T,ref))
    end
    if _selected(:spmm_ext) && T<:O.NativeFloat && m==n && n<=10^5
        # Extended storage formats: SpMV against the CSR kernel (same product).
        ref=S*x
        I,J,_=findnz(S); diagonals=length(unique(J .- I))
        for (name,F) in (("ELL",O.ELLMatrix),("SELL",O.SELLMatrix),("HYB",O.HYBMatrix),("BSR",O.BSRMatrix),("DIA",O.DIAMatrix))
            name=="DIA" && diagonals>64 && continue
            A=try F(C) catch; nothing end
            A===nothing && continue
            measure(merge(base,(op="spmv_ext",format=name)),"oxiblas","kernel",()->O.sparse_mul!(y,A,x);check=_ok(T,ref))
        end
    end
    if _selected(:spgemm) && nnz(S)<=2*10^6
        B=m==n ? S : sparse(transpose(S)); ref=S*B; meta=merge(base,(op="spgemm",nnz_rhs=nnz(B)))
        measure(merge(meta,(format="SparseMatrixCSC",)),"sparsearrays","kernel",()->S*B;output_of=identity)
        CB=O.CSRMatrix(B)
        measure(merge(meta,(format="CSR",)),"oxiblas","kernel",()->O.sparse_matmul(C,CB);check=_ok(T,ref),output_of=identity)
        measure(merge(meta,(format="SparseMatrixCSC",)),"oxiblas","total",()->O.sparse_matmul(S,B);check=_ok(T,ref),output_of=identity)
    end
    # Conversions: equivalent storage reorganizations on both sides.
    if _selected(:convert)
        meta=merge(base,(op="convert",))
        measure(merge(meta,(format="SparseMatrixCSC->CSC (shared)",)),"oxiblas","convert",()->O.CSCMatrix(S);check=v->v.colptr===SparseArrays.getcolptr(S))
        rT=sparse(transpose(S))
        measure(merge(meta,(format="CSC->CSR (transpose storage)",)),"sparsearrays","convert",()->sparse(transpose(S)))
        measure(merge(meta,(format="CSC->CSR (transpose storage)",)),"oxiblas","convert",()->O.CSRMatrix(S);check=v->v.rowptr==SparseArrays.getcolptr(rT))
        measure(merge(meta,(format="CSR->SparseMatrixCSC",)),"sparsearrays","convert",()->sparse(transpose(rT)))
        measure(merge(meta,(format="CSR->SparseMatrixCSC",)),"oxiblas","convert",()->SparseMatrixCSC(C);check=_ok(T,S))
        I,J,V=findnz(S)
        measure(merge(meta,(format="CSC->COO",)),"sparsearrays","convert",()->findnz(S);output_of=v->nothing)
        measure(merge(meta,(format="CSC->COO",)),"oxiblas","convert",()->O.COOMatrix(S);check=v->length(v.values)==count(!iszero,V))
        measure(merge(meta,(format="COO->CSC",)),"sparsearrays","convert",()->sparse(I,J,V,m,n))
        Q=O.COOMatrix(S)
        measure(merge(meta,(format="COO->CSC",)),"oxiblas","convert",()->SparseMatrixCSC(Q);check=_ok(T,dropzeros(S)))
        measure(merge(meta,(format="COO triplets->COOMatrix (sort+sum)",)),"oxiblas","convert",()->O.COOMatrix(m,n,I,J,V);check=v->length(v.values)==count(!iszero,V))
        measure(merge(meta,(format="SparseMatrixCSC->SparseTensor",)),"oxiblas","convert",()->SparseTensor(S);check=v->nnz(v)==nnz(S))
        X=SparseTensor(S)
        measure(merge(meta,(format="SparseTensor->SparseMatrixCSC",)),"oxiblas","convert",()->SparseMatrixCSC(X);check=_ok(T,S))
        if T<:O.NativeFloat && m==n && n<=10^5
            for (name,F) in (("ELL",O.ELLMatrix),("SELL",O.SELLMatrix),("HYB",O.HYBMatrix),("BSR",O.BSRMatrix))
                measure(merge(meta,(format="CSR->$name",)),"oxiblas","convert",()->F(C);check=v->size(v)==size(S))
            end
        end
    end
    if _selected(:transpose)
        meta=merge(base,(op="transpose",format="materialized"))
        ref=copy(transpose(S))
        measure(meta,"sparsearrays","kernel",()->copy(transpose(S)))
        measure(meta,"oxiblas","kernel",()->O.sparse_transpose(S);check=_ok(T,ref))
    end
    if _selected(:add)
        meta=merge(base,(op="add",format="S+2S"))
        ref=S+2S
        measure(meta,"sparsearrays","kernel",()->S+2S)
        measure(meta,"oxiblas","kernel",()->O.sparse_add(S,S,one(T),T(2));check=_ok(T,ref))
    end
    if _selected(:permute) && m==n
        p=randperm(rng,n); q=randperm(rng,n); meta=merge(base,(op="permute",format="S[p,q]"))
        ref=S[p,q]
        measure(meta,"sparsearrays","kernel",()->S[p,q])
        measure(meta,"oxiblas","kernel",()->O.sparse_permute(S,p,q);check=_ok(T,ref))
    end
    if _selected(:norm)
        meta=merge(base,(op="norm",format="frobenius")); ref=norm(S)
        measure(meta,"sparsearrays","kernel",()->norm(S))
        measure(meta,"oxiblas","kernel",()->O.sparse_norm(S);check=v->abs(v-ref)<=_tol(T)*ref)
    end
    nothing
end

# ---- sparse direct solvers (low level) ------------------------------------------
# OxiBLAS kernels against their SparseArrays equivalents, phase by phase:
#   triangular solve      ldiv!/\ on (Unit)Lower/UpperTriangular (Julia, SparseArrays)
#   LU                    UMFPACK lu / lu! (refactor, symbolic reused) / \
#   Cholesky              CHOLMOD cholesky (natural order perm=1:n, and AMD) /
#                         CHOLMOD.symbolic + cholesky! / \
#   LDLᵀ                  CHOLMOD ldlt (no pivoting: comparable only where it
#                         succeeds on the indefinite case)
#   QR, Q, least squares  SPQR qr / F.Q' * B / \
#   orderings             fill nnz(L) of the symbolic factorization (CHOLMOD AMD)
#   incomplete factors    no SparseArrays equivalent: OxiBLAS only, quality as
#                         the relative residual after one preconditioned solve
# UMFPACK, CHOLMOD and SPQR support Float64/ComplexF64 only (the others are
# recorded N/A). Each solver has a size cap so that one slow call cannot hold
# the campaign; above it the row is recorded as skipped.
const DIRECT_CAPS=Dict(:lu=>10^4,:lu_blocked=>1000,:lu_static=>10^4,:mf_lu=>2*10^4,:cholesky=>2*10^4,:mf_cholesky=>2*10^4,
                       :ldlt=>5000,:qr=>5000,:incomplete=>10^4,:ordering=>2*10^4,:trisolve=>10^6)
_suitesparse(T)=T<:Union{Float64,ComplexF64}
_solve_tol(T)=real(T)===Float32 ? 1e-3 : 1e-8
_resid(S,x,b)=norm(S*x-b)/max(norm(b),floatmin(Float64))
_rescheck(T,S,b)=x->_resid(S,x,b)<=_solve_tol(T)
_fill(F::Union{O.SparseLUFactor,O.MultifrontalLUFactor,O.SparseStaticLUFactor})=_nnz(F.L)+_nnz(F.U)
_fill(F::O.SparseBlockedLUFactor)=_fill(F.factor)
_fill(F::Union{O.SparseCholeskyFactor,O.MultifrontalCholeskyFactor,O.SparseLDLFactor})=_nnz(F.L)
_fill(F::O.SparseQRFactor)=_nnz(F.R)
_fill(F)=try; F isa SparseArrays.UMFPACK.UmfpackLU ? nnz(F.L)+nnz(F.U) : F isa SparseArrays.CHOLMOD.Factor ? nnz(sparse(F.L)) : nnz(F.R); catch; missing; end
function _capped(meta,op,n)
    n<=DIRECT_CAPS[op] && return false
    emit(merge(Dict{String,Any}(string(k)=>v for (k,v) in pairs(meta)),Dict("backend"=>"oxiblas","phase"=>"kernel","threads"=>SP_THREADS,"status"=>"SKIPPED","note"=>"size above the $(op) cap ($(DIRECT_CAPS[op]))")))
    true
end
# Factor, then solve with the factor built once outside the timed region.
function _factor_solve(meta,backend,factor,solve,S,b,T;extra=Dict{String,Any}())
    r=measure(merge(meta,(phase_detail="factor",)),backend,"factor",factor)
    r===nothing && return nothing
    F=r.value; e=merge(extra,Dict{String,Any}("fill"=>_fill(F)))
    measure(merge(meta,(phase_detail="solve",)),backend,"solve",()->solve(F,b);check=_rescheck(T,S,b),extra=e)
    F
end
function bench_direct(case,kind,T,S,rng)
    m,n=size(S); base=(case=case,kind=string(kind),T=string(T),m=m,n=n,nnz_in=nnz(S),density_in=_density(S))
    b=T[_val(T,rng) for _ in 1:m]; ss=_suitesparse(T); na(meta,backend)=_unavailable(meta,backend,"UMFPACK/CHOLMOD/SPQR support Float64/ComplexF64 only")
    if kind===:lower && _selected(:trisolve)
        B=T[_val(T,rng) for _ in 1:n, _ in 1:8]; L=tril(S); U=sparse(transpose(L))
        for (name,A,uplo,W) in (("lower",L,:L,LowerTriangular),("upper",U,:U,UpperTriangular)), (tr,op) in ((:N,identity),(:T,transpose)), rhs in (b,B)
            meta=merge(base,(op="trisolve",format=name,trans=string(tr),rhs=size(rhs,2)))
            ref=op(W(A))\rhs
            measure(meta,"sparsearrays","kernel",()->op(W(A))\rhs)
            measure(meta,"oxiblas","kernel",()->O.sparse_triangular_solve(A,rhs;uplo,trans=tr);check=_ok(T,ref))
        end
    end
    if kind===:unsym
        if _selected(:lu) && !_capped(merge(base,(op="lu",)),:lu,n)
            meta=merge(base,(op="lu",format="SparseMatrixCSC"))
            _factor_solve(meta,"oxiblas",()->O.sparse_lu(S),O.solve,S,b,T)
            if ss
                F=_factor_solve(meta,REF_LABEL,()->lu(S),(F,b)->F\b,S,b,T)
                F===nothing || measure(merge(meta,(phase_detail="refactor (symbolic reused)",)),REF_LABEL,"refactor",()->lu!(F,S))
            else na(meta,REF_LABEL) end
            _accelerate_direct(meta,:LU,S,b,T)
        end
        if _selected(:mf_lu) && !_capped(merge(base,(op="multifrontal_lu",)),:mf_lu,n)
            meta=merge(base,(op="multifrontal_lu",format="SparseMatrixCSC",ordering="auto"))
            _factor_solve(meta,"oxiblas",()->O.multifrontal_lu(S),O.solve,S,b,T)
            ss ? _factor_solve(meta,REF_LABEL,()->lu(S),(F,b)->F\b,S,b,T) : na(meta,REF_LABEL)
        end
        if _selected(:lu_blocked) && !_capped(merge(base,(op="lu_blocked",)),:lu_blocked,n)
            meta=merge(base,(op="lu_blocked",format="SparseMatrixCSC"))
            _factor_solve(meta,"oxiblas",()->O.sparse_lu_blocked(S),O.solve,S,b,T)
        end
        if _selected(:lu_static) && !_capped(merge(base,(op="lu_static",)),:lu_static,n)
            meta=merge(base,(op="lu_static",format="SparseMatrixCSC",note="static pivoting (no UMFPACK equivalent: compared with lu)"))
            _factor_solve(meta,"oxiblas",()->O.sparse_lu_static(S),O.solve,S,b,T)
        end
        if _selected(:incomplete) && !_capped(merge(base,(op="incomplete_lu",)),:incomplete,n)
            for (name,f) in (("ilu0",()->O.ilu0(O.CSRMatrix(S))),("ilut",()->O.ilut(S)))
                meta=merge(base,(op=name,format="SparseMatrixCSC",note="no SparseArrays equivalent"))
                r=measure(meta,"oxiblas","factor",f)
                r===nothing || measure(merge(meta,(phase_detail="apply",)),"oxiblas","solve",()->O.solve(r.value,b);
                                       extra=Dict{String,Any}("quality"=>_resid(S,O.solve(r.value,b),b)))
            end
        end
    end
    if kind===:spd
        H=Hermitian(S)
        if _selected(:cholesky) && !_capped(merge(base,(op="cholesky",)),:cholesky,n)
            meta=merge(base,(op="cholesky_natural",format="SparseMatrixCSC",ordering="natural"))
            _factor_solve(meta,"oxiblas",()->O.sparse_cholesky(S),O.solve,S,b,T)
            ss ? _factor_solve(meta,REF_LABEL,()->cholesky(H;perm=1:n),(F,b)->F\b,S,b,T) : na(meta,REF_LABEL)
        end
        if _selected(:mf_cholesky) && !_capped(merge(base,(op="multifrontal_cholesky",)),:mf_cholesky,n)
            meta=merge(base,(op="cholesky_ordered",format="SparseMatrixCSC",ordering="minimum_degree / CHOLMOD AMD"))
            _factor_solve(meta,"oxiblas",()->O.multifrontal_cholesky(S),O.solve,S,b,T)
            if ss
                F=_factor_solve(meta,REF_LABEL,()->cholesky(H),(F,b)->F\b,S,b,T)
                F===nothing || measure(merge(meta,(phase_detail="numeric refactor (symbolic reused)",)),REF_LABEL,"refactor",()->cholesky!(F,H))
            else na(meta,REF_LABEL) end
            _accelerate_direct(merge(meta,(ordering="Accelerate default",)),:Cholesky,S,b,T)
        end
        if _selected(:ordering) && !_capped(merge(base,(op="ordering",)),:ordering,n)
            # AMD alone (the ordering), then the full symbolic analysis
            # (structure of every column of L) for each algorithm.
            measure(merge(base,(op="ordering",algorithm="amd (ordering only)",format="SparseMatrixCSC")),"oxiblas","analyze",()->O.amd_ordering(S))
            for alg in (:amd,:minimum_degree,:rcm)
                meta=merge(base,(op="ordering",algorithm=string(alg),format="SparseMatrixCSC"))
                r=measure(meta,"oxiblas","analyze",()->O.symbolic_cholesky(S;ordering=alg))
                r===nothing || emit(merge(Dict{String,Any}(string(k)=>v for (k,v) in pairs(meta)),Dict("backend"=>"oxiblas","phase"=>"fill","threads"=>SP_THREADS,"status"=>"OK","fill"=>sum(length,r.value.columns))))
            end
            if ss
                meta=merge(base,(op="ordering",algorithm="CHOLMOD AMD",format="SparseMatrixCSC"))
                A=SparseArrays.CHOLMOD.Sparse(H)
                r=measure(meta,REF_LABEL,"analyze",()->SparseArrays.CHOLMOD.symbolic(A))
                F=cholesky(H)
                emit(merge(Dict{String,Any}(string(k)=>v for (k,v) in pairs(meta)),Dict("backend"=>REF_LABEL,"phase"=>"fill","threads"=>SP_THREADS,"status"=>"OK","fill"=>_fill(F))))
            end
        end
        if _selected(:incomplete) && !_capped(merge(base,(op="incomplete_cholesky",)),:incomplete,n)
            for (name,f) in (("ic0",()->O.ic0(S)),("ict",()->O.ict(S)))
                meta=merge(base,(op=name,format="SparseMatrixCSC",note="no SparseArrays equivalent"))
                r=measure(meta,"oxiblas","factor",f)
                r===nothing || measure(merge(meta,(phase_detail="apply",)),"oxiblas","solve",()->O.solve(r.value,b);
                                       extra=Dict{String,Any}("quality"=>_resid(S,O.solve(r.value,b),b)))
            end
        end
    end
    if kind===:indef && _selected(:ldlt) && !_capped(merge(base,(op="ldlt",)),:ldlt,n)
        for ord in (:natural,:minimum_degree)
            meta=merge(base,(op="ldlt",format="SparseMatrixCSC",ordering=string(ord),pivot="bunchkaufman"))
            _factor_solve(meta,"oxiblas",()->O.sparse_ldlt(S;ordering=ord),O.solve,S,b,T)
        end
        meta=merge(base,(op="ldlt",format="SparseMatrixCSC",ordering="CHOLMOD default",pivot="none (CHOLMOD)"))
        if !ss
            na(meta,REF_LABEL)
        elseif (try; ldlt(Hermitian(S)); true; catch err; err isa LinearAlgebra.ZeroPivotException || rethrow(); false; end)
            _factor_solve(meta,REF_LABEL,()->ldlt(Hermitian(S)),(F,b)->F\b,S,b,T)
        else
            _unavailable(meta,REF_LABEL,"CHOLMOD ldlt does not pivot: zero pivot on this indefinite matrix")
        end
        _accelerate_direct(merge(meta,(ordering="Accelerate default",pivot="pivoted LDLT (Accelerate default)")),:LDLT,S,b,T)
    end
    if kind===:tall && _selected(:qr) && !_capped(merge(base,(op="qr",)),:qr,m)
        Bq=T[_val(T,rng) for _ in 1:m, _ in 1:8]
        # Normal-equations residual ‖Sᴴ(Sx-b)‖ ≤ tol·‖S‖₁²·‖b‖ (no dense reference).
        lscheck(x)=norm(S'*(S*x-b))<=1e3*_solve_tol(T)*opnorm(S,1)^2*norm(b)
        # pivot=true: rank-revealing column pivoting; pivot=false: multifrontal
        # QR in the natural order, in AMD(AᵀA) order and in COLAMD order (SPQR's).
        for (pivot,ord) in ((true,:natural),(false,:natural),(false,:amd),(false,:colamd))
            meta=merge(base,(op="qr",format="SparseMatrixCSC",pivot=string(pivot),ordering=string(ord)))
            r=measure(merge(meta,(phase_detail="factor",)),"oxiblas","factor",()->O.sparse_qr(S;pivot,ordering=ord))
            r===nothing && continue
            F=r.value; e=Dict{String,Any}("fill"=>_fill(F))
            measure(merge(meta,(phase_detail="least squares",)),"oxiblas","solve",()->O.solve(F,b);check=lscheck,extra=e)
            measure(merge(meta,(phase_detail="apply Qᴴ (8 columns)",)),"oxiblas","apply_q",()->O.sparse_apply_q!(copy(Bq),F;trans=:C))
        end
        meta=merge(base,(op="qr",format="SparseMatrixCSC",pivot="SPQR default ordering"))
        if ss
            r=measure(merge(meta,(phase_detail="factor",)),REF_LABEL,"factor",()->qr(S))
            if r!==nothing
                F=r.value
                measure(merge(meta,(phase_detail="least squares",)),REF_LABEL,"solve",()->F\b;check=lscheck,extra=Dict{String,Any}("fill"=>_fill(F)))
                measure(merge(meta,(phase_detail="apply Qᴴ (8 columns)",)),REF_LABEL,"apply_q",()->F.Q'*Bq)
            end
        else na(meta,REF_LABEL) end
        _accelerate_direct(merge(base,(op="qr",format="SparseMatrixCSC",pivot="Accelerate default ordering")),:QR,S,b,T;lscheck)
    end
    nothing
end
# Accelerate libSparse (only in the OXIBLAS_SP_BLAS=accelerate run): factor
# (a fresh factorization object per sample), solve, numeric refactor.
function _accelerate_direct(meta,kind,S,b,T;lscheck=nothing)
    HAS_AA || return nothing
    AA=AppleAccelerate
    A=AA.AASparseMatrix(SparseMatrixCSC{T,Int64}(S))
    fk=kind===:LU ? AA.SparseFactorizationLU : kind===:Cholesky ? AA.SparseFactorizationCholesky :
       kind===:LDLT ? AA.SparseFactorizationLDLT : AA.SparseFactorizationQR
    r=measure(merge(meta,(phase_detail="factor",)),"accelerate","factor",()->(F=AA.AAFactorization(A); AA.factor!(F,fk); F))
    r===nothing && return nothing
    F=r.value
    check=lscheck===nothing ? _rescheck(T,S,b) : lscheck
    measure(merge(meta,(phase_detail="solve",)),"accelerate","solve",()->AA.solve(F,b);check)
    measure(merge(meta,(phase_detail="numeric refactor (symbolic reused)",)),"accelerate","refactor",()->AA.refactor!(F,A))
    nothing
end

# Intermediates of a plan, executed step by step (diagnostics only).
function plan_intermediates(plan,ops)
    data=Any[O._is_sparse_operand(A) ? O._sparse_operand(A) : A for A in ops]; ls=copy(plan.labels); out=Any[]
    steps=plan.steps
    for (i,j,o) in steps
        C=O._einsum_pair_any(data[i],data[j],ls[i],ls[j],o,plan.dimensions)
        push!(out,(labels=String(Char.(filter(<=(256),o))),kind=C isa SparseTensor ? "sparse" : "dense",nnz=_nnz(C),density=_density(C)))
        deleteat!(data,j);deleteat!(data,i);push!(data,C);deleteat!(ls,j);deleteat!(ls,i);push!(ls,o)
    end
    out
end

function bench_einsum_matrix(case,T,S,rng)
    m,n=size(S); m==n || return nothing
    base=(case=case,T=string(T),m=m,n=n,nnz_in=nnz(S),density_in=_density(S))
    k=16; dense_ok=Float64(n)*k*sizeof(T)<=SP_DENSE_LIMIT
    D=dense_ok ? randn(rng,real(T),n,k).+zero(T) : nothing
    if dense_ok && _selected(:einsum_sd)
        ref=S*D; meta=merge(base,(op="einsum_sd",spec="ij,jk->ik",k=k,format="SparseMatrixCSC x dense"))
        measure(meta,"sparsearrays","kernel",()->S*D)
        measure(meta,"oxiblas","total",()->einsum("ij,jk->ik",S,D);check=_ok(T,ref),output_of=identity)
        P=plan_einsum("ij,jk->ik",S,D)
        measure(merge(meta,(spec="ij,jk->ik (plan reused)",)),"oxiblas","reuse",()->einsum(P,S,D);check=_ok(T,ref))
        C=zeros(T,n,k)
        measure(merge(meta,(spec="einsum! dense destination",)),"oxiblas","total",()->einsum!(C,P,S,D);check=v->_relerr(C,ref)<=_tol(T))
        measure(merge(meta,(op="contract_sd",spec="contract((2,),(1,))")),"oxiblas","total",()->contract(S,D,(2,),(1,));check=_ok(T,ref))
    end
    if dense_ok && _selected(:einsum_ds)
        Dt=permutedims(D); ref=Dt*S; meta=merge(base,(op="einsum_ds",spec="ki,ij->kj",k=k,format="dense x SparseMatrixCSC"))
        measure(meta,"sparsearrays","kernel",()->Dt*S)
        measure(meta,"oxiblas","total",()->einsum("ki,ij->kj",Dt,S);check=_ok(T,ref))
    end
    if _selected(:einsum_ss) && nnz(S)<=10^6
        ref=S*S; meta=merge(base,(op="einsum_ss",spec="ij,jk->ik",format="SparseMatrixCSC x SparseMatrixCSC"))
        measure(meta,"sparsearrays","kernel",()->S*S;output_of=identity)
        measure(meta,"oxiblas","total",()->einsum("ij,jk->ik",S,S);check=_ok(T,ref),output_of=identity)
        measure(merge(meta,(spec="ij,jk->ik + SparseMatrixCSC(result)",)),"oxiblas","total",()->SparseMatrixCSC(einsum("ij,jk->ik",S,S));check=_ok(T,ref))
        measure(merge(meta,(op="contract_ss",spec="contract((2,),(1,))")),"oxiblas","total",()->contract(S,S,(2,),(1,));check=_ok(T,ref))
        X=SparseTensor(S)
        measure(merge(meta,(spec="einsum! SparseMatrixCSC destination",)),"oxiblas","total",()->(Z=copy(S);einsum!(Z,"ij,jk->ik",X,X);Z);check=_ok(T,ref))
        # Plan construction, first execution of a new plan (code warm), reuse.
        P0=plan_einsum("ij,jk->ik",S,S); einsum(P0,S,S)
        measure(merge(meta,(spec="plan_einsum",)),"oxiblas","plan",()->plan_einsum("ij,jk->ik",S,S))
        P=plan_einsum("ij,jk->ik",S,S); t=@elapsed (R=einsum(P,S,S))
        emit(merge(Dict{String,Any}(string(k)=>v for (k,v) in pairs(meta)),Dict("backend"=>"oxiblas","phase"=>"first","threads"=>SP_THREADS,"status"=>_relerr(R,ref)<=_tol(T) ? "OK" : "ERROR","median"=>t,"minimum"=>t,"samples"=>1)))
        measure(merge(meta,(spec="ij,jk->ik (plan reused)",)),"oxiblas","reuse",()->einsum(P,S,S);check=_ok(T,ref))
    end
    if _selected(:einsum_chain) && nnz(S)<=2*10^5
        ref=S*S*S
        for optimize in (:left,:greedy,:optimal)
            P=plan_einsum("ij,jk,kl->il",S,S,S;optimize)
            meta=merge(base,(op="einsum_chain",spec="ij,jk,kl->il",format="3 x SparseMatrixCSC",optimize=string(optimize)))
            measure(meta,"oxiblas","reuse",()->einsum(P,S,S,S);check=_ok(T,ref),extra=Dict{String,Any}("intermediates"=>plan_intermediates(P,(S,S,S))))
        end
        measure(merge(base,(op="einsum_chain",spec="ij,jk,kl->il",format="3 x SparseMatrixCSC",optimize="(S*S)*S")),"sparsearrays","kernel",()->S*S*S)
        if n*k*sizeof(T)<=SP_DENSE_LIMIT
            D=randn(rng,real(T),n,k).+zero(T); refm=S*(S*D)
            for optimize in (:left,:greedy)
                P=plan_einsum("ij,jk,kl->il",S,S,D;optimize)
                meta=merge(base,(op="einsum_chain_mixed",spec="ij,jk,kl->il",format="S,S,dense",optimize=string(optimize),k=k))
                measure(meta,"oxiblas","reuse",()->einsum(P,S,S,D);check=_ok(T,refm),extra=Dict{String,Any}("intermediates"=>plan_intermediates(P,(S,S,D))))
            end
            measure(merge(base,(op="einsum_chain_mixed",spec="ij,jk,kl->il",format="S,S,dense",optimize="S*(S*D)",k=k)),"sparsearrays","kernel",()->S*(S*D))
        end
    end
    if _selected(:einsum_unary)
        for (spec,reff) in (("ii->",()->fill(tr(S))),("ii->i",()->diag(S)),("ij->i",()->vec(sum(S;dims=2))),("ij->ji",()->copy(transpose(S))),("ij->",()->fill(sum(S))))
            ref=reff(); meta=merge(base,(op="einsum_unary",spec=spec,format="SparseMatrixCSC"))
            measure(meta,"sparsearrays","kernel",reff)
            measure(meta,"oxiblas","total",()->einsum(spec,S);check=v->_relerr(v,ref)<=10*_tol(T))
        end
    end
    nothing
end

function bench_dense_regression(T,rng)
    _selected(:einsum_dd) || return nothing
    for n in (128,512)
        A=randn(rng,real(T),n,n).+zero(T); B=randn(rng,real(T),n,n).+zero(T); ref=A*B; C=zeros(T,n,n)
        base=(case="dense_$n",T=string(T),m=n,n=n,nnz_in=n*n,density_in=1.0,op="einsum_dd",spec="ij,jk->ik",format="dense x dense")
        measure(base,"oxiblas","kernel",()->O.gemm!(C,A,B);check=v->_relerr(C,ref)<=_tol(T))
        measure(base,"oxiblas","total",()->einsum("ij,jk->ik",A,B);check=_ok(T,ref))
        X=randn(rng,real(T),20,30,40).+zero(T); Y=randn(rng,real(T),40,30,10).+zero(T)
        meta=merge(base,(spec="ijk,lkm->ijlm",m=20,n=40))
        measure(meta,"oxiblas","total",()->einsum("ijk,ljm->iklm",X,Y))
        HAS_TO && measure(meta,"tensoroperations","total",()->TensorOperations.ncon([X,Y],[[-1,1,-2],[-3,1,-4]]))
    end
    nothing
end

# Probe check for large tensor results: a few output entries recomputed by
# brute force over the stored entries of the sparse inputs.
function probe_ok(result,A::SparseTensor,B,contract_a,contract_b,free_a,free_b;probes=8,rng=Random.default_rng())
    T=eltype(A); outdims=size(result)
    ca=[O._sparse_coords(i,A.dims) for i in A.indices]
    for _ in 1:probes
        o=ntuple(d->rand(rng,1:outdims[d]),length(outdims))
        s=zero(T)
        for (p,c) in enumerate(ca)
            all(d->c[free_a[d]]==o[d],eachindex(free_a)) || continue
            if B isa SparseTensor
                for (q,i) in enumerate(B.indices)
                    cb=O._sparse_coords(i,B.dims)
                    all(d->c[contract_a[d]]==cb[contract_b[d]],eachindex(contract_a)) && all(d->cb[free_b[d]]==o[length(free_a)+d],eachindex(free_b)) && (s+=A.values[p]*B.values[q])
                end
            else
                for idx in CartesianIndices(size(B))
                    t=Tuple(idx)
                    all(d->c[contract_a[d]]==t[contract_b[d]],eachindex(contract_a)) && all(d->t[free_b[d]]==o[length(free_a)+d],eachindex(free_b)) && (s+=A.values[p]*B[idx])
                end
            end
        end
        abs(result[o...]-s)<=_tol(T)*max(1,abs(s)) || return false
    end
    true
end

function bench_tensors(T,rng)
    _selected(:tensor) || return nothing
    # (name, dims, nnz, dense partner): order 3, 4, 5; small ones also have a
    # dense / matricized SparseArrays reference.
    for (name,dims,k) in (("t3_100",(100,100,100),10^4),("t3_100_dense",(100,100,100),10^5),("t4_50",(50,50,50,50),10^5),("t5_20",(20,20,20,20,20),10^4),("t3_1e4_hyper",(10^4,10^4,10^4),10^5))
        X=ds_tensor(T,dims,k,rng); base=(case=name,T=string(T),dims=collect(dims),nnz_in=nnz(X),density_in=_density(X))
        small=Float64(prod(BigInt.(dims)))*sizeof(T)<=SP_DENSE_LIMIT
        N=length(dims); last=dims[end]
        # Sparse x dense on the last mode: matricized SparseArrays reference.
        r=16; U=randn(rng,real(T),last,r).+zero(T)
        if Float64(prod(dims[1:end-1]))*r*sizeof(T)<=SP_DENSE_LIMIT
            spec=String(Char.('a':'a'+N-1))*",$(Char('a'+N-1))z->"*String(Char.('a':'a'+N-2))*"z"
            Mx=SparseMatrixCSC(SparseTensor((prod(dims[1:end-1]),last),copy(X.indices),copy(X.values)))
            ref=reshape(Mx*U,(dims[1:end-1]...,r)); meta=merge(base,(op="tensor_sd",spec=spec,format="SparseTensor x dense"))
            measure(meta,"sparsearrays","kernel",()->Mx*U)
            measure(meta,"oxiblas","total",()->einsum(spec,X,U);check=_ok(T,ref))
            if small && HAS_TO
                Xd=Array(X); ix=vcat(-collect(1:N-1),[1]); measure(meta,"tensoroperations","total",()->TensorOperations.ncon([Xd,U],[ix,[1,-N]]))
            end
        end
        # Sparse x sparse sharing two modes, sparse output.
        Y=ds_tensor(T,(dims[end-1],dims[end],dims[1]),k÷2,rng)
        spec=String(Char.('a':'a'+N-1))*","*String([Char('a'+N-2),Char('a'+N-1),'z'])*"->"*String(Char.('a':'a'+N-3))*"z"
        meta=merge(base,(op="tensor_ss",spec=spec,format="SparseTensor x SparseTensor",nnz_rhs=nnz(Y)))
        free_a=collect(1:N-2); ca=[N-1,N]; cb=[1,2]; fb=[3]
        measure(meta,"oxiblas","total",()->einsum(spec,X,Y);check=v->probe_ok(v,X,Y,ca,cb,free_a,fb;rng),output_of=identity)
        if small && HAS_TO
            measure(meta,"tensoroperations","total",()->TensorOperations.ncon([Array(X),Array(Y)],[vcat(-collect(1:N-2),[1,2]),[1,2,-(N-1)]]))
        end
        # Permutation and reduction (sparse unary).
        perm=reverse(collect(1:N))
        measure(merge(base,(op="tensor_permute",spec="reverse modes",format="SparseTensor")),"oxiblas","kernel",()->permutedims(X,perm);output_of=identity)
        spec=String(Char.('a':'a'+N-1))*"->a"
        measure(merge(base,(op="tensor_reduce",spec=spec,format="SparseTensor")),"oxiblas","total",()->einsum(spec,X);output_of=identity)
    end
    nothing
end

function bench_scaling(T,rng)
    _selected(:scaling) || return nothing
    for c in SCALING_CASES
        isempty(SP_CASES) || c.name in SP_CASES || continue
        S=ds_hypersparse(T,c.n,c.k,rng); base=(case=c.name,T=string(T),m=c.n,n=c.n,nnz_in=nnz(S),density_in=_density(S),series=c.n==10^5 ? "nnz" : "dimension")
        x=T[_val(T,rng) for _ in 1:c.n]; y=zeros(T,c.n); C=O.CSRMatrix(S); ref=S*x
        measure(merge(base,(op="scale_spmv",format="CSR")),"oxiblas","kernel",()->O.sparse_mul!(y,C,x);check=_ok(T,ref))
        measure(merge(base,(op="scale_spmv",format="SparseMatrixCSC")),"sparsearrays","kernel",()->mul!(y,S,x))
        measure(merge(base,(op="scale_convert",format="CSC->CSR")),"oxiblas","convert",()->O.CSRMatrix(S))
        measure(merge(base,(op="scale_convert",format="CSC->CSR")),"sparsearrays","convert",()->sparse(transpose(S)))
        measure(merge(base,(op="scale_spgemm",format="CSR")),"oxiblas","kernel",()->O.sparse_matmul(C,C);output_of=identity)
        measure(merge(base,(op="scale_spgemm",format="SparseMatrixCSC")),"sparsearrays","kernel",()->S*S;output_of=identity)
        refss=S*S
        measure(merge(base,(op="scale_einsum_ss",format="SparseMatrixCSC")),"oxiblas","total",()->einsum("ij,jk->ik",S,S);check=_ok(T,refss),output_of=identity)
        measure(merge(base,(op="scale_tensor",format="SparseMatrixCSC->SparseTensor")),"oxiblas","convert",()->SparseTensor(S))
    end
    nothing
end

# ---- campaign -------------------------------------------------------------

println("@@SPARSE_INFO\t",_json(Dict("threads"=>SP_THREADS,"julia"=>string(VERSION),"tensoroperations"=>HAS_TO,"omeinsum"=>false,
                                     "blas"=>string(LinearAlgebra.BLAS.get_config()),"seconds"=>SP_SECONDS,"samples"=>SP_SAMPLES)))
for c in MATRIX_CASES
    isempty(SP_CASES) || c.name in SP_CASES || continue
    for t in c.types
        isempty(SP_TYPES) || t in SP_TYPES || continue
        T=_type(t); rng=Random.Xoshiro(hash((c.name,t)))
        S=setprecision(BigFloat,128) do; c.build(T,rng); end
        # Each family behind a dynamic call: inferring this closure statically
        # compiled every family (direct solvers included) for each case and
        # type, about 3 min per pair even when one family is selected
        # (--trace-compile-timing: 174 s in this one MethodInstance). Only the
        # families that run are compiled now; the measured calls are unchanged.
        setprecision(BigFloat,128) do
            :core in c.ops && !SP_DIRECT_ONLY && _any_selected(CORE_OPS) && Base.invokelatest(bench_core,c.name,T,S,rng)
            for kind in (:spd,:unsym,:indef,:tall,:lower)
                kind in c.ops && _any_selected(DIRECT_OPS) && Base.invokelatest(bench_direct,c.name,kind,T,S,rng)
            end
            :einsum in c.ops && !SP_DIRECT_ONLY && Base.invokelatest(bench_einsum_matrix,c.name,T,S,rng)
        end
        GC.gc()
    end
end
for t in ("Float64","ComplexF32")
    (isempty(SP_TYPES) || t in SP_TYPES) && !SP_DIRECT_ONLY || continue
    T=_type(t); rng=Random.Xoshiro(hash(("tensors",t)))
    Base.invokelatest(bench_dense_regression,T,rng); Base.invokelatest(bench_tensors,T,rng); Base.invokelatest(bench_scaling,T,rng)
end
println("@@WORKER_ERRORS\t",FAILURES[])
