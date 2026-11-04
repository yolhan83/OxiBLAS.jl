# First-call latency of the paths of the precompilation workload, in a fresh
# process: load time, then the first call of each path (compilation
# included). Run once per path, from the repository root:
#   julia --project=. benchmark/latency/first_call.jl            # all
#   julia --project=. benchmark/latency/run_all.jl > out.tsv      # each path in its own process
t0=time_ns(); using OxiBLAS; tload=(time_ns()-t0)/1e9
import LinearAlgebra as LA
const O=OxiBLAS
n=64; A=randn(n,n)+n*LA.I; H=A'*A; b=randn(n); A32=Float32.(A)
m=400; S=CSRMatrix(COOMatrix(m,m,vcat(1:m,1:m-1,2:m),vcat(1:m,2:m,1:m-1),vcat(fill(4.0,m),fill(-1.0,m-1),fill(-1.0,m-1))))
U=CSRMatrix(COOMatrix(m,m,vcat(1:m,1:m-1),vcat(1:m,2:m),vcat(fill(4.0,m),fill(-1.0,m-1)))); bs=ones(m)
X3=randn(10,10,10); Y=randn(10,10)
paths=Dict(
 "gemm_f64"=>()->gemm!(similar(A),A,A),
 "gemm_f32"=>()->gemm!(similar(A32),A32,A32),
 "lu_dense"=>()->O.lu(A)\b,
 "cholesky_dense"=>()->O.cholesky(LA.Hermitian(H))\b,
 "ldiv_dense"=>()->O.ldiv(A,b),
 "sparse_backslash_spd"=>()->S\bs,
 "sparse_backslash_lu"=>()->U\bs,
 "einsum"=>()->einsum("abc,cd->abd",X3,Y),
)
sel=isempty(ARGS) ? sort(collect(keys(paths))) : ARGS
println("load\t",round(tload;digits=3))
for k in sel
    t=@elapsed paths[k](); t2=@elapsed paths[k]()
    println(k,"\t",round(t;digits=4),"\t",round(t2;digits=6))
end
