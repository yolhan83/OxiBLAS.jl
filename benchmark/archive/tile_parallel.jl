using OxiBLAS, BenchmarkTools, Random, InteractiveUtils
import LinearAlgebra as LA
module V04Tiles
using OxiBLAS: GemmConfig,GemmWorkspace,gemm!,trmv!,trsv!,SingularException
import OxiBLAS: NativeFloat,_square,_uplo,_scale!,_symget,_op,_tri_setup,_check_output_layout,_kernel_kind,_shape,_workspace_check,_gemm_dims
include(joinpath(@__DIR__,"..","..","reports","baseline-v0.4","level3.jl"))
end
const threads=Threads.nthreads()
LA.BLAS.set_num_threads(threads)
const backend=get(ENV,"OXIBLAS_REFERENCE","openblas")
if backend=="accelerate"
    @eval using AppleAccelerate
    AppleAccelerate.set_num_threads(threads)
end
const reportdir=get(ENV,"OXIBLAS_REPORT_DIR",joinpath(@__DIR__,"..","..","reports"))
const sizes="--smoke" in ARGS ? (129,) : (512,1024)
Random.seed!(20261102)
open(joinpath(reportdir,"tiles-$(backend)-$(threads)t-environment.txt"),"w") do io
    versioninfo(io); println(io,"Reference: ",LA.BLAS.get_config(),"; reference threads=",LA.BLAS.get_num_threads())
    println(io,"BenchmarkTools ",pkgversion(BenchmarkTools),"; max 5 samples/0.15 s, evals=1.")
    println(io,"Preallocated GEMM panels, output restoration outside timing; diagonal tiles/task contexts are included.")
    println(io,"Native tile workers own one panel pair each; their GEMM calls are single-threaded.")
end
function timed_row(io,op,T,n,k,method,workers,f,initial,expected,uplo)
    C=copy(initial); call=()->f(C); first=@timed call()
    istri=op in (:syrk,:syr2k,:herk,:her2k,:gemmt)
    selection=istri ? [(i,j) for j in 1:n for i in 1:j] : collect(Iterators.product(axes(C,1),axes(C,2)))
    err=sqrt(sum(abs2(C[i,j]-expected[i,j]) for (i,j) in selection))/max(LA.norm(expected),eps(real(T)))
    err<(real(T)===Float32 ? 1e-4 : 1e-11) || error("tile benchmark mismatch $((op,T,n,method,err))")
    if istri
        all(isequal(C[i,j],initial[i,j]) for j in 1:n for i in j+1:n) || error("unselected triangle modified")
    end
    trial=run(@benchmarkable($call(),setup=(copyto!($C,$initial)),evals=1);seconds=.15,samples=5)
    compile=hasproperty(first,:compile_time) ? first.compile_time : NaN
    println(io,join((op,T,n,k,method,threads,workers,BenchmarkTools.median(trial).time/1e9,minimum(trial).time/1e9,trial.memory,trial.allocs,length(trial),first.time,compile,err),','));flush(io)
    println((op,T,n,method,workers));flush(stdout)
end
open(joinpath(reportdir,"tiles-$(backend)-$(threads)t.csv"),"w") do io
    println(io,"operation,type,n,k,algorithm,threads,tile_workers,median_s,minimum_s,bytes,allocations,samples,first_call_s,compile_s,relative_error")
    for T in (Float32,Float64,ComplexF32,ComplexF64), n in sizes, op in (:symm,:hemm,:syrk,:syr2k,:herk,:her2k,:gemmt)
        T<:Real && op in (:hemm,:herk,:her2k) && continue
        k=3n÷4; rank=op in (:syrk,:syr2k,:herk,:her2k); rank2=op in (:syr2k,:her2k); herm=op in (:hemm,:herk,:her2k)
        α=op===:herk ? T(.7) : T<:Real ? T(.7) : T(.7+.3im); β=T(.2)
        A=randn(T,rank || op===:gemmt ? (n,k) : (n,n))
        B=randn(T,rank ? (n,k) : op===:gemmt ? (k,n) : (n,k))
        C=randn(T,op in (:symm,:hemm) ? size(B) : (n,n))
        herm && op!==:hemm && (C[LA.diagind(C)].=real.(C[LA.diagind(C)]))
        if op in (:symm,:hemm)
            expected=α*(herm ? LA.Hermitian(A) : LA.Symmetric(A))*B+β*C
        elseif op===:gemmt
            expected=α*A*B+β*C
        else
            rhs(X)=herm ? adjoint(X) : transpose(X)
            expected=(rank2 ? α*A*rhs(B)+(herm ? conj(α) : α)*B*rhs(A) : α*A*rhs(A))+β*(herm ? Matrix(LA.Hermitian(C)) : C)
        end
        config=GemmConfig(T)
        w=op in (:symm,:hemm) ? GemmWorkspace(T,n,k,n;config) : GemmWorkspace(T,min(n,128),min(n,128),k;config)
        methods=backend=="openblas" ? (:v04,:native,:reference_blas) : (:reference_blas,)
        for method in methods
            method===:reference_blas && op===:gemmt && continue
            workers=method===:native && !(op in (:symm,:hemm)) ? OxiBLAS._triangle_worker_count(n,k,128,(;threaded=true,config,workspace=w),rank2) : 0
            if method===:reference_blas
                bf=getfield(LA.BLAS,Symbol(op,:!))
                f=op in (:symm,:hemm) ? out->bf('L','U',α,A,B,β,out) :
                  rank2 ? (herm ? out->bf('U','N',α,A,B,real(β),out) : out->bf('U','N',α,A,B,β,out)) :
                  herm ? out->bf('U','N',real(α),A,real(β),out) : out->bf('U','N',α,A,β,out)
            else
                mod=method===:v04 ? V04Tiles : OxiBLAS; func=getfield(mod,Symbol(op,:!))
                f=op in (:symm,:hemm) ? out->func(out,A,B,α,β;workspace=w,config) :
                  op===:gemmt ? out->func(out,A,B,α,β;workspace=w,config) :
                  rank2 ? out->func(out,A,B,α,β;workspace=w,config) : out->func(out,A,α,β;workspace=w,config)
            end
            timed_row(io,op,T,n,k,method,workers,f,C,expected,:U)
        end
    end
end
