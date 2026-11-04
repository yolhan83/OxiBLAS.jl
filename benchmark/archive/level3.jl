# Measured references live outside src/. Baseline is the frozen v0.2 source.
using OxiBLAS, BenchmarkTools, Random, InteractiveUtils
import LinearAlgebra as LA
module V02
using OxiBLAS: gemm!, trmv!, trsv!, SingularException
import OxiBLAS: _square, _uplo, _scale!, _symget, _op, _tri_setup
include(joinpath(@__DIR__,"..","..","reports","baseline-v0.2","level3.jl"))
end
LA.BLAS.set_num_threads(1)
Random.seed!(20261102)
const reportdir=get(ENV,"OXIBLAS_REPORT_DIR",joinpath(@__DIR__,"..","..","reports"))
const backend=get(ENV,"OXIBLAS_REFERENCE","openblas")
if backend=="accelerate"
    @eval using AppleAccelerate
    AppleAccelerate.set_num_threads(1)
end
const sizes="--smoke" in ARGS ? (17,) : (32,128,512)
open(joinpath(reportdir,"level3-$(backend)-environment.txt"),"w") do io
    versioninfo(io)
    println(io,"BLAS reference: ",LA.BLAS.get_config(),"; threads=",LA.BLAS.get_num_threads())
    println(io,"BenchmarkTools ",pkgversion(BenchmarkTools))
    println(io,"Native Julia threads: ",Threads.nthreads())
    println(io,"All calls use one worker; reset output outside timing, evals=1, up to 7 samples/0.12 s.")
    println(io,"Baseline is reports/baseline-v0.2/level3.jl, with unchanged GEMM implementation.")
end

function measure(io,operation,T,n,k,case,method,f,initial,expected,uplo,triangular)
    output=copy(initial)
    first=@timed f(output)
    selected=triangular ? [(i,j) for j in axes(output,2) for i in axes(output,1) if (uplo=='U' ? i<=j : i>=j)] : collect(Iterators.product(axes(output,1),axes(output,2)))
    error=sqrt(sum(abs2(output[i,j]-expected[i,j]) for (i,j) in selected))/max(LA.norm(expected),eps(real(T)))
    limit=real(T)===Float32 ? 5e-4 : 5e-12
    error<=limit || Base.error("level3 benchmark mismatch: $((operation,T,n,method,error))")
    if triangular
        all(isequal(output[i,j],initial[i,j]) for j in axes(output,2) for i in axes(output,1) if (uplo=='U' ? i>j : i<j)) || Base.error("unselected triangle modified")
    end
    trial=run(@benchmarkable($f($output), setup=(copyto!($output,$initial)), evals=1);seconds=0.12,samples=7)
    compile=hasproperty(first,:compile_time) ? first.compile_time : NaN
    println(io,join((operation,T,n,k,case,method,BenchmarkTools.median(trial).time/1e9,minimum(trial).time/1e9,trial.memory,trial.allocs,length(trial),first.time,compile,error),','));flush(io)
    println((operation,T,n,case,method));flush(stdout)
end

open(joinpath(reportdir,"level3-$(backend).csv"),"w") do io
    println(io,"operation,type,n,k,case,algorithm,median_s,minimum_s,bytes,allocations,samples,first_call_s,compile_s,relative_error")
    for T in (Float32,Float64,ComplexF32,ComplexF64), n in sizes, operation in (:symm,:hemm,:trmm,:trsm,:syrk,:syr2k,:herk,:her2k,:gemmt)
        T<:Real && operation in (:hemm,:herk,:her2k) && continue
        cases="--smoke" in ARGS ? ((:L,:U,:N),) : ((:L,:U,:N),(:R,:L,:C))
        for (side,uplo,trans) in cases
            k=operation===:gemmt ? n : max(1,3n÷4); sc=side===:L ? 'L' : 'R'; uc=uplo===:U ? 'U' : 'L'
            case=string(side,uplo,trans)
            α=T<:Real ? T(0.7) : T(0.7+0.2im); β=T(-0.3)
            herm=operation in (:hemm,:herk,:her2k)
            operation===:herk && (α=T(0.7))
            tr=operation in (:syrk,:syr2k) && trans===:C ? :T : trans
            tc=tr===:N ? 'N' : tr===:T ? 'T' : 'C'
            triangular=operation in (:syrk,:syr2k,:herk,:her2k,:gemmt)
            A=randn(T,triangular && operation!==:gemmt ? (tr===:N ? (n,k) : (k,n)) : (n,n))
            operation in (:trmm,:trsm) && (A./=n; A[LA.diagind(A)].+=T(2))
            B=randn(T,operation===:gemmt ? (n,n) : triangular ? size(A) : (side===:L ? (n,k) : (k,n)))
            C=randn(T,triangular ? (n,n) : size(B))
            # Real initial Hermitian diagonals keep all reference backends on
            # the same valid workload. A separate probe tests ignored imaginary
            # diagonal data, which the local Accelerate HERK fails to discard.
            if operation in (:herk,:her2k)
                for i in 1:n
                    C[i,i]=real(C[i,i])
                end
            end
            initial=operation in (:trmm,:trsm) ? B : C
            if operation in (:symm,:hemm)
                S=herm ? LA.Hermitian(A,uplo) : LA.Symmetric(A,uplo)
                expected=α*(side===:L ? S*B : B*S)+β*C
            elseif operation in (:trmm,:trsm)
                S=tr===:N ? (uplo===:U ? LA.UpperTriangular(A) : LA.LowerTriangular(A)) : adjoint(uplo===:U ? LA.UpperTriangular(A) : LA.LowerTriangular(A))
                expected=operation===:trmm ? α*(side===:L ? S*B : B*S) : (side===:L ? S\(α*B) : (α*B)/S)
            elseif operation===:gemmt
                aa=tr===:N ? A : adjoint(A); expected=α*aa*B+β*C
            else
                aa=tr===:N ? A : tr===:T ? transpose(A) : adjoint(A)
                bb=tr===:N ? B : tr===:T ? transpose(B) : adjoint(B)
                rhs(X)=herm ? adjoint(X) : transpose(X)
                P=operation in (:syr2k,:her2k) ? α*aa*rhs(bb)+(herm ? conj(α) : α)*bb*rhs(aa) : α*aa*rhs(aa)
                expected=P+β*(herm ? Matrix(LA.Hermitian(C,uplo)) : C)
            end
            methods=backend=="openblas" ? (:v02,:native,:reference_blas) : (:reference_blas,)
            for method in methods
                operation===:gemmt && method!==:native && continue # portable BLAS has no GEMMT entry
                mod=method===:v02 ? V02 : OxiBLAS
                func=getfield(mod,Symbol(operation,:!))
                if method===:reference_blas
                    bf=getfield(LA.BLAS,Symbol(operation,:!))
                    f=operation in (:symm,:hemm) ? out->bf(sc,uc,α,A,B,β,out) :
                      operation in (:trmm,:trsm) ? out->bf(sc,uc,tc,'N',α,A,out) :
                      operation in (:syr2k,:her2k) ? out->bf(uc,tc,α,A,B,β,out) :
                      out->bf(uc,tc,herm ? real(α) : α,A,herm ? real(β) : β,out)
                    # HER2K beta is real, even when alpha is complex.
                    operation===:her2k && (f=out->bf(uc,tc,α,A,B,real(β),out))
                else
                    f=operation in (:symm,:hemm) ? out->func(out,A,B,α,β;side,uplo,threaded=false) :
                      operation in (:trmm,:trsm) ? (method===:v02 ? out->func(out,A,α;side,uplo,trans=tr) : out->func(out,A,α;side,uplo,trans=tr,threaded=false)) :
                      operation===:gemmt ? out->func(out,A,B,α,β;uplo,transA=tr,threaded=false) :
                      operation in (:syr2k,:her2k) ? out->func(out,A,B,α,β;uplo,trans=tr,threaded=false) :
                      out->func(out,A,α,β;uplo,trans=tr,threaded=false)
                end
                measure(io,operation,T,n,k,case,method,f,initial,expected,uc,triangular)
            end
        end
    end
end
