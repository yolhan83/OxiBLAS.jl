using OxiBLAS, BenchmarkTools, Random, InteractiveUtils
import LinearAlgebra as LA
include(joinpath(@__DIR__,"..","..","test","helpers","compact_references.jl"))
module V04Compact
using OxiBLAS: gemv!,trmv!,trsv!,syr!,syr2!,her!,her2!
import OxiBLAS: _square,_uplo
include(joinpath(@__DIR__,"..","..","reports","baseline-v0.4","storage.jl"))
end
LA.BLAS.set_num_threads(1)
const backend=get(ENV,"OXIBLAS_REFERENCE","openblas")
if backend=="accelerate"
    @eval using AppleAccelerate
    AppleAccelerate.set_num_threads(1)
end
const outdir=get(ENV,"OXIBLAS_REPORT_DIR",joinpath(@__DIR__,"..","..","reports"))
Random.seed!(20261102)
const sizes="--smoke" in ARGS ? (33,) : (128,1024,16384)
open(joinpath(outdir,"compact-$(backend)-environment.txt"),"w") do io
    versioninfo(io); println(io,"BLAS: ",LA.BLAS.get_config(),"; threads=",LA.BLAS.get_num_threads())
    println(io,"BenchmarkTools ",pkgversion(BenchmarkTools),"; max 7 samples/0.12 seconds, evals=1; output reset outside timing.")
    println(io,"N>1024: banded only, no quadratic baseline, no complex symmetric extension without portable BLAS counterpart.")
end
function old_compact(P)
    P isa PackedMatrix && return V04Compact.PackedMatrix(copy(P.data),P.n;kind=P.kind,uplo=P.upper ? :U : :L,unit=P.unit)
    V04Compact.BandedMatrix(copy(P.data),P.m;kl=P.kl,ku=P.ku,kind=P.kind,uplo=P.upper ? :U : :L,unit=P.unit)
end
function measure(io,op,T,n,kd,trans,method,f,setup,verify)
    setup(); first=@timed f(); error=verify(first.value)
    error< (real(T)===Float32 ? 1e-4 : 1e-11) || Base.error("compact mismatch $((op,T,n,trans,method,error))")
    sink=Ref(first.value); measured=()->(sink[]=f())
    trial=run(@benchmarkable($measured(),setup=($setup()),evals=1);seconds=.12,samples=7)
    compile=hasproperty(first,:compile_time) ? first.compile_time : NaN
    println(io,join((op,T,n,kd,trans,method,BenchmarkTools.median(trial).time/1e9,minimum(trial).time/1e9,trial.memory,trial.allocs,length(trial),first.time,compile,error),','));flush(io)
    println((op,T,n,trans,method));flush(stdout)
end
open(joinpath(outdir,"compact-$(backend).csv"),"w") do io
    println(io,"operation,type,n,bandwidth,trans,algorithm,median_s,minimum_s,bytes,allocations,samples,first_call_s,compile_s,relative_error")
    for T in (Float32,Float64,ComplexF32,ComplexF64), n in sizes, op in (:gbmv,:sbmv,:hbmv,:tbmv,:tbsv,:spmv,:hpmv,:tpmv,:tpsv,:spr,:spr2,:hpr,:hpr2), trans in ('N','C')
        band=op in (:gbmv,:sbmv,:hbmv,:tbmv,:tbsv)
        herm=op in (:hbmv,:hpmv,:hpr,:hpr2)
        symmetric=op in (:sbmv,:spmv,:spr,:spr2)
        T<:Real && herm && continue
        n>1024 && (!band || symmetric && T<:Complex) && continue
        rank=op in (:spr,:spr2,:hpr,:hpr2); rank2=op in (:spr2,:hpr2)
        triangular=op in (:tbmv,:tbsv,:tpmv,:tpsv); solve=op in (:tbsv,:tpsv)
        trans=='C' && !(triangular || op===:gbmv) && continue
        kd=min(n-1,8)
        kind=op===:gbmv ? :general : triangular ? :triangular : herm ? :hermitian : :symmetric
        if n<=1024
            A=randn(T,n,n)/n; A[LA.diagind(A)].+=T(2)
            P=band ? band_matrix(A;kl=kd,ku=kd,kind,uplo=:U) : pack_matrix(A;kind,uplo=:U)
        else
            rows=kind===:general ? 2kd+1 : kd+1
            data=randn(T,rows,n)/n
            P=BandedMatrix(data,n;kl=kd,ku=kd,kind,uplo=:U)
            P.data[P.ku+1,:].+=T(2)
        end
        oldP=old_compact(P); initial_data=copy(P.data)
        x=randn(T,n); z=randn(T,n); initial=randn(T,n)
        α=op===:hpr ? T(.7) : T<:Real ? T(.7) : T(.7+.3im); β=T(-.2)
        # Direct BLAS references for the standard scalar/type combinations.
        portable=!(symmetric && T<:Complex)
        if n<=1024
            D=Matrix(P); aa=trans=='C' ? adjoint(D) : D
            rhs(v)=herm ? adjoint(v) : transpose(v)
            expected=rank ? D+(rank2 ? α*x*rhs(z)+(herm ? conj(α) : α)*z*rhs(x) : α*x*rhs(x)) :
                     triangular ? (solve ? aa\initial : aa*initial) : α*aa*x+β*initial
        else
            expected=copy(initial)
            if triangular
                getfield(Main,Symbol(:reference_,op,:!))(expected,P,trans)
            else
                reference_compact_mv!(op,expected,P,x,α,β,trans)
            end
        end
        methods=backend=="openblas" ? (n<=1024 ? (:v04,:native,:reference_blas) : (:native,:reference_blas)) : (:reference_blas,)
        symbol_trans=trans=='N' ? :N : :C
        for method in methods
            method===:reference_blas && !portable && continue
            current=method===:v04 ? oldP : P
            buffer=copy(initial)
            setup=rank ? ()->copyto!(current.data,initial_data) : ()->copyto!(buffer,initial)
            if method===:reference_blas
                ref_func=rank || triangular ? getfield(Main,Symbol(:reference_,op,:!)) : nothing
                f=rank ? (rank2 ? ()->ref_func(current,x,z,α) : ()->ref_func(current,x,op===:hpr ? real(α) : α)) :
                  triangular ? ()->ref_func(buffer,current,trans) : ()->reference_compact_mv!(op,buffer,current,x,α,β,trans)
            else
                mod=method===:v04 ? V04Compact : OxiBLAS; func=getfield(mod,Symbol(op,:!))
                f=rank ? (rank2 ? ()->func(current,x,z,α) : ()->func(current,x,α)) :
                  triangular ? ()->func(buffer,current;trans=symbol_trans) :
                  op===:gbmv ? ()->func(buffer,current,x,α,β;trans=symbol_trans) : ()->func(buffer,current,x,α,β)
            end
            verify(_)=LA.norm((rank ? Matrix(current) : buffer)-expected)/max(LA.norm(expected),eps(real(T)))
            measure(io,op,T,n,band ? kd : n-1,trans,method,f,setup,verify)
        end
    end
end
