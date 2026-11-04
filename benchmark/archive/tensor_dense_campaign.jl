# Dense tensor campaign, in place: OxiBLAS einsum!(C, plan, ...; workspace) against TensorOperations (TTGT on
# LinearAlgebra's BLAS, OpenBLAS by default) and Tullio (generated loops).
# Same data and same process for all three; plans are built outside the timed
# region and the output is preallocated (@tensor/@tullio C[...] = ..., in place);
# each result is checked against the first reference before timing; one
# CSV line is flushed per case.
#
# Run (one thread, OpenBLAS on one thread):
#   julia -t1 --project=<env with OxiBLAS, TensorOperations, Tullio, BenchmarkTools> \
#         benchmark/archive/tensor_dense_campaign.jl OUT.csv [case,case,...] [types]
using OxiBLAS, TensorOperations, Tullio, BenchmarkTools, Random
import LinearAlgebra as LA
LA.BLAS.set_num_threads(1)
const OUT=ARGS[1]
const SELECT=length(ARGS)>=2 && !isempty(ARGS[2]) ? Symbol.(split(ARGS[2],',')) : nothing
const TYPES=length(ARGS)>=3 ? [eval(Meta.parse(t)) for t in split(ARGS[3],',')] : [Float64,ComplexF64,Float32]
const SECONDS=parse(Float64,get(ENV,"OXI_TENSOR_SECONDS","0.4"))

_time(f)=(f(); BenchmarkTools.median(run(@benchmarkable($f(),evals=1);seconds=SECONDS,samples=200)).time/1e6)
relerr(X,Y)=LA.norm(X-Y)/max(LA.norm(Y),eps(real(eltype(Y))))

# Each case: name, spec, operand builder, TensorOperations call, Tullio call.
# Dimensions: comparable work (≈ 1e7-1e8 multiply-adds) except the `small_*`
# cases, which measure per-call overhead.
cases(T)=[
 (:gemm_384,"ik,kj->ij",()->(randn(T,384,384),randn(T,384,384)),
   (C,A,B)->(@tensor C[i,j]=A[i,k]*B[k,j]),(C,A,B)->(@tullio C[i,j]=A[i,k]*B[k,j])),
 (:abc_cd_abd,"abc,cd->abd",()->(randn(T,48,48,48),randn(T,48,48)),
   (C,A,B)->(@tensor C[a,b,d]=A[a,b,c]*B[c,d]),(C,A,B)->(@tullio C[a,b,d]=A[a,b,c]*B[c,d])),
 (:abc_bd_acd,"abc,bd->acd",()->(randn(T,48,48,48),randn(T,48,48)),
   (C,A,B)->(@tensor C[a,c,d]=A[a,b,c]*B[b,d]),(C,A,B)->(@tullio C[a,c,d]=A[a,b,c]*B[b,d])),
 (:ijk_jkl_il,"ijk,jkl->il",()->(randn(T,96,32,32),randn(T,32,32,96)),
   (C,A,B)->(@tensor C[i,l]=A[i,j,k]*B[j,k,l]),(C,A,B)->(@tullio C[i,l]=A[i,j,k]*B[j,k,l])),
 (:abcd_cdef_abef,"abcd,cdef->abef",()->(randn(T,16,16,16,16),randn(T,16,16,16,16)),
   (C,A,B)->(@tensor C[a,b,e,f]=A[a,b,c,d]*B[c,d,e,f]),(C,A,B)->(@tullio C[a,b,e,f]=A[a,b,c,d]*B[c,d,e,f])),
 (:abcd_ebfd_aecf,"abcd,ebfd->aecf",()->(randn(T,16,16,16,16),randn(T,16,16,16,16)),
   (C,A,B)->(@tensor C[a,e,c,f]=A[a,b,c,d]*B[e,b,f,d]),(C,A,B)->(@tullio C[a,e,c,f]=A[a,b,c,d]*B[e,b,f,d])),
 # TensorOperations has no batch (hyper) index: the first reference of the
 # two batched cases is a loop of LinearAlgebra.mul! (OpenBLAS) over b.
 (:batched_bij_bjk,"bij,bjk->bik",()->(randn(T,32,48,48),randn(T,32,48,48)),
   (C,A,B)->(for b in 1:32; LA.mul!(view(C,b,:,:),view(A,b,:,:),view(B,b,:,:)); end; C),
   (C,A,B)->(@tullio C[b,i,k]=A[b,i,j]*B[b,j,k])),
 (:batched_ijb_jkb,"ijb,jkb->ikb",()->(randn(T,48,48,32),randn(T,48,48,32)),
   (C,A,B)->(for b in 1:32; LA.mul!(view(C,:,:,b),view(A,:,:,b),view(B,:,:,b)); end; C),
   (C,A,B)->(@tullio C[i,k,b]=A[i,j,b]*B[j,k,b])),
 (:mode2_abc_db_adc,"abc,db->adc",()->(randn(T,48,48,48),randn(T,48,48)),
   (C,A,B)->(@tensor C[a,d,c]=A[a,b,c]*B[d,b]),(C,A,B)->(@tullio C[a,d,c]=A[a,b,c]*B[d,b])),
 (:outer_i_j,"i,j->ij",()->(randn(T,1024),randn(T,1024)),
   (C,A,B)->(@tensor C[i,j]=A[i]*B[j]),(C,A,B)->(@tullio C[i,j]=A[i]*B[j])),
 (:small_ab_bc,"ab,bc->ac",()->(randn(T,8,8),randn(T,8,8)),
   (C,A,B)->(@tensor C[a,c]=A[a,b]*B[b,c]),(C,A,B)->(@tullio C[a,c]=A[a,b]*B[b,c])),
 (:small_abc_bd_acd,"abc,bd->acd",()->(randn(T,6,6,6),randn(T,6,6)),
   (C,A,B)->(@tensor C[a,c,d]=A[a,b,c]*B[b,d]),(C,A,B)->(@tullio C[a,c,d]=A[a,b,c]*B[b,d])),
]
chain(T)=(:chain_ij_jk_kl,"ij,jk,kl->il",()->(randn(T,192,192),randn(T,192,192),randn(T,192,192)),
          (D,A,B,C)->(@tensor D[i,l]=A[i,j]*B[j,k]*C[k,l]),(D,A,B,C)->(@tullio D[i,l]=A[i,j]*B[j,k]*C[k,l]))

Random.seed!(20261104)
open(OUT,"w") do io
    println(io,"case,type,oxiblas_ms,ref1_ms,tullio_ms,ratio_vs_ref1,ratio_vs_best,relerr,oxiblas_bytes")   # ref1 = TensorOperations (OpenBLAS loop for batched_*)
    flush(io)
    for T in TYPES, (name,spec,build,fto,ftu) in vcat(cases(T),[chain(T)])
        SELECT===nothing || name in SELECT || continue
        X=build(); plan=plan_einsum(spec,X...); ws=EinsumWorkspace(T)
        shape=Tuple(plan.dimensions[l] for l in plan.output)
        Cr=zeros(T,shape); Co=zeros(T,shape); Ct=zeros(T,shape)
        fto(Cr,X...); einsum!(Co,plan,X...;workspace=ws)
        e=relerr(Co,Cr)
        e<=1000*eps(real(T))*max(1,length(Cr))^(1/4) || error("OxiBLAS result differs: $name $T err=$e")
        fox=()->einsum!(Co,plan,X...;workspace=ws)
        tox=_time(fox); tto=_time(()->fto(Cr,X...)); ttu=_time(()->ftu(Ct,X...))
        fox(); bytes=@allocated fox()
        line=join((name,T,round(tox;sigdigits=4),round(tto;sigdigits=4),round(ttu;sigdigits=4),
                   round(tox/tto;digits=2),round(tox/min(tto,ttu);digits=2),round(e;sigdigits=2),bytes),',')
        println(io,line); flush(io); println(line); flush(stdout)
    end
end
