# Prepared only. Execute after explicit approval of a named campaign.
using OxiBLAS, BenchmarkTools, Random, InteractiveUtils
import LinearAlgebra as LA
length(ARGS)==1 || error("usage: incremental_svd.jl NEW_OUTPUT_DIRECTORY")
out=abspath(ARGS[1]); ispath(out) && error("refusing to overwrite results")
mkpath(out); Random.seed!(826); LA.BLAS.set_num_threads(1)
open(joinpath(out,"environment.txt"),"w") do io
    versioninfo(io;verbose=true)
    println(io,"Native threads=",Threads.nthreads(),"; OxiBLAS=",pkgversion(OxiBLAS))
    println(io,"Reference backend=",LA.BLAS.get_config(),"; reference threads=",LA.BLAS.get_num_threads())
    println(io,"Data seed=826; projection seed=0x0b1a5; evals=1; <=7 samples; <=0.15s/case; single samples may exceed budget.")
    println(io,"Each update uses a fresh copied state in untimed setup. First call includes that copy.")
    println(io,"References use the stored approximation plus the batch, not discarded original directions.")
end
function measure_isvd(io,name,T,m,n,r,p,base,update,augmented)
    first=@timed update(copy(base)); snapshot=svd_snapshot(first.value)
    reference=LA.svd(augmented;full=false); k=min(base.maxrank,length(reference.S))
    expected=reference.U[:,1:k]*LA.Diagonal(reference.S[1:k])*reference.Vt[1:k,:]
    residual=LA.norm(snapshot.U*LA.Diagonal(snapshot.S)*snapshot.Vt-expected)/max(LA.norm(expected),eps(real(T)))
    orthogonality=max(LA.norm(snapshot.U'*snapshot.U-LA.I),LA.norm(snapshot.Vt*snapshot.Vt'-LA.I))
    limit=1000max(m,n)*eps(real(T))
    residual<=limit && orthogonality<=limit || error("incremental SVD reference guard failed: $((name,T,m,n,residual,orthogonality))")
    # setup runs outside the measured update, preventing repeated dimension growth.
    trial=run(@benchmarkable($update(state),setup=(state=copy($base)),evals=1);seconds=0.15,samples=7)
    compile=hasproperty(first,:compile_time) ? first.compile_time : NaN
    println(io,join((name,T,m,n,r,p,Threads.nthreads(),base.reorthogonalize,residual,orthogonality,
                    snapshot.last_truncation_norm,snapshot.last_projection_residual,first.time,compile,
                    BenchmarkTools.median(trial).time/1e9,minimum(trial).time/1e9,
                    trial.memory,trial.allocs,length(trial),Base.summarysize(first.value)),','))
    flush(io)
end
function measure_isvd_init(io,name,T,m,n,r,initialize,reference)
    first=@timed initialize(); state=svd_snapshot(first.value)
    residual=LA.norm(state.U*LA.Diagonal(state.S)*state.Vt-reference)/max(LA.norm(reference),eps(real(T)))
    orthogonality=max(LA.norm(state.U'*state.U-LA.I),LA.norm(state.Vt*state.Vt'-LA.I))
    limit=1000max(m,n)*eps(real(T))
    residual<=limit && orthogonality<=limit || error("initialization reference guard failed: $((name,T,m,n,residual,orthogonality))")
    trial=run(@benchmarkable($initialize(),evals=1);seconds=0.15,samples=7)
    compile=hasproperty(first,:compile_time) ? first.compile_time : NaN
    println(io,join((name,T,m,n,r,0,Threads.nthreads(),true,residual,orthogonality,
                    state.last_truncation_norm,state.last_projection_residual,first.time,compile,
                    BenchmarkTools.median(trial).time/1e9,minimum(trial).time/1e9,
                    trial.memory,trial.allocs,length(trial),Base.summarysize(first.value)),','))
    flush(io)
end
open(joinpath(out,"incremental-svd.csv"),"w") do io
    println(io,"case,type,m,n,rank,batch,threads,reorthogonalize,relative_reference_error,orthogonality,local_truncation_norm,local_projection_residual,first_including_copy_s,compile_s,median_update_s,min_update_s,bytes,allocations,samples,retained_state_bytes")
    for T in (Float32,Float64,ComplexF32,ComplexF64), (m,n,r) in ((64,32,4),(256,128,8)), p in (1,8)
        A=randn(T,m,r)*randn(T,r,n)
        for reorthogonalize in (false,true)
            base=incremental_svd(A;rank=r,maxrank=r,method=:randomized,oversampling=4,reorthogonalize)
            initial=svd_snapshot(base); represented=initial.U*LA.Diagonal(initial.S)*initial.Vt
            rows=randn(T,p,n); columns=randn(T,m,p)
            measure_isvd(io,:append_rows,T,m,n,r,p,base,F->svd_add_rows!(F,rows),vcat(represented,rows))
            measure_isvd(io,:append_columns,T,m,n,r,p,base,F->svd_add_columns!(F,columns),hcat(represented,columns))
            # No new residual direction: still requires the coupling SVD.
            rows=randn(T,p,r)*initial.Vt
            columns=initial.U*randn(T,r,p)
            measure_isvd(io,:within_span_rows,T,m,n,r,p,base,F->svd_add_rows!(F,rows),vcat(represented,rows))
            measure_isvd(io,:within_span_columns,T,m,n,r,p,base,F->svd_add_columns!(F,columns),hcat(represented,columns))
            p==1 && measure_isvd(io,:reset_state,T,m,n,r,0,base,
                                 F->initialize_svd!(F,A;rank=r,method=:randomized,oversampling=4),A)
            p==1 && measure_isvd(io,:recondition,T,m,n,r,0,base,
                                 F->reorthogonalize_svd!(F),represented)
        end
    end
    for T in (Float32,Float64,ComplexF32,ComplexF64), (m,n,r) in ((64,32,4),(128,64,8))
        A=randn(T,m,r)*randn(T,r,n)
        for method in (:jacobi,:randomized)
            measure_isvd_init(io,Symbol("initialize_dense_",method),T,m,n,r,
                              ()->incremental_svd(A;rank=r,method,oversampling=4),A)
        end
        diagonal=zeros(T,m,n)
        for i in 1:r
            diagonal[i,i]=T(i+1)
        end
        sparse=CSRMatrix(diagonal)
        for method in (:randomized,:lanczos)
            measure_isvd_init(io,Symbol("initialize_sparse_",method),T,m,n,r,
                              ()->incremental_svd(sparse;rank=r,method,oversampling=4),diagonal)
        end
    end
end
