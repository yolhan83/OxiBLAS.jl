using OxiBLAS, Random, Statistics
import LinearAlgebra as LA
LA.BLAS.set_num_threads(1)
const O=OxiBLAS
@eval OxiBLAS function _svd_block_trial!(B,V,rtol,b,passes)
    m,n=size(B);blocks=cld(n,b);slots=iseven(blocks) ? blocks : blocks+1
    pairs=slots÷2;schedule=collect(1:slots);flags=zeros(UInt8,1)
    ws=[_svd_block_workspace(eltype(B),m,n,b,true)];sweeps=0
    for _ in 1:passes
        changed=false
        for _ in 1:slots-1
            changed=_svd_block_round!(B,V,b,rtol,schedule,flags,ws,blocks,pairs,1) | changed
            tail=schedule[end]
            for k in slots:-1:3
                schedule[k]=schedule[k-1]
            end
            schedule[2]=tail
        end
        sweeps+=1;!changed && break
    end
    sweeps+_jacobi_svd!(B,V,rtol,100-sweeps)
end
function trial(A,b,passes)
    T=eltype(A);R=real(T);B,scale=O._normalized_copy(A);n=size(A,2);V=O._identity(T,n)
    sweeps=O._svd_block_trial!(B,V,8eps(R)*sqrt(R(n)),b,passes)
    (;B,V,sweeps,scale)
end
for T in (Float32,Float64,ComplexF32,ComplexF64),n in parse.(Int,split(get(ENV,"OXIBLAS_SVD_TUNE_SIZES","192,512"),','))
    string(T) in split(get(ENV,"OXIBLAS_SVD_TUNE_TYPES","Float64,ComplexF32"),',') || continue
    A=randn(MersenneTwister(0x5ad),T,n,n)
    for (b,passes) in ((16,0),(8,3),(16,3),(16,6),(32,3),(32,6))
        trial(A,b,passes);rs=[@timed(trial(A,b,passes)) for _ in 1:3]
        result=last(rs).value
        residual=LA.norm(A-result.B*result.V'*result.scale)/LA.norm(A)
        residual<128eps(real(T))*n || error("block rotation residual $residual")
        println("JACOBI_TUNE ",T," n=",n," block=",b," passes=",passes,
                " seconds=",median(getproperty.(rs,:time))," bytes=",minimum(getproperty.(rs,:bytes)),
                " total_passes=",result.sweeps," residual=",residual)
        flush(stdout)
    end
end
