# SPDX-License-Identifier: Apache-2.0
# Static contiguous ownership. A task may migrate without changing its range.
@inline _task_range(n,worker,workers)=fld((worker-1)*n,workers)+1:fld(worker*n,workers)

function _foreach_range(f,n,workers)
    workers=min(n,workers)
    workers<=1 && return f(1:n,1)
    @sync for worker in 1:workers
        let worker=worker
            Threads.@spawn f(_task_range(n,worker,workers),worker)
        end
    end
    nothing
end

@inline function _operation_workers(n,work,threaded,threshold)
    threaded && work>=threshold ? min(n,Threads.nthreads()) : 1
end
