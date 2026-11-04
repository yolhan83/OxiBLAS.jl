# SPDX-License-Identifier: Apache-2.0
struct GemmTuning
    config::GemmConfig
    kernel::Symbol
    element_type::String
    dimensions::NTuple{3,Int}
    cpu::String
    timings_ns::Vector{Float64}
    candidates::Vector{GemmConfig}
end
_tuning_cpu()=string(Sys.ARCH,':',Sys.CPU_NAME,':',VERSION,':',Threads.nthreads())

"""Explicit, opt-in native GEMM calibration. Calling this function runs timings;
loading the package never calibrates or benchmarks. Candidate results are checked
against the native reference before their measured time is accepted.
"""
function autotune_gemm(::Type{T},m::Integer,n::Integer,k::Integer;kernel=:auto,threaded=true,repeats=3,
                       candidates=[GemmConfig(T;mc=mc,kc=kc,nc=nc) for mc in (96,192,384,576), kc in (128,256,448), nc in (256,1024,2048)]) where {T<:NativeFloat}
    min(m,n,k)>0 && repeats isa Integer && repeats>0 && !isempty(candidates) || throw(ArgumentError("invalid tuning dimensions/options"))
    kind=_kernel_kind(kernel); A=Matrix{T}(undef,m,k); B=Matrix{T}(undef,k,n)
    for i in eachindex(A)
        A[i]=T((mod(17i,101)-50)/101)
    end
    for i in eachindex(B)
        B[i]=T((mod(29i,97)-48)/97)
    end
    C=Matrix{T}(undef,m,n); reference=similar(C); gemm_reference!(reference,A,B)
    timings=Float64[]
    for config in candidates
        w=GemmWorkspace(T,m,n,k;config,workers=threaded ? Threads.nthreads() : 1)
        gemm!(C,A,B;kernel=kind,threaded,workspace=w,config)
        tolerance=64eps(real(T))*max(1,k)*max(one(real(T)),maximum(abs,reference))
        maximum(abs(C[i]-reference[i]) for i in eachindex(C))<=tolerance || throw(ArgumentError("GEMM tuning candidate failed numerical validation"))
        samples=Float64[]
        for _ in 1:repeats
            start=time_ns(); gemm!(C,A,B;kernel=kind,threaded,workspace=w,config)
            push!(samples,Float64(time_ns()-start))
        end
        sort!(samples); push!(timings,samples[cld(repeats,2)])
    end
    best=argmin(timings)
    GemmTuning(candidates[best],kind,string(T),(Int(m),Int(n),Int(k)),_tuning_cpu(),timings,collect(candidates))
end

"""Save an explicitly measured configuration in a simple versioned text file."""
function save_gemm_tuning(path::AbstractString,tuning::GemmTuning)
    open(path,"w") do io
        println(io,"OxiBLAS tuning 1"); println(io,tuning.cpu); println(io,tuning.element_type)
        println(io,tuning.kernel); println(io,join(tuning.dimensions,','))
        c=tuning.config
        println(io,join((c.mc,c.kc,c.nc,c.parallel_threshold,c.small_threshold),','))
    end
    path
end
"""Load tuning only for the same CPU, Julia version, thread count and type.
Returns the config/kernel/shape; it never starts a new calibration.
"""
function load_gemm_tuning(::Type{T},path::AbstractString) where {T<:NativeFloat}
    lines=readlines(path)
    length(lines)==6 && lines[1]=="OxiBLAS tuning 1" || throw(ArgumentError("invalid GEMM tuning file"))
    lines[2]==_tuning_cpu() && lines[3]==string(T) || throw(ArgumentError("GEMM tuning environment differs"))
    kind=_kernel_kind(Symbol(lines[4])); dims=parse.(Int,split(lines[5],',')); c=parse.(Int,split(lines[6],','))
    length(dims)==3 && length(c)==5 && minimum(dims)>0 || throw(ArgumentError("invalid tuning dimensions/config"))
    config=GemmConfig(T;mc=c[1],kc=c[2],nc=c[3],parallel_threshold=c[4],small_threshold=c[5])
    (;config,kernel=kind,dimensions=Tuple(dims))
end
