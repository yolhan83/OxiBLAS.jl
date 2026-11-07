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

# ==== Spectral Autotuning (SVD & Schur) =========================================

struct SpectralConfig
    svd_algorithm::Symbol
    schur_algorithm::Symbol
    svd_bidiag_threshold::Int
    schur_multishift_threshold::Int
end

struct SpectralTuning
    config::SpectralConfig
    element_type::String
    dimension::Int
    cpu::String
    svd_timings_ns::Dict{Symbol,Float64}
    schur_timings_ns::Dict{Symbol,Float64}
end

const _ACTIVE_SPECTRAL_TUNING = Dict{DataType,SpectralConfig}()

function get_spectral_config(::Type{T}) where T
    get(_ACTIVE_SPECTRAL_TUNING, T, nothing)
end

function set_spectral_tuning!(tuning::SpectralTuning)
    T = eval(Meta.parse(tuning.element_type))
    _ACTIVE_SPECTRAL_TUNING[T] = tuning.config
    tuning
end

function clear_spectral_tuning!()
    empty!(_ACTIVE_SPECTRAL_TUNING)
end

"""Explicit, opt-in native spectral algorithm calibration (SVD and Schur).
Runs timings for candidate algorithms on an n × n matrix; loading the package
never calibrates or benchmarks. Candidate results are checked numerically before acceptance.
"""
function autotune_spectral(::Type{T}, n::Integer=128; threaded::Bool=true, repeats::Integer=3) where {T<:NativeFloat}
    n > 0 && repeats > 0 || throw(ArgumentError("invalid spectral tuning dimension/options"))
    
    # 1. SVD Calibration
    A = Matrix{T}(undef, n, n)
    for i in eachindex(A)
        A[i] = T((mod(17i, 101) - 50) / 101)
    end
    
    svd_cands = (:golub_kahan, :jacobi, :divide_conquer)
    svd_timings = Dict{Symbol,Float64}()
    
    for algo in svd_cands
        try
            F = svd_bidiagonal(copy(A); algorithm=algo, threaded)
            tol = 64eps(real(T)) * sqrt(real(T)(n))
            err = norm(F.U * _LA.Diagonal(F.S) * F.Vt - A) / max(norm(A), floatmin(real(T)))
            err <= tol || continue
            
            samples = Float64[]
            for _ in 1:repeats
                start = time_ns()
                svd_bidiagonal(copy(A); algorithm=algo, threaded)
                push!(samples, Float64(time_ns() - start))
            end
            sort!(samples)
            svd_timings[algo] = samples[cld(repeats, 2)]
        catch
        end
    end
    
    best_svd = isempty(svd_timings) ? :golub_kahan : argmin(svd_timings)
    
    # 2. Schur Calibration
    schur_cands = (:standard, :multishift)
    schur_timings = Dict{Symbol,Float64}()
    
    for algo in schur_cands
        try
            F = if algo === :multishift
                schur_multishift(copy(A); threaded)
            else
                schur_factor(copy(A); algorithm=:standard, threaded)
            end
            tol = 64eps(real(T)) * max(1, n)
            err = norm(F.Z * F.T * adjoint(F.Z) - A) / max(norm(A), floatmin(real(T)))
            err <= tol || continue
            
            samples = Float64[]
            for _ in 1:repeats
                start = time_ns()
                if algo === :multishift
                    schur_multishift(copy(A); threaded)
                else
                    schur_factor(copy(A); algorithm=:standard, threaded)
                end
                push!(samples, Float64(time_ns() - start))
            end
            sort!(samples)
            schur_timings[algo] = samples[cld(repeats, 2)]
        catch
        end
    end
    
    best_schur = isempty(schur_timings) ? (n >= 192 ? :multishift : :standard) : argmin(schur_timings)
    
    svd_cutoff = (T <: Complex) ? 4 : 8
    schur_cutoff = (T <: Complex) ? 192 : 256
    
    config = SpectralConfig(best_svd, best_schur, svd_cutoff, schur_cutoff)
    tuning = SpectralTuning(config, string(T), Int(n), _tuning_cpu(), svd_timings, schur_timings)
    set_spectral_tuning!(tuning)
    tuning
end

"""Save an explicitly measured spectral configuration in a simple versioned text file."""
function save_spectral_tuning(path::AbstractString, tuning::SpectralTuning)
    open(path, "w") do io
        println(io, "OxiBLAS spectral tuning 1")
        println(io, tuning.cpu)
        println(io, tuning.element_type)
        println(io, tuning.dimension)
        c = tuning.config
        println(io, join((c.svd_algorithm, c.schur_algorithm, c.svd_bidiag_threshold, c.schur_multishift_threshold), ','))
    end
    path
end

"""Load spectral tuning only for the same CPU, Julia version, thread count and type.
Returns the SpectralConfig and activates it for automatic spectral dispatch.
"""
function load_spectral_tuning(::Type{T}, path::AbstractString) where {T<:NativeFloat}
    lines = readlines(path)
    length(lines) == 5 && lines[1] == "OxiBLAS spectral tuning 1" || throw(ArgumentError("invalid spectral tuning file"))
    lines[2] == _tuning_cpu() && lines[3] == string(T) || throw(ArgumentError("spectral tuning environment differs"))
    dim = parse(Int, lines[4])
    parts = split(lines[5], ',')
    length(parts) == 4 || throw(ArgumentError("invalid spectral tuning parameters"))
    svd_algo = Symbol(parts[1])
    schur_algo = Symbol(parts[2])
    svd_thresh = parse(Int, parts[3])
    schur_thresh = parse(Int, parts[4])
    config = SpectralConfig(svd_algo, schur_algo, svd_thresh, schur_thresh)
    tuning = SpectralTuning(config, string(T), dim, _tuning_cpu(), Dict{Symbol,Float64}(), Dict{Symbol,Float64}())
    set_spectral_tuning!(tuning)
    config
end
