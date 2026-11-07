# SPDX-License-Identifier: Apache-2.0

# Query Julia's code-generation target rather than just the physical CPU. This
# also respects julia -C generic and prevents dispatch to an unsupported ISA.
@static if isdefined(Base, :cpu_supports)
    @inline _avx2() = Sys.ARCH === :x86_64 && Base.cpu_supports(:avx2) && Base.cpu_supports(:fma)
    @inline _avx512() = Sys.ARCH === :x86_64 && Base.cpu_supports(:avx512f) && Base.cpu_supports(:fma)
    @inline _neon() = Sys.ARCH === :aarch64 && Base.cpu_supports(:neon)
else
    # Older Julia: conservative fallback. No unsafe CPUID assumptions.
    @inline _avx2() = false
    @inline _avx512() = false
    @inline _neon() = Sys.ARCH === :aarch64
end

@inline function _kernel_kind(request::Symbol)
    request in (:auto, :scalar, :neon, :avx2, :avx512) || throw(ArgumentError("unknown kernel: $request"))
    request === :scalar && return :scalar
    available = _avx512() ? :avx512 : _avx2() ? :avx2 : _neon() ? :neon : :scalar
    request === :auto && return available
    ok = request === :neon ? _neon() : request === :avx2 ? _avx2() : _avx512()
    ok || throw(ArgumentError("kernel $request is unavailable for this Julia CPU target"))
    request
end

@inline function _shape(::Type{T}, kind) where T
    if T === Float64
        kind === :neon && return (2, 4, 6) # W, MR/W, NR; upstream 8×6
        kind === :avx2 && return (4, 2, 6)
        kind === :avx512 && return (8, 2, 6)
    elseif T === Float32
        kind === :neon && return (4, 2, 8) # upstream 8×8
        kind === :avx2 && return (8, 1, 8)
        kind === :avx512 && return (16, 1, 16)
    elseif T === ComplexF64
        # The complex micro-kernel holds two accumulators per output vector:
        # 24 of NEON's 32 registers (60-62 GF/s on M5 Max; 4x4 spilled at 39),
        # 12 of AVX2's 16 (x86 shapes unmeasured).
        kind === :neon && return (1, 4, 3)
        kind === :avx2 && return (2, 2, 3)
        kind === :avx512 && return (4, 2, 6)
    elseif T === ComplexF32
        kind === :neon && return (2, 2, 4)
        kind === :avx2 && return (4, 1, 4)
        kind === :avx512 && return (8, 1, 8)
    end
    (1, 4, 4)
end

"""
    kernel_info(::Type{T}=Float64; kernel=:auto)

Report selected kernel, vector width, and micro-panel dimensions.
"""
function kernel_info(::Type{T}=Float64; kernel=:auto) where T
    kind = _kernel_kind(kernel)
    w, r, nr = _shape(T, kind)
    (; architecture=Sys.ARCH, cpu=Sys.CPU_NAME, kind, lanes=w, mr=w*r, nr)
end
