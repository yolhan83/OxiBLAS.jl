# Definitions only. Optional imports happen only when a future authorized
# compatibility/benchmark campaign explicitly calls optional_float_types().
function optional_float_types()
    requested=filter(!isempty,split(get(ENV,"OXIBLAS_OPTIONAL_FLOATS",""),','))
    all(name->name in ("DoubleFloats","Quadmath","available"),requested) || error("unknown optional float provider")
    isempty(requested) && return DataType[]
    available="available" in requested; types=DataType[]
    for name in ("DoubleFloats","Quadmath")
        (available || name in requested) || continue
        if Base.find_package(name)===nothing
            available || error("requested optional provider $name is not installed in this test environment")
            @info "Optional floating compatibility not prepared on this environment" provider=name reason="package absent"
            continue
        end
        try
            if name=="DoubleFloats"
                @eval import DoubleFloats
                R=DoubleFloats.Double64
            else
                @eval import Quadmath
                R=Quadmath.Float128
            end
            R<:AbstractFloat || error("provider type must implement AbstractFloat")
            push!(types,R,Complex{R})
        catch error
            available || rethrow()
            @info "Optional floating compatibility skipped" provider=name reason=sprint(showerror,error)
        end
    end
    types
end
