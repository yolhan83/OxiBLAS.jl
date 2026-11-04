# SPDX-License-Identifier: Apache-2.0
function _mtx_line(io)
    while !eof(io)
        line=strip(readline(io))
        !isempty(line) && !startswith(line,'%') && return line
    end
    throw(ArgumentError("unexpected end of Matrix Market file"))
end
function read_matrix_market(io::IO;T=nothing,format=:csr)
    eof(io) && throw(ArgumentError("empty Matrix Market input"))
    header=split(lowercase(strip(readline(io))))
    length(header)==5 && header[1]=="%%matrixmarket" && header[2]=="matrix" || throw(ArgumentError("invalid Matrix Market header"))
    storage,field,symmetry=header[3:5]
    storage in ("coordinate","array") && field in ("real","integer","complex","pattern") && symmetry in ("general","symmetric","hermitian","skew-symmetric") || throw(ArgumentError("unsupported Matrix Market header"))
    field=="pattern" && storage=="array" && throw(ArgumentError("pattern arrays are invalid Matrix Market"))
    TT=T===nothing ? (field=="complex" ? ComplexF64 : Float64) : T
    TT<:NativeFloat || throw(ArgumentError("unsupported Matrix Market scalar type"))
    dims=parse.(Int,split(_mtx_line(io))); expected=storage=="coordinate" ? 3 : 2
    length(dims)==expected && all(>=(0),dims) || throw(ArgumentError("invalid Matrix Market dimensions"))
    m,n=dims[1:2]; symmetry!="general" && m!=n && throw(DimensionMismatch("structured Matrix Market matrix must be square"))
    rows=Int[]; cols=Int[]; vals=TT[]
    function value(tokens)
        if field=="pattern"
            isempty(tokens) || throw(ArgumentError("pattern entry has a value"))
            return one(TT)
        elseif field=="complex"
            length(tokens)==2 || throw(ArgumentError("complex entry needs two components"))
            return convert(TT,complex(parse(Float64,replace(tokens[1],'D'=>'e','d'=>'e')),parse(Float64,replace(tokens[2],'D'=>'e','d'=>'e'))))
        else
            length(tokens)==1 || throw(ArgumentError("scalar entry needs one value"))
            return convert(TT,parse(Float64,replace(tokens[1],'D'=>'e','d'=>'e')))
        end
    end
    function entry(i,j,v)
        1<=i<=m && 1<=j<=n || throw(ArgumentError("Matrix Market index out of range"))
        symmetry=="hermitian" && i==j && !isreal(v) && throw(ArgumentError("Hermitian diagonal must be real"))
        symmetry=="skew-symmetric" && i==j && !iszero(v) && throw(ArgumentError("skew diagonal must be zero"))
        push!(rows,i); push!(cols,j); push!(vals,v)
        if i!=j && symmetry!="general"
            push!(rows,j); push!(cols,i)
            push!(vals,symmetry=="hermitian" ? conj(v) : symmetry=="skew-symmetric" ? -v : v)
        end
    end
    if storage=="coordinate"
        for _ in 1:dims[3]
            tokens=split(_mtx_line(io)); length(tokens)>=2 || throw(ArgumentError("invalid coordinate entry"))
            entry(parse(Int,tokens[1]),parse(Int,tokens[2]),value(tokens[3:end]))
        end
    else
        for j in 1:n, i in (symmetry=="general" ? (1:m) : symmetry=="skew-symmetric" ? (j+1:m) : (j:m))
            entry(i,j,value(split(_mtx_line(io))))
        end
    end
    C=COOMatrix(m,n,rows,cols,vals)
    format===:coo && return C
    format===:csr && return CSRMatrix(C)
    format===:csc && return CSCMatrix(C)
    format===:dense && return Matrix(C)
    throw(ArgumentError("Matrix Market output format must be :csr/:csc/:coo/:dense"))
end
read_matrix_market(path::AbstractString;kwargs...)=open(io->read_matrix_market(io;kwargs...),path,"r")
function write_matrix_market(io::IO,A::NativeSparse;symmetry=:general)
    symmetry in (:general,:symmetric,:hermitian,:skew) || throw(ArgumentError("unknown Matrix Market symmetry"))
    m,n=size(A); symmetry!==:general && m!=n && throw(DimensionMismatch("structured output must be square"))
    C=COOMatrix(CSRMatrix(A)); selected=Int[]
    for p in eachindex(C.values)
        i,j=C.rows[p],C.cols[p]; v=C.values[p]
        if symmetry!==:general
            expected=symmetry===:hermitian ? conj(A[j,i]) : symmetry===:skew ? -A[j,i] : A[j,i]
            v==expected || throw(ArgumentError("matrix does not have requested symmetry"))
            i<j && continue
            symmetry===:skew && i==j && continue
        end
        push!(selected,p)
    end
    complex=eltype(A)<:Complex; field=complex ? "complex" : "real"
    structure=symmetry===:skew ? "skew-symmetric" : string(symmetry)
    println(io,"%%MatrixMarket matrix coordinate ",field,' ',structure)
    println(io,m,' ',n,' ',length(selected))
    for p in selected
        print(io,C.rows[p],' ',C.cols[p],' ',repr(Float64(real(C.values[p]))))
        complex && print(io,' ',repr(Float64(imag(C.values[p]))))
        println(io)
    end
    io
end
write_matrix_market(path::AbstractString,A::NativeSparse;kwargs...)=open(io->write_matrix_market(io,A;kwargs...),path,"w")
