# SPDX-License-Identifier: Apache-2.0
# Column approximate minimum degree ordering (COLAMD), a port of colamd.c
# from SuiteSparse: COLAMD, Copyright (c) 1998-2022, Timothy A. Davis and
# Stefan Larimore, SPDX-License-Identifier: BSD-3-Clause. Algorithm of
# Davis, Gilbert, Larimore and Ng, ACM TOMS 30(3), 2004.
#
# Orders the columns of A so that the Cholesky factor of AᵀA (hence the R of
# QR and the L, U of LU with partial pivoting) stays sparse, without forming
# AᵀA: the rows of A are the initial elements; columns are eliminated by
# approximate minimum degree with row merging, aggressive row absorption,
# supercolumn detection and mass elimination; dense rows and columns are set
# aside (more than max(16, 10·√n) entries) and dense columns are ordered last.
#
# The port keeps colamd.c's 0-based conventions internally (row and column
# ids, pointers into the workspace A, EMPTY = -1); arrays are accessed at
# index+1. The Col and Row structures are split into one array per field
# (the unions keep their names: thickness/parent, score/order, prev/hash/
# headhash, degree_next/hash_next; degree/p, mark/first_column).

const _CA_EMPTY=-1
const _CA_DEAD_PRINCIPAL=-1
const _CA_DEAD_NON_PRINCIPAL=-2

mutable struct _Colamd
    A::Vector{Int}                 # workspace: column form, then row form
    # Col
    cstart::Vector{Int}; clength::Vector{Int}
    cthick::Vector{Int}            # thickness (alive) / parent (dead)
    cscore::Vector{Int}            # score (alive) / order (dead)
    cprev::Vector{Int}             # prev / hash / headhash
    cnext::Vector{Int}             # degree_next / hash_next
    # Row
    rstart::Vector{Int}; rlength::Vector{Int}
    rdeg::Vector{Int}              # degree / p
    rmark::Vector{Int}             # mark / first_column
end
@inline _ca_row_alive(S,r)=S.rmark[r+1]>=0
@inline _ca_col_alive(S,c)=S.cstart[c+1]>=0

"""    colamd_ordering(A)

Column approximate minimum degree ordering of A (COLAMD, ported from
SuiteSparse): a column permutation `q` (`q[new] = old`) that keeps the
factors of AᵀA, the R of A[:,q] = QR and the L, U of A[p,q] = LU sparse.
Stored zeros are ignored."""
function colamd_ordering(A::NativeSparse)
    C=_as_csc(A); m,n=size(C)
    n==0 && return Int[]
    # Pattern by columns, 0-based rows, sorted, without stored zeros.
    nnz=0
    @inbounds for j in 1:n, t in C.colptr[j]:C.colptr[j+1]-1
        iszero(C.values[t]) || (nnz+=1)
    end
    Alen=2nnz+n+nnz÷5+16
    Aw=Vector{Int}(undef,Alen); p=Vector{Int}(undef,n+1); x=0
    @inbounds for j in 1:n
        p[j]=x
        for t in C.colptr[j]:C.colptr[j+1]-1
            iszero(C.values[t]) && continue
            Aw[x+1]=Int(C.rowind[t])-1; x+=1
        end
    end
    p[n+1]=x
    S=_Colamd(Aw,zeros(Int,n+1),zeros(Int,n+1),zeros(Int,n+1),zeros(Int,n+1),zeros(Int,n+1),zeros(Int,n+1),
              zeros(Int,m+1),zeros(Int,m+1),zeros(Int,m+1),zeros(Int,m+1))
    _ca_init_rows_cols!(S,m,n,p)
    n_col2,max_deg=_ca_init_scoring!(S,m,n,p)     # p is the degree list head from here on
    _ca_find_ordering!(S,m,n,Alen,p,n_col2,max_deg,2nnz)
    _ca_order_children!(S,n,p)
    [p[k]+1 for k in 1:n]
end

# Column and row structures; the row form of A follows the column form.
# The input columns are sorted and without duplicates.
function _ca_init_rows_cols!(S::_Colamd,n_row,n_col,p)
    A=S.A
    @inbounds begin
        for col in 0:n_col-1
            S.cstart[col+1]=p[col+1]; S.clength[col+1]=p[col+2]-p[col+1]
            S.cthick[col+1]=1; S.cscore[col+1]=0
            S.cprev[col+1]=_CA_EMPTY; S.cnext[col+1]=_CA_EMPTY
        end
        for row in 0:n_row-1
            S.rlength[row+1]=0; S.rmark[row+1]=-1
        end
        for col in 0:n_col-1, q in p[col+1]:p[col+2]-1
            S.rlength[A[q+1]+1]+=1
        end
        if n_row>0
            S.rstart[1]=p[n_col+1]; S.rdeg[1]=S.rstart[1]; S.rmark[1]=-1
            for row in 1:n_row-1
                S.rstart[row+1]=S.rstart[row]+S.rlength[row]
                S.rdeg[row+1]=S.rstart[row+1]; S.rmark[row+1]=-1
            end
        end
        for col in 0:n_col-1, q in p[col+1]:p[col+2]-1
            r=A[q+1]; A[S.rdeg[r+1]+1]=col; S.rdeg[r+1]+=1
        end
        for row in 0:n_row-1
            S.rmark[row+1]=0; S.rdeg[row+1]=S.rlength[row+1]
        end
    end
    S
end

# Kills dense or empty columns and rows, initial scores, degree lists.
function _ca_init_scoring!(S::_Colamd,n_row,n_col,head)
    A=S.A
    dense_row_count=max(16,floor(Int,10*sqrt(Float64(n_col))))
    dense_col_count=max(16,floor(Int,10*sqrt(Float64(min(n_row,n_col)))))
    max_deg=0; n_col2=n_col; n_row2=n_row
    @inbounds begin
        # Empty columns last, in their natural order.
        for c in n_col-1:-1:0
            if S.clength[c+1]==0
                n_col2-=1; S.cscore[c+1]=n_col2; S.cstart[c+1]=_CA_DEAD_PRINCIPAL
            end
        end
        # Dense columns last, in their natural order.
        for c in n_col-1:-1:0
            _ca_col_alive(S,c) || continue
            if S.clength[c+1]>dense_col_count
                n_col2-=1; S.cscore[c+1]=n_col2
                for q in S.cstart[c+1]:S.cstart[c+1]+S.clength[c+1]-1
                    S.rdeg[A[q+1]+1]-=1
                end
                S.cstart[c+1]=_CA_DEAD_PRINCIPAL
            end
        end
        for r in 0:n_row-1
            deg=S.rdeg[r+1]
            if deg>dense_row_count || deg==0
                S.rmark[r+1]=-1; n_row2-=1
            else
                max_deg=max(max_deg,deg)
            end
        end
        # Initial scores: Σ (row degree - 1) over the live rows of the column.
        for c in n_col-1:-1:0
            _ca_col_alive(S,c) || continue
            score=0; cp=S.cstart[c+1]; new_cp=cp
            for q in cp:cp+S.clength[c+1]-1
                row=A[q+1]
                _ca_row_alive(S,row) || continue
                A[new_cp+1]=row; new_cp+=1
                score+=S.rdeg[row+1]-1; score=min(score,n_col)
            end
            col_length=new_cp-cp
            if col_length==0
                n_col2-=1; S.cscore[c+1]=n_col2; S.cstart[c+1]=_CA_DEAD_PRINCIPAL
            else
                S.clength[c+1]=col_length; S.cscore[c+1]=score
            end
        end
        for c in 0:n_col
            head[c+1]=_CA_EMPTY
        end
        # Reverse order: low column indices first in each list (ties).
        for c in n_col-1:-1:0
            _ca_col_alive(S,c) || continue
            score=S.cscore[c+1]; next_col=head[score+1]
            S.cprev[c+1]=_CA_EMPTY; S.cnext[c+1]=next_col
            next_col!=_CA_EMPTY && (S.cprev[next_col+1]=c)
            head[score+1]=c
        end
    end
    n_col2,max_deg
end

function _ca_clear_mark!(S::_Colamd,tag_mark,max_mark,n_row)
    if tag_mark<=0 || tag_mark>=max_mark
        @inbounds for r in 0:n_row-1
            _ca_row_alive(S,r) && (S.rmark[r+1]=0)
        end
        tag_mark=1
    end
    tag_mark
end

# Orders the principal columns (approximate minimum degree with row merging,
# aggressive absorption, supercolumns). Returns the number of garbage
# collections.
function _ca_find_ordering!(S::_Colamd,n_row,n_col,Alen,head,n_col2,max_deg,pfree)
    A=S.A
    max_mark=typemax(Int)-n_col
    tag_mark=_ca_clear_mark!(S,0,max_mark,n_row)
    min_score=0; ngarbage=0; k=0
    @inbounds while k<n_col2
        # Select the pivot column.
        while min_score<n_col && head[min_score+1]==_CA_EMPTY
            min_score+=1
        end
        pivot_col=head[min_score+1]
        next_col=S.cnext[pivot_col+1]; head[min_score+1]=next_col
        next_col!=_CA_EMPTY && (S.cprev[next_col+1]=_CA_EMPTY)
        pivot_col_score=S.cscore[pivot_col+1]
        S.cscore[pivot_col+1]=k     # order
        pivot_col_thickness=S.cthick[pivot_col+1]
        k+=pivot_col_thickness
        # Garbage collection if needed.
        needed_memory=min(pivot_col_score,n_col-k)
        if pfree+needed_memory>=Alen
            pfree=_ca_garbage_collection!(S,n_row,n_col,pfree); ngarbage+=1
            tag_mark=_ca_clear_mark!(S,0,max_mark,n_row)
        end
        # Pivot row pattern: union of the live rows of the pivot column.
        pivot_row_start=pfree; pivot_row_degree=0
        S.cthick[pivot_col+1]=-pivot_col_thickness
        for q in S.cstart[pivot_col+1]:S.cstart[pivot_col+1]+S.clength[pivot_col+1]-1
            row=A[q+1]
            _ca_row_alive(S,row) || continue
            for t in S.rstart[row+1]:S.rstart[row+1]+S.rlength[row+1]-1
                col=A[t+1]; col_thickness=S.cthick[col+1]
                if col_thickness>0 && _ca_col_alive(S,col)
                    S.cthick[col+1]=-col_thickness
                    A[pfree+1]=col; pfree+=1
                    pivot_row_degree+=col_thickness
                end
            end
        end
        S.cthick[pivot_col+1]=pivot_col_thickness
        max_deg=max(max_deg,pivot_row_degree)
        # Kill the rows of the pivot column.
        for q in S.cstart[pivot_col+1]:S.cstart[pivot_col+1]+S.clength[pivot_col+1]-1
            S.rmark[A[q+1]+1]=-1
        end
        pivot_row_length=pfree-pivot_row_start
        pivot_row=pivot_row_length>0 ? A[S.cstart[pivot_col+1]+1] : _CA_EMPTY
        # Approximate degree: set differences of the rows of each column of
        # the pivot row with the pivot row.
        for t in pivot_row_start:pivot_row_start+pivot_row_length-1
            col=A[t+1]
            col_thickness=-S.cthick[col+1]; S.cthick[col+1]=col_thickness
            # Remove the column from its degree list.
            cur_score=S.cscore[col+1]; prev_col=S.cprev[col+1]; next_col=S.cnext[col+1]
            if prev_col==_CA_EMPTY
                head[cur_score+1]=next_col
            else
                S.cnext[prev_col+1]=next_col
            end
            next_col!=_CA_EMPTY && (S.cprev[next_col+1]=prev_col)
            for q in S.cstart[col+1]:S.cstart[col+1]+S.clength[col+1]-1
                row=A[q+1]; row_mark=S.rmark[row+1]
                row_mark<0 && continue
                set_difference=row_mark-tag_mark
                set_difference<0 && (set_difference=S.rdeg[row+1])
                set_difference-=col_thickness
                if set_difference==0
                    S.rmark[row+1]=-1        # aggressive absorption
                else
                    S.rmark[row+1]=set_difference+tag_mark
                end
            end
        end
        # Column scores and hashing for supercolumn detection.
        for t in pivot_row_start:pivot_row_start+pivot_row_length-1
            col=A[t+1]; hash=0; cur_score=0
            cp=S.cstart[col+1]; new_cp=cp
            for q in cp:cp+S.clength[col+1]-1
                row=A[q+1]; row_mark=S.rmark[row+1]
                row_mark<0 && continue
                A[new_cp+1]=row; new_cp+=1
                hash+=row
                cur_score+=row_mark-tag_mark; cur_score=min(cur_score,n_col)
            end
            S.clength[col+1]=new_cp-cp
            if S.clength[col+1]==0
                # Mass elimination: absorbed by the pivot element.
                S.cstart[col+1]=_CA_DEAD_PRINCIPAL
                pivot_row_degree-=S.cthick[col+1]
                S.cscore[col+1]=k; k+=S.cthick[col+1]
            else
                S.cscore[col+1]=cur_score
                hash%=n_col+1
                head_column=head[hash+1]
                if head_column>_CA_EMPTY
                    first_col=S.cprev[head_column+1]; S.cprev[head_column+1]=col
                else
                    first_col=-(head_column+2); head[hash+1]=-(col+2)
                end
                S.cnext[col+1]=first_col; S.cprev[col+1]=hash
            end
        end
        _ca_detect_super_cols!(S,n_col,head,pivot_row_start,pivot_row_length)
        S.cstart[pivot_col+1]=_CA_DEAD_PRINCIPAL
        tag_mark=_ca_clear_mark!(S,tag_mark+max_deg+1,max_mark,n_row)
        # Finalize the new pivot row and the column scores.
        new_rp=pivot_row_start
        for t in pivot_row_start:pivot_row_start+pivot_row_length-1
            col=A[t+1]
            _ca_col_alive(S,col) || continue
            A[new_rp+1]=col; new_rp+=1
            A[S.cstart[col+1]+S.clength[col+1]+1]=pivot_row; S.clength[col+1]+=1
            cur_score=S.cscore[col+1]+pivot_row_degree
            max_score=n_col-k-S.cthick[col+1]
            cur_score-=S.cthick[col+1]
            cur_score=min(cur_score,max_score)
            S.cscore[col+1]=cur_score
            next_col=head[cur_score+1]
            S.cnext[col+1]=next_col; S.cprev[col+1]=_CA_EMPTY
            next_col!=_CA_EMPTY && (S.cprev[next_col+1]=col)
            head[cur_score+1]=col
            min_score=min(min_score,cur_score)
        end
        if pivot_row_degree>0
            S.rstart[pivot_row+1]=pivot_row_start
            S.rlength[pivot_row+1]=new_rp-pivot_row_start
            S.rdeg[pivot_row+1]=pivot_row_degree
            S.rmark[pivot_row+1]=0
        end
    end
    ngarbage
end

# Supercolumns among the columns of the pivot row (hash buckets).
function _ca_detect_super_cols!(S::_Colamd,n_col,head,row_start,row_length)
    A=S.A
    @inbounds for t in row_start:row_start+row_length-1
        col=A[t+1]
        _ca_col_alive(S,col) || continue
        hash=S.cprev[col+1]; head_column=head[hash+1]
        first_col=head_column>_CA_EMPTY ? S.cprev[head_column+1] : -(head_column+2)
        super_c=first_col
        while super_c!=_CA_EMPTY
            length=S.clength[super_c+1]; prev_c=super_c
            c=S.cnext[super_c+1]
            while c!=_CA_EMPTY
                if S.clength[c+1]!=length || S.cscore[c+1]!=S.cscore[super_c+1]
                    prev_c=c; c=S.cnext[c+1]; continue
                end
                cp1=S.cstart[super_c+1]; cp2=S.cstart[c+1]; same=true
                for i in 0:length-1
                    A[cp1+i+1]!=A[cp2+i+1] && (same=false; break)
                end
                if !same
                    prev_c=c; c=S.cnext[c+1]; continue
                end
                # c joins the supercolumn super_c.
                S.cthick[super_c+1]+=S.cthick[c+1]
                S.cthick[c+1]=super_c                 # parent
                S.cstart[c+1]=_CA_DEAD_NON_PRINCIPAL
                S.cscore[c+1]=_CA_EMPTY               # order
                S.cnext[prev_c+1]=S.cnext[c+1]
                c=S.cnext[c+1]
            end
            super_c=S.cnext[super_c+1]
        end
        if head_column>_CA_EMPTY
            S.cprev[head_column+1]=_CA_EMPTY
        else
            head[hash+1]=_CA_EMPTY
        end
    end
    S
end

# Compacts the live columns and rows of the workspace; returns the new pfree.
function _ca_garbage_collection!(S::_Colamd,n_row,n_col,pfree)
    A=S.A; pdest=0
    @inbounds begin
        for c in 0:n_col-1
            _ca_col_alive(S,c) || continue
            psrc=S.cstart[c+1]; S.cstart[c+1]=pdest
            for j in 0:S.clength[c+1]-1
                r=A[psrc+j+1]
                _ca_row_alive(S,r) && (A[pdest+1]=r; pdest+=1)
            end
            S.clength[c+1]=pdest-S.cstart[c+1]
        end
        # Mark the start of each live row with its ones' complement.
        for r in 0:n_row-1
            if !_ca_row_alive(S,r) || S.rlength[r+1]==0
                S.rmark[r+1]=-1
            else
                psrc=S.rstart[r+1]
                S.rmark[r+1]=A[psrc+1]           # first_column
                A[psrc+1]=-r-1
            end
        end
        psrc=pdest
        while psrc<pfree
            if A[psrc+1]<0
                r=-A[psrc+1]-1
                A[psrc+1]=S.rmark[r+1]
                S.rstart[r+1]=pdest
                for j in 0:S.rlength[r+1]-1
                    c=A[psrc+1]; psrc+=1
                    _ca_col_alive(S,c) && (A[pdest+1]=c; pdest+=1)
                end
                S.rlength[r+1]=pdest-S.rstart[r+1]
            else
                psrc+=1
            end
        end
    end
    pdest
end

# Orders the non-principal columns after their principal one (parent tree);
# p[k] = k-th column (0-based).
function _ca_order_children!(S::_Colamd,n_col,p)
    @inbounds begin
        for i in 0:n_col-1
            if S.cstart[i+1]!=_CA_DEAD_PRINCIPAL && S.cscore[i+1]==_CA_EMPTY
                parent=i
                while true
                    parent=S.cthick[parent+1]
                    S.cstart[parent+1]==_CA_DEAD_PRINCIPAL && break
                end
                c=i; order=S.cscore[parent+1]
                while true
                    S.cscore[c+1]=order; order+=1
                    S.cthick[c+1]=parent
                    c=S.cthick[c+1]
                    S.cscore[c+1]==_CA_EMPTY || break
                end
                S.cscore[parent+1]=order
            end
        end
        for c in 0:n_col-1
            p[S.cscore[c+1]+1]=c
        end
    end
    p
end
