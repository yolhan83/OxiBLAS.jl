"""Read source text only; write traceable inventories. No Julia/Rust execution.
Regex/brace inventories do not expand Rust macros, resolve Julia dispatch or
certify numerical equivalence. Unreconciled declarations stay explicit.
"""
from pathlib import Path
import csv
import hashlib
import re

ROOT=Path(__file__).resolve().parents[3]
RUST=ROOT.parents[1]/'work'/'oxiblas-upstream'
REV='8d82bb6bf1df133aa90525aa9f478ebe41514f9b'


def csv_write(path,fields,rows):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def tokens(text,name):
    return re.search(r'(?<![A-Za-z_0-9])'+re.escape(name)+r'(?![A-Za-z_0-9!])',text) is not None


def rust_code_mask(text):
    """Preserve offsets/newlines while masking comments and Rust literals."""
    result=list(text);i=0
    raw_pattern=re.compile(r'(?:br|r)(\#*)"')
    while i<len(text):
        start=i;end=None
        if text.startswith('//',i):
            found=text.find('\n',i);end=len(text) if found<0 else found
        elif text.startswith('/*',i):
            depth=1;j=i+2
            while j<len(text) and depth:
                if text.startswith('/*',j):depth+=1;j+=2
                elif text.startswith('*/',j):depth-=1;j+=2
                else:j+=1
            end=j
        else:
            raw=raw_pattern.match(text,i) if text[i] in 'rb' else None
            if raw:
                marker='"'+raw.group(1);found=text.find(marker,i+len(raw.group()))
                end=len(text) if found<0 else found+len(marker)
            elif text[i]=='"':
                j=i+1
                while j<len(text):
                    if text[j]=='\\':j+=2;continue
                    if text[j]=='"':j+=1;break
                    j+=1
                end=j
            elif text[i]=="'":
                literal=re.match(r"'(?:\\.|[^'\\\n])'",text[i:])
                if literal:end=i+len(literal.group())
        if end is None:i+=1;continue
        for j in range(start,end):
            if result[j]!='\n':result[j]=' '
        i=end
    return ''.join(result)


def rust_declarations(text):
    code=rust_code_mask(text)
    explicit=list(re.finditer(r'(?m)^[ \t]*pub(?:\([^\n)]*\))?\s+(?:(?:unsafe|async|const)\s+)*(?:extern\s+[^\n]*?\s+)?fn\s+([A-Za-z_][A-Za-z_0-9]*)',code))
    result=[(match,'pub fn') for match in explicit]
    for trait in re.finditer(r'\bpub\s+(?:unsafe\s+)?trait\s+([A-Za-z_][A-Za-z_0-9]*)[^{}]*\{',code):
        start=trait.end();depth=1;end=start
        while end<len(code) and depth:
            if code[end]=='{':depth+=1
            elif code[end]=='}':depth-=1
            end+=1
        body=code[start:end-1]
        for method in re.finditer(r'(?m)^[ \t]*(?:(?:unsafe|async|const)\s+)*fn\s+([A-Za-z_][A-Za-z_0-9]*)',body):
            prefix=body[:method.start()]
            if prefix.count('{')!=prefix.count('}'):continue
            # Re-match against the full masked source to keep global offsets.
            position=start+method.start()
            full=re.compile(r'[ \t]*(?:(?:unsafe|async|const)\s+)*fn\s+([A-Za-z_][A-Za-z_0-9]*)').match(code,position)
            result.append((full,'trait '+trait.group(1)))
    return sorted(result,key=lambda item:item[0].start())


def main():
    source=(ROOT/'src/OxiBLAS.jl').read_text().split('export ',1)[1].split('const NativeFloat',1)[0]
    # Strip comment words so every row denotes an actual exported identifier.
    source='\n'.join(line.split('#',1)[0] for line in source.splitlines())
    exports=list(dict.fromkeys(re.findall(r'\b[A-Za-z_][A-Za-z_0-9!]*',source)))
    julia={str(p.relative_to(ROOT)):p.read_text() for p in sorted((ROOT/'src').glob('*.jl'))}
    tests={str(p.relative_to(ROOT)):p.read_text() for p in sorted((ROOT/'test').glob('*.jl'))}
    bench={str(p.relative_to(ROOT)):p.read_text() for p in sorted((ROOT/'benchmark').glob('*.jl')) if not p.name.startswith('inspect_')}
    aliases={'lu_factor':'lu','cholesky_factor':'cholesky','qr_factor':'qr','ldlt_factor':'ldl',
             'bunchkaufman_factor':'ldl','gemm_extended!':'gemm','svd_bidiagonal':'svd_golub_kahan',
             'svd_divide_conquer':'svd_dc','singular_values':'svd','sparse_svd':'sparse_svd_dilation',
             'block_arnoldi_factor':'block_arnoldi','gemm_reference!':'gemm','gemm_blocked!':'gemm'}
    indirect={'eigen_tridiagonal':'eigen_hermitian','eigen_tridiagonal_dc':'eigen_hermitian_dc',
              'eigen_tridiagonal_mrrr':'eigen_hermitian_mrrr','frechet_matrix':'frechet_exp',
              'svd_rank':'incremental_svd','filter_response':'apply_filter!','bandwidth_profile':'ordering_metrics',
              'sparse_ordering':'select_sparse_ordering','compare_orderings':'select_sparse_ordering'}
    reasons={'autotune_gemm':'autotuning explicitement exclu de la campagne actuelle',
             'save_gemm_tuning':'persistance des réglages; IO hors timing numérique',
             'load_gemm_tuning':'lecture des réglages; IO hors timing numérique',
             'read_matrix_market':'IO dépendant du système de fichiers; tests dédiés, hors comparaison numérique',
             'write_matrix_market':'IO dépendant du système de fichiers; tests dédiés, hors comparaison numérique',
             'generic_threadsafe':'trait de contexte scalaire, pas un kernel numérique',
             'einsum_layout_report':'rapport de métadonnées; coût couvert par planification, pas une référence BLAS',
             'kernel_info':'introspection matérielle, pas une opération arithmétique',
             'ldl_inertia':'diagnostic du facteur; pas isolé du coût de LDL dans cette campagne',
             'apply_hessenberg_q!':'application de reflecteurs stockés; reconstruction Hessenberg déjà mesurée, timing isolé non réconcilié',
             'schur_reorder':'formes réelles et conventions de réordonnancement encore partielles; pas de comparateur sûr de blocs réconcilié',
             'eigen_general':'vecteurs généraux couverts par tests et Schur chronométré; timing isolé exclu de ce catalogue',
             'svd_truncated':'troncature déjà couverte par SVD sélectionnée/rang complet; aucun gain partiel revendiqué',
             'sparse_triangular_solve':'substitution creuse utilisée via solveurs/facteurs; pas isolée dans les timings',
             'sparse_apply_q!':'application sparse-Q couverte via sparse_qr/solve; pas isolée dans les timings',
             'arnoldi':'variante scalaire non isolée; campagne spectrale par blocs existante; pas de référence full-EVD substituée',
             'iram':'redémarrage scalaire non isolé; benchmarks par blocs/thick restart distincts, pas de comparaison artificielle',
             'shift_invert':'coût dépend du facteur et de sa réutilisation; contrat de timing isolé encore exclu',
             'lobpcg':'paramètres de convergence/init/préconditionnement spécifiques; contrat isolé à définir, pas de full-EVD substituée',
             'balance_matrix':'équilibrage traité comme étape auxiliaire; timing isolé exclu des comparaisons de solveurs',
             'balance_backtransform':'transformation des vecteurs dépend du facteur équilibré; timing isolé non préparé'}
    api=[]
    for name in exports:
        locations=[path for path,text in julia.items() if path!='src/OxiBLAS.jl' and tokens(text,name)]
        prepared_tests=[path for path,text in tests.items() if tokens(text,name)]
        stem=name.rstrip('!');alias=aliases.get(name,stem)
        cases=[path for path,text in bench.items() if tokens(text,name) or tokens(text,alias) or re.search(r':'+re.escape(alias)+r'\b',text)]
        if name in reasons:
            status='exclusion justifiée';reason=reasons[name]
        elif name[0].isupper():
            status='exclusion justifiée';reason='type/contexte/exception; coût de construction inclus là où utilisé, pas de microbenchmark de déclaration'
        elif name in indirect:
            status='cas indirect; non isolé';reason='surface réutilisée via '+indirect[name]+'; pas de temps attribué directement à cette API'
            cases=[path for path,text in bench.items() if tokens(text,indirect[name])]
        elif cases:
            status='cas préparé (preuve textuelle)';reason='appel/registre à vérifier lors de la campagne; aucune mesure ni validation inférée de ce scan'
        else:
            status='exclusion provisoire explicite';reason='adaptateur de mesure et contrat équivalent non réconciliés; pas de wrapper de remplacement inventé; couverture ciblée encore requise'
        api.append(dict(api=name,classification=status,julia=';'.join(locations),tests=';'.join(prepared_tests),benchmarks=';'.join(cases),execution='non exécuté',restriction=reason))
    csv_write(ROOT/'docs/BENCHMARK_API_AUDIT.csv',['api','classification','julia','tests','benchmarks','execution','restriction'],api)

    # Conservative per-declaration reconciliation, not a name-only coverage %.
    rows=[]
    direct={name.rstrip('!'):name for name in exports if not name[0].isupper()}
    direct.update({'gemm_transposed':'gemm!','gemm_with_par':'gemm!','contract_2d':'contract','contract_3d_2d':'contract',
                   'mixed_precision_solve':'mixed_solve','trace_hutch_plus_plus':'trace_hutch_plus_plus','trace_xtrace':'trace_xtrace'})
    family={'partial_piv.rs':'lu_factor','full_piv.rs':'lu_complete','rook_piv.rs':'lu_rook','llt.rs':'cholesky_factor',
            'bunch_kaufman.rs':'bunchkaufman_factor','ldlt.rs':'ldlt_factor','symmetric.rs':'eigen_hermitian',
            'hermitian.rs':'eigen_hermitian','complex_schur.rs':'schur_factor','schur.rs':'schur_factor'}
    for path in sorted((RUST/'crates').rglob('*.rs')):
        if '/src/' not in str(path):continue
        text=path.read_text();relative=str(path.relative_to(RUST));digest=hashlib.sha256(path.read_bytes()).hexdigest()
        for match,scope in rust_declarations(text):
            name=match.group(1);line=text.count('\n',0,match.start(1))+1
            base=re.sub(r'_(?:f32|f64|c32|c64)$','',name)
            target=direct.get(base)
            if name in ('compute','compute_blocked','compute_recursive','compute_auto'):
                target=family.get(path.name)
            if name in ('solve','solve_transpose','solve_transposed') and path.name in family:
                target='solve_transposed' if 'transpose' in name else 'solve'
            # Avoid assigning low-level SIMD/FFI methods to a generic API merely
            # because they share a leaf name.
            if '/simd/' in relative or '/ffi/' in relative:target=None
            status='Non réconcilié'
            restrictions='déclaration publique inventoriée; correspondance algorithmique non établie; macros/traits non développés par ce scan'
            if target:
                status='Adapté à Julia'
                restrictions='correspondance candidate de surface, pas preuve de fidélité algorithmique; non validé'
                if '/lapack/' in relative or '/oxiblas-lapack/' in relative:
                    status='Algorithme alternatif'
                    restrictions='algorithmes Julia existants; variantes exactes à réconcilier; non validé'
                if any(key in relative.lower() for key in ('qz','schur','mrrr','svd')):
                    status='Partiellement implémenté'
                    restrictions='variantes QZ réel/réordonnancement/small-bulge/SVD spécialisées non toutes couvertes; voir lots et tableau utilisateur'
                if name=='trace_xtrace':
                    status='Algorithme alternatif';restrictions='contrôle leave-one-out, noyau XTrace optimisé absent'
            entries=[entry for entry in api if entry['api']==target]
            row=dict(revision=REV,rust_path=relative,rust_line=line,rust_function=name,rust_scope=scope,rust_sha256=digest,
                     statut_portage=status,validation='code Julia non validé; correspondance candidate à confirmer' if target else 'aucune validation ni conclusion de présence',
                     julia_api=target or '',julia=entries[0]['julia'] if entries else '',tests=entries[0]['tests'] if entries else '',
                     benchmarks=entries[0]['benchmarks'] if entries else '',restrictions=restrictions)
            rows.append(row)
    csv_write(ROOT/'docs/UPSTREAM_FUNCTION_AUDIT.csv',list(rows[0]),rows)
    # Mechanical risk map: every source is scanned, flags are not findings.
    risks=[]
    patterns={'pointer':r'unsafe_load|unsafe_store!|Ptr\{|pointer\(', 'gc_preserve':r'GC\.@preserve',
              'simd':r'@simd|llvmcall|@generated','tasks':r'Threads\.@spawn|_foreach_range',
              'float_cast':r'Float64\(|Float32\(','inbounds':r'@inbounds'}
    for path,text in julia.items():
        for kind,pattern in patterns.items():
            for match in re.finditer(pattern,text):
                risks.append(dict(julia=path,line=text.count('\n',0,match.start())+1,category=kind,
                                  status='site repéré statiquement; pas une validation',context=text.splitlines()[text.count('\n',0,match.start())].strip()))
    csv_write(ROOT/'docs/LOW_LEVEL_SITES.csv',['julia','line','category','status','context'],risks)
    print('Source-only inventories:',len(api),'exports;',len(rows),'Rust callable declarations (pub fn and public traits);',len(risks),'risk sites.')
    print('Unreconciled Rust declarations:',sum(row['statut_portage']=='Non réconcilié' for row in rows))
    print('Provisional benchmark exclusions:',sum(row['classification']=='exclusion provisoire explicite' for row in api))


if __name__=='__main__':main()
