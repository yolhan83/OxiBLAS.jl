"""Targeted BLAS 1 / BLAS 2 / BLAS 3 campaigns. Workers run one at a time; no full report.

Usage: python3 benchmark/archive/level1_targeted.py [phase]
  OXIBLAS_TARGET=level1 (default) runs benchmark/archive/level1_targeted.jl.
  OXIBLAS_TARGET=level2 runs the BLAS LEVEL 2 group of benchmark/archive/full_worker.jl
    (full_cases.jl / full_extra_cases.jl) on Float32/Float64/ComplexF32/
    ComplexF64; OXIBLAS_L1_SELECT/TYPES/SIZES/SAMPLES/SECONDS are forwarded as
    the corresponding OXIBLAS_FULL_* filters (default sizes 64,256,1024,2048,
    9 samples). BLAS 2 kernels thread internally: in parallel mode each row is
    compared with the same operation at one thread.
  OXIBLAS_TARGET=lapack: the dense factorizations of the LAPACK groups
    (default selection LAPACK_SELECT below; spectral/SVD-based entries are
    excluded); same variables and defaults as level3.
  OXIBLAS_TARGET=sparse runs benchmark/archive/sparse_worker.jl: sparse matrices and
    sparse tensor contractions against SparseArrays (and TensorOperations when
    installed), both measured in one process per thread count (parallel
    default threads 1,6,18). Filters: OXIBLAS_SP_SELECT (operations),
    OXIBLAS_SP_CASES (data sets), OXIBLAS_SP_TYPES; budget OXIBLAS_SP_SECONDS,
    OXIBLAS_SP_SAMPLES. `table` on such a log prints the sparse report table.
  OXIBLAS_TARGET=level3: the same for the BLAS LEVEL 3 group (default sizes
    64,256,1024,2048, 9 samples within 1 s; a call longer than its share of
    the budget is measured with the samples that fit, at least one).
  OXIBLAS_L1_MODE=sequential (default): one Julia/BLAS thread, original protocol.
  OXIBLAS_L1_MODE=parallel: for each count in OXIBLAS_L1_THREADS (default
    "1,2,4,6,12,18"), run every backend with that many Julia threads and the same
    requested OpenBLAS/Accelerate thread count. Sequential and parallel
    OxiBLAS kernels are measured in the same Julia process.
  OXIBLAS_L1_BACKENDS (default "julia,openblas,accelerate") limits backends.
  OXIBLAS_L1_REPORT_DIR (default reports/) receives the log; use a scratch
    directory for trial runs so that only campaign logs land in reports/.
Other OXIBLAS_L1_* variables are forwarded to benchmark/archive/level1_targeted.jl.

Tables: python3 benchmark/archive/level1_targeted.py table AFTER.log [BEFORE.log]
  prints Markdown from the ROWS_JSON line(s). With BEFORE, OxiBLAS medians are
  shown before/after; references come from AFTER (BEFORE if AFTER has none). In parallel logs each
  *_parallel row is also compared with its sequential kernel at equal threads,
  and any other row with the same operation at one thread.
"""
import os,subprocess,datetime,pathlib,json,sys,hashlib
ROOT=pathlib.Path(__file__).resolve().parents[2]
def load(log):
 for line in pathlib.Path(log).read_text().splitlines():
  if line.startswith('ROWS_JSON '):return json.loads(line[10:])
 raise SystemExit(f'{log}: no ROWS_JSON line (unfinished campaign?)')
def table(after,before=None):
 # Row fields: op,type,n,backend,status,median,minimum,bytes,allocs,samples,hash,provider,note[,threads]
 def index(rows):return {(r[0],r[1],int(r[2]),r[3],int(r[13]) if len(r)>13 else 1):r for r in rows}
 A=index(load(after));B=index(load(before)) if before else {}
 def t(I,k):
  r=I.get(k);return float(r[5]) if r and r[4] in ('OK','INACCURATE') else None
 def ns(v,flag=''):return 'N/A' if v is None else (f'{v/1000:.1f} µs' if v>=1e4 else f'{v:.0f} ns')+flag
 keys=sorted({k[:3]+(k[4],) for k in A if k[3]=='julia'},key=lambda k:(k[1],k[0],k[3],k[2]))
 par=any(k[3]>1 for k in keys)
 head=['Opération','Type','N']+(['Threads'] if par else [])+(['OxiBLAS avant','OxiBLAS après','Gain'] if before else ['OxiBLAS'])
 head+=(['vs séquentiel'] if par else [])+['OpenBLAS','Accelerate','OxiBLAS / meilleur']
 out=['| '+' | '.join(head)+' |','|'+'---|'*3+('---:|' if par else '')+'---:|'*(len(head)-3-par)]
 for op,T,n,th in keys:
  now=t(A,(op,T,n,'julia',th));old=t(B,(op,T,n,'julia',th)) if before else None
  R=lambda b:A if (op,T,n,b,th) in A else B # references from AFTER, else from BEFORE
  refs={b:(t(R(b),(op,T,n,b,th)),R(b).get((op,T,n,b,th),['']*5)[4]=='INACCURATE') for b in ('openblas','accelerate')}
  best=min([v for v,_ in refs.values() if v is not None],default=None)
  cells=[op,T,str(n)]+([str(th)] if par else [])
  cells+=[ns(old),ns(now),'N/A' if None in (old,now) else f'{old/now:.2f}x'] if before else [ns(now)]
  if par:
   seq=t(A,(op.removesuffix('_parallel').replace('dot','dotu'),T,n,'julia',th)) if op.endswith('_parallel') else t(A,(op,T,n,'julia',1)) if th>1 else None
   cells.append('' if seq is None and not op.endswith('_parallel') else 'N/A' if None in (seq,now) else f'{seq/now:.2f}x')
  cells+=[ns(v,' †' if bad else '') for v,bad in refs.values()]
  cells.append('N/A' if None in (now,best) else f'{now/best:.2f}x')
  out.append('| '+' | '.join(cells)+' |')
 out+=['','Médianes par appel. « OxiBLAS / meilleur » < 1 : OxiBLAS plus rapide. † : résultat de référence hors tolérance 64 eps (mesuré, mais non comparable en précision).']
 if par:out.append('« vs séquentiel » : accélération sur le noyau séquentiel OxiBLAS (variante *_parallel : même processus et même nombre de threads ; autre opération : même opération à 1 thread).')
 return '\n'.join(out)
def sparse_table(log):
 # One line per OxiBLAS measurement, with the reference measured for the same
 # operation, case, type, thread count and specification (and storage
 # conversion, for conversions).
 rows=load(log)
 def key(r):
  k=[r.get(f) for f in ('op','case','T','threads','spec','k','optimize','phase_detail','trans','rhs','algorithm')]
  if r.get('op') in ('convert','scale_convert'):k.append(r.get('format'))
  return tuple(map(str,k))
 refs={}
 for r in rows:
  if r.get('backend')!='oxiblas' and r.get('status')=='OK':refs.setdefault(key(r),[]).append(r)
 def t(v):return 'N/A' if v is None else (f'{v*1e9:.0f} ns' if v<1e-5 else f'{v*1e6:.1f} µs' if v<1e-2 else f'{v*1e3:.1f} ms' if v<10 else f'{v:.1f} s')
 def b(v):return '' if v is None else (f'{v/2**20:.1f} MiB' if v>=2**20 else f'{v/1024:.1f} KiB' if v>=1024 else f'{v} B')
 head=['Opération','Cas','Type','Dimensions','nnz','Format / spécification','Phase','Threads','OxiBLAS','Alloc. OxiBLAS','Référence','Alloc. réf.','OxiBLAS / réf.','nnz sortie','Densité sortie','Remarques']
 out=['| '+' | '.join(head)+' |','|'+'---|'*len(head)]
 for r in rows:
  if r.get('backend')!='oxiblas':continue
  dims=r.get('dims') or [r.get('m'),r.get('n')]
  spec=' '.join(str(x) for x in (r.get('format'),r.get('spec'),('k='+str(r['k'])) if r.get('k') else None,('optimize='+r['optimize']) if r.get('optimize') else None) if x)
  ref=min(refs.get(key(r),[]),key=lambda x:x['median'],default=None)
  ok=r.get('status')=='OK'
  notes=[]
  if not ok:notes.append(r.get('status','')+': '+str(r.get('note',''))[:80])
  if ref:notes.append('meilleure réf. : '+ref['backend'])
  allref=sorted(refs.get(key(r),[]),key=lambda x:x['backend'])
  if ok and len(allref)>1:notes.append(', '.join(f"{x['backend']} {r['median']/x['median']:.2f}x" for x in allref if x['median']>0))
  if r.get('intermediates'):notes.append('intermédiaires '+'; '.join(f"{i['labels']}:{i['kind']} nnz={i['nnz']} d={i['density']:.2g}" for i in r['intermediates']))
  cells=[r.get('op',''),r.get('case',''),r.get('T',''),'×'.join(map(str,dims)),str(r.get('nnz_in','')),spec,r.get('phase',''),str(r.get('threads','')),
         t(r.get('median')) if ok else r.get('status',''),b(r.get('bytes')) if ok else '',t(ref['median']) if ref else 'N/A',b(ref.get('bytes')) if ref else '',
         f"{r['median']/ref['median']:.2f}x" if ok and ref and ref['median']>0 else '',str(r.get('nnz_out','')),f"{r['density_out']:.3g}" if r.get('density_out') is not None else '','; '.join(notes)]
  out.append('| '+' | '.join(cells)+' |')
 out+=['','Médianes par appel. « OxiBLAS / réf. » < 1 : OxiBLAS plus rapide. Phases : kernel (structures préparées), convert, total (appel complet, conversions comprises), plan, first (première exécution d\'un plan neuf, code déjà compilé), reuse (plan réutilisé).']
 return '\n'.join(out)
if len(sys.argv)>2 and sys.argv[1]=='table':
 rows=load(sys.argv[2])
 print(sparse_table(sys.argv[2]) if rows and isinstance(rows[0],dict) else table(*sys.argv[2:4]));raise SystemExit(0)
JULIA=os.environ.get('OXIBLAS_JULIA','julia')
TARGET=os.environ.get('OXIBLAS_TARGET','level1')
if TARGET not in ('level1','level2','level3','lapack','sparse'):raise SystemExit('OXIBLAS_TARGET must be level1, level2, level3, lapack or sparse')
LAPACK_SELECT=('lu,lu_tall,lu_complete,lu_rook,solve,solve_transposed,inverse,determinant,cholesky,potrs,potri,ldl,aasen,'
 'band_lu,band_cholesky,packed_cholesky,qr,qr_tall,qr_wide,qr_pivoted,qr_q,apply_q,least_squares,lq,ql,rq,cod,'
 'mixed_solve,expert_solve,refine_solve,expert_cholesky,expert_symmetric')
WORKER={'level1':'benchmark/archive/level1_targeted.jl','sparse':'benchmark/archive/sparse_worker.jl'}.get(TARGET,'benchmark/archive/full_worker.jl')
def level2_env():
 g=lambda k,d:os.environ.get('OXIBLAS_L1_'+k) or d
 return dict(OXIBLAS_FULL_GROUPS={'level2':'BLAS LEVEL 2','level3':'BLAS LEVEL 3','lapack':'LAPACK'}[TARGET],OXIBLAS_FULL_SELECT=g('SELECT',LAPACK_SELECT if TARGET=='lapack' else ''),OXIBLAS_FULL_TYPES=g('TYPES',''),
  OXIBLAS_FULL_SIZES=g('SIZES','64,256,1024,2048'),OXIBLAS_FULL_SAMPLES=g('SAMPLES','9'),OXIBLAS_FULL_SECONDS=g('SECONDS','1'),
  # The four native types are gated by the BLAS 2/3/LAPACK test files run with each change.
  OXIBLAS_VALIDATED_TYPES='\n'.join(('Float32','Float64','ComplexF32','ComplexF64')))
def level2_row(line,t):
 # @@OXI_ROW: group,op,T,n,threads,backend,status,median s,minimum s,bytes,allocs,first,compile,gflops,provider,note,fingerprint
 f=line.rstrip('\n').split('\t')[1:]
 sec=lambda v:str(float(v)*1e9)
 return [f[1],f[2],f[3],f[5],f[6],sec(f[7]),sec(f[8]),f[9],f[10],'',f[16] if len(f)>16 else '',f[14],f[15],t]
MODE=os.environ.get('OXIBLAS_L1_MODE','sequential')
if MODE not in ('sequential','parallel'):raise SystemExit('OXIBLAS_L1_MODE must be sequential or parallel')
THREADS=[1] if MODE=='sequential' else [int(t) for t in os.environ.get('OXIBLAS_L1_THREADS','1,6,18' if TARGET=='sparse' else '1,2,4,6,12,18').split(',') if t.strip()]
# The sparse worker measures OxiBLAS and its Julia references in one process.
# Sparse: one process per BLAS behind the SuiteSparse references (OpenBLAS,
# then Accelerate, which also brings Accelerate's native sparse solvers).
BACKENDS=[b.strip() for b in os.environ.get('OXIBLAS_SP_BLAS_LIST','openblas,accelerate').split(',') if b.strip()] if TARGET=='sparse' else [b.strip() for b in os.environ.get('OXIBLAS_L1_BACKENDS','julia,openblas,accelerate').split(',') if b.strip()]
phase=sys.argv[1] if len(sys.argv)>1 else 'initial'
stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
prefix={'level1':'blas1','level2':'blas2','level3':'blas3','lapack':'lapack','sparse':'sparse'}[TARGET]
report=pathlib.Path(os.environ.get('OXIBLAS_L1_REPORT_DIR',ROOT/'reports'))/(f'{prefix}_{phase}_{stamp}.log' if MODE=='sequential' else f'{prefix}_parallel_{phase}_{stamp}.log')
rows=[]
with report.open('x') as log:
 def write(s):log.write(s+'\n');log.flush()
 write(f'{prefix.upper()} {phase}; target={TARGET}; mode={MODE}; threads={THREADS}; one worker process at a time; evals=1; independent batch; UTC '+datetime.datetime.now(datetime.timezone.utc).isoformat())
 for k in sorted(os.environ):
  if k.startswith('OXIBLAS_L1_') or k.startswith('OXIBLAS_SP_') or k=='OXIBLAS_TARGET':write(f'ENV {k}={os.environ[k]}')
 for p in [*ROOT.glob('src/*.jl'),*ROOT.glob({'level1':'benchmark/level1_targeted*','sparse':'benchmark/archive/sparse_worker.jl'}.get(TARGET,'benchmark/full_*.jl'))]:write('SOURCE_SHA256 '+str(p.relative_to(ROOT))+' '+hashlib.sha256(p.read_bytes()).hexdigest())
 failed=False
 for threads in THREADS:
  for backend in BACKENDS:
   t=str(threads)
   env=os.environ.copy();env.update(OXIBLAS_L1_MODE=MODE,OXIBLAS_FULL_BACKEND=backend,OXIBLAS_FULL_THREADS=t,OPENBLAS_NUM_THREADS=t,VECLIB_MAXIMUM_THREADS=t,JULIA_NUM_PRECOMPILE_TASKS='1')
   if TARGET not in ('level1','sparse'):env.update(level2_env())
   if TARGET=='sparse':
    env['OXIBLAS_SP_BLAS']=backend
    # The Accelerate run only adds references for the direct solvers: OxiBLAS
    # (BLAS-independent) and the other families are measured in the OpenBLAS run.
    if backend=='accelerate':env.update(OXIBLAS_SP_REF_ONLY='1',OXIBLAS_SP_DIRECT_ONLY='1')
   cache=ROOT.parents[1]/'work/julia-depot-static'
   if cache.is_dir():env.setdefault('JULIA_DEPOT_PATH',os.pathsep.join((str(cache),str(pathlib.Path.home()/'.julia'))))
   cmd=[JULIA,'--startup-file=no','--project=benchmark',f'--threads={t}','--check-bounds=auto',WORKER]
   write('COMMAND '+repr(cmd));write(f'BACKEND {backend} THREADS {t}')
   with subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True) as p:
    print(backend,'threads',t,'PID',p.pid,flush=True)
    for line in p.stdout:
     write(line.rstrip())
     if line.startswith('@@L1\t'):rows.append(line.rstrip('\n').split('\t')[1:]+[t])
     if line.startswith('@@OXI_ROW\t'):rows.append(level2_row(line,t))
     if line.startswith('@@SPARSE_ROW\t'):rows.append(json.loads(line.split('\t',1)[1]))
    code=p.wait();write('EXIT '+str(code));print(backend,'threads',t,'EXIT',code,flush=True)
   if code:failed=True;print('Worker errors: inspect',report,flush=True)
 # Each row: the @@L1 fields followed by the Julia/BLAS thread count.
 write('ROWS_JSON '+json.dumps(rows))
print('REPORT',report,flush=True)

raise SystemExit(1 if failed else 0)
