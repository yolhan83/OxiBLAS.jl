"""Dense Jacobi/Golub-Kahan only. One worker at a time, isolated BLAS providers."""
import os,sys,subprocess,pathlib,json,hashlib,datetime
ROOT=pathlib.Path(__file__).resolve().parents[2]
SOURCE_ROOT=pathlib.Path(os.environ.get('OXIBLAS_SVD_SOURCE_ROOT',str(ROOT))).resolve()
PHASE=sys.argv[1] if len(sys.argv)>1 else 'before'
OUT=ROOT/'reports/dense_svd_20261104';OUT.mkdir(exist_ok=True)
threads=[int(t) for t in os.environ.get('OXIBLAS_SVD_THREADS','1,4,18').split(',')]
report=OUT/(PHASE+'_'+datetime.datetime.now().strftime('%H%M%S')+'.log')
failed=False
with report.open('x') as log:
 def write(s):log.write(s+'\n');log.flush()
 write('UTC '+datetime.datetime.now(datetime.timezone.utc).isoformat())
 write('SOURCE_ROOT '+str(SOURCE_ROOT))
 for p in sorted((SOURCE_ROOT/'src').glob('*.jl')):write('SOURCE_SHA256 '+p.name+' '+hashlib.sha256(p.read_bytes()).hexdigest())
 for t in threads:
  for backend in os.environ.get('OXIBLAS_SVD_BACKENDS','julia,openblas,accelerate').split(','):
   if backend=='accelerate' and sys.platform!='darwin':write('Accelerate N/A: non-macOS');continue
   env=os.environ.copy();env.update(OXIBLAS_FULL_BACKEND=backend,OXIBLAS_FULL_THREADS=str(t),OPENBLAS_NUM_THREADS=str(t if backend!='julia' else 1),VECLIB_MAXIMUM_THREADS=str(t),JULIA_NUM_PRECOMPILE_TASKS='1')
   cache=ROOT.parents[1]/'work/julia-depot-static'
   if cache.exists():env.setdefault('JULIA_DEPOT_PATH',os.pathsep.join((str(cache),str(pathlib.Path.home()/'.julia'))))
   cmd=[os.environ.get('OXIBLAS_JULIA','julia'),'--startup-file=no',f'--project={SOURCE_ROOT / "benchmark"}',f'--threads={t}','--check-bounds=auto',str(SOURCE_ROOT/'benchmark/archive/dense_svd_worker.jl')]
   write('COMMAND '+repr(cmd));print(PHASE,backend,t,'threads',flush=True)
   source_before={str(p.relative_to(SOURCE_ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((SOURCE_ROOT/'src').glob('*.jl'))}
   write('WORKER_SOURCE_BEFORE '+json.dumps(source_before,sort_keys=True))
   with subprocess.Popen(cmd,cwd=SOURCE_ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True) as p:
    for line in p.stdout:write(line.rstrip())
    code=p.wait();write('EXIT '+str(code));failed|=code!=0;print('EXIT',code,flush=True)
   source_after={str(p.relative_to(SOURCE_ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((SOURCE_ROOT/'src').glob('*.jl'))}
   write('WORKER_SOURCE_AFTER '+json.dumps(source_after,sort_keys=True))
   if source_before!=source_after:
    write('SOURCE_CHANGED_DURING_WORKER: measurements require review')
    raise RuntimeError('Sources changed during worker; remaining measurements cancelled')
   if code!=0:
    raise RuntimeError('Worker failed; remaining measurements cancelled')
print('REPORT',report,flush=True)
raise SystemExit(1 if failed else 0)
