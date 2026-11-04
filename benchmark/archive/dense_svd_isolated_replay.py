"""Prepare, then explicitly execute an isolated baseline/current/reference replay.

Default invocation only prints the plan. Execution additionally requires
--execute and OXIBLAS_SVD_QUIET_WINDOW=1 after coordination with other agents.
"""
import argparse,datetime,hashlib,json,os,pathlib,shutil,subprocess,sys,uuid

ROOT=pathlib.Path(__file__).resolve().parents[2]
OUT=ROOT/'reports/dense_svd_20261104'

def fingerprints(base=ROOT):
 paths=list((base/'src').rglob('*.jl'))
 paths += [base/p for p in ('Project.toml','benchmark/Project.toml',
           'benchmark/Manifest.toml','benchmark/archive/full_backend.jl',
           'benchmark/archive/dense_svd_worker.jl','benchmark/archive/dense_svd_campaign.py',
           'reports/dense_svd_20261104/svd.jl.before',
           'reports/dense_svd_20261104/svd_advanced.jl.before')]
 return {str(p.relative_to(base)):hashlib.sha256(p.read_bytes()).hexdigest()
         for p in sorted(paths) if p.exists()}

def checked_rows(path):
 lines=path.read_text().splitlines()
 commands=sum(line.startswith('COMMAND ') for line in lines)
 exits=[line for line in lines if line.startswith('EXIT ')]
 if not commands or len(exits)!=commands or any(line!='EXIT 0' for line in exits):
  raise RuntimeError('Incomplete or failed worker: '+str(path))
 if any('SOURCE_CHANGED_DURING_WORKER' in line for line in lines):
  raise RuntimeError('Sources changed during measurements: '+str(path))
 rows=[json.loads(line.split('\t',1)[1]) for line in lines if line.startswith('@@DSVD\t')]
 if not rows or any(r['status']!='OK' for r in rows):
  raise RuntimeError('Missing measurements or numerical guard failed: '+str(path))
 return rows

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--execute',action='store_true')
 args=parser.parse_args()
 print('Plan: original SVD methods (Julia only), then current Julia/OpenBLAS/Accelerate;')
 print('4 types; N=64,192; square/tall/wide; vectors/values; requested threads 1,2,4,8; 7 samples.')
 print('Exactly one measurement worker at a time; source changes abort publication.')
 if not args.execute:
  print('Prepared only. No Julia process started.');return
 if os.environ.get('OXIBLAS_SVD_QUIET_WINDOW')!='1':
  parser.error('A coordinated quiet window is required: set OXIBLAS_SVD_QUIET_WINDOW=1 only after confirmation.')
 OUT.mkdir(exist_ok=True)
 stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8]
 initial=fingerprints();logs={};frozen=OUT/('frozen_'+stamp)
 for folder in ('src','ext','deps'):
  if (ROOT/folder).exists():shutil.copytree(ROOT/folder,frozen/folder)
 for name in initial:
  if name.startswith('src/'):continue
  target=frozen/name;target.parent.mkdir(parents=True,exist_ok=True)
  shutil.copy2(ROOT/name,target)
 if initial!=fingerprints() or initial!=fingerprints(frozen):
  raise RuntimeError('Sources changed while snapshot was copied; restart before measuring')
 settings=dict(OXIBLAS_SVD_THREADS='1,2,4,8',OXIBLAS_SVD_SAMPLES='7',
               OXIBLAS_SVD_SECONDS='0.20',OXIBLAS_SVD_SIZES='64,192',
               OXIBLAS_SVD_TYPES='Float32,Float64,ComplexF32,ComplexF64',
               OXIBLAS_SVD_SHAPES='square,tall,wide',OXIBLAS_SVD_SELECT='jacobi,golub_kahan')
 for phase,baseline,backends in (('before','1','julia'),('after','0','julia,openblas,accelerate')):
  if fingerprints(frozen)!=initial:raise RuntimeError('Frozen sources changed before '+phase)
  label='isolated_'+stamp+'_'+phase
  env=os.environ.copy();env.update(settings,OXIBLAS_SVD_BASELINE=baseline,OXIBLAS_SVD_BACKENDS=backends,OXIBLAS_SVD_SOURCE_ROOT=str(frozen))
  subprocess.run([sys.executable,'benchmark/archive/dense_svd_campaign.py',label],cwd=ROOT,env=env,check=True)
  paths=list(OUT.glob(label+'_*.log'))
  if len(paths)!=1:raise RuntimeError('Expected one unambiguous campaign log for '+phase)
  checked_rows(paths[0]);logs[phase]=paths[0]
  if fingerprints(frozen)!=initial:raise RuntimeError('Frozen sources changed during '+phase)
 before=checked_rows(logs['before']);after=checked_rows(logs['after'])
 key=lambda r:tuple(r[k] for k in ('algorithm','type','m','columns','values_only','threads'))
 old={key(r):r for r in before if r['backend']=='julia'}
 new={key(r):r for r in after if r['backend']=='julia'}
 if len(old)!=384 or old.keys()!=new.keys():raise RuntimeError('Incomplete native case grid')
 for k,r in new.items():
  if r['fingerprint']!=old[k]['fingerprint']:raise RuntimeError('Input mismatch')
  for backend in (('openblas','accelerate') if sys.platform=='darwin' else ('openblas',)):
   matches=[v for v in after if v['backend']==backend and all(v[f]==r[f] for f in
             ('type','m','columns','values_only','threads','fingerprint'))]
   expected={'gesdd_values'} if r['values_only'] else {'gesdd','gesvd'}
   if {v['algorithm'] for v in matches}!=expected:raise RuntimeError('Missing equivalent reference')
 record={'status':'validated_isolated_rerun','run_id':stamp,'quiet_window':'operator_confirmed',
         'recorded_at':datetime.datetime.now().astimezone().isoformat(),
         'source_sha256':initial,
         'frozen_source_root':str(frozen),'live_sources_after':fingerprints(),
         'selected_logs':{phase:[{'name':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}]
                          for phase,p in logs.items()},
         'note':'Only these complete logs qualify. Earlier provisional timings remain archived.'}
 manifest=OUT/'MEASUREMENT_VALIDITY.json'
 if manifest.exists():
  (OUT/('MEASUREMENT_VALIDITY_before_'+stamp+'.json')).write_bytes(manifest.read_bytes())
 temporary=OUT/('validity_'+stamp+'.tmp');temporary.write_text(json.dumps(record,indent=2)+'\n')
 temporary.replace(manifest)
 subprocess.run([sys.executable,'benchmark/archive/dense_svd_report.py'],cwd=ROOT,check=True)

if __name__=='__main__':main()
