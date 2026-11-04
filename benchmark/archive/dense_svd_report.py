"""Build paired comparisons from the isolated dense-SVD worker logs."""
import csv,json,math,pathlib,statistics,sys,hashlib
ROOT=pathlib.Path(__file__).resolve().parents[2]
OUT=ROOT/'reports/dense_svd_20261104'
validity=OUT/'MEASUREMENT_VALIDITY.json'
record=json.loads(validity.read_text()) if validity.exists() else {}
if record.get('status')!='validated_isolated_rerun':
 raise SystemExit('Performance report withheld: earlier measurements are provisional after reported concurrent activity; an isolated baseline/current/reference rerun is required.')
key=lambda r:tuple(r[k] for k in ('type','m','columns','values_only','threads'))
def read(phase):
 rows=[]
 for entry in record['selected_logs'][phase]:
  assert pathlib.Path(entry['name']).name==entry['name'],'Invalid log path'
  path=OUT/entry['name']
  assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256'],'Log changed after validation'
  for line in path.read_text().splitlines():
   if line.startswith('@@DSVD\t'):
    r=json.loads(line.split('\t',1)[1]);r['log']=path.name;rows.append(r)
 return rows
before=read('before')
after=read('after')
assert after,'No final measurements'
assert all(r['status']=='OK' for r in before+after),'A failed measurement requires review'
old={(r['algorithm'],*key(r)):r for r in before if r['backend']=='julia'}
refs={}
for r in after:
 if r['backend']!='julia':
  k=(r['backend'],*key(r))
  if k not in refs or r['median_s']<refs[k]['median_s']:refs[k]=r
paired=[]
for r in after:
 if r['backend']!='julia':continue
 b=old.get((r['algorithm'],*key(r)))
 assert b is not None,('Missing original measurement',key(r))
 assert b['fingerprint']==r['fingerprint'],'Input mismatch'
 ob=refs.get(('openblas',*key(r)));ac=refs.get(('accelerate',*key(r)))
 reference=min((v for v in (ob,ac) if v),key=lambda v:v['median_s'])
 assert reference['fingerprint']==r['fingerprint'],'Reference input mismatch'
 paired.append(dict(algorithm=r['algorithm'],type=r['type'],m=r['m'],n=r['columns'],vectors=not r['values_only'],threads_requested=r['threads'],
  before_ms=1e3*b['median_s'],after_ms=1e3*r['median_s'],speedup=b['median_s']/r['median_s'],
  openblas_ms=1e3*ob['median_s'] if ob else None,openblas_driver=ob['algorithm'] if ob else None,
  accelerate_ms=1e3*ac['median_s'] if ac else None,accelerate_driver=ac['algorithm'] if ac else None,
  best_backend=reference['backend'],best_driver=reference['algorithm'],best_ms=1e3*reference['median_s'],
  ratio=r['median_s']/reference['median_s'],remaining_percent=100*(r['median_s']/reference['median_s']-1),
  target_met=r['median_s']<=1.2*reference['median_s'],before_bytes=b['bytes'],after_bytes=r['bytes'],reference_bytes=reference['bytes'],
  before_allocations=b['allocations'],after_allocations=r['allocations'],reference_allocations=reference['allocations'],
  residual=r['residual'],left_orthogonality=r['left_orthogonality'],right_orthogonality=r['right_orthogonality'],
  spectrum_error=r['spectrum_error'],before_iterations=b['iterations'],after_iterations=r['iterations'],
  status=r['status'],fingerprint=r['fingerprint'],before_log=b['log'],after_log=r['log']))
with (OUT/'comparisons.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(paired[0]));w.writeheader();w.writerows(paired)
lines=['# SVD dense — campagne du 4 novembre 2026','',
 '**Cible : temps OxiBLAS ≤ 1,20× le meilleur temps comparable OpenBLAS/Accelerate.**','',
 'Les tables utilisent la médiane des appels après préchauffage, à données identiques (SHA-256 vérifié). '
 'Chaque backend et chaque configuration de threads sont isolés dans un processus Julia, exécuté seul. '
 'Pour les vecteurs, la référence est le meilleur de `gesdd` et `gesvd` ; pour les valeurs seules, `svdvals` (`gesdd`). '
 'La comparaison est à nombre de threads **demandé** identique, pas au meilleur thread count choisi a posteriori.','',
 'Machine : Apple M5 Max, macOS, Julia 1.14.0-alpha2. Les performances x86/AVX2/AVX-512 ne sont pas validées. '
 'Accelerate décide de son parallélisme effectif : un mode multithread demandé ne prouve pas exactement N travailleurs. '
 'Les journaux contiennent la configuration BLAS et la vérification du fournisseur de chaque symbole LAPACK.','',
 '| Algorithme / type | Cas | Gain médian | Rapport médian au meilleur | Rapport min–max | Cible atteinte |',
 '|---|---:|---:|---:|---:|---:|']
snapshot=record.get('frozen_source_root','unknown')
owned=('src/svd.jl','src/svd_advanced.jl','src/svd_dense_kernels.jl','src/svd_block_jacobi.jl','src/svd_dense_qr.jl')
later=[p for p in owned if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=record['source_sha256'].get(p)]
provenance=['> Version mesurée : `'+snapshot+'`.']
if later:
 provenance+=['> Le code SVD courant comporte des changements postérieurs non rebenchmarkés : `'+', '.join(later)+'`. Les chiffres décrivent exclusivement cette copie figée.']
lines[2:2]=provenance+['']
for alg in ('jacobi','golub_kahan'):
 for t in ('Float32','Float64','ComplexF32','ComplexF64'):
  rs=[r for r in paired if r['algorithm']==alg and r['type']==t]
  if rs:lines.append(f"| {alg} / {t} | {len(rs)} | {statistics.median(r['speedup'] for r in rs):.2f}× | {statistics.median(r['ratio'] for r in rs):.2f}× | {min(r['ratio'] for r in rs):.2f}–{max(r['ratio'] for r in rs):.2f}× | {sum(r['target_met'] for r in rs)}/{len(rs)} |")
lines+=['',f"**Bilan : {sum(r['target_met'] for r in paired)}/{len(paired)} cas atteignent la cible.** Tous les cas listés ont convergé et passé les contrôles numériques. La cible générale n’est donc pas déclarée atteinte.",'',
 'Modifications : statistiques de paires Jacobi fusionnées avec retour au chemin normalisé pour les colonnes très petites ; réduction LABRD native ; QR préalable des matrices hautes avec filtre des pivots ; variantes Jacobi bloquées ; '
 'SIMD portable des réflecteurs propres à la réduction SVD ; rotations QR vectorisables ; '
 'reconstruction mince des vecteurs avec application compacte WY pour les tailles suffisantes ; suppression d’une copie de V dans Jacobi. '
 'Aucun calcul délégué à une SVD externe dans le code OxiBLAS. Les références externes servent uniquement aux mesures et à la validation.','',
 'Les allocations sont les octets/objets du tas Julia observés par `@timed` (minimum des échantillons), pas les allocations internes éventuelles des bibliothèques C. '
 'Résidu = ‖A−UΣVᴴ‖F/‖A‖F ; orthogonalité = ‖UᴴU−I‖F/√k et ‖VVᴴ−I‖F/√k. '
 'Sans vecteurs, ces trois mesures sont N/A ; le spectre est comparé à LAPACK. Les comptes d’itérations natifs figurent dans le CSV.','',
 'Les premiers essais ont aussi évalué une décomposition explicite de l’arithmétique complexe et un accès réinterprété au stockage réel. '
 'Le premier n’apportait pas de gain reproductible ; le second régressait d’environ 10 %. Ils ont été retirés. '
 'Les journaux intermédiaires sont conservés.','',
 '## Avant / après — tous les cas','',
 '| Algo / type | Taille | Vecteurs | Threads demandés | Avant ms | Après ms | Gain | OpenBLAS ms | Accelerate ms | Après / meilleur | Octets Julia avant → après |',
 '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for r in paired:
 fmt=lambda x:'N/A' if x is None else f'{x:.3f}'
 lines.append(f"| {r['algorithm']} / {r['type']} | {r['m']}×{r['n']} | {'oui' if r['vectors'] else 'non'} | {r['threads_requested']} | {r['before_ms']:.3f} | {r['after_ms']:.3f} | {r['speedup']:.2f}× | {fmt(r['openblas_ms'])} | {fmt(r['accelerate_ms'])} | {r['ratio']:.2f}× | {r['before_bytes']} → {r['after_bytes']} |")
lines+=['','## Contrôles numériques','',
 '| Algorithme / type | Résidu maximal | Orthogonalité maximale | Erreur spectrale maximale |',
 '|---|---:|---:|---:|']
for alg in ('jacobi','golub_kahan'):
 for t in ('Float32','Float64','ComplexF32','ComplexF64'):
  rs=[r for r in paired if r['algorithm']==alg and r['type']==t]
  if rs:lines.append(f"| {alg} / {t} | {max(r['residual'] or 0 for r in rs):.3e} | {max(max(r['left_orthogonality'] or 0,r['right_orthogonality'] or 0) for r in rs):.3e} | {max(r['spectrum_error'] for r in rs):.3e} |")
initial=json.loads((OUT/'initial_source_hashes.json').read_text())
changed=[p for p,h in initial.items() if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h]
lines+=['','## Intégrité du périmètre','',f"Fichiers sources préexistants différents du début de campagne : `{', '.join(changed)}`. Les changements hors SVD proviennent du travail concurrent et ne sont pas attribués à cette campagne. Nouveaux fichiers privés SVD : `src/svd_dense_kernels.jl`, `src/svd_block_jacobi.jl`, `src/svd_dense_qr.jl`.",'',
 'Les tests ciblés, les données complètes (allocations, qualité, itérations, drivers et empreintes) sont conservés dans ce répertoire. '
 'Aucun fichier sparse/ordering/solveur ni kernel générique partagé n’a été modifié par cette campagne.']
(OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
print('rows',len(paired),'target_met',sum(r['target_met'] for r in paired),'changed',changed)
