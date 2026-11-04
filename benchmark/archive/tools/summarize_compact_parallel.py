#!/usr/bin/env python3
"""Build the v0.5 report from completed logs and actual CSV measurements."""
import csv,json,pathlib
root=pathlib.Path(__file__).resolve().parents[3]; reports=root/'reports'
def read(name):return list(csv.DictReader((reports/name).open()))
r=read('compact-openblas.csv');a=read('compact-accelerate.csv')
assert len(r)==424 and len(a)==140
validation=reports/'baseline-v0.5'/'validation.json'
if not validation.exists(): validation=reports/'validation.json'
v=json.loads(validation.read_text());assert v['version']=='0.5.0'
for row in r+a:
    assert float(row['relative_error'])< (1e-4 if row['type'] in ('Float32','ComplexF32') else 1e-11)
    if row['algorithm'] in ('native','reference_blas'):assert int(row['bytes'])==0
scales={n:read(f'tiles-openblas-{n}t.csv') for n in (1,2,4,8)}
assert all(len(rows)==124 for rows in scales.values())
assert len(read('tiles-accelerate-1t.csv'))==36
def compact(op,T,method,rows=r):
    return next(x for x in rows if x['operation']==op and x['type']==T and x['n']=='1024' and x['trans']=='N' and x['algorithm']==method)
def tile(n,op,T,method='native'):
    return next(x for x in scales[n] if x['operation']==op and x['type']==T and x['n']=='1024' and x['algorithm']==method)
content=['# Validation compact/inter-tuiles v0.5 — 2 novembre 2026','',
'Les kernels BLAS 2 exploitent les coefficients stockés et les BLAS 3 de rang k '
'peuvent distribuer leurs tuiles indépendantes. Le code numérique est natif Julia, '
'avec SIMD LLVM sur les mises à jour compactes réelles contiguës.','',
'## Suite complète','',
'| Configuration locale Apple M5 Max arm64 | Tests réussis |','|---|---:|']
labels={'4t':'Julia 1.14-alpha2, natif, 4 threads','generic-1t':'Julia 1.14-alpha2, CPU générique, 1 thread','julia113-4t':'Julia 1.13.3, natif, 4 threads'}
for x in v['tests']:content.append(f'| {labels[x["configuration"]]} | {x["passed"]} |')
content+=['',
'Bornes vérifiées, dépôts de cache primaires isolés. Les journaux sont '
'`reports/tests-v05-final-*.log`. La suite précédente est conservée et inclut '
'les quatre types ; les nouveaux contrôles couvrent N/T/C, U/L, vues inversées '
'et buffers non contigus, alias des coefficients, padding et diagonales unité '
'interdits à la lecture, α/β sans lecture, singularité avant mutation, '
'coefficients hermitiens à partie imaginaire NaN, zéros structurels avec Inf, '
'poids diagonaux extrêmes/subnormaux et solutions de référence BigFloat.','',
'Un adaptateur de sortie trace les tâches ayant écrit chaque tuile. Il vérifie '
'le nombre de tâches distinctes, les régions disjointes et les appels partageant '
'un workspace. Des contrôles indépendants denses/BLAS complètent les 48 '
'groupes de fixtures Rust (428 valeurs, quatre types) réellement compilés et '
'exécutés au commit 8d82bb6. Le Rust n’est jamais utilisé par le paquet.','',
'## Stockage compact','',
'424 lignes v0.4/native/OpenBLAS et 140 lignes Accelerate, tailles 128/1024 '
'et bandes 16 384 de largeur 8, N/C lorsque pertinent. Les extensions '
'symétriques complexes sans équivalent BLAS portable sont vérifiées contre '
'la matrice dense aux tailles 128/1024. Les grandes bandes évitent toute '
'expansion dense et toute exécution quadratique de la baseline.','',
'Médianes à N=1024, largeur de bande 8, sans transposition ; microsecondes. '
'Les références hermitiennes utilisent ComplexF64, les autres lignes Float64.','',
'| Opération | v0.4 | Native | OpenBLAS | Accelerate | Gain v0.4/native |',
'|---|---:|---:|---:|---:|---:|']
for op in ('gbmv','sbmv','hbmv','tbmv','tbsv','spmv','hpmv','tpmv','tpsv','spr','spr2','hpr','hpr2'):
    T='ComplexF64' if op in ('hbmv','hpmv','hpr','hpr2') else 'Float64'
    rows=[compact(op,T,'v04'),compact(op,T,'native'),compact(op,T,'reference_blas'),compact(op,T,'reference_blas',a)]
    gain=float(rows[0]['median_s'])/float(rows[1]['median_s'])
    content.append('| '+op.upper()+' | '+' | '.join(f'{float(x["median_s"])*1e6:.2f}' for x in rows)+f' | {gain:.2f}× |')
content+=['',
f'Les {sum(x["algorithm"]=="native" for x in r)} appels natifs mesurés (sans alias) allouent zéro octet selon '
'BenchmarkTools. Les références mesurées allouent également zéro octet dans '
'ces wrappers. Le dispatch SIMD à largeur typée évite les allocations par '
'colonne ; les conversions d’options et recherches de fonctions sont hors '
'chronométrage. Les alias imposent encore des snapshots du stockage utile. '
'Les gains face à la v0.4 ne signifient pas que les BLAS externes sont dépassés '
'dans toutes les tailles/types.','',
'## Scalabilité inter-tuiles','',
'Quatre processus distincts à 1/2/4/8 threads : 124 lignes chacun, '
'v0.4/native/OpenBLAS, tailles 512/1024, K=3N/4. Accelerate fournit 36 lignes '
'supplémentaires à un thread. SYMM/HEMM sont des contrôles conservant le GEMM '
'parallèle existant ; leur algorithme n’est pas réécrit. GEMMT n’a pas de ligne '
'BLAS externe dans cette campagne.','',
'SYRK à N=1024, K=768, bloc 128 ; médianes en millisecondes et allocations '
'incluant contextes/tampons diagonaux. Panneaux GEMM réutilisés.','',
'| Threads | Float64 ms | Gain vs 1 thread | Octets Float64 | ComplexF32 ms | Gain vs 1 thread |',
'|---|---:|---:|---:|---:|---:|']
for n in (1,2,4,8):
    d=tile(n,'syrk','Float64');c=tile(n,'syrk','ComplexF32')
    gd=float(tile(1,'syrk','Float64')['median_s'])/float(d['median_s'])
    gc=float(tile(1,'syrk','ComplexF32')['median_s'])/float(c['median_s'])
    content.append(f'| {n} | {1000*float(d["median_s"]):.3f} | {gd:.2f}× | {d["bytes"]} | {1000*float(c["median_s"]):.3f} | {gc:.2f}× |')
content+=['',
'À huit threads, rapports v0.4/natif dans le même processus et sur les mêmes '
'entrées à N=1024 :','',
'| Opération | Float64 | ComplexF32 |','|---|---:|---:|']
for op in ('syrk','syr2k','gemmt'):
    gains=[float(tile(8,op,T,'v04')['median_s'])/float(tile(8,op,T)['median_s']) for T in ('Float64','ComplexF32')]
    content.append('| '+op.upper()+' | '+' | '.join(f'{g:.2f}×' for g in gains)+' |')
content+=['',
'Chaque tâche conserve son couple de panneaux et exécute un GEMM monothread. '
'Le parent du workspace est verrouillé pendant le plan ; les wrappers enfants '
'ont leurs propres verrous, sans réentrée sur le parent. La propriété suit '
'les tâches et supporte leur migration. Le nombre de tâches est borné par les '
'buffers, les tuiles et les threads disponibles. Les contextes et tampons '
'diagonaux restent alloués par appel, ce qui explique leur croissance avec '
'le nombre de tâches.','',
]
rr=read('rust-compact-workload/rust-compact-gbmv.csv')
jj=read('rust-compact-workload/julia-openblas.csv'); ja=read('rust-compact-workload/julia-accelerate.csv')
assert len(rr)==3 and len(jj)==6 and len(ja)==3
content+=['','## GBMV Float64 sur entrées identiques au Rust','',
'Largeur 8, un thread, neuf échantillons après warmup, mêmes coefficients et '
'α/β. Les vecteurs complets Rust sont vérifiés contre un BLAS indépendant. '
'Les formats physiques restent ceux de chaque API. Médianes en microsecondes :','',
'| N | Rust | Julia native | OpenBLAS | Accelerate |','|---|---:|---:|---:|---:|']
for x in rr:
    n=x['n']; rows=[next(y for y in jj if y['n']==n and y['algorithm']==method) for method in ('native','reference_blas')]
    rows.append(next(y for y in ja if y['n']==n))
    assert all(float(y['relative_error'])<1e-11 for y in rows)
    content.append('| '+n+' | '+' | '.join(f'{1e6*float(y["median_s"]):.2f}' for y in [x,*rows])+' |')
content+=['',
'Ce jeu ne concerne qu’un noyau Float64 monothread ; il ne permet pas de '
'conclure à une supériorité générale. Scripts, résultats et vecteurs binaires '
'sont conservés dans `reports/rust-compact-workload/` et `benchmark/`.','']
content+=['',
'## Assembleur et provenance','',
'Les deux kernels AXPY compacts réels inspectés contiennent des instructions '
'FMLA vectorielles et aucun appel de fonction. LLVM/assembleur, environnements '
'et bibliothèques par symbole sont conservés dans `reports/compact-*` et '
'`reports/blas2-3-providers-*.txt`. Seul NEON a été exécuté sur le matériel local. '
'La validation x86 et la CI distante restent à faire.','',
'Les sources v0.4 comparées sont figées dans `reports/baseline-v0.4/`. '
'Les sources GEMM, SIMD commun et BLAS 1 sont inchangées ; leurs campagnes '
'historiques gardent leurs propres hashes. Les manifests courants '
'`reports/source-sha256.txt`, `compact-source-sha256.txt`, '
'`tiles-source-sha256.txt` et `artifact-sha256.txt` identifient les livrables.','',
'## Limites et suite','',
'Les campagnes restent courtes et locales, sans généralisation statistique '
'à toutes les machines. Les kernels SIMD complexes/stridés et le parallélisme '
'compact restent à optimiser. Les lots indépendants, la parallélisation des '
'factorisations et des autres BLAS 1, les microkernels triangulaires, le tuning '
'et les extensions LAPACK/sparse ne sont pas déclarés terminés.','',
'Voir [conception](COMPACT_PARALLEL_DESIGN.md), [protocole](PERFORMANCE.md), '
'[API](API.md), [état](STATUS.md) et '
'[continuation actualisée](CONTINUATION_2026-11-02.md).','']
(root/'docs/COMPACT_PARALLEL_VALIDATION.md').write_text('\n'.join(content))
print(root/'docs/COMPACT_PARALLEL_VALIDATION.md')
