#!/usr/bin/env python3
"""Summarize the recorded validation run (optionally copy logs from argv[1])."""
import csv, hashlib, json, math, pathlib, re, shutil, subprocess, sys

root=pathlib.Path(__file__).resolve().parents[3]
reports=root/'reports'
labels=['tests-final-4t','tests-final-generic-1t','tests-final-julia113-4t']
if len(sys.argv)>1:
    logs=pathlib.Path(sys.argv[1])
    for label in labels:
        shutil.copyfile(logs/(label+'.log'),reports/(label+'.log'))

def rows(name):
    with (reports/name).open() as f:
        return list(csv.DictReader(f))

native=rows('gemm-openblas-8t-large.csv')
accelerate=rows('gemm-accelerate-1t-large.csv')
rust=rows('rust-gemm.csv')
rustjulia=rows('julia-rust-workload.csv')
tests=[]
for label in labels:
    log=(reports/(label+'.log')).read_text()
    match=re.search(r'OxiBLAS native package\s*\|\s*(\d+)\s+(\d+)\s+([\d.]+)s',log)
    if not match or match[1]!=match[2]:
        raise RuntimeError('Missing successful test summary: '+label)
    tests.append({'configuration':label,'pass':int(match[1]),'seconds':float(match[3])})

def pick(data,T,n,method,threads):
    return next(r for r in data if r['type']==T and r['m']==r['n']==r['k']==str(n) and r['algorithm']==method and r['threads']==str(threads))

asm=[]
for path in sorted(reports.glob('neon-*-native.s')):
    if 'before' in path.name:continue
    text=path.read_text()
    calls=len(re.findall(r'^\s*(?:bl|blr)\s',text,re.M))
    fmas=len(re.findall(r'\bfmla',text))
    if calls or not fmas: raise RuntimeError('Unexpected assembly: '+path.name)
    asm.append({'file':path.name,'fmla':fmas,'calls':calls})

content=['# Validation de l’incrément 0.1 — 2 novembre 2026','',
    'Apple M5 Max, 18 cœurs logiques, macOS arm64. Tests sous Julia 1.14.0-alpha2 '
    '(LLVM 23.1.4) et Julia 1.13.3. Les résultats concernent le code identifié dans '
    '`reports/baseline-v0.1/source-sha256.txt`. Le portage complet reste inachevé ; voir [STATUS.md](STATUS.md).','',
    '## Correction','',
    '| Configuration | Réussis | Durée Test, hors précompilation séparée |',
    '|---|---:|---:|']
pretty=['Julia 1.14, CPU natif, 4 threads, bornes vérifiées',
        'Julia 1.14, CPU générique, 1 thread, bornes vérifiées',
        'Julia 1.13.3, CPU natif, 4 threads, bornes vérifiées']
for p,t in zip(pretty,tests):
    content.append(f'| {p} | {t["pass"]} | {t["seconds"]:.1f} s |')
content += ['',
    'La suite inclut quatre types, N/T/C, matrices rectangulaires et vides, '
    'vues inversées/non contiguës, canaris autour des sorties, alias, cas α/β sans '
    'lecture, espaces de travail partagés, résidus de factorisation, moindres carrés, '
    'stockages compactés, CSR/CG et 36 fixtures issues du Rust original. Les tests '
    'ne constituent pas une preuve de stabilité sur toutes les entrées extrêmes.',
    '', 'Les références Rust proviennent du commit '
    '`8d82bb6bf1df133aa90525aa9f478ebe41514f9b`, compilé en release avec '
    '`-C target-cpu=native`, LTO fin. Linux/Windows et les noyaux x86 restent à '
    'valider sur leur matériel ; les jobs CI fournis n’ont pas été lancés à distance.',
    '', '## GEMM 4096 × 4096 × 4096', '',
    'Débit médian en GFLOP/s. Alpha=0,7, bêta=0. Panneaux Julia réutilisés ; '
    'tous les calculs sont vérifiés avant mesure. OpenBLAS est mesuré à 1/8 threads '
    'et Accelerate en mode monothread, dans des processus successifs. Certains '
    'essais de cette taille ne comportent qu’un échantillon : ces chiffres sont '
    'des observations locales, sans garantie statistique ni supériorité générale.', '',
    '| Type | Julia 1 thread | Julia 8 threads | OpenBLAS 1 thread | OpenBLAS 8 threads | Accelerate 1 thread |',
    '|---|---:|---:|---:|---:|---:|']
bench=[]
for T in ('Float32','Float64','ComplexF32','ComplexF64'):
    vals=[pick(native,T,4096,'simd',1),pick(native,T,4096,'simd',8),
          pick(native,T,4096,'reference_blas',1),pick(native,T,4096,'reference_blas',8),
          pick(accelerate,T,4096,'reference_blas',1)]
    content.append('| '+T+' | '+' | '.join(f'{float(v["gflops"]):.1f}' for v in vals)+' |')
    bench.append({'type':T,'n':4096,'rows':vals})
content += ['',
    f'La campagne finale conserve {len(native)} lignes avec OpenBLAS et '
    f'{len(accelerate)} avec Accelerate, tailles 16–4096 et rectangles. '
    'Les CSV contiennent temps minimaux et médians, GFLOP/s, nombre '
    'd’échantillons, allocations, temps du premier appel, compilation et erreur '
    'relative. Les variantes référence/cache/packed/register sont également '
    'mesurées sur les tailles prévues par le protocole.', '',
    '| Type | Erreur relative maximale observée, campagne OpenBLAS |', '|---|---:|']
for T in ('Float32','Float64','ComplexF32','ComplexF64'):
    error=max(float(r['relative_error']) for r in native if r['type']==T)
    content.append(f'| {T} | {error:.3g} |')
alloc=pick(native,'Float64',1024,'simd',1)
content += ['',
    f'Le contrôle d’un appel Float64 1024 avec workspace alloue encore '
    f'{alloc["bytes"]} octets en {alloc["allocations"]} allocations à un thread. '
    'Les panneaux ne sont pas réalloués à chaque micro-noyau. Les tâches '
    'multithread ajoutent leurs propres allocations, enregistrées dans les CSV. '
    'Trafic mémoire et compteurs matériels non mesurés.', '',
    '## Comparaison Rust sur entrées identiques', '',
    'Float64, un thread, minima en millisecondes, cinq échantillons. '
    '« API » inclut la construction des panneaux par appel, comme l’API Rust ; '
    '« workspace » les réutilise. Les entrées déterministes et α/β sont identiques. '
    'Cette comparaison ne porte que sur GEMM Float64.', '',
    '| N | Rust OxiBLAS | Julia API | Julia workspace | OpenBLAS |',
    '|---|---:|---:|---:|---:|']
checksum_error=0.0
for rr in rust:
    n=rr['n']
    jj=[next(r for r in rustjulia if r['n']==n and r['algorithm']==mode)
        for mode in ('julia_api','julia_workspace','openblas')]
    checksum_error=max(checksum_error,*(abs(float(r['checksum'])-float(rr['checksum'])) for r in jj))
    content.append('| '+n+' | '+' | '.join(f'{float(r["minimum_s"])*1000:.4f}' for r in [rr,*jj])+' |')
if checksum_error>1e-6:raise RuntimeError('Rust deterministic checksum mismatch')
content += ['',f'Écart maximal entre checksums agrégés : {checksum_error:.3g}. '
    'Les produits Julia sont également comparés aux matrices de référence, '
    'et les 36 fixtures testent les valeurs élément par élément.', '',
    '## Instructions machine et restriction des backends', '',
    '| Micro-noyau inspecté | Instructions FMLA dans la fonction | Appels de fonction |',
    '|---|---:|---:|']
for a in asm:content.append(f'| {a["file"]} | {a["fmla"]} | {a["calls"]} |')
content += ['',
    'Le compte inclut boucle déroulée, reste et épilogue. Les sauvegardes ABI '
    'de registres en entrée/sortie ne sont pas des appels numériques externes. '
    'Le code LLVM et l’assembleur sont conservés dans `reports/`. '
    'L’évolution avant/après inlining est conservée séparément ; ses premières '
    'mesures sont exploratoires.', '',
    'L’audit source ne relève aucun appel numérique BLAS/LAPACK/ccall dans '
    '`src/` ; seuls les types de wrappers LinearAlgebra sont importés. '
    'Voir `reports/backend-audit.txt`. Cet audit n’est pas une preuve formelle.', '',
    '## Reproduction', '',
    'Voir les commandes du [README](../README.md), le protocole '
    '[PERFORMANCE.md](PERFORMANCE.md), `benchmark/benchmarks.jl`, '
    '`benchmark/compare_rust.jl`, `tools/run_rust_reference.py` et les '
    'environnements enregistrés dans `reports/`. Les journaux Test sont livrés '
    'avec les résultats. Les limites fonctionnelles restent détaillées dans '
    '[STATUS.md](STATUS.md).', '']
(root/'docs/VALIDATION.md').write_text('\n'.join(content))
(reports/'baseline-v0.1'/'validation.json').write_text(json.dumps({'tests':tests,'benchmark_rows':len(native),
    'accelerate_rows':len(accelerate),'assembly':asm,'checksum_max_abs_error':checksum_error,
    'large_gemm':bench},indent=2)+'\n')
print(root/'docs/VALIDATION.md')
