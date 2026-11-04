#!/usr/bin/env python3
"""Summarize the actual current BLAS 3 campaign, requiring complete evidence."""
import csv,json,pathlib
root=pathlib.Path(__file__).resolve().parents[3]; reports=root/'reports'
def read(name):return list(csv.DictReader((reports/name).open()))
r=read('level3-openblas.csv'); a=read('level3-accelerate.csv')
if len(r)!=492 or len(a)!=156:raise RuntimeError(f'Incomplete campaigns: {len(r)}, {len(a)}')
for x in r+a:
    if not float(x['relative_error'])< (5e-4 if x['type'] in ('Float32','ComplexF32') else 5e-12):raise RuntimeError('Incorrect benchmark row')
def pick(rows,T,operation,method):
    return next(x for x in rows if x['type']==T and x['operation']==operation and x['n']=='512' and x['case']=='LUN' and x['algorithm']==method)
def gain(T,op):return float(pick(r,T,op,'v02')['median_s'])/float(pick(r,T,op,'native')['median_s'])
v=json.loads((reports/'baseline-v0.3'/'validation.json').read_text())
if v['version']!='0.3.0':raise RuntimeError('Current tests must be recorded first')
content=['# Validation BLAS 3 v0.3 — 2 novembre 2026','',
'P0 est terminé pour les vérifications locales : revalidation des sources v0.2, '
'rapport spectral ajouté, workflows Linux/macOS/Windows présents et renforcés '
'avec conservation des journaux. Les workflows distants ne sont pas exécutés '
'ici. AVX2/AVX-512 restent non validés sur du matériel x86.','',
'## Suite complète sur les sources finales','',
'| Configuration locale Apple M5 Max arm64 | Tests réussis |','|---|---:|']
labels={'4t':'Julia 1.14-alpha2, CPU natif, 4 threads','generic-1t':'Julia 1.14-alpha2, CPU générique, 1 thread','julia113-4t':'Julia 1.13.3, CPU natif, 4 threads'}
for t in v['tests']:content.append(f'| {labels[t["configuration"]]} | {t["passed"]} |')
content+=['',
'Bornes vérifiées dans les trois configurations. Journaux `reports/tests-v03-final-*.log`, '
'hashes des sources `reports/baseline-v0.3/source-sha256.txt`, hashes des tests/scripts/docs/CI '
'`reports/baseline-v0.3/artifact-sha256.txt`. Les tests couvrent les quatre types, côtés/triangles, '
'N/T/C, vues inversées et adjointes, alias, triangles/diagonales unité interdits '
'à la lecture, sorties sans lecture pour β=0, entrées ignorées pour α=0/K=0, '
'singularités avant mutation, canaris et workspaces partagés entre tâches. '
'Les références de calcul sont indépendantes du Rust (BLAS/LAPACK Julia et résidus).','',
'Les deux processus Julia 1.14-alpha ont émis un message non fatal de cache '
'objcache/LMDB (MDB_INVALID), conservé dans les logs ; les suites ont néanmoins '
'terminé avec succès. La validation stable Julia 1.13.3 est consignée séparément.','',
'## Mesures locales','',
'492 lignes v0.2/native/OpenBLAS et 156 lignes Accelerate. Tailles 32/128/512, '
'côtés L/R, triangles U/L et transpositions pertinentes, un thread, jusqu’à '
'sept échantillons. Remise à zéro/restauration des sorties hors chronométrage. '
'Les API comprennent l’allocation des panneaux. Les bibliothèques de référence '
'sont confinées aux tests/benchmarks. Environnements et CSV dans `reports/`.','',
'Rapport temps v0.2 / temps natif, médianes à N=512, côté gauche, triangle '
'supérieur, sans transposition ; A ou B rectangulaire 512×384 selon l’opération. Un rapport >1 indique un gain local.','',
'| Type | SYMM | TRMM | TRSM | SYRK | SYR2K |','|---|---:|---:|---:|---:|---:|']
for T in ('Float32','Float64','ComplexF32','ComplexF64'):
    content.append('| '+T+' | '+' | '.join(f'{gain(T,op):.2f}×' for op in ('symm','trmm','trsm','syrk','syr2k'))+' |')
content+=['',
'| Type complexe | HEMM | HERK | HER2K |','|---|---:|---:|---:|']
for T in ('ComplexF32','ComplexF64'):content.append('| '+T+' | '+' | '.join(f'{gain(T,op):.2f}×' for op in ('hemm','herk','her2k'))+' |')
content+=['',
'Temps TRMM correspondants, millisecondes ; la comparaison BLAS externe reste '
'indispensable car un progrès face à la v0.2 ne signifie pas la parité avec BLAS.','',
'| Type | Julia native | OpenBLAS | Accelerate |','|---|---:|---:|---:|']
for T in ('Float32','Float64','ComplexF32','ComplexF64'):
    vals=[pick(r,T,'trmm','native'),pick(r,T,'trmm','reference_blas'),pick(a,T,'trmm','reference_blas')]
    content.append('| '+T+' | '+' | '.join(f'{1000*float(x["median_s"]):.3f}' for x in vals)+' |')
old=pick(r,'Float64','syr2k','v02'); new=pick(r,'Float64','syr2k','native')
content+=['',
f'SYR2K Float64 N=512 alloue {int(old["bytes"]):,} octets dans la v0.2 '
f'et {int(new["bytes"]):,} dans la nouvelle API (panneaux compris). '
'Le tampon de produit est limité à une tuile diagonale de 128² éléments. '
'Les snapshots d’alias demeurent nécessaires. TRSM échange le faible scratch '
'de l’ancienne résolution scalaire contre des panneaux GEMM réutilisables.','',
'Les petits cas ne gagnent pas systématiquement : notamment certaines mises à '
'jour de rang k à N=32 et TRMM ComplexF32 gauche à N=32. Les temps natifs '
'restent supérieurs à BLAS externe dans de nombreux cas. Les feuilles SIMD '
'spécialisées, l’autotuning, GEMM ComplexF32 et le parallélisme entre tuiles '
'restent des priorités. Ces mesures courtes sont locales ; aucune généralisation '
'à d’autres machines ou grandes dimensions n’est faite.','',
'## Contrat hermitien des références','',
'Les mesures HERK/HER2K partent d’une diagonale réelle. Un contrôle distinct '
'avec partie imaginaire initiale arbitraire relève 16/16 réussites OpenBLAS '
'et 16/16 échecs de la diagonale imaginaire dans l’Accelerate chargé ici ; '
'voir `reports/hermitian-contract-*.csv` et `benchmark/hermitian_reference_probe.jl`. '
'Le code natif passe ce contrôle dans les deux processus, ainsi que les tests '
'avec partie imaginaire NaN. Le contrat d’effacement de cette partie '
'imaginaire est donné par [Netlib CHERK](https://netlib.org/blas/cherk.f). '
'Ce constat ne porte que sur les versions/environnements enregistrés ici.','',
'## Provenance et suite','',
'`reports/baseline-v0.2/level3.jl` conserve la source exacte comparée. Les '
'sources hardware/SIMD/GEMM sont inchangées depuis la campagne v0.1 ; les '
'mesures GEMM historiques restent identifiées par leurs propres hashes. '
'Les rapports spectraux ne prétendent pas contenir une campagne complète de temps.','',
'Voir [BLAS3_DESIGN.md](BLAS3_DESIGN.md), [PERFORMANCE.md](PERFORMANCE.md), '
'[SPECTRAL_VALIDATION.md](SPECTRAL_VALIDATION.md) et [STATUS.md](STATUS.md). '
'La suite P1 comprend ROTM/ROTMG, les produits scalaires étendus, les noyaux '
'BLAS 2 banded/packed, la validation x86 et le tuning. Le portage complet '
'OxiBLAS reste inachevé.','']
(root/'docs/BLAS3_VALIDATION.md').write_text('\n'.join(content))
print(root/'docs/BLAS3_VALIDATION.md')
