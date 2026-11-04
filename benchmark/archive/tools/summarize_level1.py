#!/usr/bin/env python3
"""Summarize the actual v0.4 test/benchmark evidence; no inferred timings."""
import csv,json,pathlib,re
root=pathlib.Path(__file__).resolve().parents[3]; reports=root/'reports'
def read(name):return list(csv.DictReader((reports/name).open()))
r=read('level1-openblas.csv'); a=read('level1-accelerate.csv')
b=read('baseline-level1-generic/level1-openblas.csv')
assert len(r)==152 and len(a)==76
v=json.loads((reports/'baseline-v0.4'/'validation.json').read_text());assert v['version']=='0.4.0'
for row in r+a:
    assert float(row['relative_error'])<= (5e-5 if row['type']=='Float32' else 5e-12)
def pick(rows,op,T,n='1000000',flag='0',method='native',stride='1'):
    return next(x for x in rows if all(x[k]==val for k,val in dict(operation=op,type=T,n=n,flag=flag,algorithm=method,stride=stride).items()))
content=['# Validation BLAS 1 complémentaire v0.4 — 2 novembre 2026','',
'ROTM/ROTMG réels et produits scalaires étendus sont implémentés en Julia. '
'Les formes SIMD réutilisent les primitives LLVM du paquet ; aucune délégation '
'numérique BLAS/LAPACK/Rust n’est ajoutée dans src/.','',
'## Tests sur les sources finales','',
'| Configuration locale Apple M5 Max arm64 | Réussis |','|---|---:|']
labels={'4t':'Julia 1.14-alpha2, natif, 4 threads','generic-1t':'Julia 1.14-alpha2, CPU générique, 1 thread','julia113-4t':'Julia 1.13.3, natif, 4 threads'}
for t in v['tests']:content.append(f'| {labels[t["configuration"]]} | {t["passed"]} |')
content+=['',
'Bornes vérifiées. Chaque processus dispose d’un dépôt de cache primaire '
'isolé ; les dépôts existants servent de repli en lecture. Journaux '
'`reports/tests-v04-final-*.log`, manifeste source `reports/baseline-v0.4/source-sha256.txt` '
'et manifeste d’artefacts `reports/baseline-v0.4/artifact-sha256.txt`. Les tests v0.3 '
'GEMM/BLAS/LAPACK/sparse sont conservés dans la suite complète.','',
'Les nouveaux tests vérifient les deux précisions réelles pertinentes, '
'les formes de H, les paramètres implicites non lus, les vues inversées, '
'les canaris, les alias de paramètres, le refus d’alias entre sorties, '
'les restes SIMD et la protection des tableaux sous GC. DSDOT/SDSDOT sont '
'comparés à BLAS, à une accumulation indépendante ordonnée et à BigFloat. '
'ROTMG est contrôlé par annulation et congruence de la forme quadratique '
'à 512 bits, sur poids extrêmes, subnormaux, redimensionnements multiples '
'et formes indéfinies admissibles. Les dépassements de plage sont explicites '
'et laissent l’état mutable intact.','',
'## Mesures de temps et allocations','',
'152 lignes avec OpenBLAS et 76 avec Accelerate. Tailles vectorielles '
'32/4096/1 000 000, strides 1/3/-2, trois formes ROTM, un thread, jusqu’à '
'onze échantillons. Les symboles de référence sont associés à leurs vrais '
'fournisseurs dans les environnements enregistrés ; ils proviennent bien '
'd’OpenBLAS puis d’Accelerate. Les sorties sont restaurées hors mesure. '
'Les résultats lus sont rendus observables au compilateur.','',
'ROTM contigu, flag 0, N=1 000 000, médianes en millisecondes :','',
'| Type | Boucle avant SIMD | Native SIMD | OpenBLAS | Accelerate | Gain local avant/après |',
'|---|---:|---:|---:|---:|---:|']
for T in ('Float32','Float64'):
    rows=[pick(b,'rotm',T),pick(r,'rotm',T),pick(r,'rotm',T,method='reference_blas'),pick(a,'rotm',T,method='reference_blas')]
    gain=float(rows[0]['median_s'])/float(rows[1]['median_s'])
    content.append('| '+T+' | '+' | '.join(f'{1000*float(x["median_s"]):.3f}' for x in rows)+f' | {gain:.2f}× |')
content+=['',
'Produits étendus contigus, N=1 000 000, médianes en millisecondes :','',
'| Opération | Native | OpenBLAS | Accelerate |','|---|---:|---:|---:|']
for op,T in (('dsdot','Float64'),('sdsdot','Float32')):
    rows=[pick(r,op,T),pick(r,op,T,method='reference_blas'),pick(a,op,T,method='reference_blas')]
    content.append('| '+op.upper()+' | '+' | '.join(f'{1000*float(x["median_s"]):.3f}' for x in rows)+' |')
content+=['',
'Les appels natifs de cette campagne allouent zéro octet selon BenchmarkTools. '
'Les buffers d’entrée/sortie sont préparés hors mesure. La voie ordonnée des '
'produits étendus reste plus lente que les références BLAS mesurées ; la précision '
'étendue ne garantit pas une sommation exacte. Les vues non contiguës utilisent '
'les boucles génériques. Les nombres sont des observations locales de courte '
'durée, sans extrapolation à d’autres architectures.','',
'ROTMG mesure des lots de 256 entrées variées ; les CSV indiquent le temps '
'du lot, et non celui d’un seul scalaire. Les sorties de chaque entrée sont '
'conservées afin d’éviter le repli de constantes et les durées sous la '
'résolution de l’horloge. La génération par exposants protégés coûte davantage '
'que la référence sur les cas ordinaires ; son objectif est de préserver les '
'cas extrêmes et de signaler les résultats non représentables.','',
'## Contrôle du Rust et de l’assembleur','',
'Le programme Rust figé a été compilé et exécuté : '
'`reports/rust-level1.csv`, `reports/rust-level1-reference.log`, '
'`benchmark/rust_level1_reference.rs`. Il concorde sur les cas ordinaires. '
'Les deux contre-exemples de redimensionnement violent l’annulation avec le '
'Rust et la respectent avec Julia. Les tests enregistrent explicitement '
'cette divergence ; aucune valeur incorrecte du Rust n’est adoptée comme cible.','',
'Les six noyaux paquets Float32/Float64 et flags -1/0/1 inspectés contiennent '
'chacun 14 instructions vectorielles FMLA et aucun appel de fonction. LLVM, '
'assembleur et comptages sont conservés dans `reports/level1-assembly-*` '
'et `reports/rotm-packet-*`. Le comptage inclut les boucles déroulées, '
'les restes et les épilogues. Les racines restent protégées en dehors du noyau.','',
'## Provenance et limites','',
'La source et la campagne avant SIMD sont archivées dans '
'`reports/baseline-level1-generic/` et vérifiées contre le hash mesuré. '
'Les hashes et résultats v0.3 restent dans `reports/baseline-v0.3/`. '
'Les sources GEMM et BLAS 3 sont inchangées : les campagnes historiques '
'ne sont pas présentées comme de nouvelles mesures de v0.4.','',
'Les entrées ROTMG doivent être finies. Si une sortie ou un coefficient non nul '
'déborde ou sous-déborde de la plage représentable, une erreur est retournée '
'avant mutation. La validation physique x86 et la CI distante restent à faire. '
'Les variantes parallèles BLAS 1, les noyaux banded/packed BLAS 2 et le tuning '
'des produits étendus restent à poursuivre.','',
'Voir [LEVEL1_DESIGN.md](LEVEL1_DESIGN.md), [API.md](API.md), '
'[PERFORMANCE.md](PERFORMANCE.md) et [STATUS.md](STATUS.md). Les conventions '
'primaires et les écarts avec l’amont sont référencés dans le document de conception.','']
cache_logs='\n'.join((reports/t['log']).read_text() for t in v['tests'])
if 'objcache error' in cache_logs:
    content+=['Un message non fatal objcache/LMDB de Julia 1.14-alpha est conservé '
'dans les journaux ; il n’a pas empêché les suites de terminer.','']
(root/'docs/LEVEL1_VALIDATION.md').write_text('\n'.join(content))
print(root/'docs/LEVEL1_VALIDATION.md')
