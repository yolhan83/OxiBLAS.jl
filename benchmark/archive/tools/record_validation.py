#!/usr/bin/env python3
"""Record current source hashes and successful final tests; keep old reports historical.
Usage: python3 tools/record_validation.py /path/to/work/logs
"""
import hashlib, json, pathlib, re, shutil, sys
root=pathlib.Path(__file__).resolve().parents[3]
reports=root/'reports'; logs=pathlib.Path(sys.argv[1])
version=re.search(r'^version = "([^"]+)"', (root/'Project.toml').read_text(),re.M)[1]
labels=('4t','generic-1t','julia113-4t')
release='v'+''.join(version.split('.')[:2])
tests=[]
for label in labels:
    name=f'tests-{release}-final-{label}.log'
    data=(logs/name).read_text()
    m=re.search(r'OxiBLAS native package\s*\|\s*(\d+)\s+(\d+)\s+((?:\d+m)?[\d.]+s)',data)
    if not m or m[1]!=m[2] or 'Test Failed' in data or 'Error During Test' in data:
        raise RuntimeError('Missing clean success summary: '+name)
    duration=m[3]; parts=duration[:-1].split('m'); seconds=float(parts[-1])+(60*float(parts[0]) if len(parts)>1 else 0)
    tests.append(dict(configuration=label,passed=int(m[1]),seconds=seconds,log=name))
    shutil.copyfile(logs/name,reports/name)
sourcepaths=[root/'Project.toml',*sorted((root/'src').glob('*.jl'))]
hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sourcepaths}
(reports/'source-sha256.txt').write_text(''.join(f'{h}  {p}\n' for p,h in hashes.items()))
# Include scripts, tests, workflows and docs in a separate reproducibility manifest.
files=sorted(p for folder in ('src','test','benchmark','tools','docs','.github') for p in (root/folder).rglob('*') if p.is_file())
files+= [root/name for name in ('Project.toml','README.md','CHANGELOG.md','LICENSE','NOTICE')]
(reports/'artifact-sha256.txt').write_text(''.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(root)}\n' for p in files))
result=dict(version=version,date='2026-11-02',architecture='aarch64',tests=tests,source_sha256=hashes,x86_executed=False,remote_ci_executed=False,
            historical_gemm='baseline-v0.1/validation.json',spectral_validation='../docs/SPECTRAL_VALIDATION.md',blas3_validation='../docs/BLAS3_VALIDATION.md',level1_validation='../docs/LEVEL1_VALIDATION.md',compact_parallel_validation='../docs/COMPACT_PARALLEL_VALIDATION.md')
(reports/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(version=version,tests=tests),indent=2))
