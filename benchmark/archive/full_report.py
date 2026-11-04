"""Prepared only. Stdlib orchestration; no work occurs on import.
run_all.sh is the only supported entry point after explicit authorization.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
FIELDS = ('group operation type n threads backend status median_s min_s bytes allocations '
          'first_s compile_s gflops provider note fingerprint').split()
GROUPS = ['BLAS LEVEL 1', 'BLAS LEVEL 2', 'BLAS LEVEL 3', 'LAPACK', 'SPARSE',
          'SPECTRAL / SOLVERS / GRAPHS', 'TENSOR', 'EXTENDED PRECISION']
# Every pre-existing timing entry point. Helpers/validators and LLVM inspectors
# are classified separately; adding a script requires a deliberate inventory update.
LEGACY = {
    'benchmarks.jl': 'fixed', 'level1.jl': 'fixed', 'compact.jl': 'fixed',
    'level3.jl': 'fixed', 'spectral.jl': 'fixed', 'tile_parallel.jl': 'fixed',
    'stochastic.jl': 'directory', 'tensor_plans.jl': 'directory',
    'block_spectral.jl': 'directory', 'incremental_svd.jl': 'directory',
    'interval_spectral.jl': 'directory', 'sparse_graphs.jl': 'directory', 'sparse_svd_memory.jl': 'directory',
    'sparse_direct_complete.jl': 'reference', 'precision_complete.jl': 'reference',
    'compare_rust.jl': 'fixed', 'compare_compact_rust.jl': 'compact-reference',
}
AUXILIARY = {'level1_targeted.jl', 'level1_targeted_cases.jl', 'reference_providers.jl', 'hermitian_reference_probe.jl',
             'compare_extended_reference.jl', 'full_backend.jl', 'full_cases.jl',
             'full_worker.jl', 'full_legacy.jl', 'full_extra_cases.jl', 'full_qz_schur_cases.jl', 'selected_svd_reference.jl', 'full_selected_svd_cases.jl', 'full_arnoldi_xtrace_cases.jl', 'full_memory_triangular_cases.jl'}
RUST_SOURCES = {
    'full': 'full_rust.rs', 'gemm': 'rust_reference.rs',
    'compact': 'rust_compact_reference.rs', 'compact-bench': 'rust_compact_benchmark.rs',
    'level1': 'rust_level1_reference.rs', 'spectral': 'rust_spectral_reference.rs',
    'graph': 'rust_graph_reference/src/main.rs',
    'precision': 'rust_precision_reference/src/main.rs',
}


def section(number, name):
    print('\n' + '=' * 100 + f'\n[{number}] {name}\n' + '=' * 100, flush=True)


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError):
        return None


class Campaign:
    def __init__(self, scratch, args):
        self.scratch, self.args = Path(scratch), args
        self.rows, self.tests, self.steps, self.legacy_rows = [], [], [], []
        self.project = os.environ.get('OXIBLAS_BENCH_PROJECT', 'benchmark')
        self.test_project = os.environ.get('OXIBLAS_TEST_PROJECT', '.')
        self.threads = [1, 2, 4, 8]
        self.base_env = dict(os.environ, PYTHONUNBUFFERED='1')
        self.base_env.setdefault('OXIBLAS_OPTIONAL_FLOATS', 'available')
        cache = ROOT.parents[1] / 'work' / 'cargo-home'
        if cache.is_dir():
            self.base_env.setdefault('CARGO_HOME', str(cache))
        self.binaries = {}
        self.seen_csv = set()
        self.validation_ok = False
        self.validated_types = {}
        self.rust_revision = None
        self.rust_dirty = False

    def run(self, name, command, env=None, mandatory=True):
        print(f'\n>>> STEP {name}\n$ {shlex.join(map(str, command))}', flush=True)
        started = time.monotonic()
        status = 127
        try:
            with subprocess.Popen(list(map(str, command)), cwd=ROOT,
                                  env=env or self.base_env, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, errors='replace',
                                  bufsize=1) as process:
                parse_error = False
                for line in process.stdout:
                    print(line, end='', flush=True)
                    try:
                        if name == 'rust-checkout' and len(line.strip()) == 40 and all(c in '0123456789abcdef' for c in line.strip()):
                            self.rust_revision = line.strip()
                        if name == 'rust-worktree' and line.strip():
                            self.rust_dirty = True
                        if line.startswith('@@OXI_VALIDATED_TYPES\t'):
                            self.validated_types[name] = set(line.rstrip('\n').split('\t')[1:])
                        if line.startswith('@@OXI_ROW\t'):
                            cells = line.rstrip('\n').split('\t')[1:]
                            if len(cells) != len(FIELDS):
                                raise ValueError(f'Malformed measurement record in {name}: {line}')
                            row = dict(zip(FIELDS, cells))
                            row['step'] = name
                            self.rows.append(row)
                        elif line.startswith('@@OXI_TEST\t'):
                            counts = list(map(int, line.strip().split('\t')[1:]))
                            if len(counts) != 4:
                                raise ValueError('Malformed test counts')
                            self.tests.append((name, counts))
                    except ValueError:
                        parse_error = True
                        traceback.print_exc()
                status = process.wait()
                if parse_error and status == 0:
                    status = 1
        except (OSError, ValueError):
            traceback.print_exc()
        elapsed = time.monotonic() - started
        self.steps.append((name, status, mandatory, elapsed))
        print(f'<<< STEP {name}: exit={status}; elapsed={elapsed:.3f}s', flush=True)
        return status == 0

    def julia(self, threads, script, *args, project=None, mode="benchmark"):
        if mode not in ("test", "benchmark"):
            raise ValueError("explicit Julia execution mode must be test or benchmark")
        bounds = "yes" if mode == "test" else "auto"
        return ['julia', '--startup-file=no', f'--project={project or self.project}',
                f'--threads={threads}', f'--check-bounds={bounds}', script, *args]

    def inventory(self):
        unknown = [p.name for p in (ROOT / 'benchmark').glob('*.jl')
                   if p.name not in LEGACY and p.name not in AUXILIARY
                   and not p.name.startswith('inspect_')]
        print('Timing entry points:', ', '.join(LEGACY))
        print('LLVM/native inspectors deliberately excluded from this tests/benchmark campaign.')
        print('Full comparison registry: benchmark/archive/full_cases.jl (also covers new operations).')
        manifest=ROOT / 'docs' / 'BENCHMARK_API_AUDIT.csv'
        if manifest.is_file():
            print('PUBLIC API BENCHMARK INVENTORY (source-only; exclusions are not passed measurements)')
            print(manifest.read_text())
        else:
            self.steps.append(('missing-public-api-inventory', 1, True, 0.0))
        if unknown:
            print('ERROR: unclassified benchmark scripts:', unknown)
            self.steps.append(('benchmark-inventory', 1, True, 0.0))
        for path in sorted(p for folder in ('src','test','benchmark') for p in (ROOT / folder).rglob('*') if p.is_file() and p.suffix in ('.jl','.rs','.py','.toml','.sh') and 'Manifest' not in p.name):
            print('SOURCE_SHA256', hashlib.sha256(path.read_bytes()).hexdigest(), path.relative_to(ROOT))

    def system(self):
        print('OXIBLAS.JL — FULL TEST & BENCHMARK REPORT')
        section(1, 'SYSTEM INFORMATION')
        print('UTC:', time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
        print('OS:', platform.platform(), '; architecture:', platform.machine())
        print('CPU:', platform.processor(), '; logical CPUs:', os.cpu_count())
        print('Python:', sys.version.replace('\n', ' '))
        print('Requested thread configurations:', self.threads)
        print('Sizes:', self.base_env.get('OXIBLAS_FULL_SIZES', '16,32,64,128,256,512,1024,2048,4096'))
        controls = ('OXIBLAS_BENCH_PROJECT','OXIBLAS_TEST_PROJECT','OXIBLAS_RUST_ROOT','OXIBLAS_FULL_SIZES','OXIBLAS_FULL_SAMPLES','OXIBLAS_FULL_MEMORY_MIB','OXIBLAS_GENERIC_BITS','OXIBLAS_OPTIONAL_FLOATS','JULIA_CPU_TARGET','RUSTFLAGS','CARGO_BUILD_TARGET')
        print('Environment controls:', {key: self.base_env[key] for key in controls if key in self.base_env})
        print('Paired samples:', self.base_env.get('OXIBLAS_FULL_SAMPLES', '5'))
        print('No dependency installation or LLVM inspection is performed by this script.')
        self.run('julia-version', ['julia', '--startup-file=no', '--version'])
        self.run('rust-version', ['rustc', '--version', '--verbose'])
        self.run('cargo-version', ['cargo', '--version'])
        if sys.platform == 'darwin':
            self.run('cpu', ['sysctl', '-n', 'machdep.cpu.brand_string'], mandatory=False)
            self.run('macos', ['sw_vers'], mandatory=False)
        elif sys.platform.startswith('linux'):
            self.run('cpu', ['lscpu'], mandatory=False)
        self.inventory()

    def test_suite(self):
        section(2, 'COMPLETE JULIA TEST SUITE')
        for threads in self.threads:
            self.run(f'tests-{threads}t', self.julia(threads, 'test/legacy/run_all_supervised.jl', project=self.test_project, mode='test'))
        section(3, 'TEST SUMMARY')
        for name, counts in self.tests:
            print(name, dict(zip(('Passed', 'Failed', 'Errors', 'Skipped'), counts)))
        expected = {f'tests-{t}t' for t in self.threads}
        missing = expected - {name for name, _ in self.tests}
        if missing:
            print('UNKNOWN counts (test process did not finish):', ', '.join(sorted(missing)))
            self.steps.append(('missing-test-summaries', 1, True, 0.0))
        if self.tests:
            totals = [sum(row[i] for _, row in self.tests) for i in range(4)]
            print('Totals over completed process summaries (assertions repeated at each thread setting):', totals)
        test_steps = [status for name, status, required, elapsed in self.steps if name in expected]
        self.validation_ok = (not missing and len(test_steps) == len(self.threads)
                              and all(status == 0 for status in test_steps)
                              and all(counts[0] > 0 and counts[1] == counts[2] == 0 for _, counts in self.tests))
        if self.validation_ok:
            sets = [self.validated_types.get(name, set()) for name in expected]
            common = set.intersection(*sets)
            self.base_env['OXIBLAS_VALIDATED_TYPES'] = '\n'.join(sorted(common))
            providers = []
            if any('Double' in name for name in common):
                providers.append('DoubleFloats')
            if any('Float128' in name for name in common):
                providers.append('Quadmath')
            self.base_env['OXIBLAS_OPTIONAL_FLOATS'] = ','.join(providers)
            print('Benchmark optional providers restricted to validated types:', providers)
            if not common:
                self.validation_ok = False
            print('Numerical types tested in every thread configuration:', sorted(common))
        print('NUMERICAL VALIDATION GATE:', 'OPEN' if self.validation_ok else 'CLOSED — performance stages blocked')
        if not self.validation_ok:
            self.steps.append(('numerical-validation-gate', 1, True, 0.0))

    def prepare_rust(self):
        upstream = Path(os.environ.get('OXIBLAS_RUST_ROOT', str(ROOT.parents[1] / 'work' / 'oxiblas-upstream'))).resolve()
        if not (upstream / 'Cargo.toml').is_file():
            print('ERROR: Rust checkout is unavailable:', upstream)
            self.steps.append(('rust-checkout-missing', 1, True, 0.0))
            return
        if not self.run('rust-checkout', ['git', '-C', upstream, 'rev-parse', 'HEAD']):
            return
        expected_revision = os.environ.get('OXIBLAS_RUST_REVISION', '8d82bb6bf1df133aa90525aa9f478ebe41514f9b')
        if self.rust_revision != expected_revision:
            print('ERROR: Rust revision mismatch:', self.rust_revision, 'expected:', expected_revision)
            self.steps.append(('rust-revision-mismatch', 1, True, 0.0))
            return
        if not self.run('rust-worktree', ['git', '-C', upstream, 'status', '--porcelain', '--untracked-files=no']):
            return
        if self.rust_dirty:
            print('ERROR: tracked Rust source changes; HEAD alone does not identify the compiled source')
            self.steps.append(('rust-worktree-modified', 1, True, 0.0))
            return
        crate = self.scratch / 'rust'; crate.mkdir()
        manifest = ['[package]', 'name="oxiblas-full-report"', 'version="0.1.0"',
                    'edition="2021"', 'publish=false', '[workspace]', '[dependencies]']
        for package in ('core', 'matrix', 'blas', 'lapack', 'sparse'):
            path = upstream / 'crates' / f'oxiblas-{package}'
            features = ', features=["parallel","f128"]' if package == 'core' else ', features=["parallel"]' if package in ('blas', 'lapack') else ''
            manifest.append(f'oxiblas-{package}={{path={json.dumps(str(path))}{features}}}')
        manifest += ['num-complex="0.4"']
        for name, source in RUST_SOURCES.items():
            dest = crate / f'{name}.rs'; shutil.copyfile(ROOT / 'benchmark' / source, dest)
            manifest += ['[[bin]]', f'name="{name}"', f'path="{name}.rs"']
        path = crate / 'Cargo.toml'; path.write_text('\n'.join(manifest) + '\n')
        print('Generated temporary Rust manifest:\n', path.read_text())
        print('Rust source checkout:', upstream)
        # Separate builds let independent binaries survive another adapter failure.
        for name in RUST_SOURCES:
            if self.run('rust-build-' + name, ['cargo', 'build', '--offline', '--release', '--manifest-path', path,
                                             '--target-dir', self.scratch / 'rust-target', '--bin', name]):
                self.binaries[name] = self.scratch / 'rust-target' / 'release' / (name + ('.exe' if os.name == 'nt' else ''))

    def paired(self):
        for threads in self.threads:
            for backend in ('julia', 'openblas', 'accelerate'):
                if backend == 'accelerate' and sys.platform != 'darwin':
                    print(f'Accelerate {threads} requested threads: N/A (non-macOS)')
                    continue
                env = dict(self.base_env, OXIBLAS_FULL_BACKEND=backend,
                           OXIBLAS_FULL_THREADS=str(threads), OPENBLAS_NUM_THREADS=str(threads if backend != 'julia' else 1),
                           VECLIB_MAXIMUM_THREADS=str(threads))
                self.run(f'paired-{backend}-{threads}t', self.julia(threads, 'benchmark/archive/full_worker.jl'), env)
            if 'full' in self.binaries:
                env = dict(self.base_env, OXIBLAS_FULL_THREADS=str(threads), RAYON_NUM_THREADS=str(threads))
                self.run(f'paired-rust-{threads}t', [self.binaries['full']], env)

    def artifacts(self, directory, name):
        """Embed all textual intermediates in the only persistent report."""
        directory = Path(directory)
        if not directory.exists():
            return
        for path in ([directory] if directory.is_file() else sorted(directory.rglob('*'))):
            if not path.is_file() or path in self.seen_csv:
                continue
            self.seen_csv.add(path)
            print(f'\n--- ARTIFACT {name}/{path.name if directory.is_file() else path.relative_to(directory)} SHA256={hashlib.sha256(path.read_bytes()).hexdigest()} ---')
            if path.suffix not in ('.csv', '.txt', '.log', '.sha256', '.toml'):
                print(f'Binary intermediate: {path.stat().st_size} bytes; hash retained; data not rendered as text.')
                continue
            content = path.read_text(errors='replace'); print(content, end='' if content.endswith('\n') else '\n')
            if path.suffix == '.csv':
                for index, row in enumerate(csv.DictReader(content.splitlines())):
                    if any(k in row for k in ('median_s', 'median_update_s', 'minimum_s', 'min_s')):
                        self.legacy_rows.append((str(Path(name) / (path.name if directory.is_file() else path.relative_to(directory))), index, row))

    def legacy(self):
        print('\nADDITIONAL EXISTING BENCHMARK VARIANTS — raw output retained below.')
        print('Legacy timings are not merged into paired speedups without an equivalent-input/provider adapter.')
        compact_dir = self.scratch / 'rust-compact'; compact_dir.mkdir()
        destinations = {'gemm': [self.scratch / 'rust-gemm' / 'fixtures'],
                        'compact': [self.scratch / 'rust-compact.csv'],
                        'compact-bench': [compact_dir], 'level1': [self.scratch / 'rust-level1.csv'],
                        'spectral': [self.scratch / 'rust-spectral.csv'], 'graph': [self.scratch / 'rust-graph.csv']}
        for name, arguments in destinations.items():
            if name in self.binaries:
                self.run('legacy-rust-' + name, [self.binaries[name], *arguments])
        self.artifacts(compact_dir, 'rust-compact')
        if 'precision' in self.binaries:
            probe = self.scratch / 'rust-precision-probe.csv'
            ok = self.run('rust-precision-probe', [self.binaries['precision'], 'probe', probe])
            if ok and self.run('rust-precision-oracle', self.julia(1, 'benchmark/archive/compare_extended_reference.jl', probe, mode='test')):
                self.run('rust-precision-bench', [self.binaries['precision'], 'bench', self.scratch / 'rust-precision.csv'])
        for threads in self.threads:
            for script, kind in LEGACY.items():
                # Existing reference-capable scripts run in isolated OpenBLAS and
                # Accelerate processes; native-only scripts are not relabelled.
                backends = ('openblas', 'accelerate') if kind in ('fixed', 'compact-reference', 'reference') else ('julia',)
                for backend in backends:
                    if backend == 'accelerate' and sys.platform != 'darwin':
                        continue
                    destination = self.scratch / f'{Path(script).stem}-{backend}-{threads}t'
                    if kind in ('fixed', 'compact-reference'):
                        destination.mkdir()
                    env = dict(self.base_env, OXIBLAS_FULL_BACKEND=backend,
                               OXIBLAS_FULL_THREADS=str(threads), OXIBLAS_REFERENCE=backend,
                               OXIBLAS_REPORT_DIR=str(destination), OXIBLAS_COMPACT_REFERENCE_DIR=str(compact_dir))
                    args = []
                    if kind == 'compact-reference':
                        for fixture in compact_dir.glob('gbmv-*.bin'):
                            shutil.copyfile(fixture, destination / fixture.name)
                        env['OXIBLAS_COMPACT_REFERENCE_DIR'] = str(destination)
                    if kind == 'directory':
                        args = [destination]
                    elif kind == 'reference':
                        # Both native and reference modes are full legacy cases.
                        for mode in ('native', 'reference'):
                            target = destination / mode
                            self.run(f'legacy-{script}-{backend}-{mode}-{threads}t',
                                     self.julia(threads, 'benchmark/archive/full_legacy.jl', script, target, mode, backend), env)
                        self.artifacts(destination, destination.name)
                        continue
                    self.run(f'legacy-{script}-{backend}-{threads}t',
                             self.julia(threads, 'benchmark/archive/full_legacy.jl', script, *args), env)
                    # compact-reference writes beside its generated Rust fixtures.
                    self.artifacts(destination, destination.name)
                    if script == 'sparse_graphs.jl' and (self.scratch / 'rust-graph.csv').exists():
                        self.run(f'graph-input-comparator-{threads}t', ['python3', 'benchmark/archive/compare_graph_inputs.py',
                                 destination / 'cases-fnv64.csv', self.scratch / 'rust-graph.csv'])
            for script in ('reference_providers.jl', 'hermitian_reference_probe.jl'):
                for backend in ('openblas', 'accelerate'):
                    if backend == 'accelerate' and sys.platform != 'darwin':
                        continue
                    destination = self.scratch / f'{Path(script).stem}-{backend}-{threads}t'; destination.mkdir()
                    env = dict(self.base_env, OXIBLAS_FULL_BACKEND=backend, OXIBLAS_FULL_THREADS=str(threads),
                               OXIBLAS_REFERENCE=backend, OXIBLAS_REPORT_DIR=str(destination))
                    self.run(f'probe-{script}-{backend}-{threads}t', self.julia(threads, 'benchmark/archive/full_legacy.jl', script, mode='test'), env)
                    self.artifacts(destination, destination.name)
        for path in sorted(self.scratch.glob('rust-*.csv')):
            self.artifacts(path, path.name)
        self.artifacts(self.scratch / 'rust-gemm', 'rust-gemm')

    def tables(self):
        grouped = {}
        for row in self.rows:
            key = tuple(row[k] for k in ('group', 'operation', 'type', 'n', 'threads'))
            if row['backend'] in grouped.setdefault(key, {}):
                raise ValueError(f'Duplicate backend measurement: {key}, {row["backend"]}')
            grouped[key][row['backend']] = row
        for section_no, group in enumerate(GROUPS, 4):
            section(section_no, group + ' BENCHMARKS')
            for key, backends in grouped.items():
                if key[0] != group:
                    continue
                _, op, ty, n, threads = key
                print(f'\n{op} {ty} — N={n} — {threads} requested threads')
                print('Operation                    Julia (ms)       Rust (ms)   OpenBLAS (ms) Accelerate (ms)')
                print('-' * 96)
                measured = {}
                julia = backends.get('julia')
                for backend, row in backends.items():
                    # Equal type/shape/thread keys are insufficient: verify inputs.
                    comparable = not julia or row['fingerprint'] == julia['fingerprint']
                    if not comparable:
                        row['note'] += '; input fingerprint mismatch — comparison rejected'
                        row['status'] = 'ERROR'
                    if row['status'] == 'OK' and comparable:
                        measured[backend] = number(row['median_s'])
                cells = ['N/A' if measured.get(b) is None else f'{measured[b]*1000:.6g}'
                         for b in ('julia', 'rust', 'openblas', 'accelerate')]
                print(f'{op:<28}' + ''.join(f'{cell:>15}' for cell in cells))
                native = measured.get('julia')
                for backend in ('julia', 'rust', 'openblas', 'accelerate'):
                    row = backends.get(backend)
                    if not row:
                        print(f'  {backend}: N/A — ' + ('non-macOS' if backend == 'accelerate' and sys.platform != 'darwin' else 'no equivalent adapter/result available'))
                        continue
                    print(f'  {backend}: {row["status"]}; provider={row["provider"]}; {row["note"]}; input={row["fingerprint"]}')
                    if row['status'] == 'OK':
                        print(f'    median_s={row["median_s"]}; min_s={row["min_s"]}; first_s={row["first_s"]}; compile_s={row["compile_s"]}; GFLOP/s={row["gflops"]}')
                        if backend != 'rust':
                            print(f'    Julia allocations: bytes={row["bytes"]}; count={row["allocations"]}')
                        if backend != 'julia' and native and measured.get(backend):
                            print(f'    speedup OxiBLAS.jl vs {backend}: {measured[backend]/native:.4g}x (reference/Julia median)')
            # Every additional legacy measurement gets four visible columns. It
            # is retained individually, never paired solely on names and sizes.
        print('\nLEGACY VARIANT TABLES (not paired; raw CSVs above retain all metadata)')
        print('Operation | Julia (ms) | Rust (ms) | OpenBLAS (ms) | Accelerate (ms) | source / contract')
        for source, index, row in self.legacy_rows:
            operation = row.get('operation', row.get('case', 'gemm'))
            algorithm = row.get('algorithm', '')
            seconds = number(row.get('median_s', row.get('median_update_s')))
            is_rust = source.startswith('rust-')
            is_native = algorithm not in ('reference_blas', 'v02', 'v04', 'openblas', 'accelerate', 'lapack', 'reference') and not is_rust
            value = f'{seconds*1000:.6g}' if seconds is not None and is_native and '-reference-' not in source and '/reference/' not in source else 'N/A'
            rust_value = f'{seconds*1000:.6g}' if seconds is not None and is_rust else 'N/A'
            print(f'{operation} | {value} | {rust_value} | N/A | N/A | {source} row={index+1}; {algorithm}; unpaired variant; see raw timings (minimum-only rows not relabelled median)')
        return grouped

    def summary(self, grouped):
        section(12, 'MULTITHREAD SCALING')
        baseline = {}
        for key, backends in grouped.items():
            for backend, row in backends.items():
                if key[-1] == '1' and row['status'] == 'OK':
                    baseline[(key[:-1], backend)] = number(row['median_s'])
        for key, backends in grouped.items():
            for backend, row in backends.items():
                base = baseline.get((key[:-1], backend)); current = number(row['median_s'])
                if base and current and row['status'] == 'OK':
                    print(f'{key}: {backend}; t1/tN={base/current:.5g}; requested threads={key[-1]}; Accelerate actual per-call workers unmeasured')
        section(13, 'ALLOCATIONS')
        print('Measured Julia heap allocations include wrapper/runtime allocations, not foreign Rust/BLAS/framework heap allocations.')
        for row in self.rows:
            if row['status'] == 'OK' and row['backend'] != 'rust':
                print(' | '.join(row[k] for k in ('group','operation','type','n','threads','backend','bytes','allocations')))
        section(14, 'FINAL SUMMARY')
        failed = [name for name, status, required, _ in self.steps if status and required]
        errors = [row for row in self.rows if row['status'] == 'ERROR']
        print('Mandatory commands:', len([s for s in self.steps if s[2]]), '; failed:', len(failed))
        print('Failed stages:', failed)
        print('Completed benchmark records:', sum(row['status'] == 'OK' for row in self.rows), '; failed records:', len(errors))
        print('Unavailable/skipped records:', sum(row['status'] in ('N/A','SKIP') for row in self.rows))
        missing = sum(sum(b not in values or values[b]['status'] != 'OK' for b in ('rust','openblas','accelerate')) for values in grouped.values())
        print('Missing comparisons (including explicit N/A):', missing)
        print('Legacy variant records retained:', len(self.legacy_rows))
        print('Performance regressions vs v0.5: NOT DETERMINED — historical data are preserved; no equal-contract baseline inferred.')
        for backend in ('rust','openblas','accelerate'):
            ratios=[]
            for values in grouped.values():
                j, r = values.get('julia'), values.get(backend)
                if j and r and j['status']==r['status']=='OK' and number(j['median_s']) and number(r['median_s']):
                    ratios.append(number(r['median_s'])/number(j['median_s']))
            if ratios:
                print(f'Julia vs {backend}: {len(ratios)} comparable cases; geometric mean reference/Julia={math.exp(sum(map(math.log,ratios))/len(ratios)):.5g}x; individual tables are authoritative')
            else:
                print(f'Julia vs {backend}: N/A — no successful equivalent measurements')
        print('No LLVM inspection performed. Review the full report before proposing fixes or a further campaign.')
        print('FINAL RESULT:', 'FAILED' if failed or errors else 'COMPLETED (N/A is not a passed comparison)')
        return 1 if failed or errors else 0


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='oxiblas-full-') as scratch:
        campaign=Campaign(scratch, None)
        try:
            campaign.system()
            campaign.test_suite()
            print('\nRUST RELEASE PREPARATION (offline; errors preserved; no implicit network installation)')
            if campaign.validation_ok:
                campaign.prepare_rust()
            if campaign.validation_ok:
                campaign.paired()
                campaign.legacy()
            else:
                print('All performance stages blocked: numerical test validation was not established.')
            grouped=campaign.tables()
            return campaign.summary(grouped)
        except Exception:
            traceback.print_exc()
            section(14, 'FINAL SUMMARY — ORCHESTRATION FAILURE')
            print('Report is partial; failure is not success. Completed steps:', campaign.steps)
            return 1


if __name__ == '__main__':
    sys.exit(main())
