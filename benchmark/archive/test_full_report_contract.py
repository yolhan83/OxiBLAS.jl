"""PREPARED ONLY: runner contract tests with mocks; no Julia/Rust subprocesses.
Future authorized command: python3 -m unittest discover -s test -p test_full_report_contract.py
"""
import contextlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('full_report',Path(__file__).resolve().parents[2]/'benchmark/archive/full_report.py')
report=importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


class LauncherContract(unittest.TestCase):
    def test_bounds_modes_and_historical_wrapper(self):
        campaign=report.Campaign('/tmp/not-used',None)
        self.assertIn('--check-bounds=yes',campaign.julia(2,'test/legacy/run_all_supervised.jl',mode='test'))
        self.assertIn('--check-bounds=auto',campaign.julia(2,'benchmark/archive/full_worker.jl'))
        self.assertIn('--check-bounds=auto',campaign.julia(2,'benchmark/archive/full_legacy.jl','compact.jl'))
        with self.assertRaises(ValueError):campaign.julia(2,'unused',mode='unknown')

    def test_failed_gate_blocks_performance_and_rust_build(self):
        called=[]
        class ClosedCampaign:
            def __init__(self,*args):self.validation_ok=False;self.steps=[]
            def system(self):called.append('system')
            def test_suite(self):called.append('tests')
            def prepare_rust(self):raise AssertionError('build entered after failed gate')
            def paired(self):raise AssertionError('benchmark entered after failed gate')
            def legacy(self):raise AssertionError('legacy benchmark entered after failed gate')
            def tables(self):return {}
            def summary(self,grouped):return 1
        with patch.object(report,'Campaign',ClosedCampaign),patch('sys.argv',['run_all.sh']),contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(report.main(),1)
        self.assertEqual(called,['system','tests'])

    def test_errors_and_exit_codes_are_retained(self):
        class Process:
            stdout=io.StringIO('diagnostic from combined stdout/stderr\n@@OXI_ROW\tmalformed\n')
            def __enter__(self):return self
            def __exit__(self,*args):return False
            def wait(self):return 0
        campaign=report.Campaign('/tmp/not-used',None);output=io.StringIO()
        with patch.object(report.subprocess,'Popen',return_value=Process()),contextlib.redirect_stdout(output),contextlib.redirect_stderr(output):
            self.assertFalse(campaign.run('bad-row',['not-executed']))
        self.assertIn('diagnostic from combined stdout/stderr',output.getvalue())
        self.assertIn('Malformed measurement record',output.getvalue())
        self.assertNotEqual(campaign.steps[-1][1],0)

    def test_all_four_columns_survive_missing_providers(self):
        campaign=report.Campaign('/tmp/not-used',None)
        row=dict(zip(report.FIELDS,['BLAS LEVEL 3','gemm','Float64','16','1','julia','OK',
                                  '.001','.0009','0','0','.01','.001','1','native','test fixture','same-input']))
        campaign.rows=[row];output=io.StringIO()
        with contextlib.redirect_stdout(output):campaign.tables()
        for name in ('Julia (ms)','Rust (ms)','OpenBLAS (ms)','Accelerate (ms)'):
            self.assertIn(name,output.getvalue())
        self.assertIn('N/A',output.getvalue())


if __name__=='__main__':unittest.main()
