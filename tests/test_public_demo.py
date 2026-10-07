import tempfile
import json
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from src.demo import generate_synthetic_data
from src.graph import build_handover_graph
from src.configuration import load_config, PROJECT_ROOT
from scripts.audit_public import inspect_body
from scripts.export_demo_report import export_report


class PublicDemoTests(unittest.TestCase):
    def test_graph_excludes_future_handovers_and_aggregates_directionally(self):
        events = pd.DataFrame([
            ['2025-01-01', 'SYN0001-10', 'SYN0001-20', 3],
            ['2025-01-02', 'SYN0001-10', 'SYN0001-20', 4],
            ['2025-01-02', 'SYN0001-20', 'SYN0001-10', 2],
            ['2025-01-03', 'SYN0001-10', 'SYN0001-20', 1000],
        ], columns=['eventdate', 'CELL_source', 'CELL_target', 'handover_count'])
        graph = build_handover_graph(events, ['SYN0001-10', 'SYN0001-20', 'SYN0002-10'], end='2025-01-03')
        self.assertEqual(graph['SYN0001-10']['SYN0001-20']['weight'], 7)
        self.assertEqual(graph['SYN0001-20']['SYN0001-10']['weight'], 2)
        self.assertIn('SYN0002-10', graph)

    def test_graph_rejects_invalid_counts(self):
        events = pd.DataFrame({'CELL_source': ['a'], 'CELL_target': ['b'], 'handover_count': [-1]})
        with self.assertRaisesRegex(ValueError, 'nonnegative'):
            build_handover_graph(events)

    def test_generator_is_reproducible_and_contains_only_fictional_cells(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / 'a', Path(directory) / 'b'
            paths1 = generate_synthetic_data(a, days=3)
            paths2 = generate_synthetic_data(b, days=3)
            for p1, p2 in zip(paths1, paths2):
                self.assertEqual(p1.read_bytes(), p2.read_bytes())
            self.assertEqual(generate_synthetic_data(a, days=3), paths1)
            frame = pd.read_csv(paths1[0])
            self.assertEqual(frame['CELL_'].nunique(), 12)
            self.assertTrue(frame['CELL_'].str.startswith('SYN').all())
            self.assertTrue(np.isfinite(frame.select_dtypes('number')).all().all())

    def test_config_resolves_project_paths_without_changing_working_directory(self):
        config = load_config(PROJECT_ROOT / 'configs/demo.yaml')
        self.assertEqual(Path(config['data']['data_dir']), PROJECT_ROOT / 'data/demo/raw')

    def test_public_audit_detects_private_paths_and_cached_notebook_outputs(self):
        body = ('/' + 'home/' + 'example/private').encode()
        self.assertIn('private_absolute_path', inspect_body(Path('config.yaml'), body))
        notebook = {'cells': [{'outputs': [{'text': 'cached output'}], 'execution_count': 1}]}
        self.assertIn('cached_notebook_output', inspect_body(Path('analysis.ipynb'), json.dumps(notebook).encode()))

    def test_report_export_rejects_non_synthetic_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'source_manifest.json').write_text(json.dumps({'synthetic': False}))
            result_file = root / 'result.json'
            result_file.write_text(json.dumps({'config': {'data': {'data_dir': str(root)}}}))
            with self.assertRaisesRegex(ValueError, 'synthetic'):
                export_report(result_file, root / 'public')
            self.assertFalse((root / 'public').exists())


if __name__ == '__main__':
    unittest.main()
