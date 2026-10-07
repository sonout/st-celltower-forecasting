import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
import yaml
from src.preprocessing import main as preprocess
from src.pipeline import TelecomPipeline
from src.predictions import load_prediction_data
from src.datasets import resolve_dataset, verify_manifest


class PipelineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.raw = self.root / 'raw'
        self.raw.mkdir()
        times = pd.date_range('2025-01-01', periods=64, freq='15min')
        records = [dict(CELL_=cell, eventdate=stamp, KPI=float(i + offset))
                   for cell, offset in [('SYN0001-10', 1), ('SYN0001-20', 100)]
                   for i, stamp in enumerate(times)]
        pd.DataFrame(records).to_csv(self.raw / 'CELL_test.csv', index=False)
        self.config = {
            'data': {'data_dir': str(self.raw), 'output_dir': str(self.root / 'prepared'),
                     'input_steps': 4, 'output_steps': 2, 'granularity_min': 15,
                     'essential_columns': ['CELL_', 'eventdate'], 'kpi_columns': ['KPI'],
                     'add_timefeats': False, 'lag_hours': [], 'rolling_avg_hours': None},
            'data_split': {'train_start': '2025-01-01', 'train_end': '2025-01-01 08:00',
                           'val_start': '2025-01-01 08:00', 'val_end': '2025-01-01 12:00',
                           'test_start': '2025-01-01 12:00', 'test_end': '2025-01-01 16:00'},
            'target_kpis': ['KPI'], 'horizons': {'15m': 1, '30m': 2},
            'experiment': {'name': 'regression', 'device': 'cpu', 'models': ['ha', 'seasonal_naive', 'random_forest', 'xgboost', 'arima'],
                           'output_dir': str(self.root / 'results'), 'random_seed': 42,
                           'store_predictions': True, 'store_predictions_folder': str(self.root / 'predictions'),
                           'sample_limit': 4,
                           'model_overrides': {'random_forest': {'model_params': {'n_estimators': 2, 'max_depth': 2}},
                                               'xgboost': {'model_params': {'n_estimators': 2, 'max_depth': 2}},
                                               'arima': {'model_params': {'p': 0, 'd': 0, 'q': 0, 'seasonal': False}},
                                               'seasonal_naive': {'model_params': {'seasonal_period': 4}}}},
        }
        config_path = self.root / 'config.yaml'
        config_path.write_text(yaml.safe_dump(self.config))
        preprocess(str(config_path))
        self.config_path = config_path

    def test_all_sequence_baselines_and_prediction_provenance(self):
        pipeline = TelecomPipeline(self.config)
        result = pipeline.run_experiment()
        self.assertEqual(set(result), set(self.config['experiment']['models']))
        saved = json.loads(pipeline.results_path.read_text())
        self.assertIn('model_params', saved['model_configs']['random_forest'])
        self.assertEqual(saved['dataset_metadata']['schema_version'], 2)
        self.assertIn('torch', saved['runtime'])
        predictions = sorted((self.root / 'predictions').glob('*.npz'))
        self.assertEqual(len(predictions), 5)
        for path in predictions:
            data = load_prediction_data(path)
            self.assertEqual(data['predictions'].shape, (4, 2, 2, 1))
            self.assertTrue(np.isfinite(data['predictions']).all())

    def test_seasonal_naive_uses_only_observed_test_history(self):
        pipeline = TelecomPipeline(self.config)
        pipeline._load_data()
        model = pipeline._load_models(['seasonal_naive'])['seasonal_naive']
        model.train(pipeline.X_train, pipeline.Y_train)
        actual = model.predict(pipeline.X_test)
        np.testing.assert_equal(actual[:, 0], pipeline.X_test[:, 0])
        np.testing.assert_equal(actual[:, 1], pipeline.X_test[:, 1])

    def test_preprocessing_creates_new_version_without_changing_old_data(self):
        root = Path(self.config['data']['output_dir'])
        first = resolve_dataset(root)
        manifest = verify_manifest(first)
        preprocess(str(self.config_path))
        second = resolve_dataset(root)
        self.assertNotEqual(first, second)
        self.assertEqual(verify_manifest(first), manifest)

    def test_manifest_detects_modified_ground_truth(self):
        dataset = resolve_dataset(self.config['data']['output_dir'])
        with (dataset / 'Y_test.npy').open('ab') as stream:
            stream.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'changed after preprocessing'):
            verify_manifest(dataset)

    def test_stale_kpi_config_is_rejected(self):
        self.config['target_kpis'] = ['OTHER']
        with self.assertRaisesRegex(ValueError, 'target KPIs differ'):
            TelecomPipeline(self.config)._load_data()

    def test_unverifiable_legacy_predictions_are_rejected(self):
        path = self.root / 'legacy.npz'
        np.savez(path, predictions=np.zeros((1, 1, 1, 1)), metadata=json.dumps({'preprocessing_metadata_path': 'overwritten.yaml'}))
        with self.assertRaisesRegex(ValueError, 'no dataset fingerprint'):
            load_prediction_data(path)


if __name__ == '__main__':
    unittest.main()
