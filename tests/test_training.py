import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import pickle
import networkx as nx
import numpy as np
import torch
from models.pytorch_base_model import PyTorchBaseModel
from models.graph_wavenet import GraphWaveNetModel


class Probe(PyTorchBaseModel):
    def build_model(self, static_features=None):
        return torch.nn.Linear(1, 1, bias=False)


class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.config = {'n_targets': 1, 'output_steps': 1, 'device': 'cpu', 'epochs': 2, 'patience': 1}

    def test_default_mse_and_singleton(self):
        model = Probe(self.config)
        self.assertEqual(model.weight_alpha, 0)
        self.assertEqual(model._weighted_loss(torch.tensor([0.]), torch.tensor([1.])).item(), 1)

    def test_weighted_validation_independent_of_batch_size_and_order(self):
        model = Probe({**self.config, 'weight_alpha': 2})
        model.model = model.build_model()
        with torch.no_grad():
            model.model.weight.zero_()
        x = np.zeros((7, 1), dtype=np.float32)
        y = np.array([[0.], [1.], [2.], [3.], [10.], [30.], [100.]], dtype=np.float32)
        model.target_mean = torch.tensor(y.mean(axis=0))
        model.target_std = torch.tensor(y.std(axis=0))
        expected = model._weighted_loss(torch.zeros_like(torch.from_numpy(y)), torch.from_numpy(y)).item()
        for batch_size in (1, 2, 3, 7):
            model.batch_size = batch_size
            for shuffle in (False, True):
                actual = model._validate_epoch(model._create_dataloader(x, y, shuffle=shuffle))
                self.assertAlmostEqual(actual, expected, delta=expected * 1e-6)

    def test_constant_weighted_targets_stay_finite(self):
        model = Probe({**self.config, 'epochs': 1, 'weight_alpha': 2})
        x, y = np.ones((1, 1), dtype=np.float32), np.ones((1, 1), dtype=np.float32)
        model.train(x, y, x, y)
        self.assertTrue(np.isfinite(model.predict(x)).all())

    def test_early_stopping_restores_independent_snapshot(self):
        model = Probe(self.config)
        model.model = model.build_model()
        with torch.no_grad():
            model.model.weight.zero_()
        def advance(loader, optimizer):
            with torch.no_grad():
                model.model.weight.add_(1)
            return 0
        with patch.object(model, '_train_epoch', side_effect=advance), patch.object(model, '_validate_epoch', side_effect=[1., 2.]):
            model.train(np.zeros((2, 1), dtype=np.float32), np.ones((2, 1), dtype=np.float32),
                        np.zeros((2, 1), dtype=np.float32), np.ones((2, 1), dtype=np.float32))
        self.assertEqual(model.model.weight.item(), 1)

    def test_graph_checkpoint_restores_scalers_and_graph_without_original_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            graph_path = root / 'graph.pkl'
            with graph_path.open('wb') as stream:
                pickle.dump(nx.complete_graph(['a', 'b']), stream)
            config = {**self.config, 'epochs': 1, 'weight_alpha': 2, 'graph_path': str(graph_path),
                      'cell_order': ['a', 'b'], 'input_steps': 4, 'output_steps': 2,
                      'blocks': 1, 'layers': 2, 'apt_layer': False, 'randomadj': False,
                      'nhid': 2, 'residual_channels': 2, 'dilation_channels': 2,
                      'skip_channels': 4, 'end_channels': 8, 'batch_size': 2, 'dropout': 0}
            rng = np.random.default_rng(42)
            x = rng.normal(size=(4, 4, 2, 1)).astype(np.float32)
            y = rng.normal(size=(4, 2, 2, 1)).astype(np.float32)
            model = GraphWaveNetModel(config)
            model.train(x, y, x[:2], y[:2])
            before = model.predict(x[:2])
            checkpoint = root / 'model.pt'
            model.save(checkpoint)
            graph_path.unlink()
            restored = GraphWaveNetModel.load(checkpoint)
            np.testing.assert_allclose(restored.predict(x[:2]), before, rtol=1e-6, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
