import unittest
import numpy as np
import pandas as pd
from src.preprocessing import create_complete_index, impute_missing_values, create_sequences, split_sequences, create_static_features


class PreprocessingTests(unittest.TestCase):
    def frame(self):
        times = pd.date_range('2025-01-01', periods=80, freq='15min')
        index = pd.MultiIndex.from_product([['SYN0001-10', 'SYN0001-24'], times], names=['CELL_', 'eventdate'])
        values = np.arange(len(index), dtype=float)
        return pd.DataFrame({'a': values, 'b': values + 1000}, index=index)

    def sequences(self):
        return create_sequences(self.frame(), 4, 3, ['a', 'b'], ['b'])

    def test_multifeature_cell_order_is_preserved(self):
        x, y, stamps, cells = self.sequences()
        frame = self.frame()
        for cell_idx, cell in enumerate(cells):
            np.testing.assert_equal(x[0, 0, cell_idx], frame.loc[(cell, stamps[0][0]), ['a', 'b']])
            np.testing.assert_equal(y[0, 0, cell_idx, 0], frame.loc[(cell, stamps[0][1] + pd.Timedelta(minutes=15)), 'b'])

    def test_ratio_splits_have_disjoint_targets(self):
        x, y, stamps, _ = self.sequences()
        splits = split_sequences(x, y, stamps, .6, .2)
        for left, right in [('train', 'val'), ('val', 'test')]:
            self.assertLess(splits[f'timestamps_{left}'][-1][2], splits[f'timestamps_{right}'][0][1] + pd.Timedelta(minutes=15))

    def test_date_splits_use_entire_target_interval(self):
        x, y, stamps, _ = self.sequences()
        boundaries = {'train_start': '2025-01-01', 'train_end': '2025-01-01 08:00',
                      'val_start': '2025-01-01 08:00', 'val_end': '2025-01-01 12:00',
                      'test_start': '2025-01-01 12:00', 'test_end': '2025-01-01 20:00'}
        splits = split_sequences(x, y, stamps, data_split=boundaries)
        for split in ('train', 'val', 'test'):
            for _, origin, end in splits[f'timestamps_{split}']:
                self.assertGreaterEqual(origin + pd.Timedelta(minutes=15), pd.Timestamp(boundaries[f'{split}_start']))
                self.assertLess(end, pd.Timestamp(boundaries[f'{split}_end']))

    def test_imputation_never_reads_future_values(self):
        raw = pd.DataFrame({'CELL_': ['a'] * 4, 'eventdate': pd.date_range('2025-01-01', periods=4, freq='15min'), 'a': [np.nan, 1., np.nan, 9.]})
        filled = impute_missing_values(create_complete_index(raw, 15), ['a'])
        np.testing.assert_array_equal(filled['a'].to_numpy(), [0., 1., 1., 9.])

    def test_static_features_are_numeric_and_finite_for_unknown_band(self):
        values = create_static_features(['SYN0001-10', 'SYN0001-24']).to_numpy(dtype=np.float32)
        self.assertTrue(np.isfinite(values).all())


if __name__ == '__main__':
    unittest.main()
