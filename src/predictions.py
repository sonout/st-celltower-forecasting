"""Load predictions only when their original dataset can be verified."""

import json
from pathlib import Path
import numpy as np
import yaml
from .datasets import verify_manifest


def load_prediction_data(prediction_file: Path) -> dict:
    with np.load(prediction_file, allow_pickle=False) as archive:
        predictions = archive['predictions']
        pred_metadata = json.loads(archive['metadata'].item())
    if 'dataset_metadata' not in pred_metadata or not pred_metadata.get('dataset_manifest'):
        raise ValueError('Legacy predictions have no dataset fingerprint. Their ground truth cannot be verified; regenerate predictions using a versioned dataset.')
    metadata_path = Path(pred_metadata['preprocessing_metadata_path'])
    data_dir = metadata_path.parent
    verify_manifest(data_dir, pred_metadata['dataset_manifest'])
    metadata = yaml.safe_load(metadata_path.read_text())
    if metadata != pred_metadata['dataset_metadata']:
        raise ValueError('Dataset metadata differs from the snapshot saved with predictions.')
    x_test = np.load(data_dir / 'X_test.npy', mmap_mode='r')
    y_test = np.load(data_dir / 'Y_test.npy', mmap_mode='r')
    limit = pred_metadata.get('sample_limit')
    if limit:
        x_test, y_test = x_test[:limit], y_test[:limit]
    if predictions.shape != y_test.shape:
        raise ValueError('Prediction dimensions differ from the saved dataset targets.')
    return {'predictions': predictions, 'x_test': x_test, 'y_test': y_test,
            'metadata': metadata, 'pred_metadata': pred_metadata}
