"""Export verified synthetic metrics and one forecast figure, never raw private results."""

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from src.datasets import file_sha256, verify_manifest
from src.predictions import load_prediction_data
from src.utils import calculate_metrics

MODEL_FIELDS = {
    'name', 'model_params', 'dropout', 'blocks', 'layers', 'gcn_bool', 'addaptadj',
    'adjtype', 'randomadj', 'aptonly', 'kernel_size', 'apt_layer', 'weight_alpha',
    'residual_channels', 'dilation_channels', 'skip_channels', 'end_channels',
    'learning_rate', 'weight_decay', 'epochs', 'batch_size', 'patience', 'num_workers',
    'device', 'input_steps', 'output_steps', 'random_seed', 'graph_sha256',
}


def export_report(result_file, output_dir):
    result = json.loads(Path(result_file).read_text())
    config = result['config']
    source_dir = Path(config['data']['data_dir'])
    source = json.loads((source_dir / 'source_manifest.json').read_text())
    if source.get('synthetic') is not True:
        raise ValueError('Only explicitly synthetic data may be exported.')
    verify_manifest(source_dir, source)
    metadata = result['dataset_metadata']
    if not all(cell.startswith('SYN') for cell in metadata['cell_order']):
        raise ValueError('The dataset does not contain exclusively fictional cell identifiers.')
    verify_manifest(Path(result['dataset_path']), result['dataset_manifest'])
    prediction_dir = Path(config['experiment']['store_predictions_folder'])
    graph_prediction = None
    for model, targets in result['results'].items():
        paths = list(prediction_dir.glob(f'{model}_*_{result["run_id"]}.npz'))
        if len(paths) != 1:
            raise ValueError(f'Expected exactly one prediction archive for {model}.')
        loaded = load_prediction_data(paths[0])
        if loaded['pred_metadata']['dataset_manifest'] != result['dataset_manifest']:
            raise ValueError('Results and predictions have different dataset fingerprints.')
        for target, horizons in targets.items():
            column = metadata['target_columns'].index(target)
            for horizon, stored in horizons.items():
                step = config['horizons'][horizon] - 1
                actual = calculate_metrics(loaded['y_test'][:, step, :, column].flatten(),
                                           loaded['predictions'][:, step, :, column].flatten())
                if any(not np.isclose(actual[k], value, rtol=1e-6, atol=1e-6)
                       for k, value in stored.items()):
                    raise ValueError(f'Metric verification failed for {model}/{target}/{horizon}.')
        if model == 'graph_wavenet':
            graph_prediction = loaded
    # Whitelist fields rather than publishing resolved local paths or raw result metadata.
    epochs = result['model_configs'].get('graph_wavenet', {}).get('epochs')
    command = 'python main.py --demo --models ' + ' '.join(result['results'])
    if epochs is not None:
        command += f' --epochs {epochs}'
    if config['experiment'].get('sample_limit'):
        command += f' --sample-limit {config["experiment"]["sample_limit"]}'
    report = {
        'synthetic': True,
        'purpose': 'Execution demonstration only; not a benchmark on real telecom data.',
        'command': command,
        'source': source,
        'run_id': result['run_id'],
        'seed': config['experiment']['random_seed'],
        'horizons': config['horizons'],
        'evaluation': 'Exact forecast steps, aggregated across test origins and cells; MAPE excludes zero targets.',
        'dataset': {key: metadata[key] for key in (
            'granularity_min', 'input_steps', 'output_steps', 'n_cells', 'n_features',
            'n_targets', 'feature_columns', 'target_columns', 'data_split',
            'train_sequences', 'val_sequences', 'test_sequences', 'imputation', 'split_strategy')},
        'dataset_sha256': result['dataset_manifest']['sha256'],
        'model_configs': {name: {key: value for key, value in cfg.items() if key in MODEL_FIELDS}
                          for name, cfg in result['model_configs'].items()},
        'metrics': result['results'],
        'result_file_sha256': file_sha256(Path(result_file)),
        'runtime_source': 'training result' if 'runtime' in result else 'report export environment; original run did not record versions',
        'runtime': result.get('runtime', {'python': platform.python_version(), **{package: version(package) for package in (
            'torch', 'numpy', 'pandas', 'scipy', 'scikit-learn', 'statsmodels', 'xgboost', 'networkx')}},
        ),
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'demo_metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    if graph_prediction is not None:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        loaded = graph_prediction
        sample, cell, target = 0, 0, metadata['target_columns'].index('MAX_DL_ACTIVE_UE')
        history = loaded['x_test'][sample, :, cell, metadata['feature_columns'].index('MAX_DL_ACTIVE_UE')]
        truth = loaded['y_test'][sample, :, cell, target]
        pred = loaded['predictions'][sample, :, cell, target]
        delta = metadata['granularity_min'] / 60
        fig, ax = plt.subplots(figsize=(9, 3.8), constrained_layout=True)
        ax.plot(np.arange(-len(history) + 1, 1) * delta, history, color='#536675', label='Observed history')
        ax.plot(np.arange(1, len(truth) + 1) * delta, truth, color='#16805d', label='Synthetic future')
        ax.plot(np.arange(1, len(pred) + 1) * delta, pred, color='#d75e2c', label='Graph WaveNet forecast')
        ax.axvline(0, color='#adb8c2', linestyle='--', linewidth=1)
        ax.set(xlabel='Hours from forecast origin', ylabel='Active users (synthetic)',
               title=f'Fictional cell {metadata["cell_order"][cell]} · first test origin · {epochs}-epoch demonstration')
        ax.grid(alpha=.15)
        ax.legend(frameon=False, loc='upper left')
        fig.savefig(output / 'forecast_example.png', dpi=160)
        plt.close(fig)
    print(f'Exported verified, synthetic-only metrics and figure to {output}.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result-file', required=True)
    parser.add_argument('--output-dir', default=str(ROOT / 'assets'))
    args = parser.parse_args()
    export_report(args.result_file, args.output_dir)


if __name__ == '__main__':
    main()
