"""Run a configured telecom forecasting experiment."""

import argparse
from pathlib import Path
from src.pipeline import TelecomPipeline
from src.configuration import load_config, DEFAULT_CONFIG, PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--config', help='Experiment YAML; defaults to configs/experiment_example.yaml.')
    source.add_argument('--demo', action='store_true', help='Generate fictional data and run the synthetic demo.')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda:0'])
    parser.add_argument('--data-dir', help='Prepared dataset root or a specific version directory.')
    parser.add_argument('--dataset-id', help='Select a version instead of latest.json.')
    parser.add_argument('--output-dir', help='Results directory.')
    parser.add_argument('--models', nargs='+', help='graph_wavenet historical_average seasonal_naive random_forest arima xgboost')
    parser.add_argument('--graph-path', help='Handover graph file for Graph WaveNet.')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--num-workers', type=int)
    parser.add_argument('--weight-alpha', type=float)
    parser.add_argument('--sample-limit', type=int, help='Use the first N samples of each split for a smoke test.')
    parser.add_argument('--allow-legacy-data', action='store_true', help='Permit old overlapping datasets for diagnostics.')
    args = parser.parse_args()
    config_path = (PROJECT_ROOT / 'configs/demo.yaml') if args.demo else Path(args.config).resolve() if args.config else DEFAULT_CONFIG
    data_dir = Path(args.data_dir).resolve() if args.data_dir else None
    output_dir = Path(args.output_dir).resolve() if args.output_dir else None
    if args.demo:
        from src.demo import prepare_demo
        prepare_demo()
    config = load_config(config_path)
    experiment = config['experiment']
    for key in ('device', 'models', 'sample_limit'):
        value = getattr(args, key)
        if value is not None:
            experiment[key] = value
    if args.sample_limit is not None and args.sample_limit < 1:
        parser.error('--sample-limit must be positive.')
    if args.allow_legacy_data:
        experiment['allow_legacy_data'] = True
    if data_dir is not None:
        config['data']['output_dir'] = str(data_dir)
    if args.dataset_id:
        config['data']['dataset_id'] = args.dataset_id
    if output_dir is not None:
        experiment['output_dir'] = str(output_dir)
        experiment['store_predictions_folder'] = str(output_dir / 'predictions')
    overrides = experiment.setdefault('model_overrides', {}).setdefault('graph_wavenet', {})
    if args.graph_path:
        overrides['graph_path'] = str(Path(args.graph_path).resolve())
    for key in ('epochs', 'batch_size', 'num_workers', 'weight_alpha'):
        value = getattr(args, key)
        if value is not None:
            overrides[key] = value
    print(f"Running experiment: {experiment['name']}")
    pipeline = TelecomPipeline(config)
    pipeline.run_experiment()
    print(f'Experiment completed. Results saved to: {pipeline.results_path}')


if __name__ == '__main__':
    main()
