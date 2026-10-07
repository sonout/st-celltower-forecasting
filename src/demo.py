"""Reproducible fictional telecom observations; no private source data is used."""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .configuration import PROJECT_ROOT, load_config
from .datasets import file_sha256, verify_manifest
from .graph import build_handover_graph, save_graph


def generate_synthetic_data(output_dir, n_cells=12, days=6, seed=42):
    root = Path(output_dir)
    params = {'n_cells': n_cells, 'days': days, 'seed': seed, 'generator_version': 1}
    if (root / 'source_manifest.json').exists():
        manifest = json.loads((root / 'source_manifest.json').read_text())
        if manifest.get('parameters') != params:
            raise ValueError('Existing synthetic data uses different parameters; choose a new output directory.')
        verify_manifest(root, manifest)
        return root / 'CELL_synthetic.csv', root / 'HO_synthetic.csv'
    if n_cells < 2 or days < 3:
        raise ValueError('Synthetic examples need at least two cells and three days.')
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'CELL_synthetic.csv').exists() or (root / 'HO_synthetic.csv').exists():
        raise FileExistsError('Synthetic files already exist without a manifest; choose a new directory.')
    rng = np.random.default_rng(seed)
    cells = [f'SYN{i // 3 + 1:04d}-{i % 3 + 1}0' for i in range(n_cells)]
    times = pd.date_range('2025-01-01', periods=days * 96, freq='15min')
    offsets = rng.uniform(-0.6, 0.6, n_cells)
    previous = np.full(n_cells, 18., dtype=float)
    records, handovers = [], []
    for t, stamp in enumerate(times):
        daily = 18 + 7 * np.sin(2 * np.pi * t / 96 - 0.6 + offsets)
        neighboring = (np.roll(previous, 1) + np.roll(previous, -1)) / 2
        load = np.clip(0.35 * previous + 0.15 * neighboring + 0.5 * daily + rng.normal(0, 0.6, n_cells), 1, None)
        throughput = np.clip(42000 - 650 * load + 3000 * np.sin(2 * np.pi * t / 96) + rng.normal(0, 450, n_cells), 1000, None)
        utilization = np.clip(2.1 * load + rng.normal(0, 1.5, n_cells), 0, 100)
        for i, cell in enumerate(cells):
            records.append({'eventdate': stamp.isoformat(), 'CELL_': cell,
                            'MAX_DL_ACTIVE_UE': round(float(load[i]), 3),
                            'DL_USER_THPUT_KBPS': round(float(throughput[i]), 3),
                            'PERC_USE_PRBS_DL': round(float(utilization[i]), 3)})
            if t % 4 == 0:
                for direction, scale in [(1, 1.), (-1, .6)]:
                    handovers.append({'eventdate': stamp.isoformat(), 'CELL_source': cell,
                                      'CELL_target': cells[(i + direction) % n_cells],
                                      'handover_count': int(rng.poisson(3 + scale * load[i]))})
        previous = load
    kpi_path, ho_path = root / 'CELL_synthetic.csv', root / 'HO_synthetic.csv'
    pd.DataFrame(records).to_csv(kpi_path, index=False)
    pd.DataFrame(handovers).to_csv(ho_path, index=False)
    manifest = {'synthetic': True, 'parameters': params, 'sha256': {
        kpi_path.name: file_sha256(kpi_path), ho_path.name: file_sha256(ho_path),
    }}
    (root / 'source_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return kpi_path, ho_path


def prepare_demo():
    from .preprocessing import main as preprocess
    config_path = PROJECT_ROOT / 'configs/demo.yaml'
    config = load_config(config_path)
    kpi_path, ho_path = generate_synthetic_data(config['data']['data_dir'])
    cells = pd.read_csv(kpi_path, usecols=['CELL_'])['CELL_'].unique()
    graph = build_handover_graph(pd.read_csv(ho_path), cells,
                                start=config['data_split']['train_start'], end=config['data_split']['train_end'])
    path = config['experiment']['model_overrides']['graph_wavenet']['graph_path']
    save_graph(graph, PROJECT_ROOT / path)
    preprocess(str(config_path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', default=str(PROJECT_ROOT / 'data/demo/raw'))
    parser.add_argument('--cells', type=int, default=12)
    parser.add_argument('--days', type=int, default=6)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    for path in generate_synthetic_data(args.output_dir, args.cells, args.days, args.seed):
        print(path)


if __name__ == '__main__':
    main()
