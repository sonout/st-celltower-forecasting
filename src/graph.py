"""Build a directed graph from successful handover counts, using training dates only."""

import argparse
import pickle
from pathlib import Path
import networkx as nx
import numpy as np
import pandas as pd

VENDOR_COUNT_COLUMNS = ('pmhoexesucclteintraf', 'pmhoexesucclteinterf', 'pmhoexesucclteinterflb')


def build_handover_graph(events, cells=None, start=None, end=None,
                         count_columns=None, out_quantile=0, in_quantile=0):
    """Sum positive directed counts; optionally retain edges passing both quantiles."""
    frame = events.copy()
    if start is not None or end is not None:
        stamps = pd.to_datetime(frame['eventdate'])
        if start is not None:
            frame = frame.loc[stamps >= pd.Timestamp(start)]
        if end is not None:
            frame = frame.loc[pd.to_datetime(frame['eventdate']) < pd.Timestamp(end)]
    if count_columns is None:
        count_columns = ['handover_count'] if 'handover_count' in frame else list(VENDOR_COUNT_COLUMNS)
    required = ['CELL_source', 'CELL_target', *count_columns]
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f'Missing handover columns: {sorted(missing)}')
    if frame[['CELL_source', 'CELL_target']].isna().any().any():
        raise ValueError('Handover cell identifiers must be present.')
    counts = frame[count_columns].apply(pd.to_numeric, errors='raise').to_numpy()
    if not np.isfinite(counts).all() or (counts < 0).any():
        raise ValueError('Handover counts must be finite and nonnegative.')
    frame['weight'] = counts.sum(axis=1)
    weights = frame.groupby(['CELL_source', 'CELL_target'], sort=True)['weight'].sum()
    known = set(cells) if cells is not None else set(frame['CELL_source']) | set(frame['CELL_target'])
    graph = nx.DiGraph()
    graph.add_nodes_from(sorted(known))
    graph.add_weighted_edges_from((a, b, float(w)) for (a, b), w in weights.items()
                                  if w > 0 and a != b and a in known and b in known)
    if not (0 <= out_quantile <= 1 and 0 <= in_quantile <= 1):
        raise ValueError('Pruning quantiles must be between zero and one.')
    outgoing, incoming = set(), set()
    for node in graph:
        for edges, quantile, keep in ((list(graph.out_edges(node, data='weight')), out_quantile, outgoing),
                                      (list(graph.in_edges(node, data='weight')), in_quantile, incoming)):
            if edges:
                threshold = np.quantile([w for _, _, w in edges], quantile)
                keep.update((a, b) for a, b, w in edges if w >= threshold)
    graph.remove_edges_from(set(graph.edges) - (outgoing & incoming))
    graph.graph.update(training_start=str(start) if start is not None else None,
                       training_end=str(end) if end is not None else None,
                       weight_definition='sum of successful handover counts',
                       out_quantile=out_quantile, in_quantile=in_quantile)
    return graph


def save_graph(graph, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as stream:
        pickle.dump(graph, stream)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', nargs='+', required=True, help='Handover CSV files.')
    parser.add_argument('--cells-file', required=True, help='KPI CSV defining the modeled cells.')
    parser.add_argument('--output', default='data/processed/handover_graph.pkl')
    parser.add_argument('--train-start')
    parser.add_argument('--train-end', required=True, help='Exclusive cutoff; exclude validation/test events.')
    parser.add_argument('--count-columns', nargs='+')
    parser.add_argument('--out-quantile', type=float, default=0)
    parser.add_argument('--in-quantile', type=float, default=0)
    args = parser.parse_args()
    frames = [pd.read_csv(path) for path in args.input]
    cells = pd.read_csv(args.cells_file, usecols=['CELL_'])['CELL_'].unique()
    graph = build_handover_graph(pd.concat(frames, ignore_index=True), cells,
                                start=args.train_start, end=args.train_end, count_columns=args.count_columns,
                                out_quantile=args.out_quantile, in_quantile=args.in_quantile)
    save_graph(graph, args.output)
    print(f'Saved {graph.number_of_nodes()} cells and {graph.number_of_edges()} directed edges to {args.output}')


if __name__ == '__main__':
    main()
