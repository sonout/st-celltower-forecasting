# Data contract and availability

The original telecom observations, real cell identifiers, handover records,
trained checkpoints, and private experiment outputs are not distributed.
This repository does not grant rights to redistribute third-party telecom data.
Use only data you are authorized to process and share.

`sample/` contains small, hand-written synthetic CSV schema fixtures with
fictional `SYN...` identifiers. They are examples, not an anonymized extract,
and are too short for the default 48-step forecasting window. Run
`python main.py --demo` to generate a complete six-day, twelve-cell example.
The generator uses only a fixed seed and mathematical signals; it does not
read the private dataset or estimate parameters from it.

## KPI observations

Raw files are named `CELL_*.csv`. Required fields:

| Column | Meaning |
| --- | --- |
| `eventdate` | Timestamp on a regular 15-minute grid; use one consistent timezone. |
| `CELL_` | Unique network-cell identifier, not necessarily a whole tower. |
| `MAX_DL_ACTIVE_UE` | Downlink active-user KPI, used as an input and demo target. |
| `DL_USER_THPUT_KBPS` | Downlink user throughput in kbps, used as an input and demo target. |
| `PERC_USE_PRBS_DL` | Downlink physical-resource-block utilization (%), demo input. |

Other numeric KPI columns can be selected in the experiment YAML.
The default static-feature parser expects `<site>-<sector><band>`, for example
`SYN0001-10`. Unknown band codes have capacity zero; adapt this parser if your
identifier scheme or band-to-capacity mapping differs. Identifiers without
that suffix need a custom static-feature mapping.

Duplicate cell/timestamp rows are removed. Missing values are forward-filled
within each cell, with zero for initial gaps. Do not interpret imputed targets
as independently measured observations when assessing a private dataset.

## Handover observations

Each row has `eventdate`, `CELL_source`, `CELL_target`, and `handover_count`.
The count is the number of successful handovers in that observation interval.
The graph builder also accepts explicitly selected vendor counter columns;
see `python -m src.graph --help`. Counts must be finite and nonnegative.
Aggregate training-period counts by ordered source/target pair. Preserve all
modeled KPI cells, including isolated cells; ignore edges to cells outside
the modeled set. Validation/test handovers are excluded from graph estimation.

## Prepared sequences

Preprocessing writes a new immutable directory beneath `output_dir/datasets/`:

- `X_train.npy`, `X_val.npy`, `X_test.npy`: `(samples, input_steps, cells, input_features)`.
- `Y_train.npy`, `Y_val.npy`, `Y_test.npy`: `(samples, output_steps, cells, target_kpis)`.
- `static_features.npy`: `(cells, static_features)` in the same cell order.
- `timestamps_*.pkl`: input start, forecast origin, and final forecast timestamp for each sample.
- `metadata.yaml`: ordered cells/features/targets, split dates, shapes, and preprocessing settings.
- `manifest.json`: dataset identifier and SHA-256 hashes of the prepared files.

For the demo, input windows have 48 steps and eight features (three KPIs and
five cyclic/time features); targets have 12 steps and two KPIs over twelve
cells. Date intervals are start-inclusive and end-exclusive. Forecast windows
crossing a split boundary are removed; historical input may precede a split.

All generated data, private raw files, and result directories are ignored by
Git. Only this document and the explicitly synthetic `sample/` fixtures are
included in a public source export.
