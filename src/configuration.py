"""Configuration paths are relative to the project, independent of working directory."""

from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / 'configs' / 'experiment_example.yaml'
MODEL_CONFIG_DIR = PROJECT_ROOT / 'configs' / 'models'


def load_config(path=DEFAULT_CONFIG):
    path = Path(path).resolve()
    with path.open() as stream:
        config = yaml.safe_load(stream)
    for key in ('data_dir', 'output_dir'):
        value = Path(config['data'][key])
        config['data'][key] = str(value if value.is_absolute() else PROJECT_ROOT / value)
    for key in ('output_dir', 'store_predictions_folder'):
        if key in config['experiment']:
            value = Path(config['experiment'][key])
            config['experiment'][key] = str(value if value.is_absolute() else PROJECT_ROOT / value)
    return config
