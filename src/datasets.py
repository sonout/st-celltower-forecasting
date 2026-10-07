"""Resolve and fingerprint immutable preprocessing outputs."""

import hashlib
import json
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_dataset(output_dir, dataset_id=None) -> Path:
    root = Path(output_dir)
    if dataset_id:
        path = root / 'datasets' / dataset_id
    elif (root / 'latest.json').exists():
        pointer = json.loads((root / 'latest.json').read_text())
        path = root / pointer['path']
    else:
        path = root
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('Dataset path must stay inside the configured data directory.')
    if not (path / 'metadata.yaml').exists():
        raise FileNotFoundError(f'No dataset found at {path}. Run python -m src.preprocessing first.')
    return path


def verify_manifest(path: Path, expected=None) -> dict:
    manifest_path = path / 'manifest.json'
    if expected is None and not manifest_path.exists():
        return {}
    manifest = expected if expected is not None else json.loads(manifest_path.read_text())
    for name, expected in manifest['sha256'].items():
        if Path(name).name != name:
            raise ValueError('Manifest entries must be filenames.')
        if file_sha256(path / name) != expected:
            raise ValueError(f'Dataset file changed after preprocessing: {path / name}')
    return manifest
