"""Check candidate public files or reachable Git blobs without printing matched values."""

import argparse
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
RULES = {
    'private_absolute_path': re.compile(rb'/(?:data/shared|home|Users|net/home)/[A-Za-z0-9_.-]+/'),
    'real_cell_identifier': re.compile(rb'\b[0-9]{10}-[0-9]{2}\b'),
    'credential_format': re.compile(rb'(?:AKIA|ASIA)[A-Z0-9]{16}|gh[pousr]_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9]{32,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
    'credential_assignment': re.compile(rb'(?i)(?:password|api[_-]?key|access[_-]?token|secret[_-]?key)\s*[=:]\s*[\x22\x27][^\x22\x27\r\n]{8,}[\x22\x27]'),
}
PRIVATE_SUFFIXES = {'.npy', '.npz', '.parquet', '.pkl', '.pickle', '.pt', '.pth', '.ckpt', '.pem', '.key'}


def candidate_paths():
    checkout = subprocess.run(['git', 'rev-parse', '--show-toplevel'], cwd=ROOT,
                              capture_output=True, text=True)
    if checkout.returncode == 0 and Path(checkout.stdout.strip()).resolve() == ROOT:
        data = subprocess.check_output(['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard'], cwd=ROOT)
        return sorted({Path(p.decode()) for p in data.split(b'\0') if p and (ROOT / p.decode()).is_file()})
    # A clean export intentionally has no .git yet. Exclude generated/private local
    # directories, but still inspect unexpected root files (including credentials).
    ignored = {'.git', '.local', '.venv', 'venv', '__pycache__', '.pytest_cache',
               'build', 'dist', 'results', 'gwn_data', 'condor'}
    paths = []
    for path in ROOT.rglob('*'):
        relative = path.relative_to(ROOT)
        if not path.is_file() or any(part in ignored or part.endswith('.egg-info') for part in relative.parts):
            continue
        if relative.parts[0] == 'data' and relative.as_posix() != 'data/README.md' and not relative.as_posix().startswith('data/sample/'):
            continue
        if path.suffix in {'.pyc', '.pyo', '.log'}:
            continue
        paths.append(relative)
    return sorted(paths)


def inspect_body(path, body):
    categories = [name for name, pattern in RULES.items() if pattern.search(body)]
    if path.suffix.lower() in PRIVATE_SUFFIXES or path.name in ('.env', 'credentials', 'id_rsa', 'id_ed25519'):
        categories.append('private_artifact_filename')
    if path.suffix.lower() == '.csv' and not path.as_posix().startswith('data/sample/'):
        categories.append('dataset_outside_synthetic_sample')
    if path.suffix == '.ipynb':
        try:
            cells = json.loads(body)['cells']
            if any(c.get('outputs') or c.get('execution_count') is not None for c in cells):
                categories.append('cached_notebook_output')
        except (ValueError, KeyError, TypeError):
            categories.append('invalid_notebook')
    return categories


def audit_paths(paths, root=ROOT):
    return [{'path': path.as_posix(), 'categories': matches}
            for path in paths if (matches := inspect_body(path, (root / path).read_bytes()))]


def audit_history():
    if not (ROOT / '.git').exists():
        raise ValueError('This source export has no Git history; use the working-tree audit instead.')
    entries = subprocess.check_output(['git', 'rev-list', '--objects', '--all'], cwd=ROOT).decode().splitlines()
    findings, scanned = [], 0
    for entry in entries:
        object_id, _, name = entry.partition(' ')
        if not name or subprocess.check_output(['git', 'cat-file', '-t', object_id], cwd=ROOT).strip() != b'blob':
            continue
        body = subprocess.check_output(['git', 'cat-file', 'blob', object_id], cwd=ROOT)
        scanned += 1
        matches = inspect_body(Path(name), body)
        if matches:
            findings.append({'blob': object_id[:12], 'path': name, 'categories': matches})
    return scanned, findings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history', action='store_true', help='Inspect all reachable historical file blobs.')
    args = parser.parse_args()
    if args.history:
        count, findings = audit_history()
    else:
        paths = candidate_paths()
        count, findings = len(paths), audit_paths(paths)
    print(json.dumps({'scope': 'history' if args.history else 'working_tree', 'files_or_blobs_scanned': count,
                      'findings': findings, 'limitations': 'Pattern checks cannot establish absence of every secret or data redistribution rights.'}, indent=2))
    raise SystemExit(bool(findings))


if __name__ == '__main__':
    main()
