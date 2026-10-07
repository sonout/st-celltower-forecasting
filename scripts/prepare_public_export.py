"""Copy reviewed source to a new directory without Git history or private artifacts."""

import argparse
from pathlib import Path
import shutil
from audit_public import ROOT, candidate_paths, audit_paths

TOP_LEVEL = {'.gitignore', 'README.md', 'LICENSE', 'THIRD_PARTY_NOTICES.md', 'pyproject.toml',
             'requirements.txt', 'requirements-notebooks.txt', 'main.py'}
PUBLIC_DIRECTORIES = {'.github', 'assets', 'configs', 'docs', 'models', 'notebooks', 'scripts', 'src', 'tests', 'third_party'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, help='New directory; existing directories are never overwritten.')
    args = parser.parse_args()
    target = Path(args.output).resolve()
    if target.exists() or target.is_relative_to(ROOT):
        parser.error('Choose a new directory outside this repository.')
    paths = [p for p in candidate_paths() if p.as_posix() in TOP_LEVEL or p.parts[0] in PUBLIC_DIRECTORIES
             or p.as_posix() == 'data/README.md' or p.as_posix().startswith('data/sample/')]
    findings = audit_paths(paths)
    if findings:
        parser.error('Public file audit has findings; run scripts/audit_public.py before exporting.')
    for path in paths:
        if (ROOT / path).is_symlink():
            parser.error(f'Refusing to export a symlink: {path}')
    target.mkdir(parents=True)
    for path in paths:
        destination = target / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / path, destination)
    print(f'Exported {len(paths)} reviewed files to {target}; no Git history, real data, or checkpoints were copied.')


if __name__ == '__main__':
    main()
