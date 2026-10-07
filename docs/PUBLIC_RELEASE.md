# Public release and history review

The current source is intended for publication as `st-celltower-forecasting`.
Changing the project title does not rename an existing Git remote.

## Audit findings

The review scanned 144 reachable historical file blobs and flagged 38. Existing
history contains internal absolute paths, real cell identifiers, and cached
notebook outputs. No matches were found for the checked credential formats
or credential assignments. Pattern checks are not a complete secrets audit;
they also cannot determine data redistribution rights or identify every
organization name hidden in free text or rendered images.

Cleaning the current files does not remove those historical blobs. Keep the
existing research repository private. A clean source export is provided so
the public project can start with reviewed files and a fresh history, without
modifying or force-pushing the research repository.

```bash
python scripts/audit_public.py
python scripts/audit_public.py --history  # Expected findings in the research repo.
python scripts/prepare_public_export.py --output ../st-celltower-forecasting-public
```

The export refuses an existing output directory and copies only an explicit
set of source, configuration, documentation, synthetic fixture, and figure
paths. It does not copy `.git`, private data, results, local archives, or model
checkpoints. The original notebooks/configurations and legacy baseline source
are backed up locally under `.local/public-prep-archive/`, which is ignored.
These backups remain recoverable and should stay private.

## Publish the reviewed export

Inspect and test the export before creating a new remote repository. From the
export directory, after replacing the remote URL with your new public repo:

```bash
git init -b main
git add .
git diff --cached --stat
git commit -m "Initial public release of spatio-temporal telecom forecasting"
git remote add origin <new-public-repository-url>
git push -u origin main
```

Do not point these commands at the private research remote. If preserving the
old history is essential, it needs a separate review and an explicitly
authorized history rewrite; that operation is not performed by these tools.

Only synthetic demonstration results and figures are included. A real-data
benchmark would require a new run of the final configuration and permission
to share the data-derived information. The root MIT license covers original
project components; adapted Graph WaveNet source retains upstream licenses,
documented in `THIRD_PARTY_NOTICES.md`.
