"""Preview or commit/push one experiment's results; large assets use existing LFS rules."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SMALL = {'.json', '.csv', '.png', '.svg', '.txt', '.yml', '.yaml'}
LARGE = {'.pt', '.npy', '.npz'}


def git(*args, check=True):
    return subprocess.run(['git', '-C', str(ROOT), *args], check=check, text=True, capture_output=True)


def select_files(run, checkpoints=False, arrays=False):
    run = run.resolve()
    try: relative = run.relative_to((ROOT / 'results').resolve())
    except ValueError as exc: raise ValueError('--run must be inside this repository results/ directory.') from exc
    if not relative.parts: raise ValueError('Select a specific experiment, not all results/.')
    if not run.is_dir(): raise FileNotFoundError(run)
    selected = []
    for path in sorted(run.rglob('*')):
        if not path.is_file() or path.is_symlink(): continue
        if '__pycache__' in path.parts or path.name.startswith('.') or path.suffix == '.tmp': continue
        if path.suffix in SMALL:
            if path.stat().st_size > 50 * 1024**2: raise ValueError(f'Unexpected large metadata/image file: {path}')
        elif path.suffix == '.pt' and checkpoints: pass
        elif path.suffix in {'.npy', '.npz'} and arrays: pass
        else: continue
        if path.stat().st_size >= 2 * 1024**3: raise ValueError(f'Asset exceeds supported 2 GiB LFS file size: {path}')
        selected.append(path)
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path('results/formal_phase1_30ep_seed42'))
    parser.add_argument('--checkpoints', action='store_true')
    parser.add_argument('--arrays', action='store_true')
    parser.add_argument('--commit', action='store_true', help='Commit locally without pushing.')
    parser.add_argument('--push', action='store_true', help='Commit and push; without this/--commit, preview only.')
    parser.add_argument('--message', default=None)
    args = parser.parse_args()
    run = args.run if args.run.is_absolute() else ROOT / args.run
    files = select_files(run, args.checkpoints, args.arrays)
    for path in files: print(path.relative_to(ROOT).as_posix())
    print(f'{len(files)} files; {sum(p.stat().st_size for p in files):,} bytes selected.')
    if not (args.commit or args.push):
        print('Preview only. Add --push to commit and upload, or --commit for a local commit.')
        return
    if not files: raise ValueError('No matching result files found.')
    if Path(git('rev-parse', '--show-toplevel').stdout.strip()).resolve() != ROOT.resolve():
        raise RuntimeError('This TwoPhase directory must be its own Git repository.')
    remote = git('remote', 'get-url', 'origin').stdout.strip()
    normalized = remote.removesuffix('.git').rstrip('/')
    if not normalized.endswith('EvaWang007/auv_traj_exp_Two_Phase') and not normalized.endswith('EvaWang007:auv_traj_exp_Two_Phase'):
        raise RuntimeError('Origin does not point to the intended TwoPhase repository.')
    if git('diff', '--cached', '--quiet', check=False).returncode != 0:
        raise RuntimeError('There are already staged changes. Commit or unstage them before syncing results.')
    if args.checkpoints or args.arrays:
        git('lfs', 'version')
        for path in files:
            if path.suffix in LARGE:
                attr = git('check-attr', 'filter', '--', path.relative_to(ROOT).as_posix()).stdout
                if not attr.rstrip().endswith(': lfs'): raise RuntimeError(f'LFS tracking missing: {path}')
    branch = git('branch', '--show-current').stdout.strip()
    if not branch: raise RuntimeError('Detached HEAD: switch to a branch before syncing results.')
    for start in range(0, len(files), 50):
        # Explicit user-selected results are ignored by default; only this allowlist is force-added.
        git('add', '--force', '--', *[p.relative_to(ROOT).as_posix() for p in files[start:start+50]])
    if git('diff', '--cached', '--quiet', check=False).returncode != 0:
        result = git('commit', '-m', args.message or f'Save training results: {run.name}')
        print(result.stdout)
    else: print('No new changes to commit.')
    if args.push:
        result = git('push', '--set-upstream', 'origin', branch)
        print(result.stdout); print(result.stderr)


if __name__ == '__main__':
    try: main()
    except subprocess.CalledProcessError as exc:
        print(exc.stdout or '', file=sys.stderr); print(exc.stderr or '', file=sys.stderr)
        sys.exit(exc.returncode)
