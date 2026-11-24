"""Upload a completed backup to GitHub Releases; verify/assemble it after download.

Only the upload subcommand performs network writes. Authentication is handled by gh.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = 'EvaWang007/auv_traj_exp_Two_Phase'
ASSET_LIMIT = 2 * 1024**3
CHUNK_SIZE = 1024**3


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024**2), b''): digest.update(chunk)
    return digest.hexdigest()


def latest_backup():
    paths = list((ROOT.parent/'auv_backups').glob('twophase_full_*.tar.gz'))
    if not paths: raise FileNotFoundError('No backup found. Pass --archive /absolute/path/to/backup.tar.gz')
    return max(paths, key=lambda p: p.stat().st_mtime)


def prepare(archive, asset_limit=ASSET_LIMIT, chunk_size=CHUNK_SIZE):
    archive = archive.resolve()
    if not archive.is_file() or not archive.name.endswith('.tar.gz'):
        raise ValueError('Provide an existing .tar.gz backup.')
    if not archive.stat().st_size: raise ValueError('Backup is empty.')
    checksum = sha256(archive)
    folder = archive.parent/(archive.name+'.release-assets')
    folder.mkdir(exist_ok=True)
    paths = []
    if archive.stat().st_size < asset_limit:
        paths = [archive]
    else:
        with archive.open('rb') as source:
            index = 0
            while True:
                first = source.read(min(8*1024**2, chunk_size))
                if not first: break
                path = folder/(archive.name+f'.part{index:04d}')
                temporary = path.with_name(path.name+'.tmp')
                remaining = chunk_size-len(first)
                with temporary.open('wb') as target:
                    target.write(first)
                    while remaining:
                        chunk = source.read(min(8*1024**2, remaining))
                        if not chunk: break
                        target.write(chunk); remaining -= len(chunk)
                temporary.replace(path); paths.append(path); index += 1
    assets = [dict(name=p.name, bytes=p.stat().st_size, sha256=sha256(p)) for p in paths]
    manifest = dict(version=1, repository=REPOSITORY, archive_name=archive.name,
                    archive_bytes=archive.stat().st_size, archive_sha256=checksum,
                    split=len(paths)>1, assets=assets)
    manifest_path = folder/'backup_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    sums = folder/'SHA256SUMS'
    sums.write_text(''.join(f"{a['sha256']}  {a['name']}\n" for a in assets), encoding='ascii')
    return manifest, paths+[manifest_path, sums]


def gh_api(endpoint, method='GET', payload=None):
    command = ['gh', 'api', endpoint, '--method', method]
    if payload is not None: command += ['--input', '-']
    result = subprocess.run(command, input=json.dumps(payload) if payload is not None else None,
                            capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def check_assets(release, files):
    remote = {a['name']: a for a in release.get('assets', [])}
    missing = []
    for path in files:
        asset = remote.get(path.name)
        if asset is None:
            missing.append(path); continue
        if asset.get('state') != 'uploaded' or asset['size'] != path.stat().st_size:
            raise RuntimeError(f"Remote asset {path.name} is incomplete/different. Keep the original backup; remove only the failed release asset before retrying, or use a new --tag.")
        if asset.get('digest') and asset['digest'] != 'sha256:'+sha256(path):
            raise RuntimeError('Remote digest mismatch: '+path.name)
    return missing


def upload(archive, tag=None):
    if shutil.which('gh') is None: raise RuntimeError('Install gh, then run: gh auth login --hostname github.com --git-protocol https --web')
    repository = gh_api('repos/'+REPOSITORY)
    if not repository.get('permissions', {}).get('push'):
        raise PermissionError('The gh account cannot write this repository. Log in with your authorized GitHub account.')
    manifest, files = prepare(archive)
    if len(files)>1000: raise ValueError('Too many release assets.')
    tag = tag or 'backup-'+archive.name.removesuffix('.tar.gz')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', tag): raise ValueError('Use a simple alphanumeric release tag.')
    endpoint = 'repos/'+REPOSITORY+'/releases/tags/'+quote(tag, safe='')
    marker = 'Backup manifest SHA256: '+sha256(files[-2])
    print(f'Public destination: {REPOSITORY}, release {tag}', flush=True)
    print(f'Backup: {archive} ({manifest["archive_bytes"]:,} bytes); upload assets: {len(files)}', flush=True)
    try:
        release = gh_api(endpoint)
    except subprocess.CalledProcessError as exc:
        if '404' not in (exc.stderr or ''): raise
        release = gh_api('repos/'+REPOSITORY+'/releases', 'POST', dict(
            tag_name=tag, target_commitish='main', name='Training backup: '+archive.name,
            body='Owner-requested public backup of completed training.\n\n'+marker+
                 '\n\nDownload all assets and run server/backup_release.py verify --folder DOWNLOAD_DIRECTORY. '
                 'The archived code/provenance records the training version; this release tag records the upload.\n',
            draft=True))
    if marker not in release.get('body', ''):
        raise RuntimeError('This tag belongs to another snapshot; choose a new --tag.')
    missing = check_assets(release, files)
    if missing and not release['draft']:
        raise RuntimeError('Existing published release is incomplete/different. Use a new --tag; it will not be overwritten.')
    for path in missing:
        print('Uploading '+path.name, flush=True)
        subprocess.run(['gh', 'release', 'upload', tag, str(path), '--repo', REPOSITORY], check=True)
    release = gh_api(endpoint)
    if check_assets(release, files): raise RuntimeError('Not all expected assets are present; release remains draft.')
    if release['draft']:
        release = gh_api('repos/'+REPOSITORY+'/releases/'+str(release['id']), 'PATCH', dict(draft=False))
    print('UPLOAD COMPLETE: '+release['html_url'], flush=True)
    print('RELEASE_TAG='+tag, flush=True)
    print('On the analysis computer, download all release assets and verify them before extraction.', flush=True)


def safe_file(folder, name):
    if not name or Path(name).name != name or '/' in name or '\\' in name or name in ['.', '..']:
        raise ValueError('Invalid manifest filename.')
    path = folder/name
    if path.is_symlink(): raise ValueError('Refusing symlink file: '+name)
    return path


def verify(folder):
    folder = folder.resolve()
    manifest = json.loads((folder/'backup_manifest.json').read_text(encoding='utf-8'))
    paths = []
    for record in manifest['assets']:
        path = safe_file(folder, record['name'])
        if path.stat().st_size != record['bytes'] or sha256(path) != record['sha256']:
            raise ValueError('Downloaded asset failed verification: '+record['name'])
        paths.append(path)
    target = safe_file(folder, manifest['archive_name'])
    if manifest['split'] and not target.exists():
        temporary = target.with_name(target.name+'.assembling.tmp')
        if temporary.is_symlink(): raise ValueError('Refusing symlink temporary file.')
        with temporary.open('wb') as destination:
            for part in paths:
                with part.open('rb') as source: shutil.copyfileobj(source, destination, 8*1024**2)
        if temporary.stat().st_size != manifest['archive_bytes'] or sha256(temporary) != manifest['archive_sha256']:
            raise ValueError('Reassembled archive checksum mismatch.')
        temporary.replace(target)
    if target.stat().st_size != manifest['archive_bytes'] or sha256(target) != manifest['archive_sha256']:
        raise ValueError('Full archive checksum mismatch.')
    print('VERIFIED: '+str(target), flush=True)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ['prepare', 'upload']:
        child = commands.add_parser(name)
        child.add_argument('--archive', type=Path)
        if name == 'upload': child.add_argument('--tag')
    child = commands.add_parser('verify'); child.add_argument('--folder', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'verify': verify(args.folder); return
    archive = args.archive.resolve() if args.archive else latest_backup()
    if args.command == 'upload': upload(archive, args.tag)
    else:
        manifest, files = prepare(archive)
        print(json.dumps(manifest, indent=2))
        print('Prepared locally only; no upload performed. Assets:',*[str(p) for p in files],sep='\n')


if __name__ == '__main__':
    try: main()
    except subprocess.CalledProcessError as exc:
        print(exc.stderr or f'Command failed: {exc.cmd}', file=sys.stderr)
        raise SystemExit(exc.returncode)
