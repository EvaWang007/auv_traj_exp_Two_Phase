"""Import only manifest-verified data files from a separately supplied training archive."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''): value.update(chunk)
    return value.hexdigest()


def import_data(archive, root=ROOT):
    manifest = json.loads((root/'server/data_manifest.json').read_text(encoding='utf-8'))
    sidecar = Path(str(archive)+'.sha256')
    if sidecar.exists():
        if sha256(archive) != sidecar.read_text(encoding='ascii').split()[0]:
            raise ValueError('Archive checksum mismatch.')
    destination = root/'data'; destination.mkdir(exist_ok=True)
    found=set()
    with tarfile.open(archive, 'r|gz') as source:
        for member in source:
            parts=PurePosixPath(member.name).parts
            if len(parts)!=3 or parts[1]!='data' or parts[2] not in manifest: continue
            if not member.isfile(): raise ValueError('Only regular data files are accepted.')
            name=parts[2]
            if name in found: raise ValueError('Duplicate archive member: '+name)
            found.add(name); expected=manifest[name]
            if member.size != expected['bytes']: raise ValueError('Size mismatch: '+name)
            path=destination/name
            if path.is_symlink(): raise ValueError('Refusing symlink destination: '+name)
            if path.exists():
                if sha256(path)!=expected['sha256']: raise FileExistsError('Different existing data; refusing overwrite: '+str(path))
                print('Verified existing '+name,flush=True); continue
            temporary=destination/(name+'.import.tmp')
            if temporary.is_symlink(): raise ValueError('Refusing symlink temporary file.')
            digest=hashlib.sha256()
            with source.extractfile(member) as incoming, temporary.open('wb') as outgoing:
                for chunk in iter(lambda:incoming.read(8*1024*1024),b''):
                    digest.update(chunk); outgoing.write(chunk)
            if digest.hexdigest()!=expected['sha256']: raise ValueError('Data checksum mismatch: '+name)
            temporary.replace(path); print('Imported '+name,flush=True)
    missing=set(manifest)-found
    if missing: raise ValueError('Archive is missing data files: '+', '.join(sorted(missing)))
    print(f'All {len(found)} data files verified. No training was started.',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,required=True)
    args=parser.parse_args(); import_data(args.archive.resolve())


if __name__=='__main__': main()
