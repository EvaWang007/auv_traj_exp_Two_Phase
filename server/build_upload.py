"""Build a self-contained upload archive without historical checkpoints or trial data."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''): value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT.parent/'auv_twophase_gpu_phase1_30ep.tar.gz')
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    start=time.monotonic()
    inventory={p.name:dict(bytes=p.stat().st_size,sha256=sha256(p)) for p in sorted((ROOT/'data').iterdir()) if p.is_file()}
    (ROOT/'server/data_manifest.json').write_text(json.dumps(inventory,indent=2),encoding='utf-8')
    names=['formal_phase1.py','formal_phase1_evaluate.py','phase1_model.py','train_phase1.py','prepare_phase1.py',
           'routes.py','phase2_control.py','run_phase2.py','verify_phase2.py','README.txt','README.md','SERVER_PHASE1_30EP.txt','SERVER_RTX3090_IMAGE.txt',
           'TRAINING_SPEC_REVIEW.txt','training_data_inventory.json']
    paths=[ROOT/name for name in names]
    paths += [p for p in sorted((ROOT/'server').iterdir()) if p.is_file() and p.suffix in ['.py','.sh','.json','.txt','.yml']]
    paths += [p for p in sorted((ROOT/'data').iterdir()) if p.is_file()]
    assert all(p.is_file() for p in paths)
    # All archive members are under one relative, portable top-level directory.
    with tarfile.open(args.output,'w:gz',compresslevel=1) as archive:
        for path in paths:
            archive.add(path,arcname=str(Path(ROOT.name)/path.relative_to(ROOT)),recursive=False)
            print('Packed '+str(path.relative_to(ROOT)),flush=True)
    checksum=sha256(args.output)
    Path(str(args.output)+'.sha256').write_text(checksum+'  '+args.output.name+'\n',encoding='ascii')
    print(json.dumps(dict(archive=str(args.output.resolve()),bytes=args.output.stat().st_size,sha256=checksum,
                          files=len(paths),elapsed_s=time.monotonic()-start),indent=2),flush=True)


if __name__=='__main__': main()
