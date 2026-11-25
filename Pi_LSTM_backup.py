"""Package completed Pi_LSTM/CPI comparison, parameters, code and original data."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import tarfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'server'))
from backup_release import sha256, prepare


def package(args):
    pi = args.pi_run.resolve(); reference = args.reference_run.resolve()
    selection = json.loads((pi/'selected_checkpoint.json').read_text(encoding='utf-8'))
    run = json.loads((pi/'run.json').read_text(encoding='utf-8'))
    if selection['smoke'] or selection['epochs_completed'] != run['config']['epochs']:
        raise ValueError('Only completed formal Pi_LSTM runs can be archived for publication.')
    if sha256(pi/selection['checkpoint']) != selection['sha256'] or sha256(pi/'scalers.npz') != selection['scalers_sha256']:
        raise ValueError('Selected model/scalers changed.')
    if not args.comparison.is_dir():
        raise FileNotFoundError('Complete the paired test first, then pass its output directory as --comparison.')
    verified = json.loads((args.comparison/'verification.json').read_text(encoding='utf-8'))
    if not verified['passed'] or verified.get('smoke_only') or verified['dataset'] != 'full_original_held_out_test':
        raise ValueError('Require successful full held-out paired test; example/smoke tests cannot replace it.')
    ref_selection = json.loads((reference/'selected_checkpoints.json').read_text(encoding='utf-8'))
    cpi = ref_selection['selected']['cpi_lstm']
    if verified['checkpoint_hashes'] != {'Pi_LSTM': selection['sha256'], 'CPI-LSTM': cpi['sha256']}:
        raise ValueError('Comparison belongs to different checkpoints.')
    if sha256(reference/cpi['checkpoint']) != cpi['sha256'] or sha256(reference/'scalers.npz') != selection['scalers_sha256']:
        raise ValueError('Reference CPI model or normalization changed.')
    if not args.skip_data:
        for filename, info in run['data_identity'].items():
            path = args.data/filename
            if sha256(path) != info['sha256']:
                raise ValueError('Training data changed: '+filename)
    export = pi/'export'; export.mkdir(exist_ok=True)
    table = []
    with np.load(pi/'scalers.npz') as z:
        for key in z.files:
            for index,value in enumerate(z[key].reshape(-1)):
                table.append(dict(parameter=key,index=index,value=float(value)))
    pd.DataFrame(table).to_csv(export/'normalization_parameters.csv',index=False)
    pip = subprocess.run([sys.executable,'-m','pip','freeze'],capture_output=True,text=True,check=True)
    (export/'backup_environment_pip_freeze.txt').write_text(pip.stdout,encoding='utf-8')
    git = subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True)
    (export/'backup_code_commit.txt').write_text(git.stdout if git.returncode==0 else 'Git unavailable; source files included.\n',encoding='utf-8')
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    archive = args.output_dir/f'Pi_LSTM_full_{stamp}.tar.gz'
    if archive.exists(): raise FileExistsError(archive)
    info = dict(created_at=datetime.now().astimezone().isoformat(),pi_run=str(pi),reference_run=str(reference),
                data_included=not args.skip_data,comparison=str(args.comparison.resolve()),
                selected_checkpoint=selection,reference_cpi_checkpoint=cpi,
                training_provenance='Pi_LSTM_run/run.json contains exact training code/data/scaler hashes; code/ contains packaging-time source.',
                public_release_warning='Archive contains model weights, run data, logs and optionally original training data.')
    (export/'backup_info.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
    sources = [(pi,'Pi_LSTM_run'),(reference,'CPI_reference_run'),(args.comparison,'comparison')]
    if not args.skip_data:sources.append((args.data,'data'))
    # Optional paired stress/normal figures can be included without replacing the full test.
    for index,extra in enumerate(args.extra):sources.append((extra,f'extra_{index+1}_{extra.name}'))
    for source,_ in sources:
        if not source.is_dir():raise FileNotFoundError(source)
        if args.output_dir.resolve().is_relative_to(source.resolve()):
            raise ValueError('Backup destination must be outside archived source directories.')
    def checked(member):
        if member.issym() or member.islnk():raise ValueError('Refusing archive symlink: '+member.name)
        if member.name.endswith(('.tmp','.lock')) or '/__pycache__/' in member.name:return None
        return member
    temporary=archive.with_name(archive.name+'.tmp')
    with tarfile.open(temporary,'w:gz') as tar:
        for source,name in sources:
            print('Archiving '+str(source),flush=True)
            tar.add(source,arcname='Pi_LSTM_experiment/'+name,filter=checked)
        code=[]
        for suffix in ['*.py','*.txt','*.md']:
            code.extend(ROOT.glob(suffix))
        code.extend(p for p in (ROOT/'server').rglob('*') if p.is_file() and p.suffix in ['.py','.sh','.json','.txt','.yml'])
        for path in sorted(set(code)):
            if '__pycache__' not in path.parts:
                tar.add(path,arcname='Pi_LSTM_experiment/code/'+path.relative_to(ROOT).as_posix(),filter=checked)
    temporary.replace(archive)
    checksum=sha256(archive)
    Path(str(archive)+'.sha256').write_text(checksum+'  '+archive.name+'\n',encoding='ascii')
    prepare(archive)
    (args.output_dir/'Pi_LSTM_latest_archive.txt').write_text(str(archive.resolve())+'\n',encoding='utf-8')
    print('BACKUP COMPLETE: '+str(archive.resolve()),flush=True)
    return archive


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pi-run',type=Path,default=ROOT/'results/Pi_LSTM_30ep_seed42')
    parser.add_argument('--reference-run',type=Path,default=ROOT/'results/formal_phase1_30ep_seed42')
    parser.add_argument('--comparison',type=Path,default=ROOT/'results/Pi_LSTM_vs_CPI_test')
    parser.add_argument('--data',type=Path,default=ROOT/'data')
    parser.add_argument('--output-dir',type=Path,default=ROOT.parent/'auv_backups')
    parser.add_argument('--extra',type=Path,action='append',default=[],help='Optional additional example-comparison directory.')
    parser.add_argument('--skip-data',action='store_true',help='Omit original training arrays; requires the original backup for later retraining.')
    package(parser.parse_args())
