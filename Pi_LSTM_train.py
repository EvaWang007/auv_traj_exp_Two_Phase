"""Independent first-stage Pi_LSTM training; never retrains existing CPI models."""
import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
import torch
from formal_phase1 import (HERE, atomic_json, atomic_save, canonical_hash, configure, cpu_tree,
                          hash_file, inspect_data, make_loader, publish_model, read_checkpoint,
                          restore_rng, rng_state, run_epoch, run_lock, seed_all)
from Pi_LSTM_model import PiLSTM


def prepare(data, reference, config, smoke=False):
    ids, description, identity = inspect_data(data)
    ref = json.loads((reference/'run.json').read_text(encoding='utf-8'))
    selection = json.loads((reference/'selected_checkpoints.json').read_text(encoding='utf-8'))
    if identity != ref['data_identity']:
        raise ValueError('Data differ from the CPI training run. Use the original data for paired comparison.')
    if selection['signature'] != ref['signature']:
        raise ValueError('Reference run/selection mismatch.')
    scaler_hash = hash_file(reference/'scalers.npz')
    if scaler_hash != selection['scalers_sha256']:
        raise ValueError('Reference normalization changed.')
    with np.load(reference/'scalers.npz') as z:
        scales = {k: z[k] for k in z.files}
    if smoke:
        ids = {k: v[:min(64, len(v))] for k, v in ids.items()}
    code = {n: hash_file(HERE/n) for n in ['Pi_LSTM_train.py','Pi_LSTM_model.py','formal_phase1.py','phase1_model.py']}
    payload = dict(config=config, data_identity=identity, scalers_sha256=scaler_hash,
                   reference_signature=ref['signature'], code=code, smoke=smoke)
    return ids, scales, dict(**payload, signature=canonical_hash(payload))


def train(args):
    config = json.loads(args.config.read_text(encoding='utf-8'))
    if args.smoke:
        config.update(epochs=1, num_workers=0, batch_size=16)
    if config['checkpoint_selection'] != 'validation_loss':
        raise ValueError('This experiment selects checkpoints by validation position loss.')
    device = configure(config, args.device)
    ids, scales, metadata = prepare(args.data, args.reference_run, config, args.smoke)
    with run_lock(args.output):
        record_path = args.output/'run.json'
        if record_path.exists():
            old = json.loads(record_path.read_text(encoding='utf-8'))
            if old['signature'] != metadata['signature']:
                raise ValueError('Resume code/config/data changed; use a new output directory.')
            if not args.resume:
                raise FileExistsError('Existing run: pass --resume.')
        else:
            if (args.output/'last.pt').exists():
                raise RuntimeError('Checkpoint without run metadata.')
            atomic_json(record_path, metadata)
        scaler = args.output/'scalers.npz'
        if not scaler.exists(): shutil.copy2(args.reference_run/'scalers.npz', scaler)
        assert hash_file(scaler) == metadata['scalers_sha256']
        seed_all(config['seed']); generator = torch.Generator().manual_seed(config['seed'])
        model = PiLSTM(scales).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
        history = []; best = None; score = float('inf'); first = 1
        last = args.output/'last.pt'
        if last.exists():
            saved = read_checkpoint(last)
            assert saved['signature'] == metadata['signature']
            model.load_state_dict(saved['state_dict']); optimizer.load_state_dict(saved['optimizer'])
            restore_rng(saved['rng'], generator)
            history = saved['history']; best = saved['best_checkpoint']; score = best['best_score']; first = saved['epoch']+1
        loaders = {k: make_loader(args.data, ids[k], scales, config, k=='train', generator if k=='train' else None, device)
                   for k in ['train','val']}
        tensor_scales = {k: torch.as_tensor(v, device=device, dtype=torch.float32) for k,v in scales.items()}
        for epoch in range(first, config['epochs']+1):
            began = time.monotonic()
            tr = run_epoch(model, loaders['train'], tensor_scales, 'lstm', {}, device, config, optimizer)
            va = run_epoch(model, loaders['val'], tensor_scales, 'lstm', {}, device, config)
            history.append(dict(epoch=epoch, **{'train_'+k:v for k,v in tr.items()},
                                **{'val_'+k:v for k,v in va.items()}, elapsed_s=time.monotonic()-began))
            if va['loss'] < score:
                score = va['loss']
                best = cpu_tree(dict(state_dict=model.state_dict(), scalers=scales, epoch=epoch,
                                     best_score=score, signature=metadata['signature'], model='Pi_LSTM'))
            atomic_save(last, dict(state_dict=model.state_dict(), optimizer=optimizer.state_dict(),
                        epoch=epoch, rng=rng_state(generator), history=history, best_checkpoint=best, signature=metadata['signature']))
            publish_model(args.output/'best.pt', best)
            pd.DataFrame(history).to_csv(args.output/'training_history.csv', index=False)
            atomic_json(args.output/'status.json', dict(completed_epochs=epoch, target_epochs=config['epochs'], best_epoch=best['epoch']))
            print(f'Pi_LSTM epoch={epoch}/{config["epochs"]} train_ADE={tr["ADE_m"]:.4f} val_ADE={va["ADE_m"]:.4f} val_FDE={va["FDE_m"]:.4f}', flush=True)
        saved = read_checkpoint(last); best = saved['best_checkpoint']
        publish_model(args.output/'best.pt', best)
        publish_model(args.output/'final.pt', dict(state_dict=saved['state_dict'], scalers=scales, epoch=saved['epoch'], signature=metadata['signature']))
        pd.DataFrame(saved['history']).to_csv(args.output/'training_history.csv', index=False)
        atomic_json(args.output/'selected_checkpoint.json', dict(checkpoint='best.pt', best_epoch=best['epoch'],
                    sha256=hash_file(args.output/'best.pt'), scalers_sha256=metadata['scalers_sha256'], signature=metadata['signature'],
                    test_used_for_selection=False, epochs_completed=saved['epoch'], smoke=args.smoke))
        print(f'Training complete; best epoch={best["epoch"]}. No test data used.', flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=HERE/'data')
    parser.add_argument('--reference-run', type=Path, default=HERE/'results/formal_phase1_30ep_seed42')
    parser.add_argument('--output', type=Path, default=HERE/'results/Pi_LSTM_30ep_seed42')
    parser.add_argument('--config', type=Path, default=HERE/'server/Pi_LSTM_30ep.json')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--smoke', action='store_true', help='One epoch / 64 windows per split; use a separate output folder.')
    train(parser.parse_args())
