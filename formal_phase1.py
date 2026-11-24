"""GPU-capable phase-one training, resumable PSO, and epoch-boundary recovery.

Historical experiment entry points are deliberately unchanged.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from phase1_model import ARRAYS, Phase1CPI, Phase1Dataset, losses
from train_phase1 import fit_scalers

HERE = Path(__file__).resolve().parent
MODEL_NAMES = {'lstm': 'LSTM', 'cpi_lstm': 'CPI-LSTM', 'pso_cpi_lstm': 'PSO-CPI-LSTM'}
EXPECTED = {'history': (120, 16), 'plan': (80, 2), 'position': (80, 2),
            'velocity': (80, 2), 'current': (80, 2), 'bias': (2,), 'DR_start': (2,),
            'water_state': (3,), 'last_fix_bias': (2,), 'last_fix_current': (2,), 'outage_age': ()}


class DirectLSTM(nn.Module):
    """Exactly the architecture used in the previous five-epoch LSTM baseline."""
    def __init__(self, scales):
        super().__init__()
        self.hist_rnn = nn.LSTM(16, 128, batch_first=True)
        self.ctrl_rnn = nn.LSTM(2, 96, batch_first=True)
        self.head = nn.Sequential(nn.Linear(224, 128), nn.ReLU(), nn.Linear(128, 160))
        self.register_buffer('relative_mean', torch.as_tensor(scales['relative_mean']))
        self.register_buffer('relative_std', torch.as_tensor(scales['relative_std']))

    def forward(self, batch):
        h, _ = self.hist_rnn(batch['x'])
        u, _ = self.ctrl_rnn(batch['u'])
        relative = self.head(torch.cat([h[:, -1], u[:, -1]], -1)).view(-1, 80, 2)
        return {'position': batch['DR_start'][:, None] + self.relative_mean + self.relative_std * relative}


def hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def cpu_tree(value):
    if torch.is_tensor(value): return value.detach().cpu().clone()
    if isinstance(value, dict): return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list): return [cpu_tree(v) for v in value]
    if isinstance(value, tuple): return tuple(cpu_tree(v) for v in value)
    return value


def atomic_save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    torch.save(cpu_tree(value), temporary)
    temporary.replace(path)


def read_checkpoint(path):
    # These are locally generated, trusted checkpoints containing NumPy/RNG state.
    return torch.load(path, map_location='cpu', weights_only=False)


def publish_model(path, payload):
    """Rebuild a derived model file, preserving its bytes on a completed-run resume."""
    path = Path(path)
    if path.exists():
        try:
            old = read_checkpoint(path)
            if (old.get('signature') == payload.get('signature') and old.get('epoch') == payload.get('epoch')
                    and old.get('best_epoch') == payload.get('best_epoch')
                    and set(old['state_dict']) == set(payload['state_dict'])
                    and all(torch.equal(old['state_dict'][k], v.detach().cpu()) for k, v in payload['state_dict'].items())
                    and set(old['scalers']) == set(payload['scalers'])
                    and all(np.array_equal(old['scalers'][k], v) for k, v in payload['scalers'].items())):
                return
        except (OSError, RuntimeError, ValueError, EOFError, KeyError):
            pass
    atomic_save(path, payload)


@contextmanager
def run_lock(folder):
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / '.python_run.lock').open('a+b') as handle:
        if os.name == 'nt':
            import msvcrt
            handle.seek(0); handle.write(b'0'); handle.flush(); handle.seek(0)
            try: msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc: raise RuntimeError('Another process is using this output directory.') from exc
        else:
            import fcntl
            try: fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc: raise RuntimeError('Another process is using this output directory.') from exc
        try: yield
        finally:
            if os.name == 'nt':
                handle.seek(0); msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else: fcntl.flock(handle, fcntl.LOCK_UN)


def configure(config, device_name):
    torch.set_num_threads(config['cpu_threads'])
    torch.use_deterministic_algorithms(config['deterministic'])
    torch.backends.cudnn.deterministic = config['deterministic']
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if config['precision'] != 'float32': raise ValueError('This release intentionally uses float32; AMP is not enabled.')
    device = torch.device(device_name)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is required but unavailable. Activate the GPU environment; do not fall back to CPU for the formal run.')
    if device.type not in ['cpu', 'cuda']: raise ValueError('Supported devices: cpu, cuda, cuda:N')
    return device


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def rng_state(generator):
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                shuffle=generator.get_state())


def restore_rng(state, generator):
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch']); generator.set_state(state['shuffle'])
    if state['cuda'] and torch.cuda.is_available(): torch.cuda.set_rng_state_all(state['cuda'])


def inspect_data(data, full_hash=True):
    with np.load(data / 'splits.npz') as z: ids = {k: z[k] for k in ['train', 'val', 'test']}
    with (data / 'dataset_config.json').open(encoding='utf-8') as handle: description = json.load(handle)
    assert description['history_len'] == 120 and description['pred_len'] == 80 and description['dt'] == .2
    manifest = pd.read_csv(data / 'windows.csv')
    n = len(manifest)
    assert np.array_equal(manifest.row.to_numpy(), np.arange(n))
    for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        assert not np.intersect1d(ids[a], ids[b]).size
        assert not set(manifest.iloc[ids[a]].trajectory) & set(manifest.iloc[ids[b]].trajectory)
    assert np.array_equal(np.sort(np.concatenate(list(ids.values()))), np.arange(n))
    for name, ix in ids.items():
        assert len(ix) == description['window_counts'][name]
        assert (manifest.iloc[ix]['split'] == name).all()
    identity = {}
    for name, shape in EXPECTED.items():
        path = data / (name + '.npy'); array = np.load(path, mmap_mode='r')
        assert array.shape == (n, *shape) and array.dtype == np.float32, (name, array.shape, array.dtype)
        sample = array[ids['train'][::max(1, len(ids['train']) // 32)]]
        assert np.isfinite(sample).all(), name
        identity[path.name] = dict(bytes=path.stat().st_size, shape=list(array.shape))
        if full_hash: identity[path.name]['sha256'] = hash_file(path)
    for name in ['dataset_config.json', 'splits.npz', 'windows.csv', 'source_NOAA.csv']:
        identity[name] = dict(bytes=(data / name).stat().st_size, sha256=hash_file(data / name))
    if (data / 'normal_records.npz').exists():
        path = data / 'normal_records.npz'
        identity[path.name] = dict(bytes=path.stat().st_size, sha256=hash_file(path))
    assert identity['source_NOAA.csv']['sha256'] == description['source_sha256']
    expected_path = HERE / 'server/data_manifest.json'
    if full_hash and expected_path.exists():
        expected = json.loads(expected_path.read_text(encoding='utf-8'))
        for name, record in expected.items():
            assert name in identity and identity[name]['sha256'] == record['sha256'], f'Data checksum mismatch: {name}'
    return ids, description, identity


def fit_all_scalers(data, indices, description):
    scales = fit_scalers(data, indices)
    scales['train_mean_current'] = np.asarray(description['train_mean_current_mps'], np.float32)
    position = np.load(data / 'position.npy', mmap_mode='r')
    anchor = np.load(data / 'DR_start.npy', mmap_mode='r')
    count = 0; total = np.zeros(2); square = np.zeros(2)
    for k in range(0, len(indices), 512):
        ix = indices[k:k + 512]
        values = position[ix].astype(np.float64) - anchor[ix, None, :]
        total += values.sum((0, 1)); square += (values ** 2).sum((0, 1)); count += len(ix) * 80
    mean = total / count
    scales['relative_mean'] = mean.astype(np.float32)
    scales['relative_std'] = (np.sqrt(np.maximum(square / count - mean ** 2, 0)) + 1e-6).astype(np.float32)
    assert all(np.isfinite(v).all() for v in scales.values())
    return scales


def make_model(kind, scales):
    return DirectLSTM(scales) if kind == 'lstm' else Phase1CPI(scales['current_std'], scales['bias_std'])


def make_loader(data, indices, scales, config, shuffle=False, generator=None, device=None):
    cpu_scales = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in scales.items()}
    dataset = Phase1Dataset(data, indices, cpu_scales)
    return DataLoader(dataset, batch_size=config['batch_size'], shuffle=shuffle,
                      num_workers=config['num_workers'], pin_memory=device is not None and device.type == 'cuda',
                      generator=generator or torch.Generator().manual_seed(config['seed']))


def move_batch(batch, device):
    return {k: v.to(device, non_blocking=device.type == 'cuda') for k, v in batch.items()}


def objective(pred, batch, scales, kind, params):
    if kind == 'lstm':
        return (pred['position'] - batch['position']).square().mean() / scales['position_scale'].square()
    terms = losses(pred, batch, scales)
    return terms[1] + params['current'] * terms[2] + params['bias'] * terms[3] + params['physics'] * terms[4]


def run_epoch(model, loader, scales, kind, params, device, config, optimizer=None):
    model.train(optimizer is not None)
    # Aggregate on device; avoid synchronizing every individual batch for metrics.
    sums = torch.zeros(4, dtype=torch.float64, device=device); n = 0
    with torch.set_grad_enabled(optimizer is not None):
        for batch_id, batch in enumerate(loader, 1):
            batch = move_batch(batch, device); pred = model(batch)
            value = objective(pred, batch, scales, kind, params)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True); value.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'], error_if_nonfinite=True)
                optimizer.step()
            delta = pred['position'].detach() - batch['position']; d = delta.norm(dim=-1)
            size = len(d); n += size
            sums += torch.stack([value.detach().double() * size, d.double().sum(),
                                 d[:, -1].double().sum(), delta.double().square().sum()])
            if optimizer is not None and batch_id % config['log_every_batches'] == 0:
                print(f'  {kind} batch={batch_id}/{len(loader)} loss={value.detach().item():.6f}', flush=True)
    result = sums.cpu().numpy()
    if not n or not np.isfinite(result).all(): raise FloatingPointError('Empty epoch or non-finite loss/metrics.')
    return dict(loss=float(result[0] / n), ADE_m=float(result[1] / (n * 80)),
                FDE_m=float(result[2] / n), position_RMSE_m=float(np.sqrt(result[3] / (n * 80))))


def fit_model(kind, params, data, ids, scales, config, device, folder, epochs, signature,
              resume=False, selection='validation_loss'):
    folder.mkdir(parents=True, exist_ok=True)
    last = folder / 'last.pt'; best_path = folder / 'best.pt'
    if last.exists() and not resume: raise FileExistsError(f'{last}: pass --resume to continue.')
    if not last.exists() and best_path.exists(): raise RuntimeError(f'Incomplete epoch transaction in {folder}; keep last.pt and best.pt together.')
    seed_all(config['seed'])
    generator = torch.Generator().manual_seed(config['seed'])
    train = make_loader(data, ids['train'], scales, config, True, generator, device)
    val = make_loader(data, ids['val'], scales, config, device=device)
    model = make_model(kind, scales).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
    tensor_scales = {k: torch.as_tensor(v, device=device, dtype=torch.float32) for k, v in scales.items()}
    history = []; best_score = float('inf'); best_epoch = 0; first_epoch = 1
    metadata = dict(stage=1, model=kind, epochs=epochs, seed=config['seed'], device=str(device),
                    architecture='direct DR-relative positions' if kind == 'lstm' else 'history-only state/current plus candidate-control-conditioned motion',
                    params=params, checkpoint_selection=selection, signature=signature, formal_config=config)
    if last.exists():
        saved = read_checkpoint(last)
        if saved['signature'] != signature or saved['config']['params'] != params:
            raise ValueError('Resume configuration, code, or data differs from the saved run; use a new output directory.')
        model.load_state_dict(saved['state_dict']); optimizer.load_state_dict(saved['optimizer'])
        history = saved['history']; best_score = saved['best_score']; best_epoch = saved['best_epoch']
        first_epoch = saved['epoch'] + 1
        restore_rng(saved['rng'], generator)
        # best checkpoint is embedded in last.pt, making interrupted writes recoverable.
        publish_model(best_path, saved['best_checkpoint'])
        print(f'RESUME {folder.name}: completed={saved["epoch"]}/{epochs}', flush=True)
    elif any(p.name not in ['last.pt.tmp', 'best.pt.tmp', 'final.pt.tmp'] for p in folder.iterdir()):
        raise FileExistsError(f'Unexpected existing files in {folder}')
    best_checkpoint = read_checkpoint(best_path) if best_path.exists() else None
    for ep in range(first_epoch, epochs + 1):
        began = time.monotonic()
        tr = run_epoch(model, train, tensor_scales, kind, params, device, config, optimizer)
        va = run_epoch(model, val, tensor_scales, kind, params, device, config)
        score = va['loss'] if selection == 'validation_loss' else va['ADE_m'] + .5 * va['FDE_m']
        record = dict(epoch=ep, **{'train_' + k: v for k, v in tr.items()}, **{'val_' + k: v for k, v in va.items()}, elapsed_s=time.monotonic() - began)
        history.append(record)
        if score < best_score:
            best_score = score; best_epoch = ep
            best_checkpoint = cpu_tree(dict(state_dict=model.state_dict(), scalers=scales, config=metadata,
                                             epoch=ep, best_epoch=ep, signature=signature, best_score=score))
        checkpoint = dict(state_dict=model.state_dict(), optimizer=optimizer.state_dict(), rng=rng_state(generator),
                          scalers=scales, config=metadata, epoch=ep, best_epoch=best_epoch, best_score=best_score,
                          history=history, best_checkpoint=best_checkpoint, signature=signature)
        # last.pt is the authoritative epoch transaction. Derived files can be rebuilt.
        atomic_save(last, checkpoint)
        publish_model(best_path, best_checkpoint)
        pd.DataFrame(history).to_csv(folder / 'training_history.csv', index=False)
        atomic_json(folder / 'status.json', dict(completed_epochs=ep, target_epochs=epochs, best_epoch=best_epoch, last_metrics=record))
        print(f'{kind} epoch={ep}/{epochs} train_loss={tr["loss"]:.6f} val_loss={va["loss"]:.6f} val_ADE={va["ADE_m"]:.3f} val_FDE={va["FDE_m"]:.3f} elapsed={record["elapsed_s"]:.1f}s', flush=True)
    final = read_checkpoint(last)
    publish_model(best_path, final['best_checkpoint'])
    publish_model(folder / 'final.pt', {k: final[k] for k in ['state_dict', 'scalers', 'config', 'epoch', 'best_epoch', 'signature']})
    pd.DataFrame(final['history']).to_csv(folder / 'training_history.csv', index=False)
    atomic_json(folder / 'status.json', dict(completed_epochs=final['epoch'], target_epochs=epochs, best_epoch=final['best_epoch'], complete=True))
    return read_checkpoint(best_path)


def pso_search(data, ids, scales, config, device, out, signature, resume):
    pso = config['pso']; folder = out / 'pso_search'; folder.mkdir(exist_ok=True)
    if pso['mode'] == 'fixed':
        selected = dict(params=pso['fixed_hparams'], mode='fixed_prior_smoke_parameters', search_performed=False)
        atomic_json(folder / 'selected.json', selected)
        return selected['params']
    if pso['mode'] != 'search': raise ValueError('pso.mode must be search or fixed')
    names = ['lr', 'current', 'bias', 'physics']
    low = np.array([pso['bounds'][k][0] for k in names], float)
    high = np.array([pso['bounds'][k][1] for k in names], float)
    low[0] = np.log10(low[0]); high[0] = np.log10(high[0])
    def decode(x): return {k: float(10 ** x[j] if j == 0 else x[j]) for j, k in enumerate(names)}
    # A separate RNG isolates swarm evolution and subsets from neural training randomness.
    rng = np.random.default_rng(config['seed'])
    small_ids = {k: np.sort(rng.choice(ids[k], min(len(ids[k]), pso[k + '_windows']), replace=False)) for k in ['train', 'val']}
    state_path = folder / 'swarm.pt'
    if state_path.exists():
        if not resume: raise FileExistsError('Existing PSO state; pass --resume.')
        state = read_checkpoint(state_path)
        if state['signature'] != signature: raise ValueError('PSO resume signature mismatch')
        rng.bit_generator.state = state['rng']
    else:
        positions = rng.uniform(low, high, (pso['population'], len(names)))
        positions[0] = [np.log10(config['default_hparams']['lr'])] + [config['default_hparams'][k] for k in names[1:]]
        state = dict(iteration=0, next_particle=0, positions=positions, velocities=np.zeros_like(positions),
                     personal=positions.copy(), scores=np.full(pso['population'], np.inf),
                     global_position=positions[0].copy(), global_score=float('inf'), records=[], signature=signature,
                     rng=rng.bit_generator.state)
        atomic_save(state_path, state)
    np.savez(folder / 'subset_indices.npz', **small_ids)
    for iteration in range(state['iteration'], pso['iterations']):
        for particle in range(state['next_particle'], pso['population']):
            params = decode(state['positions'][particle])
            trial = folder / f'iter{iteration+1:02d}_particle{particle+1:02d}'
            trial_signature = canonical_hash(dict(parent=signature, iteration=iteration, particle=particle, params=params))
            saved = fit_model('pso_cpi_lstm', params, data, small_ids, scales, config, device, trial,
                              pso['candidate_epochs'], trial_signature, resume, selection='validation_ADE_plus_half_FDE')
            score = saved['best_score']
            state['records'].append(dict(iteration=iteration+1, particle=particle+1, fitness=score, best_epoch=saved['best_epoch'], **params))
            if score < state['scores'][particle]:
                state['scores'][particle] = score; state['personal'][particle] = state['positions'][particle].copy()
            if score < state['global_score']:
                state['global_score'] = score; state['global_position'] = state['positions'][particle].copy()
            state['next_particle'] = particle + 1; state['rng'] = rng.bit_generator.state
            atomic_save(state_path, state)
            pd.DataFrame(state['records']).to_csv(folder / 'search_history.csv', index=False)
            print(f'PSO iteration={iteration+1}/{pso["iterations"]} particle={particle+1}/{pso["population"]} fitness={score:.3f}', flush=True)
        if iteration + 1 < pso['iterations']:
            r1, r2 = rng.random(state['positions'].shape), rng.random(state['positions'].shape)
            state['velocities'] = pso['inertia'] * state['velocities'] + pso['cognitive'] * r1 * (state['personal']-state['positions']) + pso['social'] * r2 * (state['global_position']-state['positions'])
            state['positions'] = np.clip(state['positions'] + state['velocities'], low, high)
        state['iteration'] = iteration + 1; state['next_particle'] = 0; state['rng'] = rng.bit_generator.state
        atomic_save(state_path, state)
    # Derived output is rebuilt even when a resumed swarm is already complete.
    pd.DataFrame(state['records']).to_csv(folder / 'search_history.csv', index=False)
    params = decode(state['global_position'])
    atomic_json(folder / 'selected.json', dict(params=params, mode='search', fitness=state['global_score'],
                evaluations=len(state['records']), fitness_definition='minimum validation ADE + 0.5 FDE over candidate epochs', test_used=False))
    return params


def runtime_info(device):
    return dict(python=sys.version, torch=torch.__version__, numpy=np.__version__, pandas=pd.__version__,
                device=str(device), cuda_runtime=torch.version.cuda,
                gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                gpu_memory_bytes=torch.cuda.get_device_properties(device).total_memory if device.type == 'cuda' else None)


def code_identity():
    return {name: hash_file(HERE / name) for name in ['formal_phase1.py', 'formal_phase1_evaluate.py', 'phase1_model.py', 'train_phase1.py']}


def train_all(data, config, device, out, resume):
    print('Checking dataset checksums and train/validation separation...', flush=True)
    ids, description, identity = inspect_data(data)
    signature = canonical_hash(dict(config=config, data=identity, code=code_identity()))
    run_path = out / 'run.json'
    if run_path.exists():
        if not resume: raise FileExistsError('Run already exists. Use --resume or a new output directory.')
        if json.loads(run_path.read_text(encoding='utf-8'))['signature'] != signature:
            raise ValueError('Run signature mismatch: code, configuration or data changed.')
    else:
        atomic_json(run_path, dict(signature=signature, config=config, data_identity=identity, code=code_identity(), runtime=runtime_info(device)))
        snapshot = out / 'provenance'; snapshot.mkdir(exist_ok=True)
        for name in code_identity(): shutil.copy2(HERE / name, snapshot / name)
    scaler_meta = out / 'scalers_identity.json'
    if (out / 'scalers.npz').exists() and scaler_meta.exists():
        if hash_file(out / 'scalers.npz') != json.loads(scaler_meta.read_text())['sha256']:
            raise ValueError('Saved scalers were modified; do not mix normalization between runs.')
        with np.load(out / 'scalers.npz') as z: scales = {k: z[k] for k in z.files}
    else:
        print('Fitting normalization on training windows only...', flush=True)
        scales = fit_all_scalers(data, ids['train'], description)
        temporary = out / 'scalers.tmp.npz'; np.savez(temporary, **scales); temporary.replace(out / 'scalers.npz')
        atomic_json(scaler_meta, dict(sha256=hash_file(out / 'scalers.npz'), fit_on='training split only'))
    selections = {}
    for kind in config['models']:
        params = config['default_hparams'] if kind != 'pso_cpi_lstm' else pso_search(data, ids, scales, config, device, out, signature, resume)
        print(f'FINAL TRAINING {kind}: {config["epochs"]} epochs, params={params}', flush=True)
        saved = fit_model(kind, params, data, ids, scales, config, device, out / kind,
                          config['epochs'], signature, resume, config['checkpoint_selection'])
        selections[kind] = dict(checkpoint=f'{kind}/best.pt', best_epoch=saved['best_epoch'],
                                sha256=hash_file(out / kind / 'best.pt'), params=params)
        atomic_json(out / 'training_status.json', dict(completed_models=list(selections), target_models=config['models'], epochs_per_final_model=config['epochs']))
    atomic_json(out / 'selected_checkpoints.json', dict(signature=signature, selected=selections, test_used_for_selection=False,
                                                       epochs_completed_per_final_model=config['epochs'], scalers_sha256=hash_file(out / 'scalers.npz')))
    print('All models trained. Test set has not been evaluated; run --mode evaluate.', flush=True)


def preflight(data, config, device, out):
    ids, description, identity = inspect_data(data)
    print('Data checks passed; testing forward/backward on TRAINING samples only...', flush=True)
    size = min(config['batch_size'], len(ids['train']))
    scales = fit_all_scalers(data, ids['train'][:size], description)
    quick_config = dict(config, num_workers=0)
    batch = move_batch(next(iter(make_loader(data, ids['train'][:size], scales, quick_config, device=device))), device)
    tensor_scales = {k: torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in scales.items()}
    checks = {}
    if device.type == 'cuda': torch.cuda.reset_peak_memory_stats(device)
    for kind in ['lstm', 'cpi_lstm']:
        seed_all(config['seed']); model = make_model(kind, scales).to(device)
        value = objective(model(batch), batch, tensor_scales, kind, config['default_hparams'])
        value.backward()
        assert torch.isfinite(value) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        checks[kind] = dict(loss=float(value.detach()), backward_finite=True, parameter_device=str(next(model.parameters()).device))
        if device.type == 'cuda':
            reference = make_model(kind, scales)
            reference.load_state_dict(cpu_tree(model.state_dict()))
            model.eval(); reference.eval()
            with torch.no_grad():
                gpu_pred = model(batch)['position'].cpu()
                cpu_pred = reference({k: v.cpu() for k, v in batch.items()})['position']
            torch.testing.assert_close(gpu_pred, cpu_pred, rtol=1e-4, atol=1e-3)
            checks[kind]['cpu_gpu_forward_close'] = True
            del reference, gpu_pred, cpu_pred
        del model, value
    if device.type == 'cuda': torch.cuda.synchronize(device)
    atomic_json(out / 'preflight.json', dict(passed=True, runtime=runtime_info(device), checks=checks,
                data_fingerprint=canonical_hash(identity), window_counts={k: len(v) for k, v in ids.items()},
                checked_batch_size=size, peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
                formal_training_started=False, preflight_only_uses_training_samples=True))
    print(json.dumps(dict(preflight='PASSED', runtime=runtime_info(device)), indent=2), flush=True)


def load_config(path):
    config = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    assert config['epochs'] > 0 and config['batch_size'] > 0 and config['num_workers'] >= 0
    assert config['models'] and len(set(config['models'])) == len(config['models'])
    assert all(k in MODEL_NAMES for k in config['models'])
    assert config['checkpoint_selection'] == 'validation_loss'
    if 'pso_cpi_lstm' in config['models'] and config['pso']['mode'] == 'search':
        for k in ['population', 'iterations', 'candidate_epochs', 'train_windows', 'val_windows']:
            assert config['pso'][k] > 0
        for lo, hi in config['pso']['bounds'].values(): assert 0 < lo <= hi
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['preflight', 'train', 'evaluate'], default='train')
    parser.add_argument('--config', type=Path, default=HERE / 'server/phase1_30ep.json')
    parser.add_argument('--data', type=Path, default=HERE / 'data')
    parser.add_argument('--output', type=Path, default=HERE / 'results/formal_phase1_30ep_seed42')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    config = load_config(args.config); device = configure(config, args.device)
    with run_lock(args.output):
        if args.mode == 'preflight': preflight(args.data, config, device, args.output)
        elif args.mode == 'train': train_all(args.data, config, device, args.output, args.resume)
        else:
            from formal_phase1_evaluate import evaluate_run
            evaluate_run(args.data, config, device, args.output)


if __name__ == '__main__':
    main()
