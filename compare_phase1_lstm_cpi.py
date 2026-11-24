"""Paired five-epoch phase-one comparison with immutable offline test checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from phase1_model import Phase1CPI, Phase1Dataset, losses
from train_phase1 import fit_scalers

HERE = Path(__file__).resolve().parent
MODES = ['all', 'normal', 'outage_0_16s', 'outage_16_60s', 'outage_over_60s']
COLORS = {'LSTM': '#d97706', 'CPI-LSTM': '#007c91'}


class Phase1LSTM(nn.Module):
    """Same history/control encoder sizes, direct learned positions, no physics."""
    def __init__(self, mean, std):
        super().__init__()
        self.hist_rnn = nn.LSTM(16, 128, batch_first=True)
        self.ctrl_rnn = nn.LSTM(2, 96, batch_first=True)
        self.head = nn.Sequential(nn.Linear(224, 128), nn.ReLU(), nn.Linear(128, 160))
        self.register_buffer('relative_mean', torch.as_tensor(mean))
        self.register_buffer('relative_std', torch.as_tensor(std))

    def forward(self, batch):
        h, _ = self.hist_rnn(batch['x'])
        u, _ = self.ctrl_rnn(batch['u'])
        relative = self.head(torch.cat([h[:, -1], u[:, -1]], -1)).view(-1, 80, 2)
        return {'position': batch['DR_start'][:, None] + self.relative_mean + relative * self.relative_std}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_epoch(model, loader, scales, kind, optimizer=None):
    model.train(optimizer is not None)
    n = 0
    totals = np.zeros(4)
    with torch.set_grad_enabled(optimizer is not None):
        for b in loader:
            p = model(b)
            delta = p['position'] - b['position']
            objective = losses(p, b, scales)[0] if kind == 'CPI-LSTM' else delta.square().mean() / scales['position_scale'].square()
            if optimizer is not None:
                optimizer.zero_grad()
                objective.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 2.)
                optimizer.step()
            d = delta.detach().norm(dim=-1)
            size = len(d)
            totals += [objective.detach().item() * size, d.mean().item() * size,
                       d[:, -1].mean().item() * size, delta.detach().square().mean().item() * size]
            n += size
    values = totals / n
    assert np.isfinite(values).all()
    return dict(loss=values[0], ADE_m=values[1], FDE_m=values[2], coordinate_RMSE_m=np.sqrt(values[3]))


def masks(ages):
    return [np.ones(len(ages), bool), ages == 0, (ages > 0) & (ages <= 16),
            (ages > 16) & (ages <= 60), ages > 60]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE / 'results/phase1_lstm_vs_cpi_5ep_seed42')
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    (out / 'provenance').mkdir()
    for name in ['compare_phase1_lstm_cpi.py', 'phase1_model.py', 'train_phase1.py', 'prepare_phase1.py']:
        shutil.copy2(HERE / name, out / 'provenance' / name)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    data = HERE / 'data'
    with np.load(data / 'splits.npz') as z:
        ids = {k: z[k] for k in z.files}
    for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        assert not np.intersect1d(ids[a], ids[b]).size
    scales = fit_scalers(data, ids['train'])
    # Direct-output normalization is also fitted on training windows only.
    position = np.load(data / 'position.npy', mmap_mode='r')
    anchor = np.load(data / 'DR_start.npy', mmap_mode='r')
    total = np.zeros(2); square = np.zeros(2); count = 0
    for i in range(0, len(ids['train']), 512):
        ix = ids['train'][i:i + 512]
        relative = position[ix].astype(np.float64) - anchor[ix, None, :]
        total += relative.sum((0, 1)); square += (relative ** 2).sum((0, 1)); count += relative.shape[0] * 80
    mean = total / count
    scales['relative_mean'] = mean.astype(np.float32)
    scales['relative_std'] = (np.sqrt(np.maximum(square / count - mean ** 2, 0)) + 1e-6).astype(np.float32)
    np.savez(out / 'scalers.npz', **scales)
    tensors = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in scales.items()}
    datasets = {k: Phase1Dataset(data, ix, tensors) for k, ix in ids.items()}
    config = dict(epochs=5, seed=42, batch_size=128, optimizer='Adam', lr=.001, gradient_clip=2.,
                  history_hidden=128, control_hidden=96, dt=.2, horizon_steps=80,
                  windows={k: len(v) for k, v in ids.items()}, device='cpu',
                  checkpoint_selection='minimum own validation objective over epochs 1..5',
                  lstm='direct DR-relative positions; position MSE only; no physics or auxiliary labels',
                  cpi='unchanged Phase1CPI; position + 0.1 current + 0.2 bias + 0.02 physics',
                  test='frozen offline candidate-plan prediction; no optimizer, adaptation, or test-based selection',
                  prior_test_exposure='Existing project test split has been evaluated before; this is a repeated paired evaluation, not a new blind test.',
                  dataset=json.loads((data / 'dataset_config.json').read_text(encoding='utf-8')))
    (out / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    began = time.monotonic()
    selected = {}
    history = []
    # Both are retrained from scratch; separate identical shuffle generators pair minibatches.
    for kind in COLORS:
        torch.manual_seed(42); np.random.seed(42)
        model = Phase1LSTM(mean.astype(np.float32), scales['relative_std']) if kind == 'LSTM' else Phase1CPI(scales['current_std'], scales['bias_std'])
        folder = out / kind.lower().replace('-', '_'); folder.mkdir()
        loaders = {k: DataLoader(datasets[k], batch_size=128, shuffle=k == 'train', num_workers=0,
                                 generator=torch.Generator().manual_seed(42)) for k in ['train', 'val']}
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        best = float('inf'); best_epoch = None
        parameters = sum(p.numel() for p in model.parameters())
        for ep in range(1, 6):
            train = run_epoch(model, loaders['train'], tensors, kind, optimizer)
            val = run_epoch(model, loaders['val'], tensors, kind)
            row = dict(model=kind, epoch=ep, **{'train_' + k: v for k, v in train.items()}, **{'val_' + k: v for k, v in val.items()})
            history.append(row)
            if val['loss'] < best:
                best = val['loss']; best_epoch = ep
                torch.save(dict(state_dict=model.state_dict(), model=kind, epoch=ep, scalers=scales, config=config), folder / 'best.pt')
            pd.DataFrame(history).to_csv(out / 'training_history.csv', index=False)
            print(f'{kind} epoch={ep}/5 train_loss={train["loss"]:.6f} val_loss={val["loss"]:.6f} val_ADE={val["ADE_m"]:.3f} val_FDE={val["FDE_m"]:.3f} elapsed={time.monotonic()-began:.0f}s', flush=True)
        torch.save(dict(state_dict=model.state_dict(), model=kind, epoch=5, scalers=scales, config=config), folder / 'final.pt')
        selected[kind] = dict(folder=folder.name, best_epoch=best_epoch, parameters=parameters)
    # Commit selection for BOTH models before accessing test samples.
    (out / 'selected_checkpoints.json').write_text(json.dumps(selected, indent=2), encoding='utf-8')
    predictions = {}; hashes = {}; checks = {}
    loader = DataLoader(datasets['test'], batch_size=128, shuffle=False)
    with torch.inference_mode():
        for kind, info in selected.items():
            path = out / info['folder'] / 'best.pt'
            frozen = path.with_name('frozen.pt'); shutil.copy2(path, frozen)
            hashes[kind] = digest(frozen)
            model = Phase1LSTM(scales['relative_mean'], scales['relative_std']) if kind == 'LSTM' else Phase1CPI(scales['current_std'], scales['bias_std'])
            model.load_state_dict(torch.load(frozen, map_location='cpu', weights_only=False)['state_dict'])
            model.eval().requires_grad_(False)
            before = {k: v.clone() for k, v in model.state_dict().items()}
            sample = next(iter(loader)); original = model(sample)
            poison = {k: v.clone() for k, v in sample.items()}
            for k in ['position', 'current', 'velocity', 'bias']: poison[k].fill_(99999.)
            for k, v in model(poison).items(): torch.testing.assert_close(v, original[k], rtol=0, atol=0)
            if kind == 'CPI-LSTM':
                poison['u'] *= -1; poison['plan'] *= -1
                altered = model(poison)
                for k in ['bias', 'current']: torch.testing.assert_close(altered[k], original[k], rtol=0, atol=0)
            chunks = []
            for batch in loader: chunks.append(model(batch)['position'].numpy())
            predictions[kind] = np.concatenate(chunks)
            for k, v in model.state_dict().items(): torch.testing.assert_close(v, before[k], rtol=0, atol=0)
            assert hashes[kind] == digest(frozen)
            checks[kind] = dict(label_poison_invariance=True, weights_unchanged=True, checkpoint_unchanged=True)
            print(f'Frozen test complete: {kind}, {len(predictions[kind])} windows', flush=True)
    truth = np.asarray(position[ids['test']]); ages = np.load(data / 'outage_age.npy', mmap_mode='r')[ids['test']]
    np.savez_compressed(out / 'test_predictions.npz', row=ids['test'], truth=truth, age=ages,
                        lstm=predictions['LSTM'], cpi_lstm=predictions['CPI-LSTM'])
    metrics = []; per_window = []; horizon = []
    for kind, p in predictions.items():
        delta = p.astype(np.float64) - truth.astype(np.float64)
        d = np.linalg.norm(delta, axis=-1)
        per_window.append(pd.DataFrame(dict(model=kind, row=ids['test'], outage_age_s=ages, ADE_m=d.mean(1), FDE_m=d[:, -1])))
        for mode, mask in zip(MODES, masks(ages)):
            metrics.append(dict(model=kind, mode=mode, windows=int(mask.sum()), ADE_m=d[mask].mean(), FDE_m=d[mask, -1].mean(),
                                position_RMSE_m=np.sqrt((d[mask] ** 2).mean()), P90_window_ADE_m=np.percentile(d[mask].mean(1), 90)))
            horizon.append(pd.DataFrame(dict(model=kind, mode=mode, horizon_s=np.arange(1, 81) * .2, mean_error_m=d[mask].mean(0))))
    frame = pd.DataFrame(metrics); frame.to_csv(out / 'test_metrics.csv', index=False)
    windows = pd.concat(per_window); windows.to_csv(out / 'per_window_metrics.csv', index=False)
    horizons = pd.concat(horizon); horizons.to_csv(out / 'horizon_errors.csv', index=False)
    make_plots(out, frame, windows, horizons, predictions, truth, ages, ids['test'], history)
    verification = dict(passed=True, epochs_completed={'LSTM': 5, 'CPI-LSTM': 5}, selected=selected,
                        frozen_sha256=hashes, checks=checks, test_windows=len(truth),
                        train_only_normalization=True, test_used_for_selection=False,
                        elapsed_s=time.monotonic() - began, offline_prediction_only=True)
    (out / 'verification.json').write_text(json.dumps(verification, indent=2), encoding='utf-8')
    overall = frame[frame['mode'] == 'all']
    lines = ['Phase-one LSTM vs CPI-LSTM: 5 epochs each', '', overall.to_string(index=False), '',
             'ADE: mean Euclidean position error over 80 future steps and all test windows (m).',
             'FDE: mean Euclidean error at 16 seconds (m). Position RMSE: sqrt(mean squared Euclidean error).',
             'P90: 90th percentile of per-window ADE. Overlapping windows are not independent trials.',
             'Trajectory examples: first test window in each actual outage group, fixed without inspecting errors.',
             'Both models trained from scratch, identical input/splits/shuffle seed/Adam/lr/batch size/5 epochs.',
             'LSTM uses direct position MSE; CPI uses the original four-term loss and physical integration.',
             'Validation losses across architectures have different meanings; compare ADE/FDE curves.',
             'Both best checkpoints chosen on their own validation objective, then frozen before paired test.',
             'This is offline candidate-control trajectory prediction, not model-controlled waypoint tracking.',
             'The existing test set was exposed in earlier project runs; this is not a new blind test.',
             'Single seed, five epochs: results do not establish converged or multi-seed superiority.', '',
             json.dumps(verification, indent=2)]
    (out / 'report.txt').write_text('\n'.join(lines), encoding='utf-8')
    print(overall.to_string(index=False), flush=True)
    print('VERIFIED ' + str(out.resolve()), flush=True)


def make_plots(out, metrics, windows, horizons, predictions, truth, ages, rows, history):
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), layout='constrained')
    for ax, key, label in zip(axes, ['ADE_m', 'FDE_m', 'position_RMSE_m', 'P90_window_ADE_m'], ['ADE', 'FDE (16 s)', 'Position RMSE', 'P90 window ADE']):
        part = metrics[metrics['mode'] == 'all']
        bars = ax.bar(part.model, part[key], color=[COLORS[k] for k in part.model], width=.6)
        ax.bar_label(bars, fmt='%.2f', padding=4); ax.set(ylabel='Error (m)', title=label); ax.set_ylim(0, part[key].max() * 1.2); ax.grid(axis='y', alpha=.2)
    fig.suptitle('Frozen paired test | 26,000 windows | 5 training epochs each')
    fig.savefig(out / 'metrics_comparison.png', dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), layout='constrained')
    for ax, key in zip(axes, ['ADE_m', 'FDE_m']):
        for offset, kind in enumerate(COLORS):
            part = metrics[(metrics.model == kind) & (metrics['mode'] != 'all')]
            bars = ax.bar(np.arange(4) + (offset - .5) * .36, part[key], .36, label=kind, color=COLORS[kind])
            ax.bar_label(bars, fmt='%.1f', fontsize=8, padding=3)
        ax.set_xticks(range(4), ['Normal', '0 < age <= 16 s', '16 < age <= 60 s', 'Age > 60 s'])
        ax.set(title=key.replace('_', ' '), ylabel='Error (m)'); ax.legend(); ax.grid(axis='y', alpha=.2)
    fig.savefig(out / 'metrics_by_outage.png', dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
    for kind in COLORS:
        h = horizons[(horizons.model == kind) & (horizons['mode'] == 'all')]
        axes[0].plot(h.horizon_s, h.mean_error_m, label=kind, color=COLORS[kind])
        values = np.sort(windows[windows.model == kind].ADE_m)
        axes[1].plot(values, np.arange(1, len(values) + 1) / len(values), label=kind, color=COLORS[kind])
    axes[0].set(xlabel='Prediction horizon (s)', ylabel='Mean position error (m)', title='Error versus prediction horizon')
    axes[1].set(xlabel='Per-window ADE (m)', ylabel='Empirical cumulative probability', title='ADE distribution')
    for ax in axes: ax.legend(); ax.grid(alpha=.2)
    fig.savefig(out / 'horizon_and_cdf.png', dpi=180); plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), layout='constrained')
    examples = []
    anchor = np.load(HERE / 'data/DR_start.npy', mmap_mode='r')
    bias = np.load(HERE / 'data/bias.npy', mmap_mode='r')
    for ax, mode, mask in zip(axes.ravel(), MODES[1:], masks(ages)[1:]):
        i = int(np.flatnonzero(mask)[0]); row = int(rows[i]); start = anchor[row] + bias[row]
        examples.append(dict(mode=mode, row=row, outage_age_s=float(ages[i])))
        reference = np.vstack([np.zeros(2), truth[i] - start])
        ax.plot(reference[:, 0], reference[:, 1], 'k-o', markevery=[-1], ms=4, lw=2, label='Truth')
        for kind, p in predictions.items():
            relative = p[i] - start
            ax.plot(relative[:, 0], relative[:, 1], color=COLORS[kind], label=kind, lw=1.7)
            ax.scatter(*relative[-1], color=COLORS[kind], marker='x', s=40)
        ax.scatter(0, 0, marker='*', color='black', s=60, label='True anchor (display only)')
        ax.set(title=f'{mode} | row {row} | age {ages[i]:.1f} s', xlabel='East relative to true anchor (m)', ylabel='North relative to true anchor (m)')
        ax.set_aspect('equal', adjustable='datalim'); ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle('Same candidate plan and truth for both frozen models | fixed examples')
    fig.savefig(out / 'trajectory_comparison.png', dpi=180); plt.close(fig)
    (out / 'trajectory_examples.json').write_text(json.dumps(examples, indent=2), encoding='utf-8')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
    h = pd.DataFrame(history)
    for kind in COLORS:
        part = h[h.model == kind]
        for ax, metric in zip(axes, ['ADE_m', 'FDE_m']):
            ax.plot(part.epoch, part['val_' + metric], '-o', color=COLORS[kind], label=kind + ' validation')
            ax.plot(part.epoch, part['train_' + metric], '--', color=COLORS[kind], alpha=.6, label=kind + ' train')
            ax.set(xlabel='Epoch', ylabel=metric.replace('_', ' ')); ax.set_xticks(range(1, 6)); ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.savefig(out / 'training_comparison.png', dpi=180); plt.close(fig)


if __name__ == '__main__':
    main()
