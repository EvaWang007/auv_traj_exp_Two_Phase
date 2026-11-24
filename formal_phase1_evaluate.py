"""Frozen, paired evaluation after ALL phase-one final training has finished."""
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from formal_phase1 import (HERE, MODEL_NAMES, atomic_json, canonical_hash, code_identity,
                          hash_file, inspect_data, make_loader, make_model, move_batch, read_checkpoint)

MODES = ['all', 'normal', 'outage_0_16s', 'outage_16_60s', 'outage_over_60s']
COLORS = {'lstm': '#d97706', 'cpi_lstm': '#007c91', 'pso_cpi_lstm': '#b91c1c'}


def mode_masks(ages):
    return [np.ones(len(ages), bool), ages == 0, (ages > 0) & (ages <= 16),
            (ages > 16) & (ages <= 60), ages > 60]


def state_hash(model):
    import hashlib
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode()); digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def evaluate_run(data, config, device, out):
    run = json.loads((out / 'run.json').read_text(encoding='utf-8'))
    selection = json.loads((out / 'selected_checkpoints.json').read_text(encoding='utf-8'))
    ids, _, identity = inspect_data(data)
    signature = canonical_hash(dict(config=config, data=identity, code=code_identity()))
    if signature != run['signature'] or selection['signature'] != signature:
        raise ValueError('Evaluation must use the original code, config, and data of the training run.')
    if set(selection['selected']) != set(config['models']): raise ValueError('Not all models have completed training.')
    if hash_file(out / 'scalers.npz') != selection['scalers_sha256']:
        raise ValueError('Normalization differs from the frozen training selection.')
    target = out / 'test'; target.mkdir(exist_ok=True)
    policy = dict(signature=signature, test_used_for_checkpoint_selection=False,
                  test_windows=len(ids['test']), prior_test_exposure=True, models={})
    # Freeze every checkpoint before accessing numerical test targets.
    for kind in config['models']:
        info = selection['selected'][kind]
        status = read_checkpoint(out / kind / 'last.pt')
        if status['epoch'] != config['epochs']: raise ValueError(f'{kind} has not completed the required epochs.')
        source = out / info['checkpoint']
        if hash_file(source) != info['sha256']: raise ValueError('Selected checkpoint was changed after selection.')
        frozen = target / (kind + '_frozen.pt')
        if frozen.exists() and hash_file(frozen) != info['sha256']:
            raise ValueError('Existing frozen checkpoint belongs to another selection.')
        if not frozen.exists():
            temporary = frozen.with_name(frozen.name + '.tmp')
            shutil.copy2(source, temporary); temporary.replace(frozen)
        policy['models'][kind] = dict(sha256=info['sha256'], best_epoch=info['best_epoch'], file=frozen.name)
    atomic_json(target / 'frozen_policy.json', policy)
    with np.load(out / 'scalers.npz') as z: scales = {k: z[k] for k in z.files}
    predictions = {}; checks = {}; auxiliary = {}
    for kind in config['models']:
        frozen = target / policy['models'][kind]['file']
        saved = read_checkpoint(frozen)
        model = make_model(kind, scales).to(device)
        model.load_state_dict(saved['state_dict']); model.eval().requires_grad_(False)
        before = state_hash(model)
        loader = make_loader(data, ids['test'], scales, config, device=device)
        positions = []; currents = []; biases = []; row_batches = []
        with torch.inference_mode():
            # Ground-truth poisoning tests are conducted only after model selection is frozen.
            batch = move_batch(next(iter(loader)), device)
            original = model(batch)
            altered = {k: v.clone() for k, v in batch.items()}
            for name in ['position', 'current', 'velocity', 'bias']: altered[name].fill_(99999.)
            poisoned = model(altered)
            for name in original: torch.testing.assert_close(original[name], poisoned[name], rtol=0, atol=0)
            if kind != 'lstm':
                altered['u'] *= -1; altered['plan'] *= -1
                changed_plan = model(altered)
                for name in ['current', 'bias']: torch.testing.assert_close(original[name], changed_plan[name], rtol=0, atol=0)
            for b in loader:
                batch = move_batch(b, device); pred = model(batch)
                positions.append(pred['position'].cpu().numpy()); row_batches.append(b['row'].numpy())
                if kind != 'lstm':
                    currents.append(pred['current'].cpu().numpy()); biases.append(pred['bias'].cpu().numpy())
        np.testing.assert_array_equal(np.concatenate(row_batches), ids['test'])
        predictions[kind] = np.concatenate(positions)
        if kind != 'lstm': auxiliary[kind] = dict(current=np.concatenate(currents), bias=np.concatenate(biases))
        assert state_hash(model) == before and hash_file(frozen) == policy['models'][kind]['sha256']
        checks[kind] = dict(label_poison_invariance=True, weights_unchanged=True, frozen_file_unchanged=True)
        np.savez_compressed(target / f'{kind}_predictions.npz', row=ids['test'], position=predictions[kind], **auxiliary.get(kind, {}))
        print(f'Frozen evaluation complete: {MODEL_NAMES[kind]}, {len(ids["test"])} windows', flush=True)
    truth = np.load(data / 'position.npy', mmap_mode='r')[ids['test']]
    true_current = np.load(data / 'current.npy', mmap_mode='r')[ids['test']]
    true_bias = np.load(data / 'bias.npy', mmap_mode='r')[ids['test']]
    ages = np.load(data / 'outage_age.npy', mmap_mode='r')[ids['test']]
    manifest = pd.read_csv(data / 'windows.csv').iloc[ids['test']]
    np.savez_compressed(target / 'test_targets.npz', row=ids['test'], truth=truth, true_current=true_current, true_bias=true_bias, age=ages)
    records = []; per_window = []; horizon = []
    for kind, p in predictions.items():
        if p.shape != truth.shape or not np.isfinite(p).all(): raise ValueError('Invalid model predictions')
        d = np.linalg.norm(p.astype(np.float64) - truth.astype(np.float64), axis=-1)
        per_window.append(pd.DataFrame(dict(model=MODEL_NAMES[kind], row=ids['test'], trajectory=manifest.trajectory.to_numpy(),
                                            outage_age_s=ages, ADE_m=d.mean(1), FDE_m=d[:, -1])))
        for mode, mask in zip(MODES, mode_masks(ages)):
            if not mask.any(): continue
            row = dict(model=MODEL_NAMES[kind], model_key=kind, mode=mode, windows=int(mask.sum()),
                       ADE_m=float(d[mask].mean()), FDE_m=float(d[mask, -1].mean()),
                       position_RMSE_m=float(np.sqrt((d[mask] ** 2).mean())),
                       P90_window_ADE_m=float(np.percentile(d[mask].mean(1), 90)))
            if kind in auxiliary:
                row['current_coordinate_RMSE_mps'] = float(np.sqrt(np.mean((auxiliary[kind]['current'][mask].astype(float) - true_current[mask]) ** 2)))
                row['bias_position_RMSE_m'] = float(np.sqrt(np.mean(np.sum((auxiliary[kind]['bias'][mask].astype(float) - true_bias[mask]) ** 2, axis=-1))))
            records.append(row)
            horizon.append(pd.DataFrame(dict(model=MODEL_NAMES[kind], model_key=kind, mode=mode,
                                             horizon_s=np.arange(1, 81)*.2, mean_error_m=d[mask].mean(0))))
    metrics = pd.DataFrame(records); metrics.to_csv(target / 'test_metrics.csv', index=False)
    windows = pd.concat(per_window, ignore_index=True); windows.to_csv(target / 'per_window_metrics.csv', index=False)
    horizons = pd.concat(horizon, ignore_index=True); horizons.to_csv(target / 'horizon_errors.csv', index=False)
    plots(data, out, predictions, truth, ages, ids['test'], metrics, windows, horizons)
    verification = dict(passed=True, epochs_per_final_model=config['epochs'], checks=checks,
                        test_windows=len(truth), test_used_for_selection=False, offline_prediction_only=True,
                        evaluation_device=str(device), prior_test_exposure=True)
    atomic_json(target / 'verification.json', verification)
    overall = metrics[metrics['mode'] == 'all']
    report = [f'第一阶段三模型正式训练：最终模型各 {config["epochs"]} epoch', '', overall.to_string(index=False), '',
              '模型按各自验证损失最低选取；PSO只使用训练/验证子集，不使用测试集。',
              'PSO候选搜索计算量单独记录，最终模型重新初始化后训练完整轮数。',
              'ADE为80步欧氏位置误差均值；FDE为16秒处欧氏位置误差均值。',
              'position_RMSE为sqrt(mean(||位置误差||²))，不是逐坐标RMSE。',
              '测试模型均冻结；这是已有留出测试集上的离线配对复测，不是新生成任务或第二阶段闭环测试。',
              '单种子；集合内部窗口重叠；不能把窗口数当作独立任务数。',
              'trajectory_comparison.png固定选每个缺失分组的第一个测试窗口，不按预测效果筛选。']
    (target / 'report_zh.txt').write_text('\n'.join(report), encoding='utf-8')
    print(overall.to_string(index=False), flush=True)


def plots(data, out, predictions, truth, ages, rows, metrics, windows, horizons):
    target = out / 'test'
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    keys = list(predictions); count = len(keys)
    overall = metrics[metrics['mode'] == 'all'].set_index('model_key').loc[keys]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), layout='constrained')
    for ax, column in zip(axes, ['ADE_m', 'FDE_m', 'position_RMSE_m', 'P90_window_ADE_m']):
        bars = ax.bar([MODEL_NAMES[k] for k in keys], overall[column], color=[COLORS[k] for k in keys])
        ax.bar_label(bars, fmt='%.2f', padding=3); ax.set(title=column, ylabel='Error (m)')
        ax.set_ylim(0, max(float(overall[column].max()) * 1.2, .1)); ax.tick_params(axis='x', labelrotation=15); ax.grid(axis='y', alpha=.2)
    fig.suptitle('Frozen paired offline test | identical history, plan, and truth')
    fig.savefig(target / 'metrics_comparison.png', dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), layout='constrained')
    for ax, column in zip(axes, ['ADE_m', 'FDE_m']):
        for j, kind in enumerate(keys):
            part = metrics[(metrics.model_key == kind) & (metrics['mode'] != 'all')].set_index('mode').reindex(MODES[1:])
            width = .8 / count
            ax.bar(np.arange(4) + (j - (count-1)/2)*width, part[column], width, label=MODEL_NAMES[kind], color=COLORS[kind])
        ax.set_xticks(range(4), ['Normal', '0 < age <= 16 s', '16 < age <= 60 s', 'Age > 60 s'])
        ax.set(title=column, ylabel='Error (m)'); ax.legend(fontsize=8); ax.grid(axis='y', alpha=.2)
    fig.savefig(target / 'metrics_by_outage.png', dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
    for kind in keys:
        h = horizons[(horizons.model_key == kind) & (horizons['mode'] == 'all')]
        axes[0].plot(h.horizon_s, h.mean_error_m, label=MODEL_NAMES[kind], color=COLORS[kind])
        values = np.sort(windows[windows.model == MODEL_NAMES[kind]].ADE_m)
        axes[1].plot(values, np.arange(1, len(values)+1) / len(values), label=MODEL_NAMES[kind], color=COLORS[kind])
    axes[0].set(xlabel='Prediction horizon (s)', ylabel='Mean position error (m)')
    axes[1].set(xlabel='Per-window ADE (m)', ylabel='Empirical cumulative probability')
    for ax in axes: ax.legend(fontsize=8); ax.grid(alpha=.2)
    fig.savefig(target / 'horizon_and_cdf.png', dpi=180); plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), layout='constrained')
    anchors = np.load(data / 'DR_start.npy', mmap_mode='r'); biases = np.load(data / 'bias.npy', mmap_mode='r')
    examples = []
    for ax, mode, mask in zip(axes.ravel(), MODES[1:], mode_masks(ages)[1:]):
        if not mask.any(): ax.set_visible(False); continue
        i = int(np.flatnonzero(mask)[0]); row = int(rows[i]); origin = anchors[row] + biases[row]
        p = np.vstack([np.zeros(2), truth[i]-origin]); ax.plot(p[:,0], p[:,1], 'k-', lw=2, label='Truth')
        for kind in keys:
            p = predictions[kind][i]-origin
            ax.plot(p[:,0], p[:,1], color=COLORS[kind], label=MODEL_NAMES[kind], lw=1.5)
            ax.scatter(*p[-1], marker='x', color=COLORS[kind], s=35)
        ax.scatter(0, 0, marker='*', color='black', s=55)
        ax.set(title=f'{mode} | row {row} | age {ages[i]:.1f} s', xlabel='East relative to true anchor (m)', ylabel='North relative to true anchor (m)')
        ax.set_aspect('equal', adjustable='datalim'); ax.grid(alpha=.2); ax.legend(fontsize=8)
        examples.append(dict(mode=mode, row=row, outage_age_s=float(ages[i])))
    fig.suptitle('Same fixed examples | true anchor used for display only')
    fig.savefig(target / 'trajectory_comparison.png', dpi=180); plt.close(fig)
    atomic_json(target / 'trajectory_examples.json', examples)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
    for kind in keys:
        history = pd.read_csv(out / kind / 'training_history.csv')
        for ax, metric in zip(axes, ['ADE_m', 'FDE_m']):
            ax.plot(history.epoch, history['val_'+metric], color=COLORS[kind], label=MODEL_NAMES[kind]+' val')
            ax.plot(history.epoch, history['train_'+metric], '--', alpha=.5, color=COLORS[kind], label=MODEL_NAMES[kind]+' train')
            ax.set(xlabel='Epoch', ylabel=metric); ax.grid(alpha=.2); ax.legend(fontsize=7)
    fig.savefig(target / 'training_comparison.png', dpi=180); plt.close(fig)
    search_path = out / 'pso_search/search_history.csv'
    if search_path.exists():
        search = pd.read_csv(search_path)
        fig, ax = plt.subplots(figsize=(8,4), layout='constrained')
        x = np.arange(1,len(search)+1)
        ax.plot(x, search.fitness, 'o-', label='Candidate best fitness')
        ax.plot(x, search.fitness.cummin(), '--', label='Swarm best')
        ax.set(xlabel='Candidate evaluation', ylabel='Validation ADE + 0.5 FDE (m)'); ax.legend(); ax.grid(alpha=.2)
        fig.savefig(target / 'pso_search.png', dpi=180); plt.close(fig)
