"""Small PSO hyperparameter search; final CPI models train exactly one epoch."""
import json
import shutil
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from phase1_model import Phase1CPI, Phase1Dataset, losses
from compare_phase1_lstm_cpi import digest, masks, MODES

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results/phase1_pso_cpi_1ep_seed42'
REFERENCE = HERE / 'results/phase1_lstm_vs_cpi_5ep_seed42'
DEFAULT = dict(lr=.001, current=.1, bias=.2, physics=.02)
NAMES = ['lr', 'current', 'bias', 'physics']
# Search log10(lr) and positive auxiliary-loss weights; no architecture changes.
LOW = np.array([np.log10(.0003), .05, .1, .005])
HIGH = np.array([np.log10(.003), .2, .3, .05])


def decode(x):
    return dict(lr=float(10 ** x[0]), current=float(x[1]), bias=float(x[2]), physics=float(x[3]))


def epoch(model, loader, scales, params, optimizer=None):
    model.train(optimizer is not None)
    total = np.zeros(3); n = 0
    with torch.set_grad_enabled(optimizer is not None):
        for batch in loader:
            pred = model(batch)
            terms = losses(pred, batch, scales)
            objective = terms[1] + params['current'] * terms[2] + params['bias'] * terms[3] + params['physics'] * terms[4]
            if optimizer is not None:
                optimizer.zero_grad(); objective.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.)
                optimizer.step()
            d = (pred['position'].detach() - batch['position']).norm(dim=-1)
            size = len(d); n += size
            total += [objective.detach().item() * size, d.mean().item() * size, d[:, -1].mean().item() * size]
    assert np.isfinite(total).all()
    return dict(loss=total[0] / n, ADE_m=total[1] / n, FDE_m=total[2] / n)


def loader(dataset, shuffle):
    return DataLoader(dataset, batch_size=128, shuffle=shuffle, num_workers=0,
                      generator=torch.Generator().manual_seed(42))


def fit(train, val, scales, params):
    torch.manual_seed(42)
    model = Phase1CPI(scales['current_std'], scales['bias_std'])
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
    tr = epoch(model, loader(train, True), scales, params, optimizer)
    va = epoch(model, loader(val, False), scales, params)
    return model, tr, va


def main():
    OUT.mkdir(parents=True, exist_ok=False); (OUT / 'provenance').mkdir()
    for filename in ['smoke_pso_phase1.py', 'phase1_model.py', 'compare_phase1_lstm_cpi.py']:
        shutil.copy2(HERE / filename, OUT / 'provenance' / filename)
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    start = time.monotonic()
    with np.load(REFERENCE / 'scalers.npz') as z: scales = {k: torch.as_tensor(z[k]) for k in z.files}
    shutil.copy2(REFERENCE / 'scalers.npz', OUT / 'scalers.npz')
    with np.load(HERE / 'data/splits.npz') as z: ids = {k: z[k] for k in z.files}
    rng = np.random.default_rng(42)
    subset = {k: np.sort(rng.choice(ids[k], size=n, replace=False)) for k, n in [('train', 8192), ('val', 2048)]}
    np.savez(OUT / 'search_indices.npz', **subset)
    small = {k: Phase1Dataset(HERE / 'data', v, scales) for k, v in subset.items()}
    full = {k: Phase1Dataset(HERE / 'data', v, scales) for k, v in ids.items()}
    config = dict(seed=42, population=4, iterations=2, search_epochs_per_candidate=1,
                  search_train_windows=8192, search_val_windows=2048,
                  final_epochs=1, final_train_windows=len(ids['train']), final_val_windows=len(ids['val']),
                  search_bounds={k: [float(10**LOW[i]) if i == 0 else float(LOW[i]), float(10**HIGH[i]) if i == 0 else float(HIGH[i])] for i, k in enumerate(NAMES)},
                  fitness='validation ADE + 0.5 FDE, in meters', inertia=.5, cognitive=1.4, social=1.4,
                  architecture='unchanged Phase1CPI; hyperparameter PSO, not neural weight optimization',
                  test_policy='frozen offline test, no test-based selection or adaptation',
                  normalization='reuse prior train-only fitted scalers', prior_test_exposure=True,
                  limitation='Tiny search and one-epoch fitness favor fast initial learning; no convergence claim.')
    (OUT / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    positions = rng.uniform(LOW, HIGH, size=(4, 4))
    positions[0] = [np.log10(DEFAULT['lr']), DEFAULT['current'], DEFAULT['bias'], DEFAULT['physics']]
    velocities = np.zeros_like(positions)
    personal = positions.copy(); scores = np.full(4, np.inf)
    global_score = np.inf; global_position = None; records = []
    for iteration in range(1, 3):
        for particle in range(4):
            params = decode(positions[particle])
            _, tr, va = fit(small['train'], small['val'], scales, params)
            score = va['ADE_m'] + .5 * va['FDE_m']
            records.append(dict(iteration=iteration, particle=particle, fitness=score, **params,
                                **{'train_' + k: v for k, v in tr.items()}, **{'val_' + k: v for k, v in va.items()}))
            if score < scores[particle]:
                scores[particle] = score; personal[particle] = positions[particle].copy()
            if score < global_score:
                global_score = score; global_position = positions[particle].copy()
            pd.DataFrame(records).to_csv(OUT / 'pso_search_history.csv', index=False)
            print(f'PSO iteration={iteration}/2 particle={particle+1}/4 fitness={score:.3f} val_ADE={va["ADE_m"]:.3f} params={params} elapsed={time.monotonic()-start:.0f}s', flush=True)
        if iteration < 2:
            r1, r2 = rng.random(positions.shape), rng.random(positions.shape)
            velocities = .5 * velocities + 1.4 * r1 * (personal - positions) + 1.4 * r2 * (global_position - positions)
            positions = np.clip(positions + velocities, LOW, HIGH)
    chosen = decode(global_position)
    (OUT / 'best_hparams.json').write_text(json.dumps(dict(params=chosen, search_validation_fitness=global_score), indent=2), encoding='utf-8')
    final = [('CPI-LSTM (1 ep)', DEFAULT), ('PSO-CPI-LSTM (1 ep)', chosen)]
    policy = {}; trained = []
    # Finish both one-epoch full-data fits before touching the test samples.
    for i, (name, params) in enumerate(final):
        model, tr, va = fit(full['train'], full['val'], scales, params)
        folder = OUT / ('cpi_1ep' if i == 0 else 'pso_cpi_1ep'); folder.mkdir()
        torch.save(dict(state_dict=model.state_dict(), params=params, epoch=1, scalers={k: v.numpy() for k, v in scales.items()}, config=config), folder / 'final.pt')
        shutil.copy2(folder / 'final.pt', folder / 'frozen.pt')
        policy[name] = dict(params=params, checkpoint=str(folder / 'frozen.pt'), sha256=digest(folder / 'frozen.pt'))
        trained.append(dict(model=name, epoch=1, **{'train_' + k: v for k, v in tr.items()}, **{'val_' + k: v for k, v in va.items()}))
        print(f'FINAL {name}: train={tr} val={va} elapsed={time.monotonic()-start:.0f}s', flush=True)
    pd.DataFrame(trained).to_csv(OUT / 'training_history.csv', index=False)
    (OUT / 'frozen_test_policy.json').write_text(json.dumps(policy, indent=2), encoding='utf-8')
    predictions = {}; checks = {}
    with torch.inference_mode():
        for name, info in policy.items():
            path = Path(info['checkpoint'])
            model = Phase1CPI(scales['current_std'], scales['bias_std'])
            model.load_state_dict(torch.load(path, weights_only=False, map_location='cpu')['state_dict'])
            model.eval().requires_grad_(False)
            before = {k: v.clone() for k, v in model.state_dict().items()}
            test = loader(full['test'], False)
            batch = next(iter(test)); pred = model(batch)
            altered = {k: v.clone() for k, v in batch.items()}
            for k in ['position', 'current', 'bias', 'velocity']: altered[k].fill_(99999.)
            poison = model(altered)
            for k in pred: torch.testing.assert_close(poison[k], pred[k], rtol=0, atol=0)
            altered['u'] *= -1; altered['plan'] *= -1
            other = model(altered)
            for k in ['current', 'bias']: torch.testing.assert_close(other[k], pred[k], rtol=0, atol=0)
            predictions[name] = np.concatenate([model(b)['position'].numpy() for b in test])
            for k, v in model.state_dict().items(): torch.testing.assert_close(v, before[k], rtol=0, atol=0)
            assert digest(path) == info['sha256']
            checks[name] = dict(label_poison_invariance=True, state_heads_plan_independent=True, weights_unchanged=True, frozen_hash_unchanged=True)
    with np.load(REFERENCE / 'test_predictions.npz') as z:
        np.testing.assert_array_equal(z['row'], ids['test'])
        truth = z['truth']; ages = z['age']
        predictions['CPI-LSTM (5 ep, best 4)'] = z['cpi_lstm']
        predictions['LSTM (5 ep, best 3)'] = z['lstm']
    np.savez_compressed(OUT / 'test_predictions.npz', row=ids['test'], truth=truth, age=ages,
                        cpi_1ep=predictions[final[0][0]], pso_cpi_1ep=predictions[final[1][0]])
    rows = []; per_window = []
    for name, p in predictions.items():
        assert p.shape == truth.shape == (26000, 80, 2) and np.isfinite(p).all()
        d = np.linalg.norm(p.astype(float) - truth.astype(float), axis=-1)
        per_window.append(pd.DataFrame(dict(model=name, row=ids['test'], ADE_m=d.mean(1), FDE_m=d[:, -1])))
        for mode, mask in zip(MODES, masks(ages)):
            rows.append(dict(model=name, mode=mode, windows=int(mask.sum()), ADE_m=float(d[mask].mean()),
                             FDE_m=float(d[mask, -1].mean()), position_RMSE_m=float(np.sqrt(np.mean(d[mask]**2))),
                             P90_window_ADE_m=float(np.percentile(d[mask].mean(1), 90))))
    metrics = pd.DataFrame(rows); metrics.to_csv(OUT / 'test_metrics.csv', index=False)
    pd.concat(per_window).to_csv(OUT / 'per_window_metrics.csv', index=False)
    plot_results(predictions, truth, ages, ids['test'], metrics, records)
    verify = dict(passed=True, final_epochs_completed=1, candidates_trained=8, search_epochs_each=1,
                  checks=checks, test_windows=26000, test_used_by_pso=False,
                  prior_five_epoch_predictions_reused=True, elapsed_s=time.monotonic()-start)
    (OUT / 'verification.json').write_text(json.dumps(verify, indent=2), encoding='utf-8')
    report = ['PSO-CPI-LSTM 第一阶段一轮试跑', '',
              'PSO 搜索学习率与三项辅助损失权重；保持原 Phase1CPI 结构和物理积分，不引入平滑项。',
              '4 粒子 × 2 次迭代；每候选在固定 8192 个训练窗口上训练 1 轮，以固定 2048 个验证窗口的 ADE+0.5FDE 选择参数。',
              '选定参数后重新初始化，在完整 91000 个训练窗口上训练 1 轮。另从头训练默认 CPI-LSTM 1 轮。',
              '测试前冻结两者；测试集 26000 个窗口未用于 PSO 和参数选择。旧的五轮结果仅作参考。',
              '注意：最终模型仅训练一轮，但搜索额外训练了八个候选，PSO 总计算量更大。',
              '这不是新生成轨迹测试，沿用已有测试集；属于离线候选控制计划预测。',
              '小规模搜索和一轮适应度偏向初期学习快的参数，不能代表多轮收敛性能。', '',
              '选定参数：' + json.dumps(chosen, ensure_ascii=False), '',
              metrics[metrics['mode']=='all'].to_string(index=False)]
    (OUT / 'report_zh.txt').write_text('\n'.join(report), encoding='utf-8')
    print(metrics[metrics['mode']=='all'].to_string(index=False), flush=True)
    print('VERIFIED '+str(OUT), flush=True)


def plot_results(predictions, truth, ages, rows, metrics, search):
    colors = ['#6b7280', '#b91c1c', '#007c91', '#d97706']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout='constrained')
    overall = metrics[metrics['mode']=='all']
    labels = ['CPI\n1 epoch', 'PSO-CPI\n1 epoch', 'CPI\n5 ep / best 4', 'LSTM\n5 ep / best 3']
    for ax, key in zip(axes, ['ADE_m', 'FDE_m', 'position_RMSE_m']):
        bars = ax.bar(labels, overall[key], color=colors)
        ax.bar_label(bars, fmt='%.2f', padding=3); ax.set(title=key, ylabel='Error (m)')
        ax.set_ylim(0, overall[key].max()*1.2); ax.grid(axis='y', alpha=.2)
    fig.suptitle('Frozen offline test | same 26,000 windows | five-epoch models are references')
    fig.savefig(OUT / 'metrics_comparison.png', dpi=180); plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), layout='constrained')
    examples = json.loads((REFERENCE / 'trajectory_examples.json').read_text())
    anchors = np.load(HERE / 'data/DR_start.npy', mmap_mode='r')
    biases = np.load(HERE / 'data/bias.npy', mmap_mode='r')
    for ax, example in zip(axes.ravel(), examples):
        row = example['row']; i = int(np.flatnonzero(rows == row)[0]); start = anchors[row]+biases[row]
        p = np.vstack([np.zeros(2), truth[i]-start]); ax.plot(p[:,0], p[:,1], 'k-', lw=2, label='Truth')
        for name, color in zip(list(predictions)[:3], colors[:3]):
            p = predictions[name][i]-start
            ax.plot(p[:,0], p[:,1], color=color, label=name, lw=1.5)
            ax.scatter(*p[-1], marker='x', color=color, s=35)
        ax.scatter(0, 0, marker='*', color='black', s=60)
        ax.set(title=f"row {row} | outage age {ages[i]:.1f} s", xlabel='East relative to true anchor (m)', ylabel='North relative to true anchor (m)')
        ax.set_aspect('equal', adjustable='datalim'); ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle('Frozen models | same four fixed examples | true anchor used for display only')
    fig.savefig(OUT / 'trajectory_comparison.png', dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), layout='constrained')
    for ax, key in zip(axes, ['ADE_m', 'FDE_m']):
        for j, (name, color) in enumerate(zip(list(predictions)[:2], colors[:2])):
            part = metrics[(metrics.model == name) & (metrics['mode'] != 'all')]
            bars = ax.bar(np.arange(4)+(j-.5)*.36, part[key], width=.36, color=color, label=name)
            ax.bar_label(bars, fmt='%.1f', fontsize=8, padding=3)
        ax.set_xticks(range(4), ['Normal', '0-16 s', '16-60 s', '>60 s'])
        ax.set(title=key, ylabel='Error (m)', xlabel='Outage age'); ax.legend(); ax.grid(axis='y', alpha=.2)
    fig.savefig(OUT / 'metrics_by_outage.png', dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
    for name, color in zip(list(predictions)[:3], colors[:3]):
        d = np.linalg.norm(predictions[name].astype(float)-truth.astype(float), axis=-1)
        axes[0].plot(np.arange(1,81)*.2, d.mean(0), color=color, label=name)
    axes[0].set(xlabel='Prediction horizon (s)', ylabel='Mean position error (m)'); axes[0].legend(fontsize=8); axes[0].grid(alpha=.2)
    s = pd.DataFrame(search)
    axes[1].plot(np.arange(1,len(s)+1), s.fitness, 'o-', label='Candidate fitness')
    axes[1].plot(np.arange(1,len(s)+1), s.fitness.cummin(), '--', label='Best so far')
    axes[1].set(xlabel='Candidate evaluation', ylabel='Search validation ADE + 0.5 FDE (m)'); axes[1].legend(); axes[1].grid(alpha=.2)
    fig.savefig(OUT / 'horizon_and_search.png', dpi=180); plt.close(fig)


if __name__ == '__main__':
    main()
