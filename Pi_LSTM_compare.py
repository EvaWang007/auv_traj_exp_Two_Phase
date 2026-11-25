"""Frozen paired comparison on the full held-out set or saved ten-task examples."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from formal_phase1 import HERE, atomic_json, hash_file, make_loader, make_model, move_batch, read_checkpoint
from formal_phase1_evaluate import mode_masks, MODES, state_hash
from Pi_LSTM_model import PiLSTM
from Pi_LSTM_train import prepare


def main(args):
    config = json.loads((args.pi_run/'run.json').read_text(encoding='utf-8'))
    if config['smoke'] and not args.allow_smoke:
        raise ValueError('Smoke checkpoints cannot be used for formal comparison. --allow-smoke is for pipeline verification only.')
    selection = json.loads((args.pi_run/'selected_checkpoint.json').read_text(encoding='utf-8'))
    if selection['epochs_completed'] != config['config']['epochs']:
        raise ValueError('Pi_LSTM training is incomplete.')
    ids, scales, checked = prepare(args.data, args.reference_run, config['config'], config['smoke'])
    if checked['signature'] != config['signature'] or selection['signature'] != config['signature']:
        raise ValueError('Training provenance differs. Do not compare changed code/config/data.')
    if hash_file(args.pi_run/'scalers.npz') != selection['scalers_sha256']:
        raise ValueError('Pi scalers changed.')
    cpi_selection = json.loads((args.reference_run/'selected_checkpoints.json').read_text(encoding='utf-8'))
    cpi_info = cpi_selection['selected']['cpi_lstm']
    paths = {'Pi_LSTM':args.pi_run/selection['checkpoint'], 'CPI-LSTM':args.reference_run/cpi_info['checkpoint']}
    hashes = {'Pi_LSTM':selection['sha256'], 'CPI-LSTM':cpi_info['sha256']}
    device = torch.device(args.device); torch.set_num_threads(4)
    models = {'Pi_LSTM':PiLSTM(scales), 'CPI-LSTM':make_model('cpi_lstm',scales)}
    states = {}
    for name,m in models.items():
        assert hash_file(paths[name]) == hashes[name]
        m.load_state_dict(read_checkpoint(paths[name])['state_dict']);m.to(device).eval().requires_grad_(False)
        states[name] = state_hash(m)
    args.output.mkdir(parents=True,exist_ok=False)
    if args.examples:
        with np.load(args.examples) as z:
            arrays = {k:z[k].copy() for k in ['history','plan','DR_start','water_state','last_fix_bias','last_fix_current','outage_age']}
            truth=z['truth'].copy();anchors=z['true_anchor'].copy()
        batch={k:torch.from_numpy(v) for k,v in arrays.items()}
        batch['x']=(batch['history']-scales['history_mean'])/scales['history_std']
        batch['u']=(batch['plan']-scales['plan_mean'])/scales['plan_std']
        batches=[batch]; rows=np.arange(len(truth)); ages=arrays['outage_age']; tasks=rows
        examples_meta=json.loads((args.examples.parent/'verification.json').read_text(encoding='utf-8'))
        assert examples_meta['scalers_sha256']==selection['scalers_sha256']
        dataset='saved_synthetic_examples'
    else:
        loader_config=dict(config['config'],num_workers=0)
        batches=make_loader(args.data,ids['test'],scales,loader_config,device=device)
        rows=ids['test'];truth=np.load(args.data/'position.npy',mmap_mode='r')[rows]
        anchors=np.load(args.data/'DR_start.npy',mmap_mode='r')[rows]+np.load(args.data/'bias.npy',mmap_mode='r')[rows]
        ages=np.load(args.data/'outage_age.npy',mmap_mode='r')[rows]
        tasks=pd.read_csv(args.data/'windows.csv').iloc[rows].trajectory.to_numpy()
        dataset='smoke_test_subset' if config['smoke'] else 'full_original_held_out_test'
    predictions={k:[] for k in models}; seen=[]
    with torch.inference_mode():
        for index,b in enumerate(batches):
            b=move_batch(b,device)
            if 'row' in b: seen.extend(b['row'].cpu().tolist())
            for name,m in models.items():
                pred=m(b)['position']
                if index==0:
                    poison={k:v.clone() for k,v in b.items()}
                    for k in ['position','current','velocity','bias']:
                        poison[k]=torch.full((len(pred),2) if k=='bias' else (len(pred),80,2),99999.,device=device)
                    torch.testing.assert_close(pred,m(poison)['position'],rtol=1e-6,atol=1e-6)
                predictions[name].append(pred.cpu().numpy())
    if not args.examples:np.testing.assert_array_equal(seen,rows)
    records=[]; windows=[]
    for name,m in models.items():
        assert state_hash(m)==states[name] and hash_file(paths[name])==hashes[name]
        p=np.concatenate(predictions[name]);predictions[name]=p
        assert p.shape==truth.shape and np.isfinite(p).all()
        d=np.linalg.norm(p.astype(float)-truth,axis=-1)
        windows.append(pd.DataFrame(dict(model=name,row=rows,trajectory=tasks,outage_age_s=ages,ADE_m=d.mean(1),FDE_m=d[:,-1])))
        for mode,mask in zip(MODES,mode_masks(ages)):
            if mask.any():records.append(dict(model=name,mode=mode,windows=int(mask.sum()),ADE_m=d[mask].mean(),FDE_m=d[mask,-1].mean(),position_RMSE_m=np.sqrt((d[mask]**2).mean())))
        np.savez_compressed(args.output/(name+'_predictions.npz'),row=rows,position=p)
    metrics=pd.DataFrame(records);metrics.to_csv(args.output/'metrics.csv',index=False)
    pd.concat(windows,ignore_index=True).to_csv(args.output/'per_window_metrics.csv',index=False)
    colors={'Pi_LSTM':'#7c3aed','CPI-LSTM':'#007c91'}
    overall=metrics[metrics['mode']=='all']
    fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    for ax,key in zip(axes,['ADE_m','FDE_m','position_RMSE_m']):
        bars=ax.bar(overall.model,overall[key],color=[colors[k] for k in overall.model]);ax.bar_label(bars,fmt='%.2f',padding=3)
        ax.set(title=key,ylabel='Error (m)',ylim=(0,overall[key].max()*1.2));ax.grid(axis='y',alpha=.2)
    fig.savefig(args.output/'metrics_comparison.png',dpi=180);plt.close(fig)
    # Select first occurrence from ten distinct tasks before looking at errors.
    examples=pd.Series(tasks).drop_duplicates().index.to_numpy()[:10]
    for j,i in enumerate(examples):
        fig,ax=plt.subplots(figsize=(7,6),layout='constrained')
        xy=np.vstack([np.zeros(2),truth[i]-anchors[i]])
        ax.plot(*xy.T,'k-',lw=2,label='Truth');ax.scatter(0,0,marker='*',color='black',label='True anchor (display only)')
        for name,p in predictions.items():
            xy=p[i]-anchors[i];ax.plot(*xy.T,color=colors[name],label=name)
        prefix='SMOKE PIPELINE ONLY | ' if config['smoke'] else ''
        ax.set(title=f'{prefix}Task {tasks[i]} | outage age {ages[i]:.1f} s',xlabel='East (m)',ylabel='North (m)')
        ax.set_aspect('equal',adjustable='datalim');ax.legend();ax.grid(alpha=.2)
        fig.savefig(args.output/f'trajectory_{j+1:02d}.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for name,folder in [('Pi_LSTM',args.pi_run),('CPI-LSTM',args.reference_run/'cpi_lstm')]:
        h=pd.read_csv(folder/'training_history.csv')
        for ax,key in zip(axes,['ADE_m','FDE_m','position_RMSE_m']):
            ax.plot(h.epoch,h['val_'+key],label=name,color=colors[name]);ax.set(xlabel='Epoch',ylabel=key);ax.grid(alpha=.2);ax.legend()
    fig.savefig(args.output/'validation_curves.png',dpi=180);plt.close(fig)
    atomic_json(args.output/'verification.json',dict(passed=True,dataset=dataset,smoke_only=config['smoke'],windows=len(rows),
                checkpoint_hashes=hashes,weights_unchanged=True,normalization_refit=False,test_used_for_selection=False,
                label_poison_tolerance=dict(rtol=1e-6,atol=1e-6),same_inputs=True,
                example_rows=rows[examples].tolist(),examples_selected_by_error=False))
    print(overall.to_string(index=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=HERE/'data')
    parser.add_argument('--reference-run',type=Path,default=HERE/'results/formal_phase1_30ep_seed42')
    parser.add_argument('--pi-run',type=Path,default=HERE/'results/Pi_LSTM_30ep_seed42')
    parser.add_argument('--output',type=Path,default=HERE/'results/Pi_LSTM_vs_CPI_test')
    parser.add_argument('--examples',type=Path,help='Optional saved predictions_and_inputs.npz; evaluates examples instead of full test set.')
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--allow-smoke',action='store_true',help='Pipeline validation only; never report its metrics as a trained-model comparison.')
    main(parser.parse_args())
