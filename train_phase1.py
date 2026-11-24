"""Phase-one offline pretraining with availability masks and precommitted controls."""
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
from torch.utils.data import DataLoader
from phase1_model import Phase1CPI,Phase1Dataset,lag_velocity,losses

HERE=Path(__file__).resolve().parent
NAMES=['total','position','current','bias','physics']


def moments(array,indices):
    count=0; total=np.zeros(array.shape[-1],np.float64); square=total.copy()
    for k in range(0,len(indices),512):
        value=np.asarray(array[indices[k:k+512]],np.float64).reshape(-1,array.shape[-1])
        total+=value.sum(0); square+=(value*value).sum(0); count+=len(value)
    mean=total/count
    return mean.astype(np.float32),(np.sqrt(np.maximum(square/count-mean**2,0))+1e-6).astype(np.float32)


def fit_scalers(data,indices):
    result={}
    for name in ['history','plan','current','bias','position']:
        array=np.load(data/(name+'.npy'),mmap_mode='r')
        mean,std=moments(array,indices)
        result[name+'_mean']=mean; result[name+'_std']=std
    result['position_scale']=np.asarray(result['position_std'].mean(),np.float32)
    result['current_scale']=np.asarray(result['current_std'].mean(),np.float32)
    return result


def epoch(model,loader,scalers,opt=None):
    model.train(opt is not None); totals=np.zeros(5); n=0; sq=0.; distance=0.; final=0.
    with torch.set_grad_enabled(opt is not None):
        for batch in loader:
            pred=model(batch); loss=losses(pred,batch,scalers)
            if opt is not None:
                opt.zero_grad(); loss[0].backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),2.); opt.step()
            size=len(batch['x']); n+=size; totals+=loss.detach().numpy()*size
            d=torch.linalg.vector_norm(pred['position']-batch['position'],dim=-1)
            distance+=float(d.sum()); final+=float(d[:,-1].sum())
            sq+=float(torch.sum((pred['current']-batch['current'])**2))
    assert np.isfinite(totals).all()
    return dict(zip(NAMES,totals/n)),dict(ADE=distance/(n*80),FDE=final/n,current_RMSE=np.sqrt(sq/(n*80*2)))


@torch.no_grad()
def evaluate(model,loader,scalers,out):
    model.eval(); storage={k:[] for k in ['position','current','bias','truth','true_current','true_bias','row','DR_start','age','baseline_mean','baseline_persistence','mean_start','persistence_start','true_start']}
    for batch in loader:
        pred=model(batch)
        water=lag_velocity(batch['water_state'],batch['plan'],.2)
        mean=scalers['train_mean_current']
        for key,value in [('position',pred['position']),('current',pred['current']),('bias',pred['bias']),
            ('truth',batch['position']),('true_current',batch['current']),('true_bias',batch['bias']),
            ('row',batch['row']),('DR_start',batch['DR_start']),('age',batch['outage_age']),
            ('true_start',batch['DR_start']+batch['bias'])]: storage[key].append(value.numpy())
        mean_start=batch['DR_start']+batch['last_fix_bias']+batch['outage_age'][:,None]*mean
        persist_start=batch['DR_start']+batch['last_fix_bias']+batch['outage_age'][:,None]*batch['last_fix_current']
        storage['mean_start'].append(mean_start.numpy()); storage['persistence_start'].append(persist_start.numpy())
        storage['baseline_mean'].append((mean_start[:,None,:]+torch.cumsum(.2*(water+mean),dim=1)).numpy())
        storage['baseline_persistence'].append((persist_start[:,None,:]+torch.cumsum(.2*(water+batch['last_fix_current'][:,None,:]),dim=1)).numpy())
    arrays={k:np.concatenate(v) for k,v in storage.items()}
    np.savez_compressed(out/'test_predictions.npz',**arrays)
    rows=[]; modes=np.select([arrays['age']==0,arrays['age']<=16,arrays['age']<=60],['normal','outage_0_16s','outage_16_60s'],default='outage_over_60s')
    for name,key in [('CPI-LSTM','position'),('Train-mean current + lag','baseline_mean'),('Last-fix current + lag','baseline_persistence')]:
        dist=np.linalg.norm(arrays[key].astype(float)-arrays['truth'].astype(float),axis=-1)
        anchor=arrays['DR_start']+arrays['bias'] if key=='position' else arrays['mean_start' if key=='baseline_mean' else 'persistence_start']
        delta=(arrays[key].astype(float)-anchor.astype(float)[:,None,:])-(arrays['truth'].astype(float)-arrays['true_start'].astype(float)[:,None,:])
        displacement=np.linalg.norm(delta,axis=-1)
        for mode in ['all','normal','outage_0_16s','outage_16_60s','outage_over_60s']:
            selected=np.ones(len(modes),bool) if mode=='all' else modes==mode
            if not selected.any(): continue
            rows.append(dict(model=name,mode=mode,windows=int(selected.sum()),ADE_m=float(dist[selected].mean()),FDE_m=float(dist[selected,-1].mean()),
                displacement_ADE_m=float(displacement[selected].mean()),displacement_FDE_m=float(displacement[selected,-1].mean())))
    metrics=pd.DataFrame(rows); metrics.to_csv(out/'test_metrics.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for name,key in [('CPI-LSTM','position'),('Train mean + lag','baseline_mean'),('Last-fix current + lag','baseline_persistence')]:
        d=np.linalg.norm(arrays[key].astype(float)-arrays['truth'].astype(float),axis=-1)
        for ax,mask in [(axes[0],np.ones(len(modes),bool)),(axes[1],modes=='outage_over_60s')]:
            ax.plot(np.arange(1,81)*.2,d[mask].mean(0),label=name)
    for ax,title in zip(axes,['All availability modes','Outage longer than 60 seconds']):
        ax.set(title=title,xlabel='Prediction horizon (s)',ylabel='Mean position error (m)'); ax.legend(fontsize=7); ax.grid(alpha=.2)
    fig.savefig(out/'test_comparison.png',dpi=170); plt.close(fig)
    return metrics,arrays


def verify_information_boundaries(model,dataset):
    batch=next(iter(DataLoader(dataset,batch_size=32,shuffle=False)))
    model.eval()
    with torch.no_grad():
        original=model(batch)
        altered={k:v.clone() for k,v in batch.items()}
        for k in ['position','velocity','current','bias']: altered[k].fill_(99999)
        rebuilt=model(altered)
        for key in original: torch.testing.assert_close(original[key],rebuilt[key],rtol=0,atol=0)
        altered['plan']=-altered['plan']; altered['u']=-altered['u']
        candidate=model(altered)
        torch.testing.assert_close(original['bias'],candidate['bias'],rtol=0,atol=0)
        torch.testing.assert_close(original['current'],candidate['current'],rtol=0,atol=0)
    return dict(label_poison_invariance=True,state_current_independent_of_candidate_plan=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=HERE/'data')
    parser.add_argument('--output',type=Path,default=HERE/'results/phase1_seed42')
    parser.add_argument('--epochs',type=int,default=20)
    args=parser.parse_args(); out=args.output; out.mkdir(parents=True,exist_ok=False)
    (out/'provenance').mkdir()
    for name in ['train_phase1.py','phase1_model.py','prepare_phase1.py','routes.py']:
        shutil.copy2(HERE/name,out/'provenance'/name)
    torch.set_num_threads(4); torch.set_num_interop_threads(1); torch.manual_seed(42); np.random.seed(42)
    began=time.monotonic(); dataset_config=json.loads((args.data/'dataset_config.json').read_text())
    with np.load(args.data/'splits.npz') as z: ids={k:z[k] for k in z.files}
    scales=fit_scalers(args.data,ids['train']); scales['train_mean_current']=np.array(dataset_config['train_mean_current_mps'],np.float32)
    np.savez(out/'scalers.npz',**scales)
    tensors={k:torch.as_tensor(v,dtype=torch.float32) for k,v in scales.items()}
    datasets={k:Phase1Dataset(args.data,v,tensors) for k,v in ids.items()}
    loaders={k:DataLoader(v,batch_size=128,shuffle=k=='train',num_workers=0) for k,v in datasets.items()}
    config=dict(stage=1,seed=42,epochs=args.epochs,hidden_dim=128,ctrl_hidden_dim=96,batch_size=128,lr=.001,
        history_len=120,pred_len=80,stride=10,dt=.2,device='cpu',loss_weights=dict(position=1.,current=.1,bias=.2,physics=.02,smooth=0.),
        data=str(args.data.resolve()),window_counts={k:len(v) for k,v in datasets.items()},
        checkpoint_selection='minimum validation total loss; no test-based selection',
        architecture='history-only current and bias heads; candidate-control encoder predicts bounded water-motion residual',
        motion_residual_bound_mps=.3,physics='lagged acceleration, recursive speed limits, integrated yaw and current',
        phase2_completed=False,dataset=dataset_config)
    (out/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    model=Phase1CPI(scales['current_std'],scales['bias_std']); opt=torch.optim.Adam(model.parameters(),lr=.001)
    records=[]; best=float('inf'); best_epoch=0
    vl,vm=epoch(model,loaders['val'],tensors)
    records.append(dict(epoch=0,**{'val_'+k:v for k,v in vl.items()},**{'val_'+k:v for k,v in vm.items()}))
    print(f'epoch=0 val_total={vl["total"]:.6f} val_ADE={vm["ADE"]:.3f}',flush=True)
    for ep in range(1,args.epochs+1):
        tl,tm=epoch(model,loaders['train'],tensors,opt); vl,vm=epoch(model,loaders['val'],tensors)
        records.append(dict(epoch=ep,**{'train_'+k:v for k,v in tl.items()},**{'val_'+k:v for k,v in vl.items()},
            **{'train_'+k:v for k,v in tm.items()},**{'val_'+k:v for k,v in vm.items()}))
        if vl['total']<best:
            best=vl['total']; best_epoch=ep
            torch.save(dict(state_dict=model.state_dict(),config=config,scalers=scales,best_epoch=ep),out/'cpi_lstm_phase1_best.pt')
        pd.DataFrame(records).to_csv(out/'training_history.csv',index=False)
        print(f'epoch={ep} train_total={tl["total"]:.6f} val_total={vl["total"]:.6f} val_ADE={vm["ADE"]:.3f} val_FDE={vm["FDE"]:.3f} elapsed={time.monotonic()-began:.1f}s',flush=True)
    torch.save(dict(state_dict=model.state_dict(),config=config,scalers=scales,epoch=args.epochs),out/'cpi_lstm_phase1_final.pt')
    saved=torch.load(out/'cpi_lstm_phase1_best.pt',map_location='cpu',weights_only=False); model.load_state_dict(saved['state_dict'])
    boundaries=verify_information_boundaries(model,datasets['test'])
    metrics,arrays=evaluate(model,loaders['test'],tensors,out)
    frame=pd.DataFrame(records); fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    axes[0].plot(frame.epoch,frame.train_total,label='Train'); axes[0].plot(frame.epoch,frame.val_total,label='Validation')
    axes[1].plot(frame.epoch,frame.val_ADE,label='Validation ADE'); axes[1].plot(frame.epoch,frame.val_FDE,label='Validation FDE')
    for ax in axes: ax.set_xlabel('Epoch'); ax.legend(); ax.grid(alpha=.2)
    fig.savefig(out/'training_curves.png',dpi=170); plt.close(fig)
    report=dict(passed=True,epochs_completed=args.epochs,best_epoch=best_epoch,elapsed_s=time.monotonic()-began,
        test_windows=len(arrays['truth']),current_RMSE_mps=float(np.sqrt(np.mean((arrays['current']-arrays['true_current'])**2))),
        bias_RMSE_m=float(np.sqrt(np.mean((arrays['bias']-arrays['true_bias'])**2))),
        normalization_fit_on_training_only=True,test_used_for_checkpoint_selection=False,
        no_actual_future_controls_as_inputs=True,phase2_completed=False,**boundaries,
        phase1_limitation=dataset_config['phase1_limitation'])
    (out/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(metrics.to_string(index=False),flush=True); print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__': main()
