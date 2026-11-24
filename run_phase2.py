"""Three rounds of observation-only rollout collection and supervised CPI fine-tuning."""
import argparse
from datetime import datetime,timezone
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
from torch.utils.data import DataLoader,ConcatDataset
from phase1_model import Phase1Dataset,losses
from phase2_control import load_model,rollout
from train_phase1 import evaluate,verify_information_boundaries

HERE=next(p for p in Path(__file__).resolve().parents if (p/'data/normal_records.npz').is_file() and (p/'phase1_model.py').is_file())
INITIAL=HERE/'results/phase1_seed42/cpi_lstm_phase1_best.pt'


def status(out,stage,**details):
    report=dict(stage=stage,updated_at_utc=datetime.now(timezone.utc).isoformat(),**details)
    (out/'status.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False),flush=True)


def run_epoch(model,loader,tensors,opt=None):
    model.train(opt is not None); totals=np.zeros(5); n=0; distance=0.; final=0.; sq=0.
    with torch.set_grad_enabled(opt is not None):
        for batch in loader:
            pred=model(batch); loss=losses(pred,batch,tensors)
            if opt is not None:
                opt.zero_grad(); loss[0].backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),2.); opt.step()
            size=len(batch['x']); n+=size; totals+=loss.detach().numpy()*size
            delta=pred['position'].detach()-batch['position']; d=torch.linalg.vector_norm(delta,dim=-1)
            distance+=float(d.sum()); final+=float(d[:,-1].sum())
            sq+=float(torch.sum((pred['current'].detach()-batch['current'])**2))
    assert n and np.isfinite(totals).all()
    return dict(zip(['total','position','current','bias','physics'],totals/n)),dict(ADE=distance/(n*80),FDE=final/n,current_RMSE=np.sqrt(sq/(n*80*2)))


def dataset_count(folder): return int(np.load(folder/'history.npy',mmap_mode='r').shape[0])


def sample_dataset(folder,limit,rng,tensors,indices=None):
    pool=np.arange(dataset_count(folder)) if indices is None else np.asarray(indices)
    chosen=rng.choice(pool,min(limit,len(pool)),replace=False)
    return Phase1Dataset(folder,chosen,tensors)


def fine_tune(out,round_id,parent,new_data,old_data,validation,scales,epochs):
    folder=out/f'round_{round_id:02d}'/'training'; folder.mkdir()
    tensors={k:torch.as_tensor(v,dtype=torch.float32) for k,v in scales.items()}
    rng=np.random.default_rng(42+round_id)
    first_ids=np.arange(91000)
    pieces=[sample_dataset(HERE/'data',27000,rng,tensors,first_ids)]
    if not old_data:
        pieces.append(sample_dataset(new_data,63000,rng,tensors))
    else:
        pieces.append(sample_dataset(new_data,36000,rng,tensors))
        each=27000//len(old_data)
        pieces.extend(sample_dataset(path,each,rng,tensors) for path in old_data)
    training=ConcatDataset(pieces)
    torch.manual_seed(42+round_id)
    train_loader=DataLoader(training,batch_size=128,shuffle=True,num_workers=0)
    val_loader=DataLoader(validation,batch_size=128,shuffle=False,num_workers=0)
    model,_,parent_saved=load_model(parent)
    config=dict(stage=2,round=round_id,parent_checkpoint=str(parent),epochs=epochs,seed=42+round_id,lr=.0001,batch_size=128,
        training_windows=len(training),validation_windows=len(validation),scalers='frozen phase-one training-only scalers',
        checkpoint_selection='minimum fixed validation prediction loss within round; closed-loop validation selects between rounds',
        loss_weights=dict(position=1.,current=.1,bias=.2,physics=.02,smooth=0.),
        architecture=parent_saved['config'].get('architecture','history-only state/current plus candidate-conditioned motion'),
        phase2_completed=True)
    (folder/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    opt=torch.optim.Adam(model.parameters(),lr=.0001)
    vl,vm=run_epoch(model,val_loader,tensors)
    records=[dict(epoch=0,**{'val_'+k:v for k,v in vl.items()},**{'val_'+k:v for k,v in vm.items()})]
    best=vl['total']; best_epoch=0
    checkpoint=folder/'cpi_lstm_phase2_best.pt'
    torch.save(dict(state_dict=model.state_dict(),scalers=scales,config=config,best_epoch=0),checkpoint)
    for ep in range(1,epochs+1):
        status(out,'training',round=round_id,epoch=ep,epochs=epochs,training_windows=len(training))
        tl,tm=run_epoch(model,train_loader,tensors,opt); vl,vm=run_epoch(model,val_loader,tensors)
        records.append(dict(epoch=ep,**{'train_'+k:v for k,v in tl.items()},**{'val_'+k:v for k,v in vl.items()},
            **{'train_'+k:v for k,v in tm.items()},**{'val_'+k:v for k,v in vm.items()}))
        if vl['total']<best:
            best=vl['total']; best_epoch=ep
            torch.save(dict(state_dict=model.state_dict(),scalers=scales,config=config,best_epoch=ep),checkpoint)
        pd.DataFrame(records).to_csv(folder/'training_history.csv',index=False)
        print(f'ROUND {round_id} EPOCH {ep}: train={tl["total"]:.6f} val={vl["total"]:.6f} val_ADE={vm["ADE"]:.3f}',flush=True)
    torch.save(dict(state_dict=model.state_dict(),scalers=scales,config=config,epoch=epochs),folder/'cpi_lstm_phase2_final.pt')
    return checkpoint,best_epoch


def closed_score(summary):
    m=summary['metrics']
    return m['outage_cross_track_RMSE_m']+.25*m['final_reference_error_m']+.01*m['control_slew_RMS']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=HERE/'results/phase2')
    parser.add_argument('--rounds',type=int,default=3)
    parser.add_argument('--epochs-per-round',type=int,default=5)
    parser.add_argument('--initial-checkpoint',type=Path,default=INITIAL)
    args=parser.parse_args(); out=args.output
    initial=args.initial_checkpoint.resolve()
    if not initial.is_file(): raise FileNotFoundError(initial)
    # The gate may already be present; refuse to overwrite any actual phase-two run.
    out.mkdir(parents=True,exist_ok=True)
    if (out/'experiment_policy.json').exists(): raise FileExistsError(out/'experiment_policy.json')
    assert json.loads((out/'gate/verification.json').read_text())['passed']
    (out/'provenance').mkdir()
    for name in ['run_phase2.py','phase2_control.py','verify_phase2.py','phase1_model.py','train_phase1.py','prepare_phase1.py','routes.py']:
        shutil.copy2(HERE/name,out/'provenance'/name)
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    began=time.monotonic()
    policy=dict(rounds=args.rounds,epochs_per_round=args.epochs_per_round,initial_checkpoint=str(initial),
        training_trajectories=700,validation_trajectories=100,test_trajectories=200,
        control_interval_s=1.,horizon_s=16.,history_s=24.,candidate_plans=32,full_model_refinement=4,
        train_sample_limit_per_round=90000,replay_first_round=[.3,.7],replay_later=[.3,.3,.4],
        training_window_stride_s=2.,outage_start_range_s=[30,120],outage_durations_s=[30,60,120,180],
        closed_loop_selection_score='mean outage cross-track RMSE + 0.25 final reference error + 0.01 normalized control slew RMS',
        no_test_based_selection=True,actual_future_controls_as_inputs=False,truth_available_to_outage_controller=False,
        normal_position_noise_std_m=.5,initial_frame='known launch-local mission frame',
        candidate_labels='separate private 16-second fixed-plan branches; actual 1-second-replanning path is not the plan label',
        validation_seed=20261123,test_seed=20261125)
    (out/'experiment_policy.json').write_text(json.dumps(policy,indent=2),encoding='utf-8')
    with np.load(HERE/'data/normal_records.npz') as z: source={k:z[k] for k in z.files}
    model,scales,_=load_model(initial)
    validation_ids=np.arange(700,800)
    status(out,'initial_closed_loop_validation',episodes=100)
    initial_summary=rollout(source,validation_ids,'cpi',model,scales,out/'initial_validation',collect=True,
        progress=lambda message:status(out,'initial_closed_loop_validation',progress=message))
    tensors={k:torch.as_tensor(v,dtype=torch.float32) for k,v in scales.items()}
    fixed_validation=ConcatDataset([Phase1Dataset(out/'initial_validation/data',np.arange(dataset_count(out/'initial_validation/data')),tensors),
        Phase1Dataset(HERE/'data',np.arange(91000,104000,5),tensors)])
    chosen=initial; best_score=closed_score(initial_summary)
    selections=[dict(round=0,checkpoint=str(initial),closed_loop_score=best_score,accepted=True,**initial_summary['metrics'])]
    old_data=[]
    for round_id in range(1,args.rounds+1):
        folder=out/f'round_{round_id:02d}'; folder.mkdir()
        model,_,_=load_model(chosen)
        status(out,'collecting_training_rollouts',round=round_id,episodes=700,parent_checkpoint=str(chosen))
        summary=rollout(source,np.arange(700),'cpi',model,scales,folder/'collection',seed=20261130+round_id,collect=True,
            progress=lambda message,r=round_id:status(out,'collecting_training_rollouts',round=r,progress=message))
        new_data=folder/'collection/data'
        candidate,best_epoch=fine_tune(out,round_id,chosen,new_data,old_data,fixed_validation,scales,args.epochs_per_round)
        candidate_model,_,_=load_model(candidate)
        status(out,'closed_loop_validation',round=round_id,checkpoint=str(candidate))
        valid=rollout(source,validation_ids,'cpi',candidate_model,scales,folder/'validation',
            progress=lambda message,r=round_id:status(out,'closed_loop_validation',round=r,progress=message))
        score=closed_score(valid); accepted=score<best_score
        if accepted: chosen=candidate; best_score=score
        selections.append(dict(round=round_id,checkpoint=str(candidate),closed_loop_score=score,accepted=accepted,
            best_prediction_epoch=best_epoch,collected_windows=summary['collected_windows'],**valid['metrics']))
        pd.DataFrame(selections).to_csv(out/'validation_selection.csv',index=False)
        (out/'selected_checkpoint.json').write_text(json.dumps(dict(checkpoint=str(chosen),score=best_score,selection_uses_test=False),indent=2),encoding='utf-8')
        old_data.append(new_data)
    # Freeze final selection before accessing any test trajectory.
    shutil.copy2(chosen,out/'selected_model.pt')
    status(out,'final_test',selected_checkpoint=str(chosen),episodes_per_policy=200)
    test_rows=[]
    models=[('DR + rolling control','dr',None),('Mean current + lag rolling control','mean',None),
        ('Last-fix current + lag rolling control','persistence',None),('Frozen phase-one CPI','cpi',initial),
        ('Selected phase-two CPI','cpi',out/'selected_model.pt')]
    for j,(name,kind,path) in enumerate(models):
        selected_model=load_model(path)[0] if path else None
        summary=rollout(source,np.arange(800,1000),kind,selected_model,scales,out/f'test_policy_{j}',seed=20261125,
            collect=j==4,progress=lambda message,n=name:status(out,'final_test',policy=n,progress=message))
        test_rows.append(dict(model=name,**summary['metrics']))
    pd.DataFrame(test_rows).to_csv(out/'closed_loop_test_comparison.csv',index=False)
    # Prediction-only comparison uses the SAME final-policy test histories and candidate branches.
    test_folder=out/'test_policy_4/data'
    test_dataset=Phase1Dataset(test_folder,np.arange(dataset_count(test_folder)),tensors)
    loader=DataLoader(test_dataset,batch_size=128,shuffle=False)
    predictive=[]; boundary=None
    for label,path in [('phase1',initial),('phase2',out/'selected_model.pt')]:
        m=load_model(path)[0]; target=out/f'prediction_test_{label}'; target.mkdir()
        if label=='phase2': boundary=verify_information_boundaries(m,test_dataset)
        frame,_=evaluate(m,loader,tensors,target); frame['checkpoint_group']=label; predictive.append(frame)
    pd.concat(predictive,ignore_index=True).to_csv(out/'prediction_test_comparison.csv',index=False)
    df=pd.DataFrame(test_rows)
    fig,axes=plt.subplots(1,2,figsize=(13,4.5),layout='constrained')
    labels=['DR','Mean current','Last-fix current','Phase 1','Selected phase 2']
    axes[0].bar(labels,df.outage_cross_track_RMSE_m); axes[0].set(ylabel='Mean episode cross-track RMSE (m)',title='During loss of ground fixes')
    axes[1].bar(labels,df.outage_position_RMSE_m); axes[1].set(ylabel='Mean episode position RMSE (m)',title='State estimation during outage')
    for ax in axes: ax.tick_params(axis='x',labelrotation=15); ax.grid(axis='y',alpha=.2)
    fig.savefig(out/'closed_loop_test_comparison.png',dpi=170); plt.close(fig)
    report=dict(passed=True,rounds_completed=args.rounds,epochs_trained=args.rounds*args.epochs_per_round,
        elapsed_s=time.monotonic()-began,selected_source_checkpoint=str(chosen),phase2_checkpoint_selected=chosen!=initial,
        final_selected_model='selected_model.pt',test_episodes_per_policy=200,test_used_for_selection=False,
        actual_future_control_input=False,controller_truth_feedback_during_outage=False,**boundary,
        limitation='Single NOAA station, overlapping within-split source windows, one training seed. Phase-one model may remain selected if no round improves closed-loop validation.')
    (out/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    status(out,'completed',**report)
    print(df.to_string(index=False),flush=True)


if __name__=='__main__': main()
