"""Fixed round-one interim test; never used for training or checkpoint selection."""
from pathlib import Path
import hashlib
import json
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from phase2_control import load_model,rollout

HERE=Path(__file__).resolve().parent
OUT=HERE/'results/phase2/round01_interim_test'


def main():
    OUT.mkdir(exist_ok=False); (OUT/'provenance').mkdir()
    for name in ['test_phase2_round1.py','phase2_control.py','phase1_model.py','routes.py']:
        shutil.copy2(HERE/name,OUT/'provenance'/name)
    checkpoint=HERE/'results/phase2/round_01/training/cpi_lstm_phase2_best.pt'
    shutil.copy2(checkpoint,OUT/'round01_frozen.pt')
    frozen,scales,saved=load_model(OUT/'round01_frozen.pt')
    phase1=HERE/'results/phase1_seed42/cpi_lstm_phase1_best.pt'
    torch.set_num_threads(2)
    with np.load(HERE/'data/normal_records.npz') as z: source={k:z[k] for k in z.files}
    policy=dict(test_trajectories=list(range(800,1000)),seed=20261125,round=1,
        checkpoint=str(checkpoint),best_epoch=saved['best_epoch'],
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        checkpoint_chosen_using_validation_only=True,interim_results_not_used_by_training_pipeline=True,
        note='User-requested interim test exposes these test cases; later reuse is a repeated evaluation, not a new blind test.')
    (OUT/'test_policy.json').write_text(json.dumps(policy,indent=2),encoding='utf-8')
    cases=[('DR + rolling control','dr',None),('Mean current + lag rolling control','mean',None),
        ('Last-fix current + lag rolling control','persistence',None),('Frozen phase-one CPI','cpi',load_model(phase1)[0]),
        ('Round-one CPI','cpi',frozen)]
    rows=[]; durations=[]
    for j,(name,kind,model) in enumerate(cases):
        print('Testing '+name,flush=True)
        folder=OUT/f'policy_{j}'
        summary=rollout(source,np.arange(800,1000),kind,model,scales,folder,seed=20261125,
            progress=lambda message:print(message,flush=True))
        rows.append(dict(model=name,**summary['metrics']))
        frame=pd.read_csv(folder/'metrics_by_duration.csv'); frame['model']=name; durations.append(frame)
    result=pd.DataFrame(rows)
    result.to_csv(OUT/'comparison.csv',index=False)
    pd.concat(durations,ignore_index=True).to_csv(OUT/'comparison_by_outage_duration.csv',index=False)
    # Verify pairing: identical missions, outage schedules, and waypoint geometry.
    reference=None
    for j in range(len(cases)):
        with np.load(OUT/f'policy_{j}/rollout.npz') as z:
            shared={key:z[key] for key in ['trajectory','waypoints','outage_start_s','outage_duration_s','outage']}
        if reference is None: reference=shared
        else:
            for key in shared: np.testing.assert_array_equal(shared[key],reference[key])
    labels=['DR','Mean current','Last-fix current','Phase 1','Round 1']
    fig,axes=plt.subplots(1,2,figsize=(13,4.5),layout='constrained')
    for ax,col,title in [(axes[0],'outage_cross_track_RMSE_m','Actual route tracking during outage'),
        (axes[1],'outage_position_RMSE_m','Position estimation during outage')]:
        ax.bar(labels,result[col]); ax.set(title=title,ylabel='Mean episode RMSE (m)'); ax.grid(axis='y',alpha=.2)
        ax.tick_params(axis='x',labelrotation=15)
    fig.savefig(OUT/'comparison.png',dpi=180); plt.close(fig)
    # Fixed examples: first test mission of each outage duration, not selected by error.
    chosen=[int(np.flatnonzero(reference['outage_duration_s']==duration)[0]) for duration in [30,60,120,180]]
    fig,axes=plt.subplots(2,2,figsize=(12,10),layout='constrained')
    for ax,i in zip(axes.ravel(),chosen):
        wp=reference['waypoints'][i]
        ax.plot(wp[:,0],wp[:,1],'k--',lw=1,label='Reference waypoints')
        for j in [1,3,4]:
            with np.load(OUT/f'policy_{j}/rollout.npz') as z:
                p=z['true_position'][i]; mask=z['outage'][i]
            ax.plot(p[:,0],p[:,1],label=cases[j][0],lw=1.2)
            ax.scatter(p[mask][0,0],p[mask][0,1],s=15)
        ax.set(title=f"Trajectory {reference['trajectory'][i]}, outage {reference['outage_duration_s'][i]} s",xlabel='East (m)',ylabel='North (m)')
        ax.set_aspect('equal',adjustable='datalim'); ax.grid(alpha=.2)
    handles,names=axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles,names,loc='outside lower center',ncol=2,fontsize=8)
    fig.savefig(OUT/'trajectory_examples.png',dpi=160); plt.close(fig)
    verification=dict(passed=True,paired_test_episodes_per_policy=200,round=1,checkpoint_epoch=saved['best_epoch'],
        seed=20261125,models=5,mission_outage_geometry_pairing_verified=True,
        checkpoint_selection_uses_test=False,ground_truth_feedback_during_outage=False,
        metrics='Per-episode outage RMSE averaged across 200 missions; not offline trajectory ADE/FDE.',
        limitations='Single station and seed; within-split NOAA windows overlap; interim test is now exposed.')
    (OUT/'verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
    print(result.to_string(index=False),flush=True)


if __name__=='__main__': main()
