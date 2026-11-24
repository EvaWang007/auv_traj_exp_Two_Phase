"""Gate phase two with observer invariance, checkpoint equivalence, and real rollouts."""
from pathlib import Path
import argparse
import json
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from phase1_model import Phase1Dataset
from phase2_control import load_model,estimate,rollout,Observer
from train_phase1 import verify_information_boundaries

HERE=Path(__file__).resolve().parent


def verify(output,checkpoint=None):
    torch.set_num_threads(4)
    model,scales,saved=load_model(checkpoint or HERE/'results/phase1_seed42/cpi_lstm_phase1_best.pt')
    tensors={k:torch.as_tensor(v,dtype=torch.float32) for k,v in scales.items()}
    dataset=Phase1Dataset(HERE/'data',np.arange(91000,91032),tensors)
    bounds=verify_information_boundaries(model,dataset)
    batch=next(iter(DataLoader(dataset,batch_size=32)))
    with torch.no_grad(): pred=model(batch)
    p,c,_=estimate(model,scales,batch['history'].numpy(),batch['last_fix_bias'].numpy(),batch['last_fix_current'].numpy(),batch['outage_age'].numpy(),batch['DR_start'].numpy())
    np.testing.assert_allclose(c,pred['current'].numpy(),rtol=0,atol=1e-5)
    np.testing.assert_allclose(p,(batch['DR_start']+pred['bias']).numpy(),rtol=0,atol=1e-5)
    # Unavailable fixes never enter observer memory, even if the caller supplied garbage.
    observer=Observer(2,scales['train_mean_current'])
    sensor=np.zeros((2,7),np.float32)
    observer.observe(0,sensor,np.zeros((2,2),np.float32),np.ones(2,bool))
    observer.observe(1,sensor,np.full((2,2),99999,np.float32),np.zeros(2,bool))
    assert np.isnan(observer.fixes[:,1]).all()
    with np.load(HERE/'data/normal_records.npz') as z: source={k:z[k] for k in z.files}
    began=time.monotonic()
    summary=rollout(source,np.arange(700,704),'cpi',model,scales,output/'smoke_rollout',seed=20261123,collect=True,progress=lambda s:print(s,flush=True))
    assert summary['collected_windows']>0
    d=output/'smoke_rollout/data'
    pos=np.load(d/'position.npy'); vel=np.load(d/'velocity.npy'); dr=np.load(d/'DR_start.npy'); bias=np.load(d/'bias.npy')
    reconstructed=(dr+bias)[:,None]+np.cumsum(.2*vel,axis=1)
    np.testing.assert_allclose(reconstructed,pos,rtol=0,atol=3e-4)
    report=dict(passed=True,phase1_best_epoch=saved['best_epoch'],cached_state_estimator_matches_checkpoint=True,
        hidden_fixes_discarded=True,branch_labels_match_executed_plans=True,smoke_episodes=4,
        collected_windows=summary['collected_windows'],elapsed_s=time.monotonic()-began,**bounds)
    (output/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=HERE/'results/phase2/gate')
    parser.add_argument('--checkpoint',type=Path,default=HERE/'results/phase1_seed42/cpi_lstm_phase1_best.pt')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False); verify(args.output,args.checkpoint)
