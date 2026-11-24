"""Normal-operation NOAA records, masked fixes and precommitted-control branches."""
import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from routes import build_waypoints

HERE=Path(__file__).resolve().parent
SEED=20261122
DT=.2
FEATURES=['DR_relative_x','DR_relative_y','water_speed','acceleration','sin_heading','cos_heading',
          'past_a_cmd','past_omega_cmd','visible_fix_minus_DR_x','visible_fix_minus_DR_y','fix_available',
          'seconds_since_visible_fix','last_fix_minus_DR_x','last_fix_minus_DR_y','last_fix_current_x','last_fix_current_y']
MODES=['normal','outage_2_16s','outage_16_60s','outage_60_180s']


def wrap(a): return (a+np.pi)%(2*np.pi)-np.pi


def normal_records(current):
    n,steps,_=current.shape; ids=1000+np.arange(n)
    wp=[build_waypoints('traj2_like' if i%2==0 else 'traj4_like',int(ids[i])) for i in range(n)]
    lengths=np.array([len(w) for w in wp]); padded=np.empty((n,max(lengths),2))
    for i,w in enumerate(wp): padded[i,:len(w)]=w; padded[i,len(w):]=w[-1]
    rng=np.random.default_rng(SEED)
    origin=padded[:,0]+rng.uniform(-8,8,(n,2)); pos=origin.copy()
    theta=rng.uniform(-.5,.5,n); v=rng.uniform(1.1,1.9,n); acc=np.zeros(n)
    dr=np.zeros((n,2)); wi=np.ones(n,dtype=int); ix=np.arange(n)
    rows=np.empty((n,steps,16),np.float32); latent=np.empty((n,steps,3),np.float32)
    for k in range(steps):
        delta=padded[ix,wi]-pos; wi+=((np.linalg.norm(delta,axis=1)<16)&(wi<lengths-1))
        delta=padded[ix,wi]-pos
        desired=np.arctan2(delta[:,1],delta[:,0]); error=wrap(desired-theta)
        omega=np.clip(2.35*error+.22*np.sin(.073*k+.31*ids)+.09*np.sin(.191*k+.7)+.06*np.cos(.113*k+.8*ids)+rng.normal(0,.035,n),-1.05,1.05)
        vd=1.75+.26*np.sin(.041*k+.7*ids)+.19*np.sin(.133*k+.3)+.12*np.cos(.227*k+.2)
        ac=np.clip(1.16*(vd-v),-.82,.82); previous=v.copy()
        acc=.68*acc+.32*ac+rng.normal(0,.042,n); v=np.clip(v+acc*DT,.35,3.)
        theta=wrap(theta+omega*DT+rng.normal(0,.003,n))
        pos+=(np.stack([v*np.cos(theta),v*np.sin(theta)],-1)+current[:,k])*DT+rng.normal(0,.02,(n,2))
        vm=v+rng.normal(0,.03,n); am=(v-previous)/DT+rng.normal(0,.02,n); tm=wrap(theta+rng.normal(0,.01,n))
        dr+=np.stack([vm*np.cos(tm),vm*np.sin(tm)],-1)*DT
        rows[:,k]=np.column_stack([np.full(n,(k+1)*DT),pos-origin,dr,vm,am,tm,current[:,k],vd,ac,omega,desired,error,np.arange(n)])
        latent[:,k]=np.column_stack([v,acc,theta])
    return rows,latent,origin


def history_features(tr,fixes,start,cut,train_mean):
    end=start+119; indexes=np.arange(start,end+1)
    visible=indexes<=cut; last=max(0,cut); earlier=max(0,last-50)
    residual=fixes-tr[:,3:5]
    prior= train_mean.copy() if earlier==last else (residual[last]-residual[earlier])/((last-earlier)*DT)
    prior=np.clip(prior,-3,3)
    age=(end-last)*DT
    h=np.column_stack([tr[start:end+1,3:5]-tr[end,3:5],tr[start:end+1,5:7],
        np.sin(tr[start:end+1,7]),np.cos(tr[start:end+1,7]),tr[start:end+1,11:13],
        np.where(visible[:,None],residual[indexes],0),visible.astype(float),np.maximum(indexes-cut,0)*DT,
        np.broadcast_to(residual[last],(120,2)),np.broadcast_to(prior,(120,2))]).astype(np.float32)
    return h,residual[last].astype(np.float32),prior.astype(np.float32),age


def candidate_plan(last_command,seed):
    rng=np.random.default_rng(seed); t=np.arange(1,81)*DT
    phase=rng.uniform(-np.pi,np.pi,2); amplitude=rng.uniform([0,0],[.10,.15])
    return np.column_stack([np.clip(last_command[0]*np.exp(-t/2)+amplitude[0]*np.sin(.8*t+phase[0]),-.82,.82),
        np.clip(last_command[1]*np.exp(-t/4)+amplitude[1]*np.sin(.4*t+phase[1]),-1.05,1.05)]).astype(np.float32)


def branch_labels(start_pos,initial_state,plan,current,seed):
    rng=np.random.default_rng(seed); pos=start_pos.astype(float).copy()
    speed,acc,theta=initial_state.astype(float); positions=[]; velocities=[]
    for k in range(80):
        acc=.68*acc+.32*plan[k,0]+rng.normal(0,.042)
        speed=np.clip(speed+DT*acc,.35,3.); theta=wrap(theta+DT*plan[k,1]+rng.normal(0,.003))
        ground=np.array([speed*np.cos(theta),speed*np.sin(theta)])+current[k]+rng.normal(0,.02,2)/DT
        pos+=DT*ground; positions.append(pos.copy()); velocities.append(ground)
    return np.asarray(positions,np.float32),np.asarray(velocities,np.float32)


def prepare(out,source):
    out.mkdir(parents=True,exist_ok=False)
    shutil.copy2(source,out/'source_NOAA.csv')
    with source.open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
    times=np.array([datetime.fromisoformat(r['time_utc'].replace('Z','+00:00')).timestamp() for r in rows])
    currents=np.array([[float(r['u_east_m_s']),float(r['v_north_m_s'])] for r in rows])
    assert len(times)==3360 and np.all(np.diff(times)==360)
    splits={'train':np.arange(700),'val':np.arange(700,800),'test':np.arange(800,1000)}
    blocks={'train':(0,2351),'val':(2353,2855),'test':(2857,3359)}
    launches=np.empty(1000,int); rng=np.random.default_rng(SEED); supports={}
    for name,ids in splits.items():
        left,right=blocks[name]; launches[ids]=rng.choice(np.arange(left,right-50+1),len(ids),replace=False)
        supports[name]=set(k for first in launches[ids] for k in range(int(first),int(first)+51))
    assert not supports['train']&supports['val'] and not supports['train']&supports['test'] and not supports['val']&supports['test']
    query=times[launches,None]+60*np.arange(1500)[None,:]*DT
    current=np.stack([np.interp(query,times,currents[:,j]) for j in range(2)],axis=-1)
    print('Generating normal-operation full records and latent vehicle states',flush=True)
    data,latent,origin=normal_records(current)
    fix_rng=np.random.default_rng(SEED+1); fixes=data[:,:,1:3]+fix_rng.normal(0,.5,(1000,1500,2)).astype(np.float32)
    np.savez_compressed(out/'normal_records.npz',data=data,true_water_state=latent,reliable_fix_xy=fixes,
        initial_world_xy=origin,launch_unix_s=times[launches],**{'split_'+k:v for k,v in splits.items()})
    mean=data[splits['train'],:,8:10].astype(float).mean((0,1)).astype(np.float32)
    # Arrays are memory-mapped to avoid expanding all windows into RAM during training.
    n=130000
    shapes={'history':(n,120,16),'plan':(n,80,2),'position':(n,80,2),'velocity':(n,80,2),'current':(n,80,2),
            'bias':(n,2),'DR_start':(n,2),'water_state':(n,3),'last_fix_bias':(n,2),'last_fix_current':(n,2),'outage_age':(n,)}
    maps={k:np.lib.format.open_memmap(out/(k+'.npy'),mode='w+',dtype='float32',shape=v) for k,v in shapes.items()}
    manifest=[]; row=0
    for i in range(1000):
        tr=data[i]; mode_rng=np.random.default_rng(np.random.SeedSequence([SEED,i,3]))
        for start in range(0,1300,10):
            end=start+119; mode=int(mode_rng.integers(0,4))
            requested=0. if mode==0 else float(mode_rng.uniform(*[(0,0),(2,16),(16,60),(60,180)][mode]))
            cut=max(0,end-int(round(requested/DT)))
            h,fb,fc,age=history_features(tr,fixes[i],start,cut,mean)
            plan=candidate_plan(tr[end,11:13],np.random.SeedSequence([SEED,i,start,4]))
            # Target branch executes this plan; subsequent normal-policy controls are never used.
            p,vel=branch_labels(tr[end,1:3],latent[i,end],plan,current[i,end+1:end+81],np.random.SeedSequence([SEED,i,start,5]))
            values=dict(history=h,plan=plan,position=p,velocity=vel,current=current[i,end+1:end+81],
                bias=tr[end,1:3]-tr[end,3:5],DR_start=tr[end,3:5],water_state=tr[end,5:8],
                last_fix_bias=fb,last_fix_current=fc,outage_age=age)
            for name,value in values.items(): maps[name][row]=value
            manifest.append(dict(row=row,trajectory=i,start=start,last_visible_fix_index=cut,mode=MODES[mode],
                outage_age_s=age,split='train' if i<700 else ('val' if i<800 else 'test')))
            row+=1
        if (i+1)%100==0: print(f'Prepared {i+1}/1000 trajectories, {row} windows',flush=True)
    for a in maps.values(): a.flush()
    frame=pd.DataFrame(manifest)
    frame['requested_mode']=frame['mode']
    ages=frame['outage_age_s'].to_numpy()
    frame['mode']=np.select([ages==0,ages<=16,ages<=60],[MODES[0],MODES[1],MODES[2]],default=MODES[3])
    frame.to_csv(out/'windows.csv',index=False)
    np.savez(out/'splits.npz',train=np.arange(91000),val=np.arange(91000,104000),test=np.arange(104000,130000))
    # Input availability audit: poisoning hidden current/fixes and future actual controls leaves inputs unchanged.
    audits=[]
    for i,start,cut in [(0,0,110),(750,400,400),(900,1000,500)]:
        original=history_features(data[i],fixes[i],start,cut,mean)[0]
        poisoned=data[i].copy(); poisoned[:,1:3]=99999; poisoned[:,8:10]=-99999
        poisoned[start+120:,10:15]=99999
        pf=fixes[i].copy(); pf[cut+1:]=99999
        rebuilt=history_features(poisoned,pf,start,cut,mean)[0]
        np.testing.assert_array_equal(original,rebuilt)
        p1=candidate_plan(data[i,start+119,11:13],123)
        p2=candidate_plan(poisoned[start+119,11:13],123)
        np.testing.assert_array_equal(p1,p2); audits.append(True)
    config=dict(stage=1,dt=.2,history_len=120,pred_len=80,stride=10,time_scale=60,
        trajectories=1000,split_counts=dict(train=700,val=100,test=200),window_counts=dict(train=91000,val=13000,test=26000),
        history_features=FEATURES,plan_features=['a_cmd','omega_cmd'],modes=MODES,
        fix_noise_std_m=.5,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        train_mean_current_mps=mean.tolist(),source_time_split_knots=blocks,
        future_control='precommitted candidate plan generated from last historical command and independent seeded exploration',
        labels='new 16-second physics rollout from true anchor state executing candidate plan, driven by NOAA current',
        phase1_limitation='Masked offline histories come from a normal-position-feedback controller; phase 2 must collect model-controlled outage rollouts.',
        cross_split_source_support_overlap=0,hidden_input_poison_tests_passed=all(audits),
        future_actual_controls_used_as_inputs=False)
    (out/'dataset_config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    print(json.dumps(config,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=HERE/'data')
    parser.add_argument('--source',type=Path,default=HERE/'data/source_NOAA.csv')
    args=parser.parse_args(); prepare(args.output,args.source)
