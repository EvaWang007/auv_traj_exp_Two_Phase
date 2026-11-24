"""Observation-only rolling controller and private NOAA simulation/label environment."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from phase1_model import Phase1CPI,lag_velocity
from routes import build_waypoints

DT=.2
KEYS=['history','plan','position','velocity','current','bias','DR_start','water_state','last_fix_bias','last_fix_current','outage_age']


def wrap(a): return (a+np.pi)%(2*np.pi)-np.pi


def load_model(path):
    saved=torch.load(path,map_location='cpu',weights_only=False)
    scales=saved['scalers']
    model=Phase1CPI(scales['current_std'],scales['bias_std']).eval()
    model.load_state_dict(saved['state_dict'])
    return model,scales,saved


def water_paths(state,plans):
    """Vectorized lag model for screening; no true state or NOAA input."""
    n,c,h,_=plans.shape
    speed=np.broadcast_to(state[:,None,0],(n,c)).copy()
    acc=np.broadcast_to(state[:,None,1],(n,c)).copy()
    theta=np.broadcast_to(state[:,None,2],(n,c)).copy()
    result=np.empty((n,c,h,2),np.float32)
    for k in range(h):
        acc=.68*acc+.32*plans[:,:,k,0]; speed=np.clip(speed+DT*acc,.35,3.)
        theta+=DT*plans[:,:,k,1]
        result[:,:,k,0]=speed*np.cos(theta); result[:,:,k,1]=speed*np.sin(theta)
    return result


class Observer:
    """Contains only onboard sensors, executed commands, and available position fixes."""
    def __init__(self,n,mean):
        # DR x/y, measured speed/acceleration/heading, executed a/omega.
        self.sensors=np.zeros((n,1500,7),np.float32)
        self.fixes=np.full((n,1500,2),np.nan,np.float32)
        self.available=np.zeros((n,1500),bool)
        self.ages=np.zeros((n,1500),np.float32)
        self.cut=np.zeros(n,int)
        self.fix_bias=np.zeros((n,2),np.float32)
        self.fix_current=np.broadcast_to(mean,(n,2)).copy().astype(np.float32)
        self.mean=np.asarray(mean,np.float32)

    def observe(self,k,sensors,fix,available):
        self.sensors[:,k]=sensors; self.available[:,k]=available
        self.fixes[available,k]=fix[available]
        for i in np.flatnonzero(available):
            self.cut[i]=k; self.fix_bias[i]=fix[i]-sensors[i,:2]
            if k>=50:
                candidates=np.flatnonzero(self.available[i,:k-49])
                if len(candidates):
                    earlier=candidates[-1]
                    old=self.fixes[i,earlier]-self.sensors[i,earlier,:2]
                    self.fix_current[i]=np.clip((self.fix_bias[i]-old)/((k-earlier)*DT),-3,3)
        self.ages[:,k]=(k-self.cut)*DT

    def features(self,k,indices):
        assert k>=119
        sensors=self.sensors[indices,k-119:k+1]
        valid=self.available[indices,k-119:k+1]
        residual=np.where(valid[:,:,None],self.fixes[indices,k-119:k+1]-sensors[:,:,:2],0)
        n=len(indices)
        return np.concatenate([sensors[:,:,:2]-sensors[:,-1:,:2],sensors[:,:,2:4],
            np.sin(sensors[:,:,4:5]),np.cos(sensors[:,:,4:5]),sensors[:,:,5:7],residual,
            valid[:,:,None].astype(np.float32),self.ages[indices,k-119:k+1,None],
            np.broadcast_to(self.fix_bias[indices,None,:],(n,120,2)),
            np.broadcast_to(self.fix_current[indices,None,:],(n,120,2))],axis=-1).astype(np.float32)


@torch.no_grad()
def estimate(model,scales,history,fix_bias,fix_current,age,dr):
    x=torch.from_numpy((history-scales['history_mean'])/scales['history_std'])
    h,_=model.hist_rnn(x); encoded=h[:,-1]; z=model.hist_shared(encoded)
    cur=torch.from_numpy(fix_current)[:,None,:]+model.current_head(z).view(-1,80,2)*model.current_std
    bias=torch.from_numpy(fix_bias+age[:,None]*fix_current)+model.bias_head(z)*model.bias_std
    return (dr+bias.numpy()).astype(np.float32),cur.numpy(),encoded


def reference_path(position,previous,target):
    delta=target-previous; length=np.maximum(np.linalg.norm(delta,axis=1),1.)
    direction=delta/length[:,None]
    distance=np.clip(np.sum((position-previous)*direction,axis=1),0,length)
    progress=np.minimum(distance[:,None]+1.75*np.arange(1,81)[None,:]*DT,length[:,None])
    return previous[:,None,:]+progress[:,:,None]*direction[:,None,:]


def plan_cost(pred,plans,reference,last_command):
    error=np.sum((pred-reference[:,None])**2,axis=-1)/25.
    normalized=plans/np.array([.82,1.05],np.float32)
    delta=np.diff(np.concatenate([np.broadcast_to(last_command[:,None,None,:],(len(plans),plans.shape[1],1,2)),plans],axis=2),axis=2)
    delta=delta/np.array([.82,1.05],np.float32)
    return error.mean(2)+2*error[:,:,-1]+.05*np.mean(normalized**2,axis=(2,3))+.1*np.mean(delta**2,axis=(2,3))


@torch.no_grad()
def choose_plan(position,current,state,last_command,previous,target,warm,seed,model=None,scales=None,encoded=None):
    """32 known-time candidates, physical screening, full-model refinement of four."""
    n=len(position); t=np.arange(1,81)*DT; rng=np.random.default_rng(seed)
    desired=np.arctan2(target[:,1]-position[:,1],target[:,0]-position[:,0])
    a=np.clip(1.16*(1.75-state[:,0]),-.82,.82)
    omega=np.clip(2.35*wrap(desired-state[:,2]),-1.05,1.05)
    nominal=np.stack([a[:,None]*np.exp(-t/2),omega[:,None]*np.exp(-t/4)],axis=-1)
    perturb=rng.normal(size=(n,32,8,2))*np.array([.08,.12])
    perturb[:,:,0]*=.5
    perturb=np.repeat(perturb,10,axis=2)
    plans=np.clip(nominal[:,None]+perturb,[-.82,-1.05],[.82,1.05]).astype(np.float32)
    plans[:,0]=nominal; plans[:,1]=warm
    plans[:,2]=np.broadcast_to(last_command[:,None,:],(n,80,2))*np.exp(-t[None,:,None]/3)
    ref=reference_path(position,previous,target)
    water=water_paths(state,plans)
    predictions=position[:,None,None,:]+np.cumsum(DT*(water+current[:,None]),axis=2)
    cost=plan_cost(predictions,plans,ref,last_command)
    best=np.argsort(cost,axis=1)[:,:4]; ix=np.arange(n)
    finalists=plans[ix[:,None],best]
    if model is not None:
        flat=finalists.reshape(-1,80,2)
        u=torch.from_numpy((flat-scales['plan_mean'])/scales['plan_std'])
        hc,_=model.ctrl_rnn(u)
        residual=.3*torch.tanh(model.motion_head(torch.cat([encoded.repeat_interleave(4,0),hc[:,-1]],-1))).view(n,4,80,2).numpy()
        fine_water=water[ix[:,None],best]+residual
        fine=position[:,None,None,:]+np.cumsum(DT*(fine_water+current[:,None]),axis=2)
        fine_cost=plan_cost(fine,finalists,ref,last_command)
        selected=np.argmin(fine_cost,axis=1)
    else: selected=np.argmin(cost[ix[:,None],best],axis=1)
    chosen=finalists[ix,selected]
    alternative=finalists[ix,(selected+1)%4]
    return chosen,alternative


def branch_batch(position,state,plans,current,seed):
    """Private target simulator, never called by the controller to choose an action."""
    rng=np.random.default_rng(seed); pos=position.astype(float).copy()
    speed=state[:,0].astype(float).copy(); acc=state[:,1].astype(float).copy(); theta=state[:,2].astype(float).copy()
    y=np.empty((len(pos),80,2),np.float32); vel=y.copy()
    for k in range(80):
        acc=.68*acc+.32*plans[:,k,0]+rng.normal(0,.042,len(pos))
        speed=np.clip(speed+DT*acc,.35,3.); theta=wrap(theta+DT*plans[:,k,1]+rng.normal(0,.003,len(pos)))
        ground=np.stack([speed*np.cos(theta),speed*np.sin(theta)],-1)+current[:,k]+rng.normal(0,.02,(len(pos),2))/DT
        pos+=DT*ground; y[:,k]=pos; vel[:,k]=ground
    return y,vel


class Collector:
    def __init__(self): self.batches=[]; self.manifests=[]

    def add(self,observer,k,indices,trajectory,plans,true_pos,true_state,current,seed):
        h=observer.features(k,indices)
        n=len(indices); times=2
        plan=np.stack(plans,axis=1).reshape(-1,80,2)
        hidden_current=np.repeat(current[:,k+1:k+81],times,axis=0)
        y,vel=branch_batch(np.repeat(true_pos,times,axis=0),np.repeat(true_state,times,axis=0),plan,hidden_current,seed)
        s=observer.sensors[indices,k]
        values=dict(history=np.repeat(h,times,axis=0),plan=plan,position=y,velocity=vel,current=hidden_current,
            bias=np.repeat(true_pos-s[:,:2],times,axis=0),DR_start=np.repeat(s[:,:2],times,axis=0),
            water_state=np.repeat(s[:,2:5],times,axis=0),last_fix_bias=np.repeat(observer.fix_bias[indices],times,axis=0),
            last_fix_current=np.repeat(observer.fix_current[indices],times,axis=0),outage_age=np.repeat(observer.ages[indices,k],times))
        self.batches.append(values)
        for local,i in enumerate(indices):
            for branch in range(times): self.manifests.append(dict(trajectory=int(trajectory[i]),anchor_index=k,
                outage_age_s=float(observer.ages[i,k]),branch=branch,origin='model_controlled_outage'))

    def save(self,out):
        out.mkdir(parents=True,exist_ok=False); n=sum(len(b['history']) for b in self.batches)
        assert n>0
        for key in KEYS:
            shape=(n,*self.batches[0][key].shape[1:])
            array=np.lib.format.open_memmap(out/(key+'.npy'),mode='w+',dtype='float32',shape=shape)
            offset=0
            for b in self.batches:
                value=b[key]; array[offset:offset+len(value)]=value; offset+=len(value)
            array.flush(); del array
        pd.DataFrame(self.manifests).to_csv(out/'windows.csv',index=False)
        (out/'collection.json').write_text(json.dumps(dict(windows=n,trajectories=len(set(r['trajectory'] for r in self.manifests)),
            future_actual_controls_as_inputs=False,labels='separate full-plan branch rollouts',
            ground_truth_used_by_outage_controller=False),indent=2),encoding='utf-8')
        self.batches.clear(); self.manifests.clear()
        return n


def tracking_metrics(position,estimate,commands,waypoints,outage,durations):
    result=[]
    for i,p in enumerate(position):
        a=waypoints[i,:-1]; delta=waypoints[i,1:]-a; length=np.linalg.norm(delta,axis=1)
        fraction=np.clip(np.sum((p[:,None]-a[None])*delta[None],axis=-1)/np.maximum(length[None]**2,1e-6),0,1)
        projected=a[None]+fraction[:,:,None]*delta[None]
        distance=np.linalg.norm(p[:,None]-projected,axis=-1).min(axis=1)
        mask=outage[i]
        estimate_error=np.linalg.norm(estimate[i]-p,axis=-1)
        traveled=1.75*300.; cumulative=np.concatenate([[0.],np.cumsum(length)])
        segment=min(int(np.searchsorted(cumulative,traveled,side='right')-1),len(length)-1)
        target=a[segment]+np.clip((traveled-cumulative[segment])/max(length[segment],1e-6),0,1)*delta[segment]
        reached=0
        for pp in p:
            if reached<len(waypoints[i])-1 and np.linalg.norm(pp-waypoints[i,reached+1])<16: reached+=1
        change=np.diff(commands[i],axis=0)/np.array([.82,1.05])
        result.append(dict(episode=i,outage_duration_s=int(durations[i]),
            outage_cross_track_RMSE_m=float(np.sqrt(np.mean(distance[mask]**2))),
            outage_cross_track_mean_m=float(distance[mask].mean()),outage_cross_track_max_m=float(distance[mask].max()),
            outage_position_RMSE_m=float(np.sqrt(np.mean(estimate_error[mask]**2))),
            final_reference_error_m=float(np.linalg.norm(p[-1]-target)),waypoint_reach_fraction=reached/(len(waypoints[i])-1),
            control_slew_RMS=float(np.sqrt(np.mean(change**2))),control_effort=float(np.mean((commands[i]/np.array([.82,1.05]))**2))))
    return pd.DataFrame(result)


def rollout(source,indices,kind,model,scales,output,seed=20261123,collect=False,progress=None):
    """Actual feedback actions use observer-only estimates; true state stays in the plant/label path."""
    n=len(indices); ix=np.arange(n)
    records=source['data'][indices]; private_current=records[:,:,8:10]
    origin=source['initial_world_xy'][indices]
    wp=np.stack([build_waypoints('traj2_like' if i%2==0 else 'traj4_like',1000+int(i))-origin[j] for j,i in enumerate(indices)])
    pos=records[:,0,1:3].astype(float).copy(); dr=records[:,0,3:5].astype(float).copy()
    actual=source['true_water_state'][indices,0].astype(float)
    speed=actual[:,0].copy(); acc=actual[:,1].copy(); theta=actual[:,2].copy()
    rng=np.random.default_rng(seed); schedule=np.random.default_rng(20261124)
    all_start=schedule.integers(30,121,1000); all_duration=np.array([30,60,120,180])[np.arange(1000)%4]
    start=all_start[indices]; duration=all_duration[indices]
    obs=Observer(n,scales['train_mean_current'])
    sensor=np.column_stack([dr,records[:,0,5:8],records[:,0,11:13]]).astype(np.float32)
    obs.observe(0,sensor,(pos+rng.normal(0,.5,(n,2))).astype(np.float32),np.ones(n,bool))
    estimate_pos=obs.fixes[:,0].copy(); cached_current=np.broadcast_to(scales['train_mean_current'],(n,80,2)).copy()
    plans=np.zeros((n,80,2),np.float32); pointers=np.zeros(n,int); wi=np.ones(n,int)
    positions=np.empty((n,1500,2),np.float32); estimates=positions.copy(); commands=positions.copy(); outage=np.zeros((n,1500),bool)
    positions[:,0]=pos; estimates[:,0]=estimate_pos; commands[:,0]=sensor[:,5:7]
    collector=Collector() if collect else None
    for k in range(1499):
        time=(k+1)*DT; absent=(time>=start)&(time<start+duration); outage[:,k]=absent
        waypoint_updated=np.zeros(n,bool)
        # All control-estimation branches receive only observer data, never pos/actual/current.
        if k>=119 and (k+1)%5==0 and absent.any():
            active=np.flatnonzero(absent); sensed=obs.sensors[active,k]
            age=obs.ages[active,k]
            encoded=None
            if kind=='cpi':
                h=obs.features(k,active)
                pp,cc,encoded=estimate(model,scales,h,obs.fix_bias[active],obs.fix_current[active],age,sensed[:,:2])
            else:
                prior=np.zeros((len(active),2),np.float32) if kind=='dr' else (np.broadcast_to(scales['train_mean_current'],(len(active),2)) if kind=='mean' else obs.fix_current[active])
                pp=sensed[:,:2]+obs.fix_bias[active]+age[:,None]*prior
                cc=np.broadcast_to(prior[:,None],(len(active),80,2)).copy()
            estimate_pos[active]=pp; cached_current[active]=cc
            reached=(np.linalg.norm(wp[active,wi[active]]-pp,axis=1)<16)&(wi[active]<wp.shape[1]-1)
            wi[active]+=reached
            waypoint_updated[active]=True
            warm=np.concatenate([plans[active,5:],np.repeat(plans[active,-1:],5,axis=1)],axis=1)
            chosen,other=choose_plan(pp,cc,sensed[:,2:5],sensed[:,5:7],wp[active,wi[active]-1],wp[active,wi[active]],warm,
                np.random.SeedSequence([seed,k,7]),model if kind=='cpi' else None,scales,encoded)
            plans[active]=chosen; pointers[active]=0
            if collect and (k+1)%10==0 and k+80<1500:
                collector.add(obs,k,active,indices,[chosen,other],pos[active],np.column_stack([speed[active],acc[active],theta[active]]),private_current[active],np.random.SeedSequence([seed,k,8]))
        normal=~absent
        estimate_pos[normal]=obs.fixes[normal,k]
        reached=(np.linalg.norm(wp[ix,wi]-estimate_pos,axis=1)<16)&(wi<wp.shape[1]-1)&(~waypoint_updated)
        wi+=reached
        target=wp[ix,wi]
        heading=np.arctan2(target[:,1]-estimate_pos[:,1],target[:,0]-estimate_pos[:,0])
        measured=obs.sensors[:,k]
        action=np.column_stack([np.clip(1.16*(1.75-measured[:,2]),-.82,.82),np.clip(2.35*wrap(heading-measured[:,4]),-1.05,1.05)])
        active=np.flatnonzero(absent)
        if len(active): action[active]=plans[active,np.minimum(pointers[active],79)]
        # Private plant step: ground truth is used here for simulation, never for action selection.
        old_speed=speed.copy(); acc=.68*acc+.32*action[:,0]+rng.normal(0,.042,n)
        speed=np.clip(speed+DT*acc,.35,3.); theta=wrap(theta+DT*action[:,1]+rng.normal(0,.003,n))
        pos+=DT*(np.stack([speed*np.cos(theta),speed*np.sin(theta)],-1)+private_current[:,k+1])+rng.normal(0,.02,(n,2))
        vm=speed+rng.normal(0,.03,n); am=(speed-old_speed)/DT+rng.normal(0,.02,n); tm=wrap(theta+rng.normal(0,.01,n))
        dr+=DT*np.stack([vm*np.cos(tm),vm*np.sin(tm)],-1)
        next_time=(k+2)*DT; available=~((next_time>=start)&(next_time<start+duration))
        noisy_fix=(pos+rng.normal(0,.5,(n,2))).astype(np.float32)
        noisy_fix[~available]=np.nan
        obs.observe(k+1,np.column_stack([dr,vm,am,tm,action]).astype(np.float32),noisy_fix,available)
        if len(active):
            q=np.minimum(pointers[active],79)
            estimate_pos[active]+=DT*(np.stack([vm[active]*np.cos(tm[active]),vm[active]*np.sin(tm[active])],-1)+cached_current[active,q])
            pointers[active]+=1
        estimate_pos[available]=noisy_fix[available]
        positions[:,k+1]=pos; estimates[:,k+1]=estimate_pos; commands[:,k+1]=action
        if progress and (k+1)%300==0: progress(f'{kind}: simulated {(k+1)*DT:.0f}/300 s, {n} episodes')
    outage[:,-1]=((300>=start)&(300<start+duration))
    output.mkdir(parents=True,exist_ok=False)
    metrics=tracking_metrics(positions,estimates,commands,wp,outage,duration)
    metrics['trajectory']=indices; metrics.to_csv(output/'episode_metrics.csv',index=False)
    metrics.groupby('outage_duration_s').mean(numeric_only=True).to_csv(output/'metrics_by_duration.csv')
    np.savez_compressed(output/'rollout.npz',true_position=positions,estimated_position=estimates,commands=commands,
        outage=outage,waypoints=wp,trajectory=indices,outage_start_s=start,outage_duration_s=duration)
    count=collector.save(output/'data') if collect else 0
    summary=dict(policy=kind,episodes=n,collected_windows=count,seed=seed,
        metrics={k:float(v) for k,v in metrics.select_dtypes(include='number').mean().items()},
        outage_controller_uses_ground_truth=False,candidates=32,refined_candidates=4,control_interval_s=1.,
        labels_from_separate_fixed_plan_branches=True)
    (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    return summary
