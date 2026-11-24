"""History-only state/current estimation plus control-conditioned motion prediction."""
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

ARRAYS=['history','plan','position','velocity','current','bias','DR_start','water_state','last_fix_bias','last_fix_current','outage_age']


class Phase1Dataset(Dataset):
    def __init__(self,folder,indices,scalers):
        self.arrays={k:np.load(folder/(k+'.npy'),mmap_mode='r') for k in ARRAYS}
        self.indices=np.asarray(indices); self.scalers=scalers

    def __len__(self): return len(self.indices)

    def __getitem__(self,i):
        index=self.indices[i]
        result={k:torch.from_numpy(np.array(a[index],dtype=np.float32,copy=True)) for k,a in self.arrays.items()}
        result['x']=(result['history']-self.scalers['history_mean'])/self.scalers['history_std']
        result['u']=(result['plan']-self.scalers['plan_mean'])/self.scalers['plan_std']
        result['row']=torch.tensor(index)
        return result


@torch.jit.script
def lag_velocity(state:torch.Tensor,plan:torch.Tensor,dt:float):
    speed=state[:,0]; acc=state[:,1]; theta=state[:,2]
    velocities=torch.jit.annotate(list[torch.Tensor],[])
    for k in range(plan.size(1)):
        acc=.68*acc+.32*plan[:,k,0]
        speed=torch.clamp(speed+dt*acc,.35,3.)
        theta=theta+dt*plan[:,k,1]
        velocities.append(torch.stack([speed*torch.cos(theta),speed*torch.sin(theta)],dim=-1))
    return torch.stack(velocities,dim=1)


class Phase1CPI(nn.Module):
    def __init__(self,current_std,bias_std):
        super().__init__()
        self.hist_rnn=nn.LSTM(16,128,batch_first=True)
        self.ctrl_rnn=nn.LSTM(2,96,batch_first=True)
        self.hist_shared=nn.Sequential(nn.Linear(128,128),nn.ReLU())
        self.current_head=nn.Linear(128,160)
        self.bias_head=nn.Linear(128,2)
        self.motion_head=nn.Sequential(nn.Linear(224,128),nn.ReLU(),nn.Linear(128,160))
        nn.init.zeros_(self.current_head.weight); nn.init.zeros_(self.current_head.bias)
        nn.init.zeros_(self.bias_head.weight); nn.init.zeros_(self.bias_head.bias)
        nn.init.zeros_(self.motion_head[-1].weight); nn.init.zeros_(self.motion_head[-1].bias)
        self.register_buffer('current_std',torch.as_tensor(current_std,dtype=torch.float32))
        self.register_buffer('bias_std',torch.as_tensor(bias_std,dtype=torch.float32))

    def forward(self,batch):
        h,_=self.hist_rnn(batch['x']); encoded=h[:,-1]
        z=self.hist_shared(encoded)
        # Neither present position nor sea current is allowed to change with a candidate plan.
        current=batch['last_fix_current'][:,None,:]+self.current_head(z).view(-1,80,2)*self.current_std
        bias_prior=batch['last_fix_bias']+batch['outage_age'][:,None]*batch['last_fix_current']
        bias=bias_prior+self.bias_head(z)*self.bias_std
        control,_=self.ctrl_rnn(batch['u'])
        residual=.3*torch.tanh(self.motion_head(torch.cat([encoded,control[:,-1]],dim=-1)).view(-1,80,2))
        water_velocity=lag_velocity(batch['water_state'],batch['plan'],.2)+residual
        ground_velocity=water_velocity+current
        position=(batch['DR_start']+bias)[:,None,:]+torch.cumsum(.2*ground_velocity,dim=1)
        return dict(position=position,current=current,bias=bias,ground_velocity=ground_velocity,
                    motion_residual=residual)


def losses(pred,batch,scalers):
    position=torch.mean((pred['position']-batch['position'])**2)/(scalers['position_scale']**2)
    current=torch.mean(((pred['current']-batch['current'])/scalers['current_std'])**2)
    bias=torch.mean(((pred['bias']-batch['bias'])/scalers['bias_std'])**2)
    physics=torch.mean((pred['ground_velocity']-batch['velocity'])**2)/(scalers['current_scale']**2)
    total=position+.1*current+.2*bias+.02*physics
    return torch.stack([total,position,current,bias,physics])
