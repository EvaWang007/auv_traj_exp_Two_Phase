"""Fixed kinematic/current priors plus directly learned position residuals."""
import torch
from torch import nn
from phase1_model import lag_velocity


class PiLSTM(nn.Module):
    def __init__(self, scales):
        super().__init__()
        self.hist_rnn = nn.LSTM(16, 128, batch_first=True)
        self.ctrl_rnn = nn.LSTM(2, 96, batch_first=True)
        self.head = nn.Sequential(nn.Linear(224, 128), nn.ReLU(), nn.Linear(128, 160))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.register_buffer('position_scale', torch.as_tensor(scales['position_scale'], dtype=torch.float32))

    def forward(self, batch):
        history, _ = self.hist_rnn(batch['x'])
        control, _ = self.ctrl_rnn(batch['u'])
        delta = self.head(torch.cat([history[:, -1], control[:, -1]], -1)).view(-1, 80, 2)*self.position_scale
        bias_prior = batch['last_fix_bias']+batch['outage_age'][:, None]*batch['last_fix_current']
        water = lag_velocity(batch['water_state'], batch['plan'], .2)
        base = (batch['DR_start']+bias_prior)[:, None]+torch.cumsum(.2*(water+batch['last_fix_current'][:, None]), dim=1)
        return {'position': base+delta, 'base_position': base, 'position_residual': delta}
