"""Meaningful model invariants; python -m unittest discover -s server -p Pi_LSTM_test.py."""
import sys
from pathlib import Path
import unittest
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from Pi_LSTM_model import PiLSTM
from phase1_model import lag_velocity


class PiLSTMTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(42)
        self.model=PiLSTM({'position_scale':np.array(20.,np.float32)})
        self.batch=dict(x=torch.randn(2,120,16),u=torch.randn(2,80,2),plan=torch.zeros(2,80,2),
            water_state=torch.tensor([[1.5,0.,0.],[1.5,0.,0.]]),DR_start=torch.zeros(2,2),
            last_fix_bias=torch.tensor([[1.,2.],[1.,2.]]),outage_age=torch.tensor([10.,10.]),
            last_fix_current=torch.tensor([[.1,.2],[.1,.2]]))

    def test_initial_prediction_equals_fixed_prior_and_ignores_targets(self):
        self.model.eval()
        with torch.no_grad():
            p=self.model(self.batch)
            expected=(self.batch['DR_start']+self.batch['last_fix_bias']+self.batch['outage_age'][:,None]*self.batch['last_fix_current'])[:,None]+torch.cumsum(.2*(lag_velocity(self.batch['water_state'],self.batch['plan'],.2)+self.batch['last_fix_current'][:,None]),1)
            torch.testing.assert_close(p['position'],expected,rtol=0,atol=0)
            poisoned=dict(self.batch,position=torch.full((2,80,2),99999.),current=torch.full((2,80,2),99999.),bias=torch.full((2,2),99999.))
            torch.testing.assert_close(p['position'],self.model(poisoned)['position'],rtol=0,atol=0)

    def test_turn_formula_responds_before_training(self):
        straight=self.model(self.batch)['position']
        turn={k:v.clone() for k,v in self.batch.items()};turn['plan'][:,:,1]=np.pi/16
        turned=self.model(turn)['position']
        self.assertGreater(float(torch.linalg.vector_norm(turned[:,-1]-straight[:,-1],dim=-1).mean().detach()),10.)

    def test_residual_can_learn_position_correction(self):
        optimizer=torch.optim.Adam(self.model.parameters(),lr=.001)
        target=self.model(self.batch)['position'].detach()+3
        before=float((self.model(self.batch)['position']-target).square().mean().detach())
        for _ in range(5):
            optimizer.zero_grad();loss=(self.model(self.batch)['position']-target).square().mean();loss.backward();optimizer.step()
        after=float((self.model(self.batch)['position']-target).square().mean().detach())
        self.assertLess(after,before)


if __name__=='__main__':unittest.main()
