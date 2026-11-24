"""Lifecycle tests on tiny synthetic fixtures, never on formal experiment data."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import formal_phase1 as formal
from phase1_model import lag_velocity
from formal_phase1_evaluate import evaluate_run


class FormalLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='auv_formal_test_')
        self.root = Path(self.temp.name); self.data = self.root / 'data'; self.data.mkdir()
        for name in ['formal_phase1.py', 'formal_phase1_evaluate.py', 'phase1_model.py', 'train_phase1.py']:
            shutil.copy2(ROOT / name, self.root / name)
        self.config = json.loads((ROOT / 'server/phase1_30ep.json').read_text())
        self.config.update(epochs=1, batch_size=8, num_workers=0, cpu_threads=2, log_every_batches=1000)
        self.config['pso'].update(population=2, iterations=2, candidate_epochs=1, train_windows=16, val_windows=8)
        self.device = formal.configure(self.config, 'cpu')
        rng = np.random.default_rng(7); n = 32
        anchor = rng.normal(0, 3, (n, 2)).astype(np.float32)
        bias = rng.normal(0, .2, (n, 2)).astype(np.float32)
        plan = rng.normal(0, .03, (n, 80, 2)).astype(np.float32)
        state = np.tile(np.array([1.5, .02, .3], np.float32), (n, 1))
        current = np.tile(np.array([.1, -.05], np.float32), (n,80,1))
        velocity = lag_velocity(torch.from_numpy(state), torch.from_numpy(plan), .2).numpy() + current
        position = (anchor + bias)[:, None] + np.cumsum(.2 * velocity, axis=1)
        history = rng.normal(0, .1, (n,120,16)).astype(np.float32)
        age = np.tile(np.array([0.,10.,30.,90.],np.float32),8)
        arrays = dict(history=history, plan=plan, water_state=state, current=current, position=position,
                      velocity=velocity, bias=bias, DR_start=anchor, last_fix_bias=bias-.02,
                      last_fix_current=np.tile(np.array([.1,-.05],np.float32),(n,1)), outage_age=age)
        for name, value in arrays.items(): np.save(self.data / (name+'.npy'), value.astype(np.float32))
        self.ids = dict(train=np.arange(16),val=np.arange(16,24),test=np.arange(24,32))
        np.savez(self.data / 'splits.npz', **self.ids)
        pd.DataFrame(dict(row=np.arange(n),trajectory=np.arange(n),split=['train']*16+['val']*8+['test']*8)).to_csv(self.data / 'windows.csv', index=False)
        source = b'time_utc,u_east_m_s,v_north_m_s\n2012-06-01T00:00:00Z,0.1,-0.05\n'
        (self.data/'source_NOAA.csv').write_bytes(source)
        self.description = dict(history_len=120,pred_len=80,dt=.2,window_counts={k:len(v) for k,v in self.ids.items()},
                                source_sha256=hashlib.sha256(source).hexdigest(),train_mean_current_mps=[.1,-.05])
        formal.atomic_json(self.data/'dataset_config.json',self.description)
        self.scales = formal.fit_all_scalers(self.data,self.ids['train'],self.description)

    def tearDown(self): self.temp.cleanup()

    def test_epoch_boundary_resume_matches_uninterrupted_weights_and_optimizer(self):
        params = self.config['default_hparams']; config = dict(self.config, epochs=2)
        continuous = self.root/'continuous'; resumed=self.root/'resumed'
        formal.fit_model('cpi_lstm',params,self.data,self.ids,self.scales,config,self.device,continuous,2,'same')
        formal.fit_model('cpi_lstm',params,self.data,self.ids,self.scales,config,self.device,resumed,1,'same')
        formal.fit_model('cpi_lstm',params,self.data,self.ids,self.scales,config,self.device,resumed,2,'same',resume=True)
        a=formal.read_checkpoint(continuous/'last.pt'); b=formal.read_checkpoint(resumed/'last.pt')
        for key in a['state_dict']: torch.testing.assert_close(a['state_dict'][key],b['state_dict'][key],rtol=0,atol=0)
        for key, values in a['optimizer']['state'].items():
            for field, value in values.items(): torch.testing.assert_close(value,b['optimizer']['state'][key][field],rtol=0,atol=0)
        self.assertEqual(a['best_epoch'],b['best_epoch'])
        self.assertEqual(a['history'][-1]['val_ADE_m'],b['history'][-1]['val_ADE_m'])

    def test_all_models_pso_freezing_and_idempotent_resume(self):
        out=self.root/'run'
        with patch.object(formal,'HERE',self.root):
            with formal.run_lock(out): formal.train_all(self.data,self.config,self.device,out,resume=True)
            choices=json.loads((out/'selected_checkpoints.json').read_text())
            hashes={k:v['sha256'] for k,v in choices['selected'].items()}
            swarm=formal.read_checkpoint(out/'pso_search/swarm.pt')
            self.assertEqual(len(swarm['records']),4)
            with np.load(out/'pso_search/subset_indices.npz') as z:
                self.assertFalse(np.intersect1d(z['train'],self.ids['test']).size)
                self.assertFalse(np.intersect1d(z['val'],self.ids['test']).size)
            with formal.run_lock(out): evaluate_run(self.data,self.config,self.device,out)
            with formal.run_lock(out): formal.train_all(self.data,self.config,self.device,out,resume=True)
            choices2=json.loads((out/'selected_checkpoints.json').read_text())
            self.assertEqual(hashes,{k:v['sha256'] for k,v in choices2['selected'].items()})
            with formal.run_lock(out): evaluate_run(self.data,self.config,self.device,out)
            self.assertTrue(json.loads((out/'test/verification.json').read_text())['passed'])
            table=pd.read_csv(out/'test/test_metrics.csv')
            self.assertEqual(len(table[table['mode']=='all']),3)
            self.assertTrue(np.isfinite(table[['ADE_m','FDE_m','position_RMSE_m']].to_numpy()).all())
            for kind in self.config['models']:
                self.assertEqual(formal.read_checkpoint(out/kind/'last.pt')['epoch'],1)
            changed=copy.deepcopy(self.config); changed['epochs']=2
            with self.assertRaises(ValueError): formal.train_all(self.data,changed,self.device,out,resume=True)

    def test_interrupted_swarm_resumes_same_candidates_and_selection(self):
        complete=self.root/'search_complete'; resumed=self.root/'search_resumed'
        complete.mkdir(); resumed.mkdir()
        reference=formal.pso_search(self.data,self.ids,self.scales,self.config,self.device,complete,'swarm-test',True)
        actual_fit=formal.fit_model
        calls=0
        def interrupt_second(*args,**kwargs):
            nonlocal calls
            calls+=1
            if calls==2: raise InterruptedError('synthetic interruption between candidates')
            return actual_fit(*args,**kwargs)
        with patch.object(formal,'fit_model',side_effect=interrupt_second):
            with self.assertRaises(InterruptedError):
                formal.pso_search(self.data,self.ids,self.scales,self.config,self.device,resumed,'swarm-test',True)
        chosen=formal.pso_search(self.data,self.ids,self.scales,self.config,self.device,resumed,'swarm-test',True)
        self.assertEqual(reference,chosen)
        a=formal.read_checkpoint(complete/'pso_search/swarm.pt')
        b=formal.read_checkpoint(resumed/'pso_search/swarm.pt')
        self.assertEqual(a['records'],b['records'])
        np.testing.assert_array_equal(a['positions'],b['positions'])
        np.testing.assert_array_equal(a['velocities'],b['velocities'])

    def test_interrupted_first_checkpoint_temporary_file_is_recoverable(self):
        folder=self.root/'incomplete_first_write'; folder.mkdir()
        (folder/'last.pt.tmp').write_bytes(b'incomplete checkpoint transaction')
        formal.fit_model('lstm',self.config['default_hparams'],self.data,self.ids,self.scales,
                         self.config,self.device,folder,1,'tmp-recovery',resume=True)
        self.assertEqual(formal.read_checkpoint(folder/'last.pt')['epoch'],1)

    @unittest.skipUnless(torch.cuda.is_available(), 'No CUDA device in this test environment')
    def test_gpu_batch_model_buffers_and_backward(self):
        device=torch.device('cuda')
        batch=next(iter(formal.make_loader(self.data,self.ids['train'][:8],self.scales,self.config,device=device)))
        tensors={k:torch.as_tensor(v,device=device) for k,v in self.scales.items()}
        for kind in self.config['models']:
            model=formal.make_model(kind,self.scales).to(device)
            moved=formal.move_batch(batch,device)
            loss=formal.objective(model(moved),moved,tensors,kind,self.config['default_hparams'])
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(all(v.device.type=='cuda' for v in model.buffers()))


if __name__ == '__main__': unittest.main(verbosity=2)
