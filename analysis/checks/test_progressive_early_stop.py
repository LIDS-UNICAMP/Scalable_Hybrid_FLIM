"""Real SPiFiL/training; fixed metric fixtures isolate stopping/resume behavior."""
import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch
from ga_search.progressive import run_config,read_config
from methods.autoencoder.progressive import ProgressiveTrainer
CONFIG=Path(__file__).resolve().parents[2]/'ga_search/config/progressive_nas_synthetic.yaml'

class EarlyStopTests(unittest.TestCase):
 def setup_config(self,root):
  return dict(mode='train',device='cpu',output_dir=str(root),image_size=32,batch_size=2,
   epochs_a=1000,epochs_b=1000,training_patience=2,training_min_delta=0.0,
   max_depth=1,genetic=dict(min_depth=1,max_depth=1),latency_warmup=0,latency_repeats=1,
   synthetic_train_samples=4,synthetic_val_samples=4,complementary_svm=False,
   genome=dict(blocks=[dict(op='conv',kernel=3,channels=4,pool_stride=2,bn=True,dropout=0.,activation='relu',connection='sequential')]))
 def test_valid_resume_preserves_phase_patience(self):
  torch.set_num_threads(1)
  with tempfile.TemporaryDirectory(dir='/tmp') as temp:
   cfg=self.setup_config(Path(temp)/'run');original=ProgressiveTrainer.train_epoch;calls=0
   def interrupt(trainer,loader):
    nonlocal calls
    calls+=1
    if calls==3:raise KeyboardInterrupt()
    return original(trainer,loader)
   metrics=dict(status='ok',J=.5,dbi=1.,dbi_norm=.5,hoyer_mean=.5,coverage=1.,dbcv=None)
   with patch('ga_search.progressive.evaluate_embeddings',return_value=metrics):
    with patch.object(ProgressiveTrainer,'train_epoch',interrupt):
     with self.assertRaises(KeyboardInterrupt):run_config(CONFIG,overrides=cfg)
    last=next(Path(cfg['output_dir']).glob('individuals/*/last.ckpt'))
    saved=torch.load(last,map_location='cpu',weights_only=False)['metadata']
    self.assertEqual(saved['phase_stale'],1)
    run_config(CONFIG,overrides=cfg,resume=True)
   history=json.load(open(next(Path(cfg['output_dir']).glob('individuals/*/history.json'))))
   self.assertEqual([(r['phase'],r['phase_epoch']) for r in history],[('A',1),('A',2),('A',3),('B',1),('B',2),('B',3)])
   events=[json.loads(x) for x in (Path(cfg['output_dir'])/'events.jsonl').read_text().splitlines()]
   self.assertEqual(sum(e['event']=='training_early_stop' for e in events),2)
 def test_invalid_stops_and_config_guards(self):
  torch.set_num_threads(1)
  for value in [0,-1,True,1.5]:
   with self.assertRaises(ValueError):read_config(CONFIG,{'training_patience':value})
  with tempfile.TemporaryDirectory(dir='/tmp') as temp:
   cfg=self.setup_config(Path(temp)/'run')
   with patch('ga_search.progressive.evaluate_embeddings',return_value=dict(status='invalid',J=1e12,coverage=.1,reason='fixture')):
    run_config(CONFIG,overrides=cfg)
   history=json.load(open(next(Path(cfg['output_dir']).glob('individuals/*/history.json'))))
   self.assertEqual(len(history),4)
