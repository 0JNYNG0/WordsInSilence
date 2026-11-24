"""Opt-in adapter for a NEW consumer instance. Does not modify the running/default web.

The inherited raw() path calls self.channels(), then still applies its final
quarantine overlap guard. TCN ranks, DTW playback and observation windows persist.
"""
from pathlib import Path
import importlib.util,time,types
P=Path(__file__).resolve().parent
def load(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
core=load('_runtime_completion_core',P/'core.py');guard=load('_runtime_completion_guard',P/'completion_guard_v2.py')
class CompletionAuditor:
 def __init__(self):
  self.encoder,self.ck=core.encoder('final_42');self.head=core.ProgressHead();self.head.load_state_dict(core.torch.load(P/'runs/final_42/model.pt',weights_only=True,map_location='cpu'));self.head.eval();self.contracts=core.read(P.parent/'word100_positive_direction_20261109/GUIDE_CONTRACTS.json')['contracts']
 def apply(self,decision,x,ts):
  start=time.perf_counter()
  with core.torch.inference_mode():p=self.head(core.torch.tensor(core.features(x,ts,self.encoder,self.ck))).softmax(-1).numpy()
  out=guard.apply_guard(decision,p,x,ts,self.contracts);out.setdefault('timing',{})['completion_identity_seconds']=time.perf_counter()-start;return out
def attach_to_isolated_consumer(instance,auditor):
 if getattr(instance,'_completion_audit_attached',False):raise ValueError('already attached')
 original=instance.channels
 def channels(self,x,ts,fid,word):return auditor.apply(original(x,ts,fid,word),x,ts)
 instance.channels=types.MethodType(channels,instance);instance._completion_audit_attached=True;return instance
