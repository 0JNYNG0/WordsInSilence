"""One frozen causal ablation, not a deployment candidate.

Keep the perturbed span but restore the original body scale. That scale is
known only because this is a synthetic paired test; never call it a real-world
solution or choose a scale using the correct label.
"""
from pathlib import Path
import sys,json,hashlib,time
sys.dont_write_bytecode=True
sys.stdout.reconfigure(encoding='utf-8')
U=Path('C:/MyUnityProjects/WordsInSilence/research/front_keyphase_v1/ops/word100_boundary_evidence_20261026')
sys.path.insert(0,str(U))
import boundary_probe as b

def main():
 import torch
 from pilot.model_masked_v1 import MaskedSignTCN
 from frozen_temporal import FrozenCTemporal
 from training_core import MODEL,padded_arrays,prediction
 b.check(U);assert b.read(U/'VERIFIED.json')['status']=='PASS'
 b.write(U/'SCALE_ABLATION_FROZEN.json',dict(at=time.time(),script_sha256=b.sha(__file__),
   comparison='Same synthetic long span, body-relative channels multiplied by modified_scale/original_scale.',
   allowed='DEV300 paired diagnosis only',not_candidate=True,no_training=True,
   invariant='All original central non-time channels exactly recovered within float32 rounding; masks and frame count unchanged.',
   no_further_tuning='Report both improvements and regressions; no label/word-dependent scale selection.'))
 rows=[r for r in b.read(b.B/'training_CONTEXT.json')['ordered_rows'] if r['role']=='development_only']
 inputs={v:{k:[] for k in b.KINDS[1:]} for v in b.VIEWS}
 maxcentral=0.;nchecked=0
 for r in rows:
  rec=b.read(b.B/'data/records'/f"{r['clip_id']}.json")
  for v in b.VIEWS:
   xy,pr,ts,span=b.prepare_native(b.arrays(rec['views'][v]['observation']));base=b.build_channels(xy,pr,ts)
   for kind in b.KINDS[1:]:
    q,meta=b.transform(xy,pr,ts,kind);x=q['x'].copy()
    x[:,:102]*=np.float32(q['body_scale']/base['body_scale'])
    off=meta['start_offset'];central=x[off:off+len(xy)]
    error=float(np.max(np.abs(central[:,:279]-base['x'][:,:279])));maxcentral=max(maxcentral,error)
    assert np.allclose(central[:,:279],base['x'][:,:279],rtol=2e-6,atol=2e-6)
    assert np.array_equal(central[1:,279],base['x'][1:,279]);nchecked+=central.size
    inputs[v][kind].append((r,x))
 torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 for name,path in [('C13',b.B/'training_C_20261013_best.pt'),('T3F13',b.T/'T3F_20261013_best.pt')]:
  ck=torch.load(path,map_location='cpu',weights_only=True);m=MaskedSignTCN(**MODEL) if name=='C13' else FrozenCTemporal(**MODEL)
  m.load_state_dict(ck['state_dict'],strict=True);m.eval();saved=b.read(U/f'{name}_STRESS_SCORES.json')
  before={k:v.detach().clone() for k,v in m.state_dict().items()};out={};counts={}
  with torch.inference_mode():
   for v in b.VIEWS:
    out[v]={};counts[v]={}
    for kind,series in inputs[v].items():
     pred=[]
     for j in range(0,len(series),32):
      batch=series[j:j+32];x,p=padded_arrays([xx for r,xx in batch]);zz=m(torch.from_numpy(x),torch.from_numpy(p)).numpy()
      pred.extend(prediction(r,z) for (r,x),z in zip(batch,zz))
     out[v][kind]=pred;old={q['clip_id']:q for q in saved['scores'][v][kind]}
     counts[v][kind]=dict(correct=sum(q['correct'] for q in pred),prior_stress_correct=sum(q['correct'] for q in old.values()),
       gained=sum(q['correct'] and not old[q['clip_id']]['correct'] for q in pred),
       lost=sum(not q['correct'] and old[q['clip_id']]['correct'] for q in pred))
     print(name,v,kind,counts[v][kind],flush=True)
  assert all(torch.equal(before[k],v) for k,v in m.state_dict().items());b.check(U)
  b.write(U/f'{name}_SCALE_FIXED_SCORES.json',dict(status='CAUSAL_DIAGNOSTIC_ONLY',scores=out,counts=counts,
    synthetic=True,model_state_unchanged=True,central_max_error=maxcentral,central_values_checked=nchecked,
    invariant='Only scalar normalization restored; added frames, masks, time phase and model identical to stress.'))
if __name__=='__main__':
 import numpy as np
 main()
