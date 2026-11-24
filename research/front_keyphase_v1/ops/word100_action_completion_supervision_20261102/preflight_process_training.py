"""Actual FIT input/label contract and untrained tensor-shape preflight once.

No optimizer, DEV predictions, old model calls or accuracy measurement.
"""
from pathlib import Path
from collections import Counter
import json,hashlib,traceback
import numpy as np
import torch
from process_head import ProcessEvidenceNet,QUERIES,make_input,batch
from evaluate_process_head import load_case,sha,read,write
U=Path(__file__).resolve().parent
def main():
 dest=U/'PROCESS_ACTUAL_INPUT_PREFLIGHT.json';assert not dest.exists()
 targets=read(U/'PROCESS_VIEW_TARGETS.json')['rows']
 inputs={r['pixel_view_key']:r for r in read(U/'PROCESS_INPUT_MANIFEST.json')['rows']}
 cache={};groups=set();points=0;intervals=0
 for r in targets:
  assert r['role']=='fit' and 1<=r['recorded_person_id']<=9
  a=load_case(r,inputs[r['pixel_view_key']]);cache[r['pixel_view_key']]=a
  groups.add((r['recorded_person_id'],r['physical_query']))
  points+=len(r['sparse_relation_targets']);intervals+=len(r['fully_retained_motion_interval_targets'])
 assert (len(targets),len(cache),len(groups),points,intervals)==(217,216,45,2360,472)
 valid=np.concatenate([a['rgb'][a['image_available']] for a in cache.values()])
 stats=dict(mean=valid.mean(0,dtype=np.float64),std=np.maximum(valid.std(0,dtype=np.float64),.001))
 seq={k:make_input(a,stats) for k,a in cache.items()}
 assert all(np.isfinite(x).all() and x.shape[1]==887 for x in seq.values())
 # Short and long actual tensors only test execution dimensions. Random model
 # scores are deliberately not stored, compared with truth, or selected on.
 keys=sorted(seq,key=lambda k:len(seq[k]));chosen=[keys[0],keys[-1]]
 torch.manual_seed(42);torch.set_num_threads(2);model=ProcessEvidenceNet().eval()
 x,l=batch([seq[k] for k in chosen]);q=torch.tensor([0,0])
 with torch.no_grad():
  p=model(x,l,q);ip=model.interval_logits(x,l,q)
  assert p['process_logits'].shape==(2,) and p['point_logits'].shape==(2,int(l.max()),2)
  assert torch.isfinite(ip).all() and all(torch.isfinite(v).all() for v in p.values())
 # Same raw blank observations must yield no geometry or RGB input evidence.
 blankrows=0
 for k,a in cache.items():
  blank=~a['image_available'];blankrows+=int(blank.sum())
  assert np.array_equal(seq[k][blank],np.zeros((int(blank.sum()),887),np.float32))
 write(dest,dict(status='PASS_ACTUAL_FIT_INPUT_AND_UNTRAINED_EXECUTION',views=217,inputs=216,source_queries=45,
  sparse_relations=points,intervals=intervals,blank_rows=blankrows,input_dimensions=887,
  process_labels=dict(Counter(str(r['observed_process_evidence']) for r in targets)),
  untrained_forward_cases=2,optimization_steps=0,development_accessed=False,accuracy_measured=False,
  old_models_called=0,pins={str(U/n):sha(U/n) for n in ['process_head.py','train_process_head.py','evaluate_process_head.py',
   'preflight_process_training.py','PROCESS_VIEW_TARGETS.json','PROCESS_INPUT_MANIFEST.json']}))
 print('Actual FIT217/216 input preflight PASS; untrained tensor execution only, no optimization or accuracy result')
if __name__=='__main__':
 try:main()
 except BaseException:
  p=U/'PROCESS_ACTUAL_PREFLIGHT_FIRST_FAILURE.txt'
  if not p.exists():p.write_text(traceback.format_exc(),encoding='utf8')
  raise
