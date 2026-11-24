"""Separate arithmetic for this diagnosis. Does not call its implementation or a model."""
from pathlib import Path
import json, hashlib
from collections import Counter
import numpy as np
U=Path(__file__).resolve().parent; P=U.parent/'word100_assignment_training_20261018'; B=U.parent/'word100_body_boundary_20261017'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
audit=read(U/'EXECUTION_AUDIT.json');assert not (U/'VERIFICATION.json').exists()
for p,h in audit['source_pins'].items():assert sha(p)==h
rows={r['clip_id']:r for r in read(P/'PREP_CONTEXT.json')['rows']};diag=read(U/'INPUT_DECOMPOSITION.json');counts={a:Counter() for a in ['P','Q']}
max_ratio_error=0.;scale_count=0;corner_checks=0
for e in diag['details']:
 r=rows[e['clip']];assert r['role']=='train_candidate';v=e['view'];t=e['temporal'];a=e['arm'];rec=read(P/'records'/f"{e['clip']}.json")
 old=load(r['views'][v]['observation']);c=load(r['views'][v]['channels']);new=load(rec['views'][v][a]);ids=c[t+'_frame_id']
 ix=np.searchsorted(old['frame_id'],ids);iy=np.searchsorted(new['frame_id'],ids);ox=old['xy_px'][ix];nx=new['xy_px'][iy];om=old['usable'][ix];nm=new['usable'][iy]
 eq=np.all(om==nm,axis=1)&np.all(np.isclose(ox,nx,rtol=0,atol=0,equal_nan=True),axis=(1,2));delta=new['x_'+t].astype(float)-c['x_'+t];changed=np.any(delta!=0,axis=1);spread=changed&eq
 assert ids[~eq].tolist()==e['direct_changed_frames'] and ids[spread].tolist()==e['scale_propagated_frames']
 cc=counts[a];cc['tensor_entries']+=1;cc['tensor_frames']+=len(ids);cc['direct_changed_frames']+=int((~eq).sum());cc['all_changed_frames']+=int(changed.sum());cc['scale_only_unchanged_frames']+=int(spread.sum());cc['changed_values']+=int(np.count_nonzero(delta));cc['spread_values']+=int(np.count_nonzero(delta[spread]))
 for side,base in enumerate([33,54]):
  scales=[]
  for xy,m,tag in [(ox,om,'old'),(nx,nm,'new')]:
   good=m[:,base,None]&m[:,[base+j for j in [5,9,13,17]]]
   d=np.sqrt(np.sum((xy[:,[base+j for j in [5,9,13,17]]]-xy[:,base,None])**2,axis=2))[good].astype(float)
   hs=float(np.median(d)) if len(d) and np.median(d)>1e-6 else None
   expected=e[tag+'_scales'][side]
   assert (hs is None)==(expected is None)
   if hs is not None:assert abs(hs-expected)<1e-5
   assert int(good.sum())==e[tag+'_support'][side]['samples'];scales.append(hs);scale_count+=1
  if all(z is not None for z in scales):
   sl=slice(102+42*side,144+42*side);o=c['x_'+t][eq,sl].astype(float);n=new['x_'+t][eq,sl].astype(float)
   if len(o):
    err=float(np.abs(n-o*(scales[0]/scales[1])).max());max_ratio_error=max(max_ratio_error,err);assert np.allclose(n,o*(scales[0]/scales[1]),rtol=2e-6,atol=2e-6)
   corner_checks+=1
assert {a:dict(c) for a,c in counts.items()}==diag['summary']
scores=read(U/'SAVED_SCORE_DIAGNOSIS.json');score_checks=0
for e in scores['paired']:
 a,b=e['comparison'].split('->');v=e['view'];counter=Counter()
 for s in [20261013,20261014,20261015]:
  def preds(arm):
   p=B/f'training_C_{s}_RESULT.json' if arm=='C' else P/f'{arm}_{s}_RESULT.json'
   return {x['clip_id']:int(np.argmax(x['logits']))==x['label_index'] for x in read(p)['predictions'][v]}
  x,y=preds(a),preds(b)
  for cid in x:counter['gain' if y[cid] and not x[cid] else 'loss' if x[cid] and not y[cid] else 'both_correct' if x[cid] else 'both_wrong']+=1;score_checks+=1
 assert dict(counter)==e['counts']
inf=read(U/'FROZEN_C_TRAIN_SENSITIVITY.json');margins=0
for e in inf['items']:
 assert rows[e['clip']]['role']=='train_candidate'
 for z in e['variants'].values():
  logits=z['logits'];y=e['label_index'];m=logits[y]-max(x for i,x in enumerate(logits) if i!=y)
  assert abs(m-z['target_margin'])<3e-6 and int(np.argmax(logits))==z['prediction'];margins+=1
assert margins==inf['forward_samples'] and inf['DEV_inference']==0 and inf['gradient_steps']==0
out={'status':'PASS','scope':'Same Codex, separate arithmetic; not independent human/agent validation. No inference repeated.',
 'pinned_files_checked':len(audit['source_pins']),'tensor_entries':len(diag['details']),'hand_scale_checks':scale_count,'unchanged_observation_ratio_checks':corner_checks,'max_ratio_error':max_ratio_error,'saved_score_pairs_checked':score_checks,'sensitivity_logits_checked':margins,
 'code_sha256':sha(__file__),'result_pins':{n:sha(U/n) for n in ['CONTRACT.json','INPUT_DECOMPOSITION.json','SAVED_SCORE_DIAGNOSIS.json','TRAIN_SCALE_INVENTORY.json','FROZEN_C_TRAIN_SENSITIVITY.json','EXECUTION_AUDIT.json']}}
(U/'VERIFICATION.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8');print(json.dumps(out,ensure_ascii=False))
