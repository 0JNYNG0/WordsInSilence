from pathlib import Path
import sys,json,hashlib,collections,importlib.util
import numpy as np,torch
from sequence_core import features,decide
from evidence_path import summarize
U=Path(__file__).resolve().parent;O=U.parent;P=O/'word100_terminal_relation_targets_20261114';S=O/'word100_goal_state_learning_20261114';D=U/'state_path'
spec=importlib.util.spec_from_file_location('current_state_path_core',D/'core.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
read=lambda p:json.loads(p.read_text('utf-8-sig'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,d):
 with(U/n).open('x',encoding='utf-8')as f:json.dump(d,f,ensure_ascii=False,indent=2)
assert read(D/'EVALUATION.json')['summary']['preliminary_gate']
write('EXTENDED_FROZEN.json',dict(
 goal='Verify admitted shape-change evidence together with unchanged endpoints on public short/prefix/speed/occlusion controls before action binding',
 expected={'normal':'observed_open_to_folded','short_goal_prefix':'observed_open_to_folded','speed075':'observed_open_to_folded','speed125':'observed_open_to_folded',
  'held_start':'shape_maintenance_observed','goal_only':'shape_maintenance_observed','reverse':'observed_opening_direction','missing_tail':'observation_insufficient','missing_all':'observation_insufficient'},
 partial='No fully observed open-to-folded claim on any reviewed middle-prefix; distinguish visible change from missing observation',
 data='Same two exposed08/09 recordings; edited controls not independent learner errors',
 no_future_after_prefix=True,old_neural_inference_repeated=False,
 pins={str(p):sha(p)for p in [Path(__file__),U/'evidence_path.py',D/'core.py',D/'EVALUATION.json',P/'EVALUATION.json',P/'transition_shape/EVALUATION.json']},
 model_pins={str(D/'models'/f'{s}.pt'):sha(D/'models'/f'{s}.pt')for s in [42,43,44]}))
arr=np.load(S/'source_arrays.npz');old=read(P/'EVALUATION.json');selected=[r for r in old['rows']if r['split']=='exposed_probe'and r['requirement']=='hand_close_to_upper_body']
models=[];torch.set_num_threads(2)
for seed in [42,43,44]:
 m=mod.StatePath();m.load_state_dict(torch.load(D/'models'/f'{seed}.pt',map_location='cpu',weights_only=True));m.eval();models.append(m)
def ctx(f,i):
 out=np.zeros((7,f.shape[-1]),np.float32)
 for j,k in enumerate(range(i-3,i+4)):
  if 0<=k<len(f):out[j]=f[k]
 return out
calls=0
def run(cid,source_ids,ts,variant,oldstates,oldavailable,prefix=False):
 global calls
 source=arr[cid+'_x'];ids=np.asarray(source_ids,int);a,b=int(ids.min()),int(ids.max())
 # The recording really stops at a prefix cut. Do not fetch its absent future.
 full=source[:b+1].copy()if prefix else source.copy();query=ids.copy()
 if variant in ['held_start','goal_only']:
  full=np.repeat(source[ids[:1]],len(ids),axis=0);query=np.arange(len(ids))
 elif variant=='reverse':
  full=full[::-1].copy();query=len(full)-1-ids
 def erase(indices):
  full[indices,60:102]=0;full[indices,144:186]=0;full[indices,216:237]=0;full[indices,258:279]=0
 if variant=='missing_all':erase(np.arange(len(full)))
 elif variant=='missing_tail':erase(np.arange(query[-max(2,len(query)//3)],len(full)))
 elif variant in ['middle_gap2','middle_long_gap']:
  # Reuse the already-declared mask from the prior saved comparison exactly.
  erase(query[~np.asarray(oldavailable,bool)])
 f,ok=features(full,'R');q=np.stack([ctx(f,int(i))for i in query]);pair=torch.tensor(np.stack([q[0],q[-1]])[None]);pot=[];probs=[]
 with torch.no_grad():
  for m in models:
   logits,_=m(pair);probs.append(logits.softmax(-1).numpy()[0]);pot.append(m.position(torch.tensor(q)).numpy());calls+=1
 available=ok[query]&np.asarray(oldavailable,bool)
 change=decide(np.asarray(probs),bool(available[0]and available[-1]and available.mean()>=.8))
 ev=summarize(change,np.asarray(pot),oldstates,available,ts)
 return dict(clip_id=cid,variant=variant,source_frames=ids.tolist(),input_source_last_frame=b if prefix else len(source)-1,
   prefix_future_excluded=prefix,change=change,positions=np.asarray(pot).tolist(),available=available.tolist(),probabilities=np.asarray(probs).tolist(),evidence=ev)
out=[]
for r in selected:
 out.append(run(r['clip_id'],r['source_frames'],r['timestamps_us'],r['variant'],r['states']['shape'],r['available']['shape'],r['variant']=='short_goal_prefix'))
prefixes=read(P/'transition_shape/EVALUATION.json')['prefixes'];dense=old['dense']
for r in prefixes:
 cid=r['clip_id'];normal=next(v for v in selected if v['clip_id']==cid and v['variant']=='normal');ids=list(range(normal['source_frames'][0],r['frame']+1));n=len(ids)
 out.append(run(cid,ids,normal['timestamps_us'][:n],'reviewed_partial_prefix',dense[cid]['R']['shape']['states'][ids[0]:ids[-1]+1],dense[cid]['R']['shape']['available'][ids[0]:ids[-1]+1],True))
cfg=read(U/'EXTENDED_FROZEN.json');checks=[]
for r in out:
 if r['variant']in cfg['expected']:
  checks.append(dict(clip_id=r['clip_id'],variant=r['variant'],expected=cfg['expected'][r['variant']],actual=r['evidence']['status'],passed=r['evidence']['status']==cfg['expected'][r['variant']]))
 elif r['variant']=='reviewed_partial_prefix':checks.append(dict(clip_id=r['clip_id'],variant=r['variant'],last=r['source_frames'][-1],expected='not_full_shape_change',actual=r['evidence']['status'],passed=not r['evidence']['observed_shape_change_to_folded']))
write('EXTENDED_COMPARISON.json',dict(rows=out,checks=checks,all_passed=all(r['passed']for r in checks),new_model_calls=calls,old_model_calls=0,web_changed=False,
   whole_action_correctness=None,source_video_count=2))
print(json.dumps(dict(cases=len(out),checks=len(checks),passed=sum(r['passed']for r in checks),failures=[r for r in checks if not r['passed']]),ensure_ascii=False))
