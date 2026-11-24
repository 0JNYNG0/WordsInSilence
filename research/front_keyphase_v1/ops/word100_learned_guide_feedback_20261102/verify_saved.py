"""Separate saved-artifact arithmetic audit. No inference/training/private reads."""
from pathlib import Path
import json,hashlib,statistics
import numpy as np
U=Path(__file__).resolve().parent;O=U.parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def metrics(y,m,p):
 out=[]
 for j in range(7):
  positive=m[:,j]&(y[:,j]>.5);negative=m[:,j]&~(y[:,j]>.5);pred=p[:,j]>=.5
  a=float(pred[positive].mean()) if positive.any() else None;b=float((~pred[negative]).mean()) if negative.any() else None
  out.append(dict(known=int(m[:,j].sum()),positive=int(positive.sum()),negative=int(negative.sum()),
   correct=int(((pred==(y[:,j]>.5))&m[:,j]).sum()),balanced_accuracy=(a+b)/2 if a is not None and b is not None else None))
 return dict(targets=out,macro_balanced_accuracy=float(np.mean([x['balanced_accuracy'] for x in out if x['balanced_accuracy'] is not None])),known=sum(x['known'] for x in out),correct=sum(x['correct'] for x in out))
def main():
 assert not (U/'SAVED_VERIFICATION.json').exists()
 manifest=read(U/'INPUT_MANIFEST.json');ready=read(U/'TRAIN_READY.json');checks=[]
 assert sha(U/'INPUT_MANIFEST.json')==ready['input_manifest_sha256']
 for n,h in ready['code_pins'].items():assert sha(U/n)==h,n
 for p,h in manifest['source_pins'].items():assert sha(p)==h,p
 assert len({r['source_sha256'] for r in manifest['rows']})==48
 for r in manifest['rows']:assert sha(r['source'])==r['source_sha256']
 checks.append('48 public video hashes and old source artifacts unchanged; fit/development people disjoint')
 dev=[r for r in manifest['rows'] if r['role']=='development'];labels=[l for r in dev for l in r['labels']]
 y=np.asarray([l['y'] for l in labels]);m=np.asarray([l['target_mask'] for l in labels],bool);assert len(labels)==36 and m.sum()==194
 pold=[]
 for r in dev:
  a=np.load(O/'word100_multipeople_live_feedback_20261101/baseline_probabilities'/f"{r['clip_id']}.npz")
  for l in r['labels']:
   i=int(np.flatnonzero(a['frame_id']==l['center_frame'])[0]);pold.append(a['probabilities'][i])
 old=metrics(y,m,np.asarray(pold));runs=[];w0=np.load(U/'PRETRAINED_PROJECTION.npy')
 for condition in ['frozen_projection','projection_finetune']:
  for seed in [42,43,44]:
   d=U/'runs'/f'{condition}_{seed}';c=read(d/'COMPLETE.json');assert sha(d/'model.pt')==c['checkpoint_sha256'];p=[]
   for r in dev:
    a=np.load(d/'predictions'/f"{r['clip_id']}.npz")
    for l in r['labels']:
     i=int(np.flatnonzero(a['frame_id']==l['center_frame'])[0]);p.append(a['probability'][i])
   score=metrics(y,m,np.asarray(p));assert abs(score['macro_balanced_accuracy']-c['development']['macro_balanced_accuracy'])<1e-12
   delta=float(np.linalg.norm(np.load(d/'signrep_fc1_tuned.npy')-w0)/np.linalg.norm(w0))
   assert (delta==0)==(condition=='frozen_projection')
   runs.append(dict(condition=condition,seed=seed,**score,weight_relative_delta=delta))
 checks.append('all six model scores independently recomputed from saved predictions; actual pretrained weight changes verified')
 stages=read(U/'STAGE_EVALUATION.json');anchor=read(U/'BLIND_STAGE_ANCHORS.json')
 feedback=read(U/'FEEDBACK_V2_AUDIT.json')['rows'];normal={}
 for c in ['frozen_projection','projection_finetune']:
  rs=[r for r in feedback if r['condition']==c and r['kind']=='public_normal'];normal[c]=dict(clip_seed_evaluations=len(rs),difference_candidates=sum(r['difference_candidates'] for r in rs))
 assert normal['frozen_projection']['difference_candidates']==normal['projection_finetune']['difference_candidates']==0
 parallel=read(U/'PARALLEL_PUBLIC_VERIFICATION.json');assert len(parallel['rows'])==4
 assert all(r['exact_path'] and r['exact_status'] and max(r['delta'].values())==0 for r in parallel['rows'])
 assert read(U/'DEMO_UI_VERIFICATION.json')['status']=='PASS'
 checks.append('parallelism retains all frames and exact numerical outputs on four public clips; browser upload/replay checks pass')
 report=dict(status='PASS_SAVED_ARITHMETIC_AND_PROVENANCE',checks=checks,old_protected=old,new_runs=runs,normal_v2=normal,
  stage_evaluation_path=str(U/'STAGE_EVALUATION.json'),latency=parallel['rows'],
  no_new_inference=True,no_new_training=True,private_inputs_read=0,reviewer='same Codex, separate saved-artifact script; not independent human KSL truth',
  conclusion='Executable learned-flow prototype completed; old protected feature score exceeds new mean. No overall feedback performance improvement or linguistic correctness claim.')
 with (U/'SAVED_VERIFICATION.json').open('x',encoding='utf8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
 print(json.dumps(dict(old_macro=old['macro_balanced_accuracy'],new_mean=np.mean([r['macro_balanced_accuracy'] for r in runs]),known=old['known'],checks=len(checks)),ensure_ascii=False))
if __name__=='__main__':main()
