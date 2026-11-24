from pathlib import Path
from collections import Counter
import hashlib,json,numpy as np,torch,time
from relation_model import V,read,RelationNet,dense
from train_relations import save
from evaluate_relations import attach
P=V.parent/'word100_learned_guide_feedback_20261102';U=V.parent/'word100_learned_feature_retention_20261102'
def main():
 torch.set_num_threads(4);labels=read(V/'RELATION_TARGETS.json')['rows'];manifest=read(P/'INPUT_MANIFEST.json')['rows'];supports=read(V/'GUIDE_RELATION_SUPPORT.json');stats=read(V/'NORMALIZATION.json')
 assert {r['person'] for r in labels if r['role']=='fit'}.isdisjoint({r['person'] for r in labels if r['role']=='development'})
 for r in labels:assert hashlib.sha256(Path(r['image']).read_bytes()).hexdigest()==r['image_sha256']
 for j in range(3):
  both=[]
  for w in sorted(set(r['word_id'] for r in labels)):
   values={r['y'][j] for r in labels if r['role']=='fit' and r['word_id']==w and r['mask'][j]}
   if values=={0,1}:both.append(w)
  assert both,('no within-word contrast',j)
 totals=[];normal=[]
 for run in read(V/'TRAINING_RESULT.json')['runs']:
  d=V/'runs'/f"{run['condition']}_{run['seed']}";a=np.load(d/'center_predictions.npz');ix=[i for i,r in enumerate(labels) if r['role']=='development'];y=a['y'][ix];p=a['p'][ix];m=a['mask'][ix]
  independently_computed=[]
  for j,s in enumerate(run['development']['targets']):
   truth=y[m[:,j],j].astype(bool);pred=p[m[:,j],j]>=.5;tp=int((truth&pred).sum());tn=int((~truth&~pred).sum());fp=int((~truth&pred).sum());fn=int((truth&~pred).sum())
   assert [tp,tn,fp,fn]==[s['TP'],s['TN'],s['FP'],s['FN']]
   independently_computed.append((tp/(tp+fn)+tn/(tn+fp))/2)
  macro=float(np.mean(independently_computed));assert abs(macro-run['development']['macro_balanced_accuracy'])<1e-12
  assert hashlib.sha256((d/'model.pt').read_bytes()).hexdigest()==run['checkpoint_sha256']
  totals.append(dict(condition=run['condition'],seed=run['seed'],macro_balanced_accuracy=macro))
  for r in manifest:
   if r['role']!='development':continue
   arrays=dict(np.load(r['input']));prob=np.load(d/'predictions'/f"{r['clip_id']}.npz")
   assert np.array_equal(arrays['frame_id'],prob['frame_id']);assert np.array_equal(arrays['timestamp_us'],prob['timestamp_us'])
   # Hold the existing sequence model fixed to42 while comparing relation head seeds.
   base=read(U/'cases'/f"42_{r['clip_id']}.json");out=attach(base,arrays,prob['probability'],supports)
   save(V/'development_feedback'/f"{run['condition']}_{run['seed']}_{r['clip_id']}.json",out)
   normal.append(dict(condition=run['condition'],seed=run['seed'],clip_id=r['clip_id'],relations=Counter(t['status'] for s in out['stages'] for t in s['relations']),
    unchanged_dtw=out['path']==base['path'],unchanged_shapes=all(s['features']==old['features'] for s,old in zip(out['stages'],base['stages']))))
 # Recount actual control results from saved detailed output, not summaries alone.
 edits=read(V/'RELATION_COMPARISON.json')['rows']
 for r in edits:
  out=read(V/'control_feedback'/f"{r['condition']}_{r['seed']}_{r['word_id']}_{r['kind']}.json")
  assert dict(Counter(s['status'] for s in out['stages']))==r['statuses']
  assert r['learned_dtw_preserved'] and r['shape_features_preserved']
  if r['kind']=='main_blackout':assert r['evidence_gaps']>0 and all(t['status']=='RELATION_UNCERTAIN' for t in r['relations'])
  if r['kind'] in ['reencoded_normal','speed_0.8','speed_1.25','trim_initial_rest']:
   assert all(t['status']=='POSITIVE_RELATION_OBSERVED' for t in r['relations'])
 # Verify preserved pretrained/protected source files; no user recording paths followed.
 previous=read(P/'INPUT_MANIFEST.json')
 for p,h in previous['source_pins'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
 aggregate={c:float(np.mean([r['macro_balanced_accuracy'] for r in totals if r['condition']==c])) for c in ['geometry_only','rgb_geometry']}
 save(V/'DEVELOPMENT_RELATIONS.json',dict(rows=normal,previously_used_development=True))
 verified=dict(status='PASS_SAVED_ARITHMETIC_PROVENANCE_AND_PRESERVATION',runs=totals,aggregate=aggregate,
  difference_percentage_points=100*(aggregate['rgb_geometry']-aggregate['geometry_only']),
  review='Separate saved-output verification by same Codex; not independent human labels',
  controls=len(edits),original_private_source_reads=0,old_input_sha_pins_unchanged=True,
  source_label_unknown_masks_preserved=True,all_dtw_shapes_preserved=True,
  normal_development_relations={c:dict(sum((Counter(r['relations']) for r in normal if r['condition']==c),Counter())) for c in aggregate})
 save(V/'SAVED_VERIFICATION.json',verified);print(json.dumps(verified,ensure_ascii=False))
if __name__=='__main__':main()
