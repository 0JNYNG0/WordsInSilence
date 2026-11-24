from pathlib import Path
import json,hashlib,collections
import numpy as np
from sequence_core import features,sequence
U=Path(__file__).resolve().parent;O=U.parent;P=O/'word100_transition_observation_audit_20261114';T=O/'word100_terminal_relation_targets_20261114'
read=lambda p:json.loads(p.read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,d):
 with(U/n).open('x',encoding='utf-8')as f:json.dump(d,f,ensure_ascii=False,indent=2)
review=read(U/'REVIEW_INPUTS.json')['rows']
write('SOURCE_REVIEW.json',dict(reviewed_sheets=6,reviewed_native_frame_panels=216,
 findings=[
 'All six FIT records show extended-to-folded fingers followed by retained folded configuration during forearm reorientation.',
 'Sweeping hand remains extended with natural flexion; opposite hand retains a folded configuration.',
 'Crossing hands retain four folded fingers while palm views change. Occluded interior fingers are not assigned new pointwise truth.',
 'A changing projected 2D joint angle can accompany a maintained hand configuration; input must include configuration context as well as endpoint differences.'
 ],rows=[dict(pair_index=r['pair_index'],reviewed=True,sequence_target_retained=True,
    hidden_finger_point_truth=None,frame_ids=r['frames'])for r in review],
 review='Same Codex visual review, not human KSL correctness',
 images={str(U/f'fit_review{rec:02}.jpg'):sha(U/f'fit_review{rec:02}.jpg')for rec in range(2,8)}))
manifest={r['clip_id']:r for r in read(O/'word100_shared_motion_learning_20261109/READY.json')['rows']}
cache={};pins={}
def load(cid):
 if cid not in cache:
  m=manifest[cid];p=Path(m['channel']);assert sha(p)==m['channel_sha256'];z=np.load(p)
  cache[cid]=(z['feedback'if'feedback'in z.files else'full'],z['frame_id']);pins[str(p)]=sha(p)
 return cache[cid]
rows=read(P/'PAIR_LABELS.json')['rows'];seq=[];meta=[]
for r in rows:
 x,fid=load(r['clip_id']);a,b=r['input_indices'];xx=x[a:b+1].copy()
 if r['reverse']:xx=xx[::-1].copy()
 if r['freeze']:xx[:]=xx[0]
 ff,ok=sequence(xx,r['hand']);seq.append(ff);meta.append({k:v for k,v in r.items()if k!='feature'})
 assert ok==r['available'];meta[-1]['sequence_available']=ok
pointx=[];pointmeta=[]
for r in read(T/'transition_shape/LABELS.json')['rows']:
 if r['domain']!='shape' or r['label']is None:continue
 x,fid=load(r['clip_id']);ii=np.flatnonzero(fid==r['frame']);assert len(ii)==1
 f,ok=features(x[ii],r['hand']);pointx.append(f[0])
 pointmeta.append(dict(clip_id=r['clip_id'],record_id=r['record_id'],split=r['split'],hand=r['hand'],frame=r['frame'],
   target=['four_open','four_closed','transition'].index(r['label']),label=r['label'],available=bool(ok[0])))
np.savez_compressed(U/'training_arrays.npz',sequence=np.stack(seq),point=np.stack(pointx))
write('LABELS.json',dict(rows=meta,point_rows=pointmeta,pins=pins,
 source_labels={str(P/'PAIR_LABELS.json'):sha(P/'PAIR_LABELS.json'),str(T/'transition_shape/LABELS.json'):sha(T/'transition_shape/LABELS.json')},
 old_probe_exposed=True,unchanged_targets=True,auxiliary_pose_does_not_replace_old_endpoint=True))
write('FROZEN.json',dict(seeds=[42,43,44],steps=1000,lr=.003,weight_decay=.01,aux_weight=.25,
 architecture='hand-normalized 59 features -> shared32 -> temporalConv3 ->GRU24; change3 and auxiliary instantaneous pose3',
 no_duration_word_or_expected_order_input=True,sample_length=16,native_sample_reuse=True,
 confidence=.8,three_seed_agreement=True,no_checkpoint_selection=True,fit_records=list(range(2,8)),
 probe_records=[8,9],baseline='Saved previous 3-seed endpoint difference learner on identical 100 conditions',
 pass_condition='24/24 probe change conditions without abstention and original endpoint unchanged; wider controls before G2 promotion',
 pins={str(U/n):sha(U/n)for n in ['training_arrays.npz','LABELS.json','SOURCE_REVIEW.json','sequence_core.py']}))
print(json.dumps(dict(sequences=len(meta),point_labels=len(pointmeta),input_dim=seq[0].shape[-1])))
