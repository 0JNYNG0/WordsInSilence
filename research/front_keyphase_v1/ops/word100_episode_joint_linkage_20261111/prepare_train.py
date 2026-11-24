from joint_core import *
from collections import Counter
assert not (U/'FROZEN.json').exists()
rr=[r for r in old.rows() if r['role']=='fit'];ann=read(U/'ALL_FIT_BOUNDARIES.json')['rows'];am={}
checks=[]
def ck(k,v):checks.append(dict(name=k,passed=bool(v)));assert v,k
ck('common899_100',len(rr)==899 and len({r['class_index'] for r in rr})==100)
for a in ann:
 ck('source_and_sheet',sha(a['sheet'])==a['sheet_sha256']);am.setdefault(a['clip_id'],[]).append(a)
counts=Counter();pc=Counter();quality=Counter();events=0;positive=np.zeros(3);total=np.zeros(3)
for r in rr:
 r['annotations']=am.get(r['clip_id'],[])
 for a in r['annotations']:ck('matching_source',r['source_sha256']==a['source_sha256']);events+=len(a['events'])
 by,bw=bc.targets(r,r['annotations']);positive+=(by*bw).sum((0,1));total+=bw.sum((0,1));ss,yy,ps,py=example_specs([r]);quality.update(yy);pc.update(py)
 counts['fine_clips']+=bool(r['annotations'])
 (U/'boundary_labels').mkdir(exist_ok=True)
 if r['annotations']:np.savez_compressed(U/'boundary_labels'/f"{r['clip_id']}.npz",y=by,w=bw)
 evalrows=read(U/'EVALUATION_LABELS.json')['rows']
ck('eval_no_person_overlap',not {r['person'] for r in rr}&{r['person'] for r in evalrows})
ck('eval_no_source_overlap',not {r['source_sha256'] for r in rr}&{r['source_sha256'] for r in evalrows})
pins={str(p):sha(p) for p in [U/'ALL_FIT_BOUNDARIES.json',U/'EVALUATION_LABELS.json',U/'NEW_OBSERVED_LABELS.json',U/'REVIEW_INPUTS.json',P/'runs/42/model.pt',P/'FULLPATH_COMPLETE.json',L/'GUARDED_COMPLETE.json']}
protections=read(P/'FROZEN.json')['protections'];protections.update({str(p):sha(p) for p in P.rglob('*') if p.is_file() and p.suffix in ['.json','.py','.pt']})
for p,h in protections.items():ck('protected_current',sha(p)==h)
cfg=dict(seeds=[42,43,44],primary_seed=42,epochs=10,steps=38,lr_encoder=.0001,lr_head=.001,common=899,pos_weight=np.clip((total-positive)/np.maximum(positive,1),1,12).tolist(),confidence=.65,probe_seed=42,probe_epochs=6,probe_heldout_annotation_ids=[3,7],probe_scope='Common encoder has previously seen FIT records. New fine losses and new quality/pair heads withhold record IDs03/07; this is annotation generalization, not globally unseen people.',quality_classes=['reviewed_nonmotion','fragment_of_one_action','complete_one_action','multiple_actions'],pair_classes=['same_action_fragments','next_repetition','different_axis_action'],pins=pins,protections=protections,selection='No model/threshold/epoch selection on evaluation. Preset seed42 primary; final3seeds. Module ablations share jointly trained weights.',adoption='Normal/speed/crossing no added warnings, no lost existing shape/direction, one-to-one automatic actions and downstream errors improve, no all-unknown success. No operational replacement.')
write(U/'FROZEN.json',cfg);write(U/'DATA_AUDIT.json',dict(common_videos=899,common_words=100,boundary_clips=counts['fine_clips'],boundary_rows=len(ann),motion_events=events,role_clips=36,derived_quality_samples=dict(quality),derived_pair_samples=dict(pc),extra_public_originals=10,eval_motion_events=sum(len(r['events']) for r in evalrows),labels_are_not_independent_expert=True,private_data=0));write(U/'PRETRAIN_CHECK.json',dict(passed=True,checks=checks));event('prepared',events=events,quality=dict(quality),pairs=dict(pc))
