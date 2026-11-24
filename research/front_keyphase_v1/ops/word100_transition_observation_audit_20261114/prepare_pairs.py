from pathlib import Path
import json,hashlib,collections
import numpy as np
from change_core import pair_features
U=Path(__file__).resolve().parent;O=U.parent;S=O/'word100_goal_state_learning_20261114';P=O/'word100_terminal_relation_targets_20261114'
read=lambda p:json.loads(p.read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,d):
 with(U/n).open('x',encoding='utf-8')as f:json.dump(d,f,ensure_ascii=False,indent=2)
audit=read(U/'AUDIT_INPUTS.json')
write('OBSERVATION_REVIEW.json',dict(reviewed_cases=len(audit['rows']),mapping_error_max_px=max(r['mapping_max_px_error']for r in audit['rows']),
 conclusions=[
 'Raw pixel landmarks and saved normalized model coordinates map to the same frame/hand within numerical precision; not a source-index or hand-side swap.',
 'All nine diagnostic hands have finite in-frame points. This availability is NOT evidence that hidden finger joints are accurately located.',
 'In closed fists, projected finger segments overlap; in a flat sweeping hand, normal finger flexion can resemble a partially bent hand. High confidence is not physical 3D truth.',
 'The old shape features omitted the wrist-MCP-PIP base bend. Adding that relation is a testable representation change, not proof of accuracy.',
 'Most importantly, static bent morphology and an ongoing closing action are different labels. A held bent shape must not itself mean unfinished motion.'
 ],full_joint_accuracy_groundtruth=False,quality_scores_not_rethresholded=True,
 direct_reviews={r['clip_id']+':'+str(r['frame'])+':'+r['hand']:dict(visible_morphology=r['label'],
  tracking='No gross whole-hand loss; precise hidden-joint accuracy unresolved',
  stage_truth_from_single_image=False)for r in audit['rows']},
 reviewer='Same Codex direct RGB/keypoint overlay review; no independent human annotation'))
write('CROSSWORD_LABELS.json',dict(rows=[
 dict(record_id=2,word='WORD2551',hand='R',start=56,end=64,change='closing',basis='Curved fingers become a compact folded outline across reviewed 56/58/60/62/64 frames'),
 dict(record_id=2,word='WORD2551',hand='R',start=64,end=68,change='shape_maintained',basis='Folded outline remains while hand/body moves'),
 dict(record_id=2,word='WORD2551',hand='R',start=68,end=72,change='opening',basis='Folded fingers spread and extend; index remains somewhat bent, no all-fingers-straight target'),
 dict(record_id=2,word='WORD2551',hand='R',start=72,end=74,change='shape_maintained',basis='Extended configuration maintained, small normal flexion allowed'),
 dict(record_id=3,word='WORD2551',hand='R',start=68,end=71,change='opening',basis='Reviewed adjacent68/69/70/71 frames show finger opening'),
 dict(record_id=3,word='WORD2551',hand='R',start=71,end=80,change='shape_maintained',basis='Extended configuration persists in71/72/74/76/78/80 frames')],
 no_exact_thumb_or_contact=True,no_word_completion_truth=True,scope='Physical handshape change/maintenance; not per-word educational acceptance',
 images={str(U/f'crossword_record{i:02}.jpg'):sha(U/f'crossword_record{i:02}.jpg')for i in [2,3]}))
src=read(S/'REVIEW_INPUTS.json')['rows'];middle=read(P/'transition_shape/LABELS.json')['rows'];manifest={r['clip_id']:r for r in read(O/'word100_shared_motion_learning_20261109/READY.json')['rows']}
cache={};pins={};rows=[]
def load(cid):
 if cid not in cache:
  m=manifest[cid];p=Path(m['channel']);assert sha(p)==m['channel_sha256'];z=np.load(p)
  key='feedback'if'feedback'in z.files else'full';cache[cid]=(z[key],z['frame_id'],z['timestamp_us']);pins[str(p)]=sha(p)
 return cache[cid]
def add(cid,record,hand,a,b,target,origin,variant='source',reverse=False,freeze=False):
 x,fid,ts=load(cid);ia=int(np.flatnonzero(fid==a)[0]);ib=int(np.flatnonzero(fid==b)[0]);assert ia<ib
 xx=x[ia:ib+1].copy();tt=ts[ia:ib+1].copy()
 if reverse:xx=xx[::-1].copy()
 if freeze:xx[:]=xx[0]
 f,ok=pair_features(xx,hand)
 rows.append(dict(index=len(rows),clip_id=cid,record_id=record,split='fit'if record<=7 else'exposed_probe',hand=hand,
  source_frames=[a,b],target=target,origin=origin,variant=variant,feature=f.tolist(),available=ok,
  native_duration_us=int(tt[-1]-tt[0]),input_indices=[ia,ib],reverse=reverse,freeze=freeze,
  semantic_completion=None,educational_error=None))
groups=collections.defaultdict(list)
for r in src:groups[r['group_id']].append(r)
for gid,rr in groups.items():
 rr=sorted(rr,key=lambda r:r['review_slot']);r=rr[0];cid=r['clip_id'];rec=r['person'];a,b=rr[0]['frame'],rr[-1]['frame'];fam=r['family']
 if fam==2:
  add(cid,rec,'R',a,b,2,'RGB_open_to_closed')
  mids=sorted({v['frame']for v in middle if v['clip_id']==cid and v['hand']=='R'and v['label']=='transition'and a<v['frame']<b})
  if mids:
   mid=mids[0];add(cid,rec,'R',a,mid,2,'RGB_open_to_bent_partial');add(cid,rec,'R',mid,b,2,'RGB_bent_to_closed')
   add(cid,rec,'R',mid,b,1,'constant_bent_pose_control',freeze=True)
 elif fam==3:add(cid,rec,'R',a,b,1,'RGB_closed_shape_during_forearm_raise')
 elif fam in [4,5]:
  for hand in ['R','L']:add(cid,rec,hand,a,b,1,'RGB_maintained_shape_during_arm_motion')
# Append second real expression from FIT only; no probe video is relabelled to
# make the development score improve. Frozen reverse augmentations are marked.
for r in read(U/'CROSSWORD_LABELS.json')['rows']:
 cid=f"NIA_SL_{r['word']}_REAL{r['record_id']:02}_F";add(cid,r['record_id'],r['hand'],r['start'],r['end'],['opening','shape_maintained','closing'].index(r['change']),'RGB_crossword_source')
for r in list(rows):
 if r['target']!=1 and not r['freeze']:
  add(r['clip_id'],r['record_id'],r['hand'],*r['source_frames'],2-r['target'],r['origin'],'time_reverse_control',reverse=True)
write('PAIR_LABELS.json',dict(rows=rows,pins=pins,
 definitions=['opening: fingers visibly move toward more extension','shape_maintained: same configuration during body motion or explicitly constant pose','closing: fingers visibly move toward folding; partial and full both qualify'],
 limitations='Change direction alone is not completion. Reversed/constant controls are not independent videos or learner errors. Local research annotations not educator acceptance labels.'))
cfg=dict(seeds=[42,43,44],steps=800,lr=.01,weight_decay=.001,confidence=.8,
 architecture='40 normalized geometry endpoint differences/means ->16 Tanh ->3; original endpoint/classifier weights unchanged',
 no_time_word_id_input=True,fit_records=list(range(2,8)),exposed_probe_records=[8,9],
 gate='Correct all known source probe changes/maintenance without confusing constant bent posture with changing posture; retain original endpoint outputs. Otherwise diagnostic only.',
 three_seeds_agree=True,checkpoint_selection=False,whole_completion_claim=False,web_changed=False,
 pins={str(U/n):sha(U/n)for n in ['PAIR_LABELS.json','change_core.py','OBSERVATION_REVIEW.json','CROSSWORD_LABELS.json']})
write('FROZEN.json',cfg)
write('CURRENT_EXECUTION.json',dict(status='DATA_REVIEWED_TRAINING_FROZEN',training_running=False))
print(json.dumps(dict(rows=len(rows),counts=dict(collections.Counter(f"{r['split']}:{r['target']}"for r in rows))),ensure_ascii=False))
