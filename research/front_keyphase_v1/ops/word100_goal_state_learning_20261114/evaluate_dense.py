"""Add a fixed dense-source replay, without training or threshold selection."""
from pathlib import Path
import json,hashlib,collections,time
import numpy as np
import torch
from state_model import features,consensus,transition_evidence,StateHeads
U=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(name,value):
 with (U/name).open('x',encoding='utf-8')as f:json.dump(value,f,ensure_ascii=False,indent=2)
plan=dict(reason='Initial evaluation used sparse reviewed snapshots. Replay actual cached contiguous source frames before claiming temporal usefulness.',
 no_retraining=True,no_selection=True,no_threshold_change=True,confidence=.8,
 variants=['normal','short_reviewed_goal_prefix','speed075','speed125','missing_middle2','missing_after_start','all_missing','only_start_pose','only_goal_pose','reverse'],
 expected='Ordered state evidence for source/short/speed/middle loss; unknown for masked-tail/all; goal-pose-only and before-goal for static controls. Reversal scores only noncyclic components.',
 limitations='Public edited controls, not genuine learner errors. No minimum duration/repetition or whole-sign correctness.',
 parents={str(U/p):sha(U/p)for p in ['EVALUATION.json','TRAINING_COMPLETE.json','TRAINING_FROZEN.json','STATE_LABELS.json']})
write('DENSE_REPLAY_FROZEN.json',plan)
meta=read(U/'LEARNING_INPUTS.json');lab={(r['clip_id'],r['frame'],r['domain'],r['hand']):r for r in meta['rows']}
models=[];torch.set_num_threads(2)
for r in read(U/'TRAINING_COMPLETE.json')['rows']:
 if r['kind']=='mlp':
    m=StateHeads('mlp');m.load_state_dict(torch.load(r['model'],weights_only=True,map_location='cpu'));m.eval();models.append(m)
groups=collections.defaultdict(list)
for r in read(U/'REVIEW_INPUTS.json')['rows']:groups[r['group_id']].append(r)
domains={2:'forearm',3:'forearm',4:'arm_region',5:'height',7:'separation'}
paths={2:['wrist_above_elbow','forearm_across'],3:['forearm_across','wrist_above_elbow'],4:['upper_arm_side','wrist_side'],5:['active_hand_higher','opposite_hand_higher'],7:['hands_near','hands_apart']}
data=np.load(meta['arrays']);rows=[];skipped=[]
for gid,rr in groups.items():
 rr=sorted(rr,key=lambda r:r['review_slot']);r=rr[0];d=domains[r['family']];path=paths[r['family']].copy()
 truth=[lab[v['clip_id'],v['frame'],d,'R']['label']for v in rr]
 if r['family']==5 and truth[0]==path[1]:path=path[::-1]
 if not all(s in truth for s in path):skipped.append(gid);continue
 start=next(i for i,s in enumerate(truth)if s==path[0]);goal=next(i for i,s in enumerate(truth)if s==path[1])
 if goal<=start:skipped.append(gid);continue
 ids=data[r['clip_id']+'_frame_id'];x=data[r['clip_id']+'_x'];a=lab[r['clip_id'],rr[start]['frame'],d,'R']['array_index'];b=lab[r['clip_id'],rr[-1]['frame'],d,'R']['array_index'];g=lab[r['clip_id'],rr[goal]['frame'],d,'R']['array_index']
 raw=x[a:b+1];src=ids[a:b+1];n=len(raw)
 cases=[('normal',raw.copy(),src.copy(),'ordered_state_change_observed'),
    ('short_reviewed_goal_prefix',x[a:g+1].copy(),ids[a:g+1].copy(),'ordered_state_change_observed')]
 for rate in [.75,1.25]:
    ix=np.unique(np.r_[np.rint(np.arange(0,n,rate)).astype(int).clip(0,n-1),0,n-1,g-a])
    # rate .75 keeps native frames here; slowed timestamps cannot create observations.
    cases.append(('speed075'if rate==.75 else'speed125',raw[ix].copy(),src[ix].copy(),'ordered_state_change_observed'))
 mid=raw.copy();lo=max(1,n//2-1);hi=min(n-1,lo+2);mid[lo:hi,186:237]=0
 cases.append(('missing_middle2',mid,src.copy(),'ordered_state_change_observed'))
 tail=raw.copy();tail[1:,186:237]=0;cases.append(('missing_after_start',tail,src.copy(),'observation_insufficient'))
 blank=raw.copy();blank[:,186:237]=0;cases.append(('all_missing',blank,src.copy(),'observation_insufficient'))
 cases.extend([('only_start_pose',np.repeat(raw[:1],n,axis=0),np.repeat(src[:1],n),'observed_before_goal'),
    ('only_goal_pose',np.repeat(x[g:g+1],n,axis=0),np.repeat(ids[g:g+1],n),'goal_pose_only')])
 if r['family']!=5:cases.append(('reverse',raw[::-1].copy(),src[::-1].copy(),'order_not_supported'))
 for name,arr,frames,expected in cases:
    t=time.perf_counter();ff,ok=features(arr,d,'R')
    with torch.no_grad():pp=[torch.softmax(m(torch.from_numpy(ff),d),-1).numpy()for m in models]
    states=consensus(pp,ok,d,.8);res=transition_evidence(states,path,ok)
    rows.append(dict(group_id=gid,clip_id=r['clip_id'],person=r['person'],split=r['split'],family=r['family'],domain=d,
       variant=name,source_frames=frames.tolist(),states=states,available=ok.tolist(),path=path,
       expected=expected,result=res,correct=res['status']==expected,seconds=time.perf_counter()-t))
summary=[]
for split in ['fit','exposed_probe']:
 for d in domains.values():
  if any(s['split']==split and s['domain']==d for s in summary):continue
  for variant in plan['variants']:
    rr=[r for r in rows if(r['split'],r['domain'],r['variant'])==(split,d,variant)]
    if rr:summary.append(dict(split=split,domain=d,variant=variant,n=len(rr),correct=sum(r['correct']for r in rr)))
write('DENSE_REPLAY.json',dict(rows=rows,summary=summary,unscored=skipped,plan_sha256=sha(U/'DENSE_REPLAY_FROZEN.json'),
    automatic_localization=False,real_user_errors=False,semantic_completion=None,web_changed=False))
print(json.dumps([s for s in summary if s['split']=='exposed_probe'],ensure_ascii=False))
