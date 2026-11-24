"""One fixed dense comparison. Reuse its saved output, never select on probe."""
from pathlib import Path
import json,hashlib,collections,time
import numpy as np
import torch
from core import TraitHeads,features,consensus,assess,CLASSES
U=Path(__file__).resolve().parent;O=U.parent;S=O/'word100_goal_state_learning_20261114'
read=lambda p:json.loads(p.read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,d):
 with(U/n).open('x',encoding='utf-8')as f:json.dump(d,f,ensure_ascii=False,indent=2,default=lambda a:a.item()if isinstance(a,np.generic)else str(a))
cfg=read(U/'FROZEN.json');lab=read(U/'LABELS.json')['rows'];done=read(U/'TRAINING_COMPLETE.json')
assert not(U/'EVALUATION.json').exists();torch.set_num_threads(2)
models=[];recovery=(U/'DENSE_PROBABILITIES.npz').exists()
cached=np.load(U/'DENSE_PROBABILITIES.npz')if recovery else None
for row in done['rows']:
 assert sha(Path(row['model']))==row['sha256']
 if not recovery:
  m=TraitHeads();m.load_state_dict(torch.load(row['model'],weights_only=True,map_location='cpu'));m.eval();models.append(m)
arr=np.load(S/'source_arrays.npz');dense={};timings=[];probabilities={}
clips=sorted({r['clip_id']for r in lab})
for cid in clips:
 x=arr[cid+'_x'];t=time.perf_counter();dense[cid]={}
 for hand in ['R','L']:
  dense[cid][hand]={}
  for d in CLASSES:
   ff,ok=features(x,d,hand)
   if recovery:pp=cached[cid+'_'+hand+'_'+d]
   else:
    with torch.no_grad():pp=np.stack([m(torch.from_numpy(ff),d).softmax(-1).numpy()for m in models])
   states=consensus(pp,ok,d);dense[cid][hand][d]=dict(states=states,available=ok.tolist())
   probabilities[cid+'_'+hand+'_'+d]=pp
 timings.append(dict(clip_id=cid,frames=len(x),seconds=None if recovery else time.perf_counter()-t))
if not recovery:np.savez_compressed(U/'DENSE_PROBABILITIES.npz',**probabilities)
state_rows=[]
for r in lab:
 pred=dense[r['clip_id']][r['hand']][r['domain']]['states'][r['array_index']]
 state_rows.append(dict(index=r['index'],split=r['split'],domain=r['domain'],label=r['label'],predicted=pred,
    observed=r['available'],correct=pred==r['label']if r['label']is not None else None))
groups=collections.defaultdict(list)
for r in read(S/'REVIEW_INPUTS.json')['rows']:
 if r['family']in [2,3]:groups[r['group_id']].append(r)
labels={(r['clip_id'],r['frame'],r['domain'],r['hand']):r['label']for r in lab}
rows=[]
for gid,rr in sorted(groups.items()):
 rr=sorted(rr,key=lambda r:r['review_slot']);r=rr[0];cid=r['clip_id'];reqname='hand_close_to_upper_body'if r['family']==2 else'raise_closed_forearm'
 req=cfg['requirements'][reqname];fids=arr[cid+'_frame_id'];ts=arr[cid+'_timestamp_us']
 ii=[int(np.flatnonzero(fids==p['frame'])[0])for p in rr];a,b=ii[0],ii[-1]
 direct_goal=[i for i,v in zip(ii,rr)if all(labels.get((cid,v['frame'],d,'R'))==value for d,value in req['goal'].items())]
 assert direct_goal;g=direct_goal[0];base=np.arange(a,b+1);n=len(base)
 cases=[('normal',base,1.,None,'required_feature_sequence_observed'),
        ('short_goal_prefix',np.arange(a,g+1),1.,None,'required_feature_sequence_observed'),
        ('speed075',base,.75,None,'required_feature_sequence_observed'),
        ('speed125',base,1.25,None,'required_feature_sequence_observed'),
        ('held_start',np.repeat(a,n),1.,None,'still_at_start_state'),
        ('goal_only',np.repeat(g,n),1.,None,'goal_posture_without_start'),
        ('reverse',base[::-1],1.,None,'reverse_feature_order'),
        ('missing_tail',base,1.,list(range(1,n)),'observation_uncertain'),
        ('missing_all',base,1.,list(range(n)),'observation_uncertain'),
        ('middle_gap2',base,1.,list(range(max(1,n//2-1),min(n-1,n//2+1))),None),
        ('middle_long_gap',base,1.,list(range(1,n-1)),'endpoint_pair_only')]
 for name,ix,speed,missing,expected in cases:
  # Timing edits keep the exact observed pose sequence. No interpolation of
  # hidden joints. Synthetic controls do not enlarge the real video count.
  nt=ts[base[:len(ix)]].copy() if len(ix)==n else ts[ix].copy()
  nt=(nt[0]+np.rint((nt-nt[0])/speed)).astype('int64')
  states={d:[dense[cid]['R'][d]['states'][i]for i in ix]for d in req['start'].keys()|req['goal'].keys()}
  av={d:np.array(dense[cid]['R'][d]['available'])[ix].copy()for d in states}
  if missing:
   for d in states:
    for k in missing:states[d][k]=None;av[d][k]=False
  result=assess(states,av,nt,req)
  rows.append(dict(group_id=gid,clip_id=cid,record_id=r['person'],split=r['split'],family=r['family'],
    requirement=reqname,variant=name,source_frames=fids[ix].tolist(),timestamps_us=nt.tolist(),
    states=states,available={d:q.tolist()for d,q in av.items()},result=result,expected=expected,
    correct=result['status']==expected if expected else None,real_learner_error=False))
summary=[]
for split in ['fit','exposed_probe']:
 for reqname in cfg['requirements']:
  for name in cfg['cases']:
   rr=[r for r in rows if(r['split'],r['requirement'],r['variant'])==(split,reqname,name)]
   summary.append(dict(split=split,requirement=reqname,variant=name,n=len(rr),
       counts=dict(collections.Counter(r['result']['status']for r in rr)),
       expected_matches=sum(r['correct']is True for r in rr),scored=sum(r['correct']is not None for r in rr)))
state_summary=[]
for split in ['fit','exposed_probe']:
 for d in CLASSES:
  rr=[r for r in state_rows if(r['split'],r['domain'])==(split,d)and r['label']is not None]
  state_summary.append(dict(split=split,domain=d,total=len(rr),supported_correct=sum(r['correct']for r in rr),
    wrong=sum(r['predicted']is not None and not r['correct']for r in rr),unknown=sum(r['predicted']is None for r in rr)))
old=read(S/'DENSE_REPLAY.json')
old_probe=[r for r in old['rows']if r['split']=='exposed_probe'and r['family']in[2,3]and r['variant']in['normal','short_reviewed_goal_prefix','speed075','speed125']]
write('EVALUATION.json',dict(rows=rows,summary=summary,state_rows=state_rows,state_summary=state_summary,
 timings=timings,model_only_seconds_median=None if recovery else float(np.median([r['seconds']for r in timings])),
 serialization_recovered_from_saved_predictions=recovery,neural_inference_repeated=False,
 cached_input=True,end_to_end_latency=False,dense=dense,old_saved_proxy_comparison=old_probe,
 baseline_scope='Old forearm-state proxy and new combined visible traits are different targets; not a same-label accuracy delta.',
 automatic_localization=False,web_changed=False,whole_action_correctness=None,
 pins={str(U/n):sha(U/n)for n in ['FROZEN.json','LABELS.json','TRAINING_COMPLETE.json','DENSE_PROBABILITIES.npz']}))
print(json.dumps(dict(states=state_summary,probe=[r for r in summary if r['split']=='exposed_probe']),ensure_ascii=False))
