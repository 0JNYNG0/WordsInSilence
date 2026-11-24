from pathlib import Path
import json,hashlib,sys,collections,time
import numpy as np
import torch
from torch import nn
T=Path(__file__).resolve().parent;U=T.parent;O=U.parent;S=O/'word100_goal_state_learning_20261114';sys.path.insert(0,str(U))
from core import features,assess
read=lambda p:json.loads(p.read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,d):
 with(T/n).open('x',encoding='utf-8')as f:json.dump(d,f,ensure_ascii=False,indent=2,default=lambda v:v.item())
classes=['four_open','four_closed','transition']
# Direct RGB review above. These annotations are morphological, not a
# requirement that every sign must show a transition or be judged incorrect.
review={2:['transition','transition','four_closed'],3:['four_open','transition','transition'],
4:['four_open','transition',None],5:['four_open','four_open','four_closed'],
6:['transition','transition',None],7:['four_open','transition','transition'],
8:['transition','transition',None],9:['four_open','transition','transition']}
arr=np.load(S/'source_arrays.npz');base=read(U/'LABELS.json')['rows'];rows=[dict(r,origin='original_visible_state')for r in base if r['domain']=='shape']
addition=read(T/'REVIEW_INPUTS.json')['rows'];counter=collections.Counter();lookup={(r['clip_id'],r['frame'],r['hand']):r for r in rows}
for r in addition:
 i=counter[r['record_id']];counter[r['record_id']]+=1;label=review[r['record_id']][i]
 if (r['clip_id'],r['frame'],'R')in lookup:
  assert lookup[r['clip_id'],r['frame'],'R']['label']==label;continue
 f,ok=features(arr[r['clip_id']+'_x'][r['array_index']:r['array_index']+1],'shape','R')
 rows.append(dict(r,hand='R',domain='shape',label=label,target=classes.index(label)if label else -1,available=bool(ok[0]),feature=f[0].tolist(),origin='RGB_transition_review'))
for i,r in enumerate(rows):r['index']=i
write('LABELS.json',dict(rows=rows,manual_review=review,reviewer='Codex RGB review, not an independent KSL educator',
 definition='Transition: visibly bending non-thumb fingers between the extended and folded prototypes. Unknown borders excluded. A normal transition is not an error.',
 evidence={str(T/f'review_{i}.jpg'):sha(T/f'review_{i}.jpg')for i in [1,2]}))
cfg=dict(seeds=[42,43,44],steps=800,lr=.01,weight_decay=.001,confidence=.8,classes=classes,
 purpose='Third visible-transition class prevents binary forced assignment. No minimum duration or motion distance.',
 fit_records=list(range(2,8)),exposed_probe_records=[8,9],no_holdout_claim=True,
 initialization='Fine-tune corresponding first experiment shape head; copy shared layer and first two output rows; add seeded third row. Common100 and original models unchanged.',
 criterion='Keep all originally correct known probe endpoint states, reduce premature closed predictions on reviewed transition prefixes, keep normal and short goal states. Do not tune on results.',
 pins={str(U/'TRAINING_COMPLETE.json'):sha(U/'TRAINING_COMPLETE.json'),str(T/'LABELS.json'):sha(T/'LABELS.json'),str(U/'core.py'):sha(U/'core.py'),str(T/'run.py'):sha(T/'run.py')})
write('FROZEN.json',cfg)
fit=[r for r in rows if r['split']=='fit'and r['available']and r['target']>=0]
for c in classes:
 assert len({r['record_id']for r in fit if r['label']==c})>=4
count=collections.Counter((r['target'],r['clip_id'])for r in fit);nc=collections.Counter(c for c,clip in count)
w=np.array([1/(3*nc[r['target']]*count[r['target'],r['clip_id']])for r in fit]);w/=w.sum()
x=torch.tensor([r['feature']for r in fit]);y=torch.tensor([r['target']for r in fit]);wt=torch.tensor(w,dtype=torch.float32)
torch.set_num_threads(2);models=[];done=[];(T/'models').mkdir(exist_ok=True)
for seed in cfg['seeds']:
 p=T/'models'/f'{seed}.pt';assert not p.exists();torch.manual_seed(seed);m=nn.Sequential(nn.Linear(12,16),nn.Tanh(),nn.Linear(16,3))
 old=torch.load(U/'models'/f'{seed}.pt',map_location='cpu',weights_only=True)
 with torch.no_grad():
  m[0].weight.copy_(old['heads.shape.0.weight']);m[0].bias.copy_(old['heads.shape.0.bias'])
  m[2].weight[:2].copy_(old['heads.shape.2.weight']);m[2].bias[:2].copy_(old['heads.shape.2.bias'])
 initial={k:v.clone()for k,v in m.state_dict().items()};opt=torch.optim.AdamW(m.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay']);start=time.perf_counter()
 for step in range(cfg['steps']):
  loss=(nn.functional.cross_entropy(m(x),y,reduction='none')*wt).sum();opt.zero_grad();loss.backward();opt.step()
 m.eval();torch.save(m.state_dict(),p);models.append(m);done.append(dict(seed=seed,path=str(p),sha256=sha(p),seconds=time.perf_counter()-start,
    fit_indices=[r['index']for r in fit],weights_changed=any(not torch.equal(initial[k],v)for k,v in m.state_dict().items()),parameters=sum(p.numel()for p in m.parameters())))
write('TRAINING_COMPLETE.json',dict(rows=done))
dense={};probs={}
for cid in sorted({r['clip_id']for r in rows}):
 dense[cid]={}
 for hand in ['R','L']:
  f,ok=features(arr[cid+'_x'],'shape',hand)
  with torch.no_grad():pp=np.stack([m(torch.from_numpy(f)).softmax(-1).numpy()for m in models])
  pred=pp.argmax(-1);known=ok&(pred==pred[:1]).all(0)&(pp.max(-1)>=.8).all(0)
  states=[classes[pred[0,i]]if known[i]else None for i in range(len(ok))]
  dense[cid][hand]=dict(states=states,available=ok.tolist());probs[cid+'_'+hand]=pp
np.savez_compressed(T/'PROBABILITIES.npz',**probs)
old_dense=read(U/'EVALUATION.json')['dense'];state_rows=[]
for r in rows:
 old=old_dense[r['clip_id']][r['hand']]['shape']['states'][r['array_index']];new=dense[r['clip_id']][r['hand']]['states'][r['array_index']]
 state_rows.append(dict(index=r['index'],clip_id=r['clip_id'],frame=r['frame'],hand=r['hand'],split=r['split'],origin=r['origin'],label=r['label'],old=old,new=new))
req={'start':{'shape':'four_open'},'goal':{'shape':'four_closed'}};traces=[]
for r in read(U/'DECOMPOSITION.json')['rows']:
 # Reuse exactly the frozen source-frame/time case descriptions from the first
 # evaluation, including artificial masks. No new interval chosen from scores.
 orig=next(q for q in read(U/'EVALUATION.json')['rows']if q['group_id']==r['group_id']and q['variant']==r['variant'])
 fid=arr[r['clip_id']+'_frame_id'];ix=[int(np.flatnonzero(fid==f)[0])for f in r['source_frames']]
 states=[dense[r['clip_id']]['R']['states'][i]if orig['available']['shape'][j]else None for j,i in enumerate(ix)]
 result=assess({'shape':states},{'shape':orig['available']['shape']},orig['timestamps_us'],req)
 traces.append(dict(clip_id=r['clip_id'],split=r['split'],variant=r['variant'],result=result,old=r['result']['parts'][0]['evidence'],
  source_frames=r['source_frames'],states=states,timestamps_us=orig['timestamps_us']))
# Real source prefixes (not frozen pictures), ending in the reviewed states.
prefixes=[];source=read(S/'REVIEW_INPUTS.json')['rows']
for r in rows:
 if r['origin']!='RGB_transition_review' or r['split']!='exposed_probe' or r['label']is None:continue
 start_frame=next(q['frame']for q in source if q['clip_id']==r['clip_id']and q['family']==2 and q['review_slot']==0)
 fid=arr[r['clip_id']+'_frame_id'];a=int(np.flatnonzero(fid==start_frame)[0]);b=r['array_index'];ts=arr[r['clip_id']+'_timestamp_us'][a:b+1]
 out={}
 for name,dd in [('old',old_dense[r['clip_id']]['R']['shape']),('new',dense[r['clip_id']]['R'])]:
  out[name]=assess({'shape':dd['states'][a:b+1]},{'shape':dd['available'][a:b+1]},ts,req)
 prefixes.append(dict(clip_id=r['clip_id'],frame=r['frame'],label=r['label'],result=out,semantic_error=None))
summary=[]
for split in ['fit','exposed_probe']:
 for origin in ['original_visible_state','RGB_transition_review']:
  rr=[r for r in state_rows if r['split']==split and r['origin']==origin and r['label']is not None]
  summary.append(dict(split=split,origin=origin,n=len(rr),old_correct=sum(r['old']==r['label']for r in rr),
   new_correct=sum(r['new']==r['label']for r in rr),new_wrong=sum(r['new']is not None and r['new']!=r['label']for r in rr),new_unknown=sum(r['new']is None for r in rr)))
write('EVALUATION.json',dict(rows=state_rows,summary=summary,dense=dense,traces=traces,prefixes=prefixes,
 existing_web_replaced=False,real_learner_error_test=False,automatic_localization=False))
print(json.dumps(dict(summary=summary,prefixes=prefixes),ensure_ascii=False))
