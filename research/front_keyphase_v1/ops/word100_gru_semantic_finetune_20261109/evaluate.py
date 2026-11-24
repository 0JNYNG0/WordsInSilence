from semantic_model import *
import statistics

VARIANTS={'normal':{},'upper':{'view':'upper'},'slow':{'scale':1.25},'fast':{'scale':.75},'reversed_main':{'reverse':True},'missing_2':{'gap':2},'missing_main':{'gap':1000}}

def source_to_index(r,q,i):
 ids=np.arange(len(r['x']))
 if q['reversed']:
  direct=np.flatnonzero(r['stage_gold']>=0);lo=int(min([direct[0]]+[p['a'] for p in r['pairs']]));hi=int(max([direct[-1]]+[p['b'] for p in r['pairs']]))
  ids[lo:hi+1]=ids[lo:hi+1][::-1]
 pos=int(np.where(ids==i)[0][0]);return int(np.argmin(abs(q['source_indices']-pos)))

def geometry(x,p):
 s=read(V/'FEEDBACK_NORMALIZATION.json');coord=x[:,:186]*np.array(s['std'])+np.array(s['mean']);pt=coord[:,:102].reshape(-1,51,2)
 a,b,axis=p['a'],p['b'],p['axis'];good=pair_available(x,a,b,axis)
 if axis==0:values=-pt[:,6,1]
 elif axis==1:
  v=pt[:,6]-pt[:,4];values=-v[:,1]/np.maximum(np.linalg.norm(v,axis=1),1e-6)
 elif axis==2:
  v=pt[:,5]-pt[:,3];rel=pt[:,[30,35,39,43,47]].mean(1)-pt[:,3];values=(v*rel).sum(1)/np.maximum((v*v).sum(1),1e-6)
 elif axis==3:values=np.linalg.norm(pt[:,9]-pt[:,30],axis=1)
 else:values=pt[:,9,1]-pt[:,30,1]
 # Fixed physical deadband, not fitted against validation labels.
 delta=float(values[b]-values[a]);floor=[.08,.10,.10,.08,.08][axis]
 pred=int(delta>floor)*2 if abs(delta)>floor else 1
 return pred,good,delta

def summarize(records):
 result={}
 for task in sorted({r['task'] for r in records}):
  rr=[r for r in records if r['task']==task];known=[r for r in rr if r['eligible']];accepted=[r for r in known if r.get('confident',True)]
  result[task]=dict(n=len(rr),raw_correct=sum(r['pred']==r['target'] for r in rr),eligible=len(known),eligible_correct=sum(r['pred']==r['target'] for r in known),
   accepted=len(accepted),accepted_correct=sum(r['pred']==r['target'] for r in accepted),withheld=len(rr)-len(accepted),
   by_target={str(t):dict(n=sum(r['target']==t for r in rr),correct=sum(r['target']==t and r['pred']==t for r in rr)) for t in sorted({r['target'] for r in rr})})
 return result

def main():
 assert (U/'TRAINING_COMPLETE.json').exists();assert not (U/'EVALUATION_COMPLETE.json').exists()
 cfg=read(U/'FROZEN.json');groups={s:i for i,g in enumerate(cfg['semantic_stage_groups']) for s in g}
 rows=[r for r in load_rows() if r['role']=='development'];assert {r['person'] for r in rows}=={11,12,13}
 models=['original']+[f'{c}_{s}' for c in cfg['conditions'] for s in cfg['seeds']]
 ev=dict(variants=VARIANTS,models=models,geometry_deadbands=[.08,.10,.10,.08,.08],shape_threshold=.5,direction_confidence=.65,order_confidence=[.35,.65],
  direction_baseline='generic body-relative endpoint geometry; prior web only used forearm direction, other axes are diagnostic baseline',
  primary='normal full view; speed/crop and synthetic separate; all original sparse anchors stay in denominator; unknown never counted correct',
  corpus='4 words x 3 previously exposed DEV IDs; no new held-out person claim')
 write(U/'EVAL_FROZEN.json',ev)
 torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 allresults={};guides=read(L/'GUIDE_SUPPORT.json')['guides'];write(U/'GUIDE_LINKS.json',dict(guides=[dict(index=i,**g) for i,g in enumerate(guides)],annotation_sha256=sha(U/'ANNOTATIONS.json'),note='Retains distinction: source sentence vs canonical video description; no language model or expert-approved error criteria.'))
 for name in models:
  model=SemanticNet()
  if name=='original':model.base.load_state_dict(torch.load(BASE,map_location='cpu',weights_only=True))
  else:model.load_state_dict(torch.load(U/'runs'/name/'model.pt',map_location='cpu',weights_only=True))
  model.eval();records=[];timings=[];predpaths=[]
  with torch.inference_mode():
   for variant,opts in VARIANTS.items():
    for r in rows:
     q=transform(r,**opts);b=collate([q]);start=time.perf_counter();o=model.base(b['x'],b['lengths'],b['word']);h=o['latent'][0];attr=o['attributes'][0].sigmoid().numpy();sp=o['stage'][0].softmax(-1).numpy()
     common=dict(model=name,variant=variant,clip_id=r['clip_id'],word=r['word'],person=r['person']);available=availability(q['x'])
     for task,mask in [('reviewed_shape',r['direct_ym']),('original_shape_labels',r['old_ym'])]:
      ys=r['y'] if task=='reviewed_shape' else r['old_y']
      for i,j in zip(*np.where(mask>0)):
       k=source_to_index(r,q,int(i));prob=float(attr[k,j]);pred=int(prob>=.5)
       records.append(dict(common,task=task,frame=int(r['frame_id'][i]),index=k,feature=int(j),target=int(ys[i,j]),pred=pred,probability=prob,eligible=bool(available[k,j]),confident=bool(prob>=.65 or prob<=.35)))
     for i in np.flatnonzero(r['stage_gold']>=0):
      k=source_to_index(r,q,int(i));s=int(sp[k].argmax());g=int(r['stage_gold'][i]);eligible=bool(q['x'][k,186+1:186+3].min()>.5 and q['x'][k,186:237].sum()>=5)
      records.append(dict(common,task='semantic_stage',frame=int(r['frame_id'][i]),index=k,guide=g,pred_guide=s,target=groups[g],pred=groups[s],eligible=eligible,confident=bool(sp[k,s]>=.65)))
     for p in q['pairs']:
      gp,good,delta=geometry(q['x'],p)
      if name=='original':pred=gp;confidence=None
      else:
       d,_=model.pair(h,p['a'],p['b'],r['word']);prob=d[p['axis']].softmax(-1).numpy();pred=int(prob.argmax());confidence=float(prob.max())
      records.append(dict(common,task='direction',axis=p['axis'],a=p['a'],b=p['b'],target=p['target'],pred=pred,eligible=good,confident=confidence is None or confidence>=.65,confidence=confidence,geometry_pred=gp,geometry_delta=delta))
     for p in q['orders']:
      good=order_available(q['x'],p['a'],p['b'],r['word'])
      if name=='original':
       ga=int(sp[p['a']].argmax());gb=int(sp[p['b']].argmax());pred=int(ga<gb);prob=None;confident=ga!=gb
      else:
       _,order=model.pair(h,p['a'],p['b'],r['word']);prob=float(order.sigmoid());pred=int(prob>=.5);confident=prob>=.65 or prob<=.35
      records.append(dict(common,task='order',a=p['a'],b=p['b'],target=p['target'],pred=pred,eligible=good,confident=confident,probability=prob))
     elapsed=time.perf_counter()-start
     if variant=='normal':timings.append(elapsed)
     if name in ['original','finetune_42','frozen_heads_42']:
      path=U/'predictions'/name/f"{r['clip_id']}_{variant}.npz";path.parent.mkdir(exist_ok=True,parents=True)
      np.savez_compressed(path,attributes=attr,stage=sp,frame_id=q['frame_id'],timestamp_us=q['timestamp_us']);predpaths.append(dict(path=str(path),sha256=sha(path)))
  # Pure post-keypoint neural timing, excludes Python evaluation/scoring loop.
  q=transform(rows[0]);b=collate([q]);lat=[]
  with torch.inference_mode():
   for rep in range(35):
    start=time.perf_counter();o=model.base(b['x'],b['lengths'],b['word'])
    if name!='original':
     for p in q['pairs']+q['orders']:model.pair(o['latent'][0],p['a'],p['b'],q['word'])
    t=time.perf_counter()-start
    if rep>=5:lat.append(t)
  result=dict(by_variant={v:summarize([r for r in records if r['variant']==v]) for v in VARIANTS},latency=dict(median=statistics.median(lat),p95=float(np.percentile(lat,95)),frames=len(q['x']),repeats=len(lat),scope='CPU4 threads, cached keypoints to base + reviewed pair heads; NOT camera/end-to-end latency'),predictions=predpaths)
  write(U/'scores'/f'{name}.json',records);write(U/'scores'/f'{name}_summary.json',result);allresults[name]=result
  event('evaluation_model_complete',model=name,normal=result['by_variant']['normal'],latency=result['latency'])
 write(U/'EVALUATION_COMPLETE.json',dict(status='FIXED_PUBLIC_COMPARISON_COMPLETE',models=allresults,frozen_sha256=sha(U/'FROZEN.json'),eval_sha256=sha(U/'EVAL_FROZEN.json'),private_used=False,web_changed=False))
if __name__=='__main__':main()
