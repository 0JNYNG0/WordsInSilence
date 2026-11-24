from common_action import *
import statistics
S=O/'word100_gru_semantic_finetune_20261109'
GROUPS={s:i for i,g in enumerate(read(S/'FROZEN.json')['semantic_stage_groups']) for s in g}
VARIANTS=['normal','upper','slow','fast','missing_2','missing_long','reversed_main']
def transform(r,v):
 x=r['x_upper' if v=='upper' else 'x_full'].copy();n=len(x);ids=np.arange(n);gold=np.flatnonzero(r['stage_gold']>=0)
 if v=='reversed_main' and len(gold):
  a=min([int(gold[0])]+[p['a'] for p in r['pairs']]);b=max([int(gold[-1])]+[p['b'] for p in r['pairs']]);ids[a:b+1]=ids[a:b+1][::-1];x=x[ids]
 if v in ['slow','fast']:
  scale=1.25 if v=='slow' else .75;ix=np.rint(np.linspace(0,n-1,round(n*scale))).astype(int);x=x[ix];ids=ids[ix]
 if v in ['missing_2','missing_long']:
  mid=int(np.median(gold)) if len(gold) else n//2;gap=2 if v=='missing_2' else n;a=max(0,mid-gap//2);b=min(n,a+gap);x=shared.mask_hand(shared.mask_hand(x,a,b,0),a,b,1)
 return x,ids
def summary(rr):
 return dict(total=len(rr),raw_correct=sum(r['pred']==r['target'] for r in rr),eligible=sum(r['eligible'] for r in rr),accepted=sum(r['eligible'] and r['confident'] for r in rr),accepted_correct=sum(r['eligible'] and r['confident'] and r['pred']==r['target'] for r in rr),opposite=sum(r['task']=='movement' and r['target']!=1 and r['eligible'] and r['pred']==2-r['target'] for r in rr))
def main():
 assert (U/'TRAINING_COMPLETE.json').exists() and not (U/'EVALUATION_COMPLETE.json').exists()
 cfg=read(U/'FROZEN.json');norm=read(U/'NORMALIZATION.json');dev=[r for r in rows() if r['role']=='development'];torch.set_num_threads(4);torch.set_num_interop_threads(1)
 write(U/'EVAL_FROZEN.json',dict(variants=VARIANTS,models=[f'{c}_{s}' for c in cfg['conditions'] for s in cfg['seeds']],denominator='All reviewed observations including missing/low confidence. 12 historical development clips; not new people test.',baseline=str(S/'scores/original.json'),old_motion=str(M/'SCORES.json'),source_indices='Original reviewed anchors mapped by nearest retained frame; repeated speed frames not independent annotations.',threshold=.65))
 results={}
 for cond in cfg['conditions']:
  for seed in cfg['seeds']:
   name=f'{cond}_{seed}';net=CommonActionNet(norm['mean'],norm['std']);net.load_state_dict(torch.load(U/'runs'/name/'model.pt',map_location='cpu',weights_only=True));net.eval();rec=[];timings=[]
   with torch.inference_mode():
    for variant in VARIANTS:
     for r in dev:
      x,ids=transform(r,variant);t=time.perf_counter();out=net.base(torch.from_numpy(x[None]),torch.tensor([len(x)]),torch.tensor([r['word']]));timings.append(time.perf_counter()-t);attr=out['attributes'][0].sigmoid().numpy();stage=out['stage'][0].softmax(-1).numpy();avail=shared.available(x)
      def ix(i):return int(np.argmin(abs(ids-int(i))))
      for task,mask,truth in [('reviewed_shape',r['direct_ym'],r['y']),('original_shape_labels',r['old_ym'],r['old_y'])]:
       for i,j in zip(*np.where(mask>0)):
        k=ix(i);p=float(attr[k,j]);rec.append(dict(task=task,variant=variant,clip_id=r['clip_id'],frame=int(r['frame_id'][i]),feature=int(j),target=int(truth[i,j]),pred=int(p>=.5),probability=p,eligible=bool(avail[k,j]),confident=p<=.35 or p>=.65))
      for i in np.flatnonzero(r['stage_gold']>=0):
       k=ix(i);pred=int(stage[k].argmax());rec.append(dict(task='semantic_stage',variant=variant,clip_id=r['clip_id'],frame=int(r['frame_id'][i]),target=GROUPS[int(r['stage_gold'][i])],pred=GROUPS[pred],eligible=bool(x[k,187:189].min()>.5 and x[k,186:237].sum()>=5),confident=bool(stage[k,pred]>=.65)))
      # All direction intervals remain, including unavailable ones. Actual reversed input changes the latent as well.
      raw=physical(x);ts=np.arange(len(x))*33333
      for pi,pair in enumerate(r['pairs']):
       aa,bb=ix(pair['a']),ix(pair['b']);a,b=min(aa,bb),max(aa,bb);target=2-pair['target'] if aa>bb else pair['target']
       g,ok,info=motion.features(raw,ts,a,b,motion.KINDS[pair['axis']]);pr=net.movement(out['latent'],torch.from_numpy(g[None]),torch.tensor([[0,a,b]])).softmax(-1)[0].numpy();pred=int(pr.argmax());conf=float(pr.max())
       rec.append(dict(task='movement',variant=variant,clip_id=r['clip_id'],pair=pi,axis=pair['axis'],a=a,b=b,target=target,pred=pred if ok else -1,confidence=conf,eligible=bool(ok),confident=bool(ok and conf>=.65),evidence=info))
      if seed==42:
       dst=U/'predictions'/name/f"{r['clip_id']}_{variant}.npz";dst.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dst,attributes=attr,stage=stage,latent=out['latent'][0].numpy(),source_indices=ids)
   result={v:{task:summary([q for q in rec if q['task']==task and q['variant']==v]) for task in ['reviewed_shape','original_shape_labels','semantic_stage','movement']} for v in VARIANTS}
   write(U/'scores'/f'{name}.json',rec);results[name]=dict(by_variant=result,neural_timing=dict(median_seconds=statistics.median(timings[1:]),p95_seconds=float(np.percentile(timings[1:],95)),scope='Base network only on cached normalized coordinates; excludes detector, DTW/UI and motion feature/head'))
   event('evaluated',model=name,normal=result['normal'])
 write(U/'EVALUATION_COMPLETE.json',dict(models=results,original_saved=read(S/'scores/original_summary.json'),previous_motion_saved=read(M/'EVALUATION_COMPLETE.json'),old_reinference=False))
 write(U/'CURRENT_EXECUTION.json',dict(status='LOCAL_COMPONENT_EVALUATION_COMPLETE_FULLPATH_PENDING',pid=None))
if __name__=='__main__':main()
