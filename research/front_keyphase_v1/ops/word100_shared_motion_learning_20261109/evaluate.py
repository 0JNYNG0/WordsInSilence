from core import *
import statistics
GROUPS={s:i for i,g in enumerate(read(S/'FROZEN.json')['semantic_stage_groups']) for s in g}
VARIANTS=['normal','upper','slow','fast','missing_2','missing_long','reversed_main']
def transform(r,variant):
 x=r['x_upper'].copy() if variant=='upper' else r['x_full'].copy();n=len(x);ids=np.arange(n);gold=np.flatnonzero(r['stage_gold']>=0)
 if variant=='reversed_main' and len(gold):
  a=min([int(gold[0])]+[p['a'] for p in r['pairs']]);b=max([int(gold[-1])]+[p['b'] for p in r['pairs']]);ids[a:b+1]=ids[a:b+1][::-1];x=x[ids]
 if variant in ['slow','fast']:
  scale=1.25 if variant=='slow' else .75;ix=np.rint(np.linspace(0,n-1,round(n*scale))).astype(int);x=x[ix];ids=ids[ix]
 if variant in ['missing_2','missing_long']:
  mid=int(np.median(gold)) if len(gold) else n//2;gap=2 if variant=='missing_2' else n;a=max(0,mid-gap//2);b=min(n,a+gap)
  x=mask_hand(mask_hand(x,a,b,0),a,b,1)
 return x,ids
def summaries(rows):
 out={}
 for task in sorted({r['task'] for r in rows}):
  rs=[r for r in rows if r['task']==task];accepted=[r for r in rs if r['eligible'] and r['confident']]
  out[task]=dict(n=len(rs),raw_correct=sum(r['pred']==r['target'] for r in rs),accepted=len(accepted),accepted_correct=sum(r['pred']==r['target'] for r in accepted),withheld=len(rs)-len(accepted),by_feature={str(f):dict(n=sum(r.get('feature')==f for r in rs),correct=sum(r.get('feature')==f and r['pred']==r['target'] for r in rs)) for f in sorted({r['feature'] for r in rs if 'feature' in r})})
 return out
def main():
 assert (U/'TRAINING_COMPLETE.json').exists();assert not (U/'EVALUATION_COMPLETE.json').exists()
 cfg=read(U/'FROZEN.json');rows=load_rows();dev=[r for r in rows if r['role']=='development'];feedback=[r for r in dev if r['word']>=0];assert len(dev)==300 and len(feedback)==12
 extra=read(U/'EXTRA_EVAL_LABELS.json');extra_by_clip={}
 for e in extra['rows']:extra_by_clip.setdefault(e['clip_id'],[]).append(e)
 write(U/'EVAL_FROZEN.json',dict(models=[f'{c}_{s}' for c in cfg['conditions'] for s in cfg['seeds']],variants=VARIANTS,shape_threshold=.5,confidence=[.35,.65],stage_groups=GROUPS,existing_baseline=str(S/'scores/original_summary.json'),no_baseline_reinference=True,word_aux_not_replacing_TCN=True,no_DEV_selection=True))
 torch.set_num_threads(4);torch.set_num_interop_threads(1);results={}
 for condition in cfg['conditions']:
  for seed in cfg['seeds']:
   name=f'{condition}_{seed}';net=SharedNet();net.load_state_dict(torch.load(U/'runs'/name/'model.pt',map_location='cpu',weights_only=True));net.eval();records=[];wordrows=[];saved=[]
   with torch.inference_mode():
    for r in dev:
     x=torch.from_numpy(r['x_full'][None]);o=net(x,torch.tensor([x.shape[1]]),torch.tensor([max(r['word'],0)]));pred=int(o['word_logits'].argmax(-1));wordrows.append(dict(clip_id=r['clip_id'],target=r['class_index'],pred=pred))
     if r['clip_id'] in extra_by_clip:
      attr=o['attributes'][0].sigmoid().numpy();av=available(r['x_full'])
      for e in extra_by_clip[r['clip_id']]:
       k=int(np.argmin(abs(r['frame_id']-e['frame'])));assert int(r['frame_id'][k])==e['frame']
       for j,t in enumerate(e['targets']):
        if t is None:continue
        p=float(attr[k,j]);records.append(dict(task='expanded_shape',variant='normal',clip_id=r['clip_id'],frame=e['frame'],feature=j,target=t,pred=int(p>=.5),probability=p,eligible=bool(av[k,j]),confident=p<=.35 or p>=.65))
    for variant in VARIANTS:
     for r in feedback:
      x,ids=transform(r,variant);o=net(torch.from_numpy(x[None]),torch.tensor([len(x)]),torch.tensor([r['word']]));attr=o['attributes'][0].sigmoid().numpy();stage=o['stage'][0].softmax(-1).numpy();avail=available(x)
      def ix(i):return int(np.argmin(abs(ids-int(i))))
      for task,mask,ys in [('reviewed_shape',r['direct_ym'],r['y']),('original_shape_labels',r['old_ym'],r['old_y'])]:
       for i,j in zip(*np.where(mask>0)):
        k=ix(i);p=float(attr[k,j]);records.append(dict(task=task,variant=variant,clip_id=r['clip_id'],frame=int(r['frame_id'][i]),feature=int(j),target=int(ys[i,j]),pred=int(p>=.5),probability=p,eligible=bool(avail[k,j]),confident=p<=.35 or p>=.65))
      for i in np.flatnonzero(r['stage_gold']>=0):
       k=ix(i);pred=int(stage[k].argmax());eligible=bool(x[k,187:189].min()>.5 and x[k,186:237].sum()>=5)
       records.append(dict(task='semantic_stage',variant=variant,clip_id=r['clip_id'],frame=int(r['frame_id'][i]),target=GROUPS[int(r['stage_gold'][i])],pred=GROUPS[pred],eligible=eligible,confident=bool(stage[k,pred]>=.65)))
      for pair in r['orders']:
       a,b=ix(pair['a']),ix(pair['b']);a,b=min(a,b),max(a,b);sa=int(stage[a].argmax());sb=int(stage[b].argmax())
       records.append(dict(task='normal_order',variant=variant,clip_id=r['clip_id'],target=int(variant!='reversed_main'),pred=int(sa<sb),eligible=bool(avail[a:b+1].any()),confident=sa!=sb))
      if seed==42:
       dst=U/'predictions'/name/f"{r['clip_id']}_{variant}.npz";dst.parent.mkdir(exist_ok=True,parents=True);np.savez_compressed(dst,attributes=attr,stage=stage,latent=o['latent'][0].numpy(),source_indices=ids);saved.append(str(dst))
    b=batch([feedback[0]]);timings=[]
    for k in range(35):
     t=time.perf_counter();net(b['x'],b['lengths'],b['word']);dt=time.perf_counter()-t
     if k>=5:timings.append(dt)
   out=dict(by_variant={v:summaries([r for r in records if r['variant']==v]) for v in VARIANTS},aux_word=dict(correct=sum(r['pred']==r['target'] for r in wordrows),total=300,trained=condition=='shared100',scope='Auxiliary readout of feedback network, not existing TCN'),latency=dict(p50_seconds=statistics.median(timings),p95_seconds=float(np.percentile(timings,95)),scope='cached coordinates to neural outputs on CPU4; not end-to-end'),predictions=saved)
   write(U/'scores'/f'{name}_records.json',records);write(U/'scores'/f'{name}_words.json',wordrows);write(U/'scores'/f'{name}.json',out);results[name]=out;event('evaluated',model=name,normal=out['by_variant']['normal'],word=out['aux_word'],latency=out['latency'])
 baseline=read(S/'scores/original_summary.json');write(U/'EVALUATION_COMPLETE.json',dict(status='FIXED_DEV_COMPARISON_COMPLETE',models=results,stored_original_baseline=baseline,extra_labels_sha256=sha(U/'EXTRA_EVAL_LABELS.json'),extra_annotation_coverage={k:extra[k] for k in ['potential_cells','known_cells','null_cells']},old_scores_not_recomputed=True,web_model_changed=False))
 write(U/'CURRENT_EXECUTION.json',dict(status='EVALUATED_PENDING_VERIFY',pid=None))
if __name__=='__main__':main()
