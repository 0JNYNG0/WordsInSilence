from core import *
import statistics,traceback
sys.path.insert(0,str(C))
previous=load('_bf_component_eval',C/'evaluate.py')
def total(rr):
 return dict(n=len(rr),correct=sum(q['pred']==q['target'] for q in rr),eligible=sum(q['eligible'] for q in rr),accepted=sum(q['eligible'] and q['confident'] for q in rr),accepted_correct=sum(q['eligible'] and q['confident'] and q['pred']==q['target'] for q in rr))
def main():
 assert (U/'TRAINING_COMPLETE.json').exists() and not (U/'EVALUATION_COMPLETE.json').exists()
 cfg=read(U/'FROZEN.json');norm=read(U/'NORMALIZATION.json');rr=rows();dev=[r for r in rr if r['role']=='development'];focus=[r for r in dev if r['word']>=0];assert len(focus)==12
 torch.set_num_threads(4);torch.set_num_interop_threads(1)
 outdir=U/'scores';outdir.mkdir(exist_ok=True);summary={}
 for cond in cfg['conditions']:
  for seed in cfg['seeds']:
   name=f'{cond}_{seed}';net=Network(cond,norm);net.load_state_dict(torch.load(U/'runs'/name/'model.pt',map_location='cpu',weights_only=True));net.eval();rec=[];times=[];wordrows=[]
   with torch.inference_mode():
    for v in previous.VARIANTS:
     for r in focus:
      x,ids=previous.transform(r,v);ts=np.arange(len(x))*33333;t=time.perf_counter();o=net.shared(torch.from_numpy(x[None]),torch.tensor([len(x)]),torch.tensor([r['word']]));times.append(time.perf_counter()-t);attr=o['attributes'][0].sigmoid().numpy();stage=o['stage'][0].softmax(-1).numpy();av=shared.available(x)
      def ix(i):return int(np.argmin(abs(ids-i)))
      for task,m,y in [('original_shape',r['old_ym'],r['old_y']),('reviewed_shape',r['direct_ym'],r['y'])]:
       for i,j in zip(*np.where(m>0)):
        k=ix(i);p=float(attr[k,j]);rec.append(dict(task=task,variant=v,clip_id=r['clip_id'],frame=int(r['frame_id'][i]),feature=int(j),target=int(y[i,j]),pred=int(p>=.5),eligible=bool(av[k,j]),confident=p<=.35 or p>=.65,probability=p))
      for i in np.flatnonzero(r['stage_gold']>=0):
       k=ix(i);pred=int(stage[k].argmax());rec.append(dict(task='semantic_stage',variant=v,clip_id=r['clip_id'],frame=int(r['frame_id'][i]),target=previous.GROUPS[int(r['stage_gold'][i])],pred=previous.GROUPS[pred],eligible=bool(x[k,187:189].min()>.5),confident=bool(stage[k,pred]>=.65)))
      raw=physical(x)
      for pi,p in enumerate(r['pairs']):
       aa,bb=ix(p['a']),ix(p['b']);a,b=sorted([aa,bb]);g,ok,info=geometry(raw,ts,a,b,p['axis'],p.get('moving','R'),p.get('anchor','L'));pr=net.movement(o['latent'],torch.from_numpy(g[None]),torch.tensor([[0,a,b]])).softmax(-1)[0].numpy();target=2-p['target'] if aa>bb else p['target']
       rec.append(dict(task='movement',variant=v,clip_id=r['clip_id'],pair=pi,axis=p['axis'],target=target,pred=int(pr.argmax()) if ok else -1,eligible=bool(ok),confident=bool(ok and pr.max()>=.65),probability=pr.tolist(),evidence=info))
      for pi,p in enumerate(r['orders']):
       aa,bb=ix(p['a']),ix(p['b']);a,b=sorted([aa,bb]);pr=float(net.ordering(o['latent'],torch.tensor([[0,a,b]]),torch.tensor([[p['guide_a'],p['guide_b']]])).sigmoid()[0]);ok=bool(av[[a,b]].any(1).all())
       rec.append(dict(task='order',variant=v,clip_id=r['clip_id'],pair=pi,target=int(aa<bb),pred=int(pr>=.5) if ok else -1,eligible=ok,confident=bool(ok and (pr<=.35 or pr>=.65)),probability=pr))
    for lo in range(0,len(dev),24):
     sample=dev[lo:lo+24];b=shared.batch(sample);o=net.shared(b['x'],b['lengths'],b['word']);pred=o['word_logits'].argmax(-1).tolist()
     extra_by_clip={}
     for q in read(B/'EXTRA_EVAL_LABELS.json')['rows']:extra_by_clip.setdefault(q['clip_id'],[]).append(q)
     for ri,(r,p) in enumerate(zip(sample,pred)):
      wordrows.append(dict(clip_id=r['clip_id'],target=r['class_index'],pred=p,eligible=True,confident=True))
      av=shared.available(r['x_full']);ap=o['attributes'][ri].sigmoid().numpy()
      for q in extra_by_clip.get(r['clip_id'],[]):
       frame=np.flatnonzero(r['frame_id']==q['frame']);assert len(frame)==1;idx=int(frame[0])
       for j,y in enumerate(q['targets']):
        if y is None:continue
        pr=float(ap[idx,j]);rec.append(dict(task='expanded_shape',variant='normal',clip_id=r['clip_id'],frame=q['frame'],feature=j,target=y,pred=int(pr>=.5),eligible=bool(av[idx,j]),confident=pr<=.35 or pr>=.65,probability=pr))
   result={v:{task:total([q for q in rec if q['variant']==v and q['task']==task]) for task in ['original_shape','reviewed_shape','expanded_shape','semantic_stage','movement','order']} for v in previous.VARIANTS}
   write(outdir/f'{name}.json',rec);write(outdir/f'{name}_words.json',wordrows);summary[name]=dict(by_variant=result,aux_word100=total(wordrows),network_seconds_median=statistics.median(times[1:]),network_seconds_p95=float(np.percentile(times[1:],95)));event('component_evaluated',model=name,normal=result['normal'],aux_word100=total(wordrows))
 write(U/'EVALUATION_COMPLETE.json',dict(models=summary,baseline_previous=str(C/'EVALUATION_COMPLETE.json'),baseline_broad=str(B/'EVALUATION_COMPLETE.json'),scope='Historical DEV; supplied reviewed component intervals. Order diagnostic trained; not automatically generated full-video correction.',old_models_rerun=False))
if __name__=='__main__':
 try:main()
 except BaseException:
  (U/f'EVAL_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),'utf-8');raise
