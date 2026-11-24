from window_core import *
import traceback,statistics
F=O/'word100_completion_fullpath_20261110'
VARIANTS=['normal','upper','slow','fast','missing_2','missing_long','reversed_main']
PADS={'clean':[0,0],'both8':[8,8],'before12':[12,0],'after12':[0,12]}
def total(rr):return dict(n=len(rr),correct=sum(q['pred']==q['target'] for q in rr),eligible=sum(q['eligible'] for q in rr),accepted=sum(q['eligible'] and q['confident'] for q in rr),accepted_correct=sum(q['eligible'] and q['confident'] and q['pred']==q['target'] for q in rr))
def transform(r,v):
 x,ids=transformed(r,'upper' if v=='upper' else 'full',v=='reversed_main');ts=r['timestamp_us'].copy()
 if v in ['slow','fast']:
  ix=np.rint(np.linspace(0,len(x)-1,round(len(x)*(1.25 if v=='slow' else .75)))).astype(int);x=x[ix];ids=ids[ix];ts=np.arange(len(x))*33333
 if v in ['missing_2','missing_long']:
  gold=np.flatnonzero(r['stage_gold']>=0);mid=int(np.median(gold)) if len(gold) else len(x)//2;gap=2 if v=='missing_2' else len(x);a=max(0,mid-gap//2);b=min(len(x),a+gap)
  for side in [0,1]:
   start=9+21*side;x[a:b,start*2:(start+21)*2]=0;x[a:b,186+start:186+start+21]=0;x[a:b,102+42*side:144+42*side]=0;x[a:b,237+21*side:258+21*side]=0
 return x,ids,ts
def truth(r,ids,a,b,p):
 classes=[];core=np.zeros(b-a+1,np.float32)
 for q in r['pairs']:
  if q['axis']!=p['axis'] or q.get('moving','R')!=p.get('moving','R'):continue
  aa=int(np.argmin(abs(ids-q['a'])));bb=int(np.argmin(abs(ids-q['b'])));lo=max(a,min(aa,bb));hi=min(b,max(aa,bb))
  if hi-lo<2:continue
  classes.append(2-q['target'] if aa>bb else q['target']);core[lo-a:hi-a+1]=1
 active=set(classes)-{1};return (3 if len(active)>1 else next(iter(active)) if active else 1),core
def main():
 assert (U/'TRAINING_COMPLETE.json').exists() and not (U/'EVALUATION_COMPLETE.json').exists();protocol=read(U/'EVAL_FROZEN.json')
 for p,h in protocol['code_pins'].items():assert sha(p)==h,p
 torch.set_num_threads(4);torch.set_num_interop_threads(1);rr=prior.rows();dev=[r for r in rr if r['role']=='development'];focus=[r for r in dev if r['word']>=0];assert len(dev)==300 and len(focus)==12
 # Frozen100-word representation is reused as an input, not retrained or selected on DEV.
 frozen=prior.Network('coupled',read(P/'NORMALIZATION.json'));frozen.load_state_dict(torch.load(P/'runs/coupled_42/model.pt',map_location='cpu',weights_only=True));frozen.eval()
 with torch.inference_mode():
  for start in range(0,len(dev),24):
   ss=dev[start:start+24];b=shared.batch(ss);out=frozen.shared(b['x'],b['lengths'],b['word'])
   for i,r in enumerate(ss):r['g_full']=geometry(prior.physical(r['x_full']),r['timestamp_us']);r['frozen_h_full']=out['latent'][i,:len(r['x_full'])].numpy()
 cfg=read(U/'FROZEN.json');summaries={}
 for mode in cfg['conditions']:
  for seed in cfg['seeds']:
   name=f'{mode}_{seed}';net=WindowNet(mode,read(U/'NORMALIZATION.json'));net.load_state_dict(torch.load(U/'runs'/name/'model.pt',weights_only=True,map_location='cpu'));net.eval();records=[];words=[];timings=[]
   with torch.inference_mode():
    for variant in VARIANTS:
     for r in focus:
      x,ids,ts=transform(r,variant);g=geometry(x,ts);t=time.perf_counter();h,_=net.encode(torch.tensor(g[None]),torch.tensor([len(g)]));h=h[0];timings.append(time.perf_counter()-t)
      for pi,p in enumerate(r['pairs']):
       aa=int(np.argmin(abs(ids-p['a'])));bb=int(np.argmin(abs(ids-p['b'])));q=torch.tensor(relations(x,p['axis'],p.get('moving','R'),p.get('anchor','L')))
       for pad,(left,right) in PADS.items():
        a=max(0,min(aa,bb)-left);b=min(len(x)-1,max(aa,bb)+right);target,core=truth(r,ids,a,b,p);logit,cp,att,valid=net.movement(h,q,a,b,p['axis']);prob=logit.softmax(-1).numpy();ok=bool(valid.float().mean()>=.8 and valid.sum()>=4)
        records.append(dict(task='movement',variant=variant,padding=pad,clip_id=r['clip_id'],pair=pi,axis=p['axis'],a=a,b=b,target=target,pred=int(prob.argmax()) if ok else -1,probability=prob.tolist(),eligible=ok,confident=bool(prob.max()>=.65 and ok),known_core_attention=float((att*torch.tensor(core)).sum()),known_core_uniform_fraction=float(core.mean()),observed_fraction=float(valid.float().mean()),attention=att.numpy().tolist(),core_probability=cp.sigmoid().numpy().tolist()))
      for pi,p in enumerate(r['orders']):
       aa=int(np.argmin(abs(ids-p['a'])));bb=int(np.argmin(abs(ids-p['b'])))
       for pad,(left,right) in PADS.items():
        a=max(0,min(aa,bb)-left);b=min(len(x)-1,max(aa,bb)+right);logit,att=net.ordering(h,a,b,p['guide_a'],p['guide_b']);prob=float(logit.sigmoid());available=shared.available(shared.normalize(x));ok=bool(available[[aa,bb]].any(1).all());pos=np.arange(a,b+1);centres=(att.numpy()*pos[:,None]).sum(0)
        records.append(dict(task='order',variant=variant,padding=pad,clip_id=r['clip_id'],pair=pi,a=a,b=b,target=int(aa<bb),pred=int(prob>=.5) if ok else -1,probability=prob,eligible=ok,confident=bool(ok and (prob<=.35 or prob>=.65)),annotated_anchors=[aa,bb],attention_centres=centres.tolist(),anchor_mae=float(np.mean(abs(centres-np.array([aa,bb]))))))
    for lo in range(0,len(dev),24):
     ss=dev[lo:lo+24];g,fh,l=batch_g(ss);h,v=net.encode(g,l);pr=net.classify_words(h,fh,l,v).argmax(-1).tolist()
     words.extend(dict(clip_id=r['clip_id'],target=r['class_index'],pred=p,eligible=True,confident=True) for r,p in zip(ss,pr))
   write(U/'scores'/f'{name}.json',records);write(U/'scores'/f'{name}_words.json',words)
   summary={v:{pad:{task:total([q for q in records if q['variant']==v and q['padding']==pad and q['task']==task]) for task in ['movement','order']} for pad in PADS} for v in VARIANTS}
   summaries[name]=dict(by_variant=summary,aux_word100=total(words),geometry_encoder_ms_median=1000*statistics.median(timings[1:]),geometry_encoder_ms_p95=1000*float(np.percentile(timings[1:],95)));event('component_evaluated',model=name,normal=summary['normal'],aux_word100=total(words))
 write(U/'EVALUATION_COMPLETE.json',dict(models=summaries,scope='Repeated historical DEV. Reviewed windows and controlled padding; automatic consumer tested separately. Missing observations remain in every denominator.'))
if __name__=='__main__':
 try:main()
 except BaseException:(U/f'EVAL_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),'utf-8');raise
