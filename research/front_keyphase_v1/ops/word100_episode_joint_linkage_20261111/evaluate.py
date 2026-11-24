from joint_core import *
import traceback,os
MODES=['baseline','interval_only','role_only','combined']
def jsonhash(v):return hashlib.sha256(json.dumps(v,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
def family(prob,a,b):
 p=np.asarray(prob)[a:b+1].mean(0);return dict(pred=int(p.argmax()),confidence=float(p.max()))
def truth_rows(annotations,clip):
 out=[]
 for r in annotations:
  if r['clip_id']!=clip:continue
  for e in r['events']:
   out.append(dict(e,axis=r['axis'],moving=r['moving'],anchor=r['anchor'],family={('WORD0739',0):2,('WORD0739',1):3,('WORD1133',2):4,('WORD1567',4):5,('WORD2523',3):7}.get((r['word_id'],r['axis'])),span=[e['onset'][0],e['offset'][1]]))
 return out
def diagnose(c,obs,base):
 oldevents=c['baseline']['phase_events']+c['baseline']['phase_partials'];oldevents=[e for e in oldevents if e['b']-e['a']>=3];union={}
 for e in oldevents+obs['selected']:union[(e['a'],e['b'],e['axis'],e['moving'])]=e
 choices={'baseline':oldevents,'interval_only':list(union.values()),'role_only':oldevents,'combined':list(union.values()),'selected_only':obs['selected']};matches={k:lc.be.match_events(v,c['truth']) for k,v in choices.items()};cache={};out=[];oldrp=legacy_role(c['baseline'])
 def dr(a,b,t):
  key=(a,b,t['axis'],t['moving'],t['anchor'])
  if key not in cache:cache[key]=integration.direction(base,c['x'],c['ts'],c['fh'],*key)
  return cache[key]
 for i,t in enumerate(c['truth']):
  a,b=t['span'];row=dict(clip_id=c['clip_id'],kind=c['kind'],truth=t,manual_direction=dr(a,b,t),manual_old_family=family(oldrp,a,b),manual_new_family=family(obs['rp'],a,b),automatic={})
  for mode,es in choices.items():
   mm=matches[mode].get(i);e=es[mm[0]] if mm else None;rp=oldrp if mode in ['baseline','interval_only'] else obs['rp'];v=dict(matched=e is not None,overlap=float(mm[1]) if mm else 0.,event=e,direction=dr(e['a'],e['b'],t) if e else None,family=family(rp,e['a'],e['b']) if e else None,both_boundaries_in_range=bool(e and t['onset'][0]<=e['a']<=t['onset'][1] and t['offset'][0]<=e['b']<=t['offset'][1]));row['automatic'][mode]=v
  out.append(row)
 return out
def main():
 assert (U/'TRAINING_COMPLETE.json').exists() and not (U/'EVALUATION_COMPLETE.json').exists();cfg=read(U/'FROZEN.json')
 for p,h in {**cfg['pins'],**cfg['protections'],**read(U/'CODE_PINS.json'),**read(U/'EVAL_PINS.json')}.items():assert sha(p)==h,p
 torch.set_num_threads(4);torch.set_num_interop_threads(1);write(U/'CURRENT_EXECUTION.json',dict(status='EVALUATING',pid=os.getpid()));rr=old.rows();wm={r['word_id']:r['word'] for r in rr if r['word']>=0};base=prior.Network('coupled',read(old.P/'NORMALIZATION.json'));base.load_state_dict(torch.load(old.P/'runs/coupled_42/model.pt',weights_only=True,map_location='cpu'));base.eval()
 runtime=lc.be.ev.prepare_runtime();saved={r['clip_id']+'_'+r['kind']:r for r in read(L/'GUARDED_COMPLETE.json')['rows'] if r['seed']==42};histtruth=read(P/'EVALUATION_LABELS.json')['historical']
 for c in runtime:
  v=saved[c['key']];assert sha(v['result'])==v['sha256'];c['baseline']=read(v['result']);c['baseline_file']=v['result'];c['clip_id']=c['row']['clip_id'];c['kind']=c['row']['kind'];c['fh'],_=lc.be.ev.frozen_forward(base,c['x'],wm[c['baseline']['word']]);c['truth']=[dict(t,span=[t['onset'][0],t['offset'][1]]) for t in histtruth if t['clip_id']==c['clip_id']] if c['kind']=='full_normal' else []
 bn=bc.BoundaryNet();bn.load_state_dict(torch.load(B/'runs/42/model.pt',weights_only=True,map_location='cpu'));bn.eval();ln=lc.LinkNet();ln.load_state_dict(torch.load(L/'runs/42/model.pt',weights_only=True,map_location='cpu'));ln.eval();extra=[];anns=read(U/'EVALUATION_LABELS.json')['rows']
 for r in read(U/'EXTRACTION_COMPLETE.json')['rows']:
  assert sha(r['raw'])==r['raw_sha256'];z=np.load(r['raw']);n=len(z['frame_id'])
  for kind in ['normal','whole_reverse']:
   idx=np.arange(n)[::-1] if kind=='whole_reverse' else np.arange(n);ts=z['timestamp_us'];rt=lc.be.ev.rt;ob=rt.consumer.quarantine(z['xy_px'][idx],z['score'][idx],ts,int(z['source_width']),int(z['source_height']));cv=rt.consumer.A.convert(ob['xy_px'],z['score'][idx],ts,z['frame_id'],int(z['source_width']),int(z['source_height']));x=cv['tcn280'];fh,st=lc.be.ev.frozen_forward(base,x,wm.get(r['word_id'],-1));bp,q=lc.be.infer(bn,x,ts,fh);ee,pp=lc.be.propose(bp,q,st);lp=le.infer(ln,x,ts);bb=dict(phase_events=ee,phase_partials=pp,guide_link_probabilities=lp.tolist());truth=truth_rows(anns,r['clip_id']);holds=[dict(axis=a['axis'],moving=a['moving'],span=s) for a in anns if a['clip_id']==r['clip_id'] for s in a['holds']]
   if kind=='whole_reverse':
    for t in truth:
     on,off=t['onset'],t['offset'];t['onset']=[n-1-off[1],n-1-off[0]];t['offset']=[n-1-on[1],n-1-on[0]];t['span']=[t['onset'][0],t['offset'][1]];t['direction']=2-t['direction']
    for h in holds:h['span']=[n-1-h['span'][1],n-1-h['span'][0]]
   dest=U/'additional_baseline'/f"{r['clip_id']}_{kind}.json";write(dest,bb);extra.append(dict(clip_id=r['clip_id'],kind=kind,x=x,ts=ts,fh=fh,baseline=bb,baseline_file=str(dest),truth=truth,holds=holds))
 contracts=ep.family_contracts();allout=[];diag=[];adiag=[];holds=[];obsrows=[]
 for seed in cfg['seeds']:
  net=JointNet();net.load_state_dict(torch.load(U/'runs'/str(seed)/'model.pt',weights_only=True,map_location='cpu'));net.eval()
  for group,items in [('historical',runtime),('additional',extra)]:
   for c in items:
    start=time.perf_counter();ob=infer(net,c['x'],c['ts'],c['fh'],c['baseline']['phase_events']+c['baseline']['phase_partials']);seconds=time.perf_counter()-start;obsfile=U/'observations'/str(seed)/f"{c['clip_id']}_{c['kind']}.json";write(obsfile,dict(candidates=ob['candidates'],selected=ob['selected'],links=ob['links'],role_probabilities=ob['rp'].tolist(),boundary_probabilities=ob['bp'].tolist(),seconds=seconds,frames=len(c['x'])));obsrows.append(dict(seed=seed,group=group,clip_id=c['clip_id'],kind=c['kind'],file=str(obsfile),sha256=sha(obsfile),seconds=seconds))
    ds=[dict(seed=seed,**v) for v in diagnose(c,ob,base)];(diag if group=='historical' else adiag).extend(ds)
    if group=='historical':
     for mode in MODES[1:]:
      t=time.perf_counter();out=supplement(c['baseline'],ob,c['x'],c['ts'],c['fh'],base,contracts[c['baseline']['word']],mode);elapsed=seconds+time.perf_counter()-t;protected={}
      for key in ['tcn_top3','path','frame_id','phase_events','phase_partials','phase_probability']:
       assert out[key]==c['baseline'][key];protected[key]=jsonhash(out[key])
      for a,b in zip(out['cards'],c['baseline']['cards']):assert a.get('shape_evidence')==b.get('shape_evidence') and a.get('motion_evidence')==b.get('motion_evidence')
      p=U/'feedback'/str(seed)/mode/f"{c['key']}.json";write(p,dict(baseline=c['baseline_file'],baseline_sha256=sha(c['baseline_file']),cards=out['cards'],bindings=out['joint_bindings'],order=out['joint_order_checks'],completion=out['joint_completion'],protected=protected));allout.append(dict(seed=seed,mode=mode,clip_id=c['clip_id'],kind=c['kind'],baseline=c['baseline_file'],result=str(p),sha256=sha(p),before=le.compact(c['baseline']),after=le.compact(out),seconds=elapsed,frames=len(c['x'])))
    else:
     for h in c['holds']:
      k=h['axis']*2+(h['moving']=='L');a,b=h['span'];valid=float((ob['q'][a:b+1,k,-2:].max(-1)>.5).mean());holds.append(dict(seed=seed,clip_id=c['clip_id'],kind=c['kind'],truth=h,coverage=valid,false_selected=[e for e in ob['selected'] if e['axis']==h['axis'] and e['moving']==h['moving'] and lc.be.ev.iou([a,b],[e['a'],e['b']])>=.5]))
   event('evaluated_group',seed=seed,group=group,clips=len(items))
 write(U/'FULLPATH_COMPLETE.json',dict(rows=allout));write(U/'HISTORICAL_DIAGNOSTIC.json',diag);write(U/'ADDITIONAL_DIAGNOSTIC.json',adiag);write(U/'HOLD_DIAGNOSTIC.json',holds);write(U/'OBSERVATIONS.json',dict(rows=obsrows));write(U/'EVALUATION_COMPLETE.json',dict(seeds=cfg['seeds'],historical_conditions=78,module_comparisons=3,additional_originals=10,additional_conditions=20,additional_component_only=True));write(U/'CURRENT_EXECUTION.json',dict(status='EVALUATED_PENDING_VERIFICATION',pid=None))
if __name__=='__main__':
 try:main()
 except BaseException:(U/f'EVAL_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),'utf-8');raise
