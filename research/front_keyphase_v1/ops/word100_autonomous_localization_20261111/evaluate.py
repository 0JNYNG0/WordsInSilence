from localization_core import *
from guide_binding import attach
import traceback,os
sys.modules['motion_core']=mc
me=load('_autoloc_previous_eval_helpers',M/'evaluate.py')
def unique(es):return list({(e['a'],e['b'],e['axis'],e['moving']):e for e in es if e['b']-e['a']>=3}.values())
def jsonhash(d):return hashlib.sha256(json.dumps(d,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
def prepare(base):
 rows=old.rows();wm={r['word_id']:r['word'] for r in rows if r['word']>=0};byclip={r['clip_id']:r for r in rows};pins={v['key']:v for v in read(M/'BASELINE_PINS.json')};out=[]
 saved={(r['clip_id'],r['kind']):r for r in read(L/'GUARDED_COMPLETE.json')['rows'] if r['seed']==42};motion={(r['seed'],r['clip_id'],r['kind']):r for r in read(M/'FULLPATH_COMPLETE.json')['rows'] if r['mode']=='motion_only'}
 for c in lc.be.ev.prepare_runtime():
  clip,kind=c['row']['clip_id'],c['row']['kind'];r=saved[clip,kind];assert sha(r['result'])==r['sha256'];before=read(r['result']);c.update(clip_id=clip,kind=kind,group='historical_replay',baseline=before,baseline_file=r['result'],truth=[],holds=[],background=[],motion_outputs={s:motion[s,clip,kind] for s in [42,43,44]});out.append(c)
 observations={(r['clip_id'],r['kind']):r for r in read(V/'OBSERVATIONS.json')['rows'] if r['seed']==42}
 for group,folder in [('prior16_replay',J),('prior16_replay',V)]:
  anns=read(folder/'EVALUATION_LABELS.json')['rows']
  for r in read(folder/'EXTRACTION_COMPLETE.json')['rows']:
   assert sha(r['raw'])==r['raw_sha256']
   for kind in ['normal','whole_reverse']:
    x,ts=me.convert(r['raw'],kind=='whole_reverse');tt,hh=me.truths(anns,r['clip_id'],len(x),kind=='whole_reverse');p=folder/'additional_baseline'/f"{r['clip_id']}_{kind}.json";ob=read(observations[r['clip_id'],kind]['file']);out.append(dict(key=r['clip_id']+'_'+kind,clip_id=r['clip_id'],kind=kind,group=group,x=x,ts=ts,truth=tt,holds=hh,background=[],baseline=read(p),baseline_file=str(p),selected=ob['selected']))
 anns=read(M/'EVALUATION_LABELS.json')['rows']
 for r in read(M/'RESERVED_PUBLIC.json')['rows']:
  rr=byclip[r['clip_id']];assert r['source_sha256']==rr['source_sha256'];x0=prior.physical(rr['x_full']);ts0=rr['timestamp_us']
  for kind in ['normal','whole_reverse','speed075','speed125']:
   rev=kind=='whole_reverse';x=x0[::-1].copy() if rev else x0;ts=(np.asarray(ts0)/({'speed075':.75,'speed125':1.25}.get(kind,1))).astype(np.int64);tt,hh=me.truths(anns,r['clip_id'],len(x),rev);p=M/'fresh_baseline'/f"{r['clip_id']}_{kind}.json";d=read(p);out.append(dict(key=r['clip_id']+'_'+kind,clip_id=r['clip_id'],kind=kind,group='prior13_replay',x=x,ts=ts,truth=tt,holds=hh,background=[],baseline=d['baseline'],baseline_file=str(p),selected=d['selected']))
 for c in out:
  p=pins[c['key']];assert sha(p['baseline_file'])==p['baseline_sha256'];c['old_cache']={tuple(r['key']):r['value'] for r in p['old_direction']};c['fh'],_=lc.be.ev.frozen_forward(base,c['x'],wm.get(c['clip_id'].split('_')[2],-1))
 # Reserved record14 is first-scored now; all baselines are frozen before scoring.
 bn=bc.BoundaryNet();bn.load_state_dict(torch.load(B/'runs/42/model.pt',weights_only=True,map_location='cpu'));bn.eval();ln=lc.LinkNet();ln.load_state_dict(torch.load(L/'runs/42/model.pt',weights_only=True,map_location='cpu'));ln.eval();jn=jc.JointNet();jn.load_state_dict(torch.load(J/'runs/42/model.pt',weights_only=True,map_location='cpu'));ad=torch.load(V/'runs/42/model.pt',weights_only=True,map_location='cpu');jn.quality.load_state_dict(ad['quality']);jn.pair.load_state_dict(ad['pair']);jn.eval();anns=read(U/'EVALUATION_LABELS.json')['rows']
 for r in read(U/'EXTRACTION_COMPLETE.json')['rows']:
  if r['person']!=14:continue
  assert sha(r['raw'])==r['raw_sha256']
  for kind in ['normal','whole_reverse','speed075','speed125']:
   rev=kind=='whole_reverse';x,ts=me.convert(r['raw'],rev);ts=(ts/({'speed075':.75,'speed125':1.25}.get(kind,1))).astype(np.int64);fh,stage=lc.be.ev.frozen_forward(base,x,-1);bp,q=lc.be.infer(bn,x,ts,fh);ee,pp=lc.be.propose(bp,q,stage);lp=le.infer(ln,x,ts);bb=dict(phase_events=ee,phase_partials=pp,guide_link_probabilities=lp.tolist());ob=jc.infer(jn,x,ts,fh,ee+pp);tt,hh=me.truths(anns,r['clip_id'],len(x),rev);bg=[]
   for ar in anns:
    if ar['clip_id']==r['clip_id']:
     for a,b in ar['background']:bg.append(dict(axis=ar['axis'],moving=ar['moving'],span=[len(x)-1-b,len(x)-1-a] if rev else [a,b]))
   key=r['clip_id']+'_'+kind;p=U/'reserved_baseline'/f'{key}.json';write(p,dict(baseline=bb,selected=ob['selected'],source_sha256=r['source_sha256']));out.append(dict(key=key,clip_id=r['clip_id'],kind=kind,group='reserved14',x=x,ts=ts,fh=fh,truth=tt,holds=hh,background=bg,baseline=bb,baseline_file=str(p),selected=ob['selected'],old_cache={}))
 event('comparison_inputs_ready',conditions=len(out));return out
def old_direction(c,base,a,b,k,moving='R',anchor='L'):
 key=(int(a),int(b),int(k),moving,anchor)
 if key not in c['old_cache']:c['old_cache'][key]=jc.integration.direction(base,c['x'],c['ts'],c['fh'],*key[:3],key[3],key[4])
 return c['old_cache'][key]
def main():
 assert not (U/'EVALUATION_COMPLETE.json').exists();cfg=read(U/'FROZEN.json')
 for p,h in {**cfg['pins'],**cfg['protections'],**read(U/'LABELS_FROZEN.json')['files']}.items():assert sha(p)==h,p
 torch.set_num_threads(4);torch.set_num_interop_threads(1);write(U/'EVAL_PINS.json',{str(p):sha(p) for p in [U/'localization_core.py',U/'evaluate.py',U/'guide_binding.py',U/'GUIDE_ENDPOINT_PROFILES.json']});write(U/'CURRENT_EXECUTION.json',dict(status='EVALUATING',pid=os.getpid()))
 base=prior.Network('coupled',read(old.P/'NORMALIZATION.json'));base.load_state_dict(torch.load(old.P/'runs/coupled_42/model.pt',weights_only=True,map_location='cpu'));base.eval();t=time.perf_counter();cases=prepare(base);preparation_seconds=time.perf_counter()-t;profiles=read(U/'GUIDE_ENDPOINT_PROFILES.json')['profiles'];results=[];frs=[];eventscores=[];holds=[]
 for seed in cfg['seeds']:
  net=Localizer(read(U/'NORMALIZATION.json'));net.load_state_dict(torch.load(U/'runs'/str(seed)/'model.pt',weights_only=True,map_location='cpu'));net.eval();motion=mc.MotionNet();motion.load_state_dict(torch.load(M/'runs'/str(seed)/'model.pt',weights_only=True,map_location='cpu'),strict=False);motion.eval()
  for c in cases:
   t=time.perf_counter();ob=infer_local(net,c['x'],c['ts'],c['fh'],cfg);inference_seconds=time.perf_counter()-t;es=unique(ob['events']+ob['partials']);meta=dict(seed=seed,group=c['group'],clip_id=c['clip_id'],kind=c['kind']);npz=U/'observations'/str(seed)/f"{c['key']}.npz";npz.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(npz,state=ob['state'],boundary=ob['boundary'],valid=ob['valid']);epath=npz.with_suffix('.json');write(epath,dict(events=es,arrays_sha256=sha(npz),frames=len(c['x']),seconds=inference_seconds))
   if c['group']=='historical_replay':
    oldc=c['baseline'];mr=c['motion_outputs'][seed];assert sha(mr['result'])==mr['sha256'];md=read(mr['result']);mout=copy.deepcopy(oldc);mout['cards']=md['cards'];pairs=[('localization_only',oldc,False),('combined',mout,True)]
    for mode,before,combined in pairs:
     t=time.perf_counter();out=attach(before,es,c['x'],c['ts'],c['fh'],base,motion,profiles,combined,c['old_cache']);seconds=time.perf_counter()-t;p=U/'feedback'/str(seed)/mode/f"{c['key']}.json"
     for a,b in zip(out['cards'],oldc['cards']):assert a.get('shape_evidence')==b.get('shape_evidence') and a.get('motion_evidence')==b.get('motion_evidence')
     protected={k:jsonhash(out[k]) for k in ['tcn_top3','path','frame_id','phase_events','phase_partials']};assert protected=={k:jsonhash(oldc[k]) for k in protected}
     write(p,dict(baseline=c['baseline_file'],baseline_sha256=sha(c['baseline_file']),cards=out['cards'],bindings=out['localization_bindings'],ledger=out['localization_ledger'],protected=protected))
     frs.append(dict(meta,mode=mode,baseline=le.compact(oldc),motion_only=mr['after'],after=le.compact(out),result=str(p),sha256=sha(p),localizer_seconds=inference_seconds,binding_seconds=seconds))
   else:
    be=unique(c['baseline']['phase_events']+c['baseline']['phase_partials']+c['selected']);ue=unique(be+es);matches={name:lc.be.match_events(ev,c['truth']) for name,ev in [('baseline',be),('localized',es),('union',ue)]}
    for i,truth in enumerate(c['truth']):
     record=dict(meta,truth=truth,automatic={});a,b=truth['span'];od=old_direction(c,base,a,b,truth['axis'],truth['moving'],truth['anchor']);nd=mc.direction(motion,c['x'],c['ts'],a,b,truth['axis'],truth['moving'],truth['anchor']);record['manual']=dict(before=od,after=mc.fused(od,nd))
     for name,ee in [('baseline',be),('localized',es),('union',ue)]:
      match=matches[name].get(i);v=dict(matched=match is not None,event=None,overlap=0.)
      if match:
       e=ee[match[0]];od=old_direction(c,base,e['a'],e['b'],e['axis'],e['moving'],truth['anchor']);nd=mc.direction(motion,c['x'],c['ts'],e['a'],e['b'],e['axis'],e['moving'],truth['anchor']);v.update(event=e,overlap=float(match[1]),before=od,after=mc.fused(od,nd),onset_error=min(abs(e['a']-z) for z in truth['onset']),offset_error=min(abs(e['b']-z) for z in truth['offset']))
      record['automatic'][name]=v
     results.append(record)
    def in_reviewed_nonmotion(e):return any(e['axis']==a['axis'] and e['moving']==a['moving'] and a['span'][0]<=e['a']<=e['b']<=a['span'][1] for a in c['holds']+c['background'])
    eventscores.append(dict(meta,truth_events=len(c['truth']),baseline_events=len(be),new_events=len(es),union_events=len(ue),baseline_matched=len(matches['baseline']),new_matched=len(matches['localized']),union_matched=len(matches['union']),known_nonmotion_baseline=sum(in_reviewed_nonmotion(e) for e in be),known_nonmotion_new=sum(in_reviewed_nonmotion(e) for e in es),unreviewed_events_not_scored_as_false=True,seconds=inference_seconds,events=es))
    for h in c['holds']:
     a,b=h['span'];k=h['axis']*2+(h['moving']=='L');known=ob['valid'][a:b+1,k];pred=ob['state'][a:b+1,k].argmax(-1);holds.append(dict(meta,truth=h,frames=b-a+1,observed=int(known.sum()),predicted_moving=int((((pred==0)|(pred==2))&known).sum()),predicted_hold=int(((pred==1)&known).sum()),predicted_background=int(((pred==3)&known).sum())))
  event('new_localization_evaluated',seed=seed,conditions=len(cases))
 write(U/'DIRECTION_RESULTS.json',results);write(U/'EVENT_RESULTS.json',eventscores);write(U/'HOLD_RESULTS.json',holds);write(U/'FULLPATH_COMPLETE.json',dict(rows=frs));write(U/'DECISION_CACHE.json',[dict(key=c['key'],baseline_file=c['baseline_file'],baseline_sha256=sha(c['baseline_file']),old_direction=[dict(key=list(k),value=v) for k,v in c['old_cache'].items()]) for c in cases]);write(U/'EVALUATION_COMPLETE.json',dict(conditions=len(cases),seeds=cfg['seeds'],preparation_seconds=preparation_seconds,old_TCN_DTW_runs=0,old_completed_predictions_reused=True,new_old_classifier_calls='Only newly proposed intervals or first-scored14 inputs',reserved14_originals=6));write(U/'CURRENT_EXECUTION.json',dict(status='EVALUATED_PENDING_VERIFICATION',pid=None))
if __name__=='__main__':
 try:main()
 except BaseException:(U/f'EVAL_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),'utf-8');raise
