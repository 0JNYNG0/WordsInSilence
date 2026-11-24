from joint_core import *
import traceback
V=U/'proposal_adapter';ev=load('_joint_diagnostics',U/'evaluate.py')
def extra_inputs(manifest,labels,base,wm,fresh):
 out=[]
 for r in read(manifest)['rows']:
  z=np.load(r['raw']);assert sha(r['raw'])==r['raw_sha256'];n=len(z['frame_id'])
  for kind in ['normal','whole_reverse']:
   ii=np.arange(n)[::-1] if kind=='whole_reverse' else np.arange(n);ts=z['timestamp_us'];rt=lc.be.ev.rt;ob=rt.consumer.quarantine(z['xy_px'][ii],z['score'][ii],ts,int(z['source_width']),int(z['source_height']));cv=rt.consumer.A.convert(ob['xy_px'],z['score'][ii],ts,z['frame_id'],int(z['source_width']),int(z['source_height']));x=cv['tcn280'];fh,st=lc.be.ev.frozen_forward(base,x,wm.get(r['word_id'],-1));truth=ev.truth_rows(labels,r['clip_id']);holds=[dict(axis=a['axis'],moving=a['moving'],span=s) for a in labels if a['clip_id']==r['clip_id'] for s in a['holds']]
   if kind=='whole_reverse':
    for t in truth:
     on,off=t['onset'],t['offset'];t.update(onset=[n-1-off[1],n-1-off[0]],offset=[n-1-on[1],n-1-on[0]],direction=2-t['direction']);t['span']=[t['onset'][0],t['offset'][1]]
    for h in holds:h['span']=[n-1-h['span'][1],n-1-h['span'][0]]
   p=(V if fresh else U)/'additional_baseline'/f"{r['clip_id']}_{kind}.json"
   if fresh:
    bp,q=lc.be.infer(BN,x,ts,fh);ee,pp=lc.be.propose(bp,q,st);lp=le.infer(LN,x,ts);bb=dict(phase_events=ee,phase_partials=pp,guide_link_probabilities=lp.tolist());write(p,bb)
   else:bb=read(p)
   out.append(dict(clip_id=r['clip_id'],kind=kind,key=r['clip_id']+'_'+kind,x=x,ts=ts,fh=fh,truth=truth,holds=holds,baseline=bb,baseline_file=str(p)))
 return out
def main():
 global BN,LN
 assert (V/'COMPLETE.json').exists() and not (V/'EVALUATION_COMPLETE.json').exists()
 for p,h in read(V/'EVAL_PINS.json').items():assert sha(p)==h,p
 torch.set_num_threads(4);torch.set_num_interop_threads(1);rr=old.rows();wm={r['word_id']:r['word'] for r in rr if r['word']>=0};base=prior.Network('coupled',read(old.P/'NORMALIZATION.json'));base.load_state_dict(torch.load(old.P/'runs/coupled_42/model.pt',weights_only=True,map_location='cpu'));base.eval();runtime=lc.be.ev.prepare_runtime();saved={r['clip_id']+'_'+r['kind']:r for r in read(L/'GUARDED_COMPLETE.json')['rows'] if r['seed']==42}
 for c in runtime:
  r=saved[c['key']];assert sha(r['result'])==r['sha256'];c.update(baseline=read(r['result']),baseline_file=r['result'],clip_id=c['row']['clip_id'],kind=c['row']['kind'],truth=[]);c['fh'],_=lc.be.ev.frozen_forward(base,c['x'],wm[c['baseline']['word']])
 BN=bc.BoundaryNet();BN.load_state_dict(torch.load(B/'runs/42/model.pt',weights_only=True,map_location='cpu'));BN.eval();LN=lc.LinkNet();LN.load_state_dict(torch.load(L/'runs/42/model.pt',weights_only=True,map_location='cpu'));LN.eval();replay=extra_inputs(U/'EXTRACTION_COMPLETE.json',read(U/'EVALUATION_LABELS.json')['rows'],base,wm,False);fresh=extra_inputs(V/'EXTRACTION_COMPLETE.json',read(V/'EVALUATION_LABELS.json')['rows'],base,wm,True);rows=[];diag=[];obsrows=[];holds=[];contracts=ep.family_contracts()
 for seed in [42,43,44]:
  net=JointNet();net.load_state_dict(torch.load(U/'runs/42/model.pt',weights_only=True,map_location='cpu'));ck=torch.load(V/'runs'/str(seed)/'model.pt',weights_only=True,map_location='cpu');net.quality.load_state_dict(ck['quality']);net.pair.load_state_dict(ck['pair']);net.eval()
  for group,items in [('historical_replay',runtime),('additional_replay',replay),('reserved_id12',fresh)]:
   for c in items:
    t=time.perf_counter();ob=infer(net,c['x'],c['ts'],c['fh'],c['baseline']['phase_events']+c['baseline']['phase_partials']);seconds=time.perf_counter()-t;p=V/'observations'/str(seed)/f"{c['key']}.json";write(p,dict(candidates=ob['candidates'],selected=ob['selected'],links=ob['links'],role_probabilities=ob['rp'].tolist(),seconds=seconds));obsrows.append(dict(seed=seed,group=group,clip_id=c['clip_id'],kind=c['kind'],file=str(p),sha256=sha(p),seconds=seconds));diag.extend(dict(seed=seed,group=group,**d) for d in ev.diagnose(c,ob,base))
    if group=='historical_replay':
     for mode in ['interval_only','role_only','combined']:
      t=time.perf_counter();out=supplement(c['baseline'],ob,c['x'],c['ts'],c['fh'],base,contracts[c['baseline']['word']],mode);elapsed=seconds+time.perf_counter()-t
      for k in ['tcn_top3','path','frame_id','phase_events','phase_partials']:assert out[k]==c['baseline'][k]
      for a,b in zip(out['cards'],c['baseline']['cards']):assert a.get('shape_evidence')==b.get('shape_evidence') and a.get('motion_evidence')==b.get('motion_evidence')
      dest=V/'feedback'/str(seed)/mode/f"{c['key']}.json";write(dest,dict(baseline=c['baseline_file'],baseline_sha256=sha(c['baseline_file']),cards=out['cards'],bindings=out['joint_bindings'],order=out['joint_order_checks'],completion=out['joint_completion']));rows.append(dict(seed=seed,mode=mode,clip_id=c['clip_id'],kind=c['kind'],result=str(dest),sha256=sha(dest),before=le.compact(c['baseline']),after=le.compact(out),seconds=elapsed))
    else:
     for h in c['holds']:holds.append(dict(seed=seed,group=group,clip_id=c['clip_id'],kind=c['kind'],truth=h,false_selected=[e for e in ob['selected'] if e['axis']==h['axis'] and e['moving']==h['moving'] and lc.be.ev.iou(h['span'],[e['a'],e['b']])>=.5]))
   event('adapter_evaluated',seed=seed,group=group,clips=len(items))
 write(V/'FULLPATH_COMPLETE.json',dict(rows=rows));write(V/'DIAGNOSTIC.json',diag);write(V/'OBSERVATIONS.json',dict(rows=obsrows));write(V/'HOLD_DIAGNOSTIC.json',holds);write(V/'EVALUATION_COMPLETE.json',dict(seeds=[42,43,44],common_encoder_parent=42,new_fine_evaluation_originals=6,new_conditions=12,new_movement_legs=10,old_replays_are_not_new_test=True))
if __name__=='__main__':
 try:main()
 except BaseException:(V/f'EVAL_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),'utf-8');raise
