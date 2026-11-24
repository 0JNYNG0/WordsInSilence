from episode_core import *
from integration import supplement,direction
import copy,traceback
def pooled_family(prob,a,b,legacy=False):
 p=np.asarray(prob)[a:b+1].mean(0)
 if legacy:
  q=np.zeros(8)
  for k,v in enumerate(p):q[MAP[k]]+=v
  p=q
 return dict(prediction=int(p.argmax()),confidence=float(p.max()),probabilities=p.tolist())
def diagnose(c,rp,events,parts,base,truth):
 out=[];x,ts,fh=c['x'],c['ts'],c['fh'];old_events=c['baseline']['phase_events']+c['baseline']['phase_partials'];candidates={name:[e for e in es if e['b']-e['a']>=3] for name,es in {'old':old_events,'new':events+parts}.items()}
 # One detected event cannot stand in for two repetitions in the evaluation.
 matches={name:lc.be.match_events(es,[dict(t,span=[t['onset'][0],t['offset'][1]]) for t in truth]) for name,es in candidates.items()}
 for ti,t in enumerate(truth):
  a,b=t['onset'][0],t['offset'][1];manual=direction(base,x,ts,fh,a,b,t['axis'],t['moving'],t['anchor']);row=dict(clip_id=c['clip_id'],kind=c['kind'],truth=t,manual_direction=manual,manual_old_family=pooled_family(c['baseline']['guide_link_probabilities'],a,b,True),manual_new_family=pooled_family(rp,a,b),automatic={})
  for name,es in candidates.items():
   match=matches[name].get(ti);matched=match is not None;e=es[match[0]] if matched else None;overlap=float(match[1]) if match else 0.
   if e:
    d=direction(base,x,ts,fh,e['a'],e['b'],t['axis'],t['moving'],t['anchor']);fam=pooled_family(rp if name=='new' else c['baseline']['guide_link_probabilities'],e['a'],e['b'],name=='old');exact=t['onset'][0]<=e['a']<=t['onset'][1] and t['offset'][0]<=e['b']<=t['offset'][1]
   else:d=None;fam=None;exact=False
   row['automatic'][name]=dict(matched=matched,overlap=overlap,event=e,direction=d,family=fam,both_boundaries_in_range=exact)
  out.append(row)
 return out
def main():
 assert (U/'TRAINING_COMPLETE.json').exists() and not (U/'EVALUATION_COMPLETE.json').exists();cfg=read(U/'FROZEN.json');ep=read(U/'EVALUATION_FROZEN.json')
 for p,h in {**cfg['pins'],**cfg['protections'],**read(U/'CODE_PINS.json'),**ep['pins']}.items():assert sha(p)==h,p
 torch.set_num_threads(4);torch.set_num_interop_threads(1);runtime=lc.be.ev.prepare_runtime();saved={r['clip_id']+'_'+r['kind']:r for r in read(L/'GUARDED_COMPLETE.json')['rows'] if r['seed']==42};contracts=read(U/'GUIDE_FAMILIES.json')['words'];labels=read(U/'EVALUATION_LABELS.json');wm={r['word_id']:r['word'] for r in old.rows() if r['word']>=0}
 base=prior.Network('coupled',read(old.P/'NORMALIZATION.json'));base.load_state_dict(torch.load(old.P/'runs/coupled_42/model.pt',weights_only=True,map_location='cpu'));base.eval()
 for c in runtime:
  r=saved[c['key']];assert sha(r['result'])==r['sha256'];c['baseline']=read(r['result']);c['baseline_file']=r['result'];c['fh'],_=lc.be.ev.frozen_forward(base,c['x'],wm[c['baseline']['word']]);c['clip_id']=c['row']['clip_id'];c['kind']=c['row']['kind'];c['truth']=[t for t in labels['historical'] if t['clip_id']==c['clip_id']] if c['kind']=='full_normal' else []
 # Additional data: only component diagnosis, no fabricated TCN/DTW or overall feedback scores.
 additional=[];bn=bc.BoundaryNet();bn.load_state_dict(torch.load(B/'runs/42/model.pt',weights_only=True,map_location='cpu'));bn.eval();ln=lc.LinkNet();ln.load_state_dict(torch.load(L/'runs/42/model.pt',weights_only=True,map_location='cpu'));ln.eval()
 for r in read(U/'EXTRACTION_COMPLETE.json')['rows']:
  assert sha(r['raw'])==r['raw_sha256'];z=np.load(r['raw']);n=len(z['frame_id'])
  for kind in ['normal','whole_reverse']:
   rev=kind=='whole_reverse';idx=np.arange(n)[::-1] if rev else np.arange(n);raw={k:z[k].copy() for k in ['timestamp_us','frame_id','source_width','source_height']};xy=z['xy_px'][idx].copy();score=z['score'][idx].copy();rt=lc.be.ev.rt
   ob=rt.consumer.quarantine(xy,score,raw['timestamp_us'],int(raw['source_width']),int(raw['source_height']));cv=rt.consumer.A.convert(ob['xy_px'],score,raw['timestamp_us'],raw['frame_id'],int(raw['source_width']),int(raw['source_height']));x,ts=cv['tcn280'],cv['timestamp_us'];fh,st=lc.be.ev.frozen_forward(base,x,wm[r['word_id']]);bp,q=lc.be.infer(bn,x,ts,fh);ee,pp=lc.be.propose(bp,q,st);lp=le.infer(ln,x,ts);bb=dict(phase_events=ee,phase_partials=pp,guide_link_probabilities=lp.tolist());truth=copy.deepcopy([t for t in labels['additional'] if t['clip_id']==r['clip_id']])
   if rev:
    for t in truth:
     on,off=t['onset'],t['offset'];t['onset']=[n-1-off[1],n-1-off[0]];t['offset']=[n-1-on[1],n-1-on[0]];t['direction']=2-t['direction']
   dest=U/'additional_baseline'/f"{r['clip_id']}_{kind}.json";write(dest,bb);additional.append(dict(clip_id=r['clip_id'],kind=kind,x=x,ts=ts,fh=fh,baseline=bb,truth=truth,baseline_file=str(dest)))
 allrows=[];diag=[];adiag=[]
 for seed in cfg['seeds']:
  net=EpisodeNet();net.load_state_dict(torch.load(U/'runs'/str(seed)/'model.pt',weights_only=True,map_location='cpu'));net.eval()
  for c in runtime:
   t=time.perf_counter();rp,bp,q=infer(net,c['x'],c['ts'],c['fh']);out=supplement(c['baseline'],rp,bp,q,c['x'],c['ts'],c['fh'],base,contracts[c['baseline']['word']]);seconds=time.perf_counter()-t
   for k in ['tcn_top3','path','frame_id','phase_events','phase_partials']:assert out[k]==c['baseline'][k]
   for a,b in zip(out['cards'],c['baseline']['cards']):assert a.get('shape_evidence')==b.get('shape_evidence') and a.get('motion_evidence')==b.get('motion_evidence')
   dest=U/'fullpath'/str(seed)/c['key']/'after.json';write(dest,out);allrows.append(dict(seed=seed,clip_id=c['clip_id'],kind=c['kind'],baseline=c['baseline_file'],baseline_sha256=sha(c['baseline_file']),result=str(dest),sha256=sha(dest),before=le.compact(c['baseline']),after=le.compact(out),seconds=seconds,frames=len(c['x'])))
   diag.extend(dict(seed=seed,**d) for d in diagnose(c,rp,out['episode_events'],out['episode_partials'],base,c['truth']))
  for c in additional:
   t=time.perf_counter();rp,bp,q=infer(net,c['x'],c['ts'],c['fh']);ee,pp=decode(rp,bp,q);seconds=time.perf_counter()-t;dest=U/'additional_results'/str(seed)/f"{c['clip_id']}_{c['kind']}.json";write(dest,dict(role_probabilities=rp.tolist(),events=ee,partials=pp,seconds=seconds,full_feedback=False));adiag.extend(dict(seed=seed,**d) for d in diagnose(c,rp,ee,pp,base,c['truth']))
  event('evaluated',seed=seed,changed=sum(r['before']!=r['after'] for r in allrows if r['seed']==seed))
 write(U/'FULLPATH_COMPLETE.json',dict(rows=allrows));write(U/'DIAGNOSTIC_SCORES.json',diag);write(U/'ADDITIONAL_DIAGNOSTIC_SCORES.json',adiag);write(U/'EVALUATION_COMPLETE.json',dict(seeds=cfg['seeds'],historical_outputs=len(allrows),diagnostic_events=len(diag),additional_diagnostic_events=len(adiag),additional_originals=4,additional_component_only=True));write(U/'CURRENT_EXECUTION.json',dict(status='EVALUATED_PENDING_VERIFICATION',pid=None))
if __name__=='__main__':
 try:main()
 except BaseException:(U/f'EVAL_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),'utf-8');raise
