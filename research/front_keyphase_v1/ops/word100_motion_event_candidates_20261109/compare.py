"""One frozen public comparison. Cached predictions retained, no private input."""
from core import *
import copy,statistics,traceback
sys.path.insert(0,str(D));consumer=load('_candidate_compare_consumer',D/'consumer.py')
B=O/'word100_shared_motion_learning_20261109';bcore=load('_candidate_frozen_observer',B/'core.py');saved_core=sys.modules.get('core');sys.modules['core']=bcore
prior=load('_candidate_prior_unit',U/'unit.py');headclass=load('_candidate_deployed_head',D/'model.py').EvidenceHeads
follow=load('_candidate_conflict_guard',P/'followup_contract.py')
if saved_core is not None:sys.modules['core']=saved_core
def expand(prob):
 out=np.zeros((len(prob),14),np.float32);out[:,0]=prob[:,0]
 for i,group in enumerate(GROUPS):out[:,group[0]+1]=prob[:,i+1]
 return out
def combine(word,oldphase,geo,d0,d1,attrs,raw,ts,fid,presence):
 first=consumer.assess(word,oldphase,d0,d1,raw,ts,fid);second=consumer.assess(word,geo,d0,d1,raw,ts,fid);cards=[];assigned=np.zeros(len(ts),bool);rescues=0
 for a,b in zip(first['cards'],second['cards']):
  use_geometry=a.get('query_indices') is None and b.get('query_indices') is not None
  chosen=copy.deepcopy(b if use_geometry else a);rescues+=int(use_geometry)
  # Inspect the selected route with the same deployed shape/direction evidence.
  one=dict(first,cards=[chosen]);out=consumer.refine(one,attrs,raw,ts,fid,geo if use_geometry else oldphase);chosen=out['cards'][0]
  bank=[]
  for route,c in [('deployed',a),('geometry',b)]:
   for e in c['phase_evidence']:
    for s in e['available_runs']:bank.append(dict(route=route,group=e['group'],span=s))
  for e,ga in zip(chosen['phase_evidence'],b['phase_evidence']):
   original=next(z for z in a['phase_evidence'] if z['group']==e['group'])
   e['available_runs']=[list(s) for s in sorted({tuple(s) for s in original['available_runs']+ga['available_runs']})]
  chosen['candidate_bank']=bank;chosen['selected_route']='geometry' if use_geometry else 'deployed'
  for e in chosen['phase_evidence']:
   for a0,b0 in e['available_runs']:assigned[a0:b0+1]=True
  cards.append(chosen)
 out=consumer.apply_coverage(dict(first,cards=cards));out=follow.apply_guard(out,attrs,raw,ts)
 # Reuse the frozen absence evidence; a conflict can only withhold success.
 # No probability from an event detector invents that event's location.
 hand=np.stack([raw[:,237:258].sum(1)>=5,raw[:,258:279].sum(1)>=5],1)
 active=np.flatnonzero(oldphase[:,1:].max(1)>.4);start=int(active[0]) if len(active) else 0
 for index,c in enumerate(out['cards']):
  groups=sorted({MAP[g]-1 for gg in c['groups'] for g in gg});c['event_presence_probability']=[float(presence[g]) for g in groups]
  sides=[0 if h=='L' else 1 for h in c['hands']];missing=consumer.assess.__globals__['runs']((~hand[:,sides].all(1))&(np.arange(len(ts))>=start),ts,300)
  if c.get('required',True) and all(presence[g]<=.35 for g in groups):
   before=any(x.get('query_indices') is not None for x in out['cards'][:index]);localized=all(e['span'] is not None for e in c['phase_evidence'])
   if not missing and before and out['return_to_rest_evidence'] and not localized:c.update(status='missing_candidate',display_status='uncertain',uncertainty_reason='learned_absence_with_observed_prior_event_and_rest',message='이 동작을 이어서 했는지 참고와 확인해 주세요.',reference_only=True)
   elif c['display_status']=='similar':c.update(status='unknown',display_status='uncertain',uncertainty_reason='presence_localization_conflict',message='이 동작을 했는지 근거가 서로 달라 아직 판단하기 어려워요.')
  c['candidate_message']=c['message'];c['educational_wrong']=False
 observable=hand.any(1);rr=consumer.assess.__globals__['runs'](observable&~assigned,ts,60)
 out.update(geometry_rescues=rescues,unassigned_observable_intervals=rr,unassigned_is_not_error=True,all_candidates_preserved=True,not_dtw_completed=True)
 return out
def main():
 assert (N/'TRAINING_COMPLETE.json').exists();assert not (N/'COMPARISON.json').exists();torch.set_num_threads(4);torch.set_num_interop_threads(1)
 write(N/'EVAL_CODE_PINS.json',{str(p):sha(p) for p in [N/'compare.py',N/'core.py',N/'EVAL_CONTRACT.json']})
 inputs=read(U/'INPUTS.json')['rows'];entries=[]
 for r in inputs:
  if r['role']!='development':continue
  z=np.load(r['path']);pred=np.load(U/'predictions/final_attempt_supervised_42'/f"{r['clip_id']}_{r['kind']}.npz")
  entries.append(dict(r,inventory='main144',raw=z['raw'],ts=z['timestamp_us'],fid=z['frame_id'],gold=z['stage_gold'],presence=pred['presence']))
 for r in read(P/'EVAL_INPUTS.json')['rows']:
  if not r['old_decision']:continue
  z=np.load(r['path']);p=np.ones(10,np.float32);d=read(U/'legacy_decisions/final_attempt_supervised_42'/f"{r['clip_id']}_{r['kind']}.json")
  for c in d['cards']:
   groups=sorted({MAP[g]-1 for gg in c['groups'] for g in gg})
   for g,v in zip(groups,c['event_presence_probability']):p[g]=v
  entries.append(dict(r,inventory='legacy93',raw=z['raw_x'],ts=z['timestamp_us'],fid=z['frame_id'],gold=None,presence=p))
 assert len(entries)==237
 observation=bcore.PoseGuideNet(bcore.STATS['guide_vectors'],bcore.STATS['guide_words'],bcore.STATS['prototype_lengths']);observation.load_state_dict(torch.load(bcore.BASE,weights_only=True,map_location='cpu'));observation.eval()
 head=headclass(read(D/'PREPARED.json')['guide_words']);head.load_state_dict(torch.load(D/'runs/42/model.pt',weights_only=True,map_location='cpu'));head.eval();web=[]
 with torch.inference_mode():
  for r in entries:
   key=f"{r['inventory']}_{r['clip_id']}_{r['kind']}";x=torch.tensor(bcore.normalize(r['raw'])[None]);n=len(r['raw']);wi=torch.tensor([r['word']]);start=time.perf_counter();o=observation(x,torch.tensor([n]),wi);h=o['latent'];p=o['attributes'].sigmoid();valid=torch.ones((1,n),dtype=torch.bool);ph,d0=head(h,p,wi,valid,torch.zeros_like(p));_,d1=head(h,p,wi,valid,torch.ones_like(p));seconds=time.perf_counter()-start
   ph=ph[0].softmax(-1).numpy();d0=d0[0].sigmoid().numpy();d1=d1[0].sigmoid().numpy();p=p[0].numpy();out=consumer.apply_coverage(consumer.refine(consumer.assess(r['word_id'],ph,d0,d1,r['raw'],r['ts'],r['fid']),p,r['raw'],r['ts'],r['fid'],ph))
   path=N/'predictions/web'/f'{key}.npz';path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,phase=ph,d0=d0,d1=d1,p=p)
   target=N/'decisions/web'/f'{key}.json';write(target,out);web.append(dict(key=key,decision=str(target),sha256=sha(target),neural_seconds=seconds));r.update(key=key,web_phase=ph,d0=d0,d1=d1,attrs=p,base_decision=out)
 write(N/'WEB_BASELINE.json',dict(rows=web,observations_checkpoint_sha256=sha(bcore.BASE),head_sha256=sha(D/'runs/42/model.pt'),TCN_rerun=False))
 results={};allrecords=[]
 for seed in [42,43,44]:
  name=f'final_{seed}';ck=torch.load(N/'runs'/name/'model.pt',weights_only=True,map_location='cpu');model=MotionProposal();model.load_state_dict(ck['model']);model.eval();rows=[];timings=[]
  with torch.inference_mode():
   for r in entries:
    gg=geometry(r['raw'],r['ts']);z=normalize(gg,ck['mean'].numpy(),ck['std'].numpy());t=time.perf_counter();pr=model(torch.tensor(z[None]),torch.tensor([len(z)]))[0].softmax(-1).numpy();sec=time.perf_counter()-t
    ph=expand(pr);out=combine(r['word_id'],r['web_phase'],ph,r['d0'],r['d1'],r['attrs'],r['raw'],r['ts'],r['fid'],r['presence'])
    target=N/'decisions'/name/f"{r['key']}.json";write(target,out);npz=N/'predictions'/name/f"{r['key']}.npz";npz.parent.mkdir(exist_ok=True,parents=True);np.savez_compressed(npz,prob=pr)
    def compact(d):return dict(display=[c['display_status'] for c in d['cards']],status=[c['status'] for c in d['cards']],required=[c.get('required',True) for c in d['cards']],phase_present=[all(e['span'] is not None for e in c['phase_evidence']) for c in d['cards']])
    row=dict(key=r['key'],clip_id=r['clip_id'],inventory=r['inventory'],kind=r['kind'],before=compact(r['base_decision']),after=compact(out),rescues=out['geometry_rescues'],unassigned=out['unassigned_observable_intervals'],decision=str(target),sha256=sha(target));rows.append(row)
    if r['inventory']=='main144' and r['kind']=='full_normal':timings.append(sec)
    if r['gold'] is not None:
     seen=set()
     for i in np.flatnonzero(r['gold']>=0):
      fid=int(r['fid'][i])
      if fid in seen:continue
      seen.add(fid);allrecords.append(dict(model=name,clip_id=r['clip_id'],kind=r['kind'],frame=fid,index=int(i),target=MAP[int(r['gold'][i])],geometry_pred=int(pr[i].argmax())))
  summary={}
  for inv,kind in sorted({(r['inventory'],r['kind']) for r in rows}):
   rr=[r for r in rows if r['inventory']==inv and r['kind']==kind];s={}
   for arm in ['before','after']:
    cc=[r[arm] for r in rr];s[arm]=dict(clips=len(cc),difference=sum('difference' in c['display'] for c in cc),uncertain=sum('uncertain' in c['display'] for c in cc),missing=sum('missing_candidate' in c['status'] for c in cc),similar=sum(all(v=='similar' for v,q in zip(c['display'],c['required']) if q) for c in cc),required_present=sum(all(v for v,q in zip(c['phase_present'],c['required']) if q) for c in cc))
   summary[inv+'/'+kind]=s
  write(N/'scores'/f'{name}_cases.json',rows);result=dict(by_kind=summary,proposal_seconds=dict(median=statistics.median(timings),p95=float(np.percentile(timings,95)),scope='Only additional geometry neural forward; no capture/render or RTMW'),cases=len(rows));write(N/'scores'/f'{name}.json',result);results[name]=result;event('evaluated',model=name,normal=summary['main144/full_normal'],shape=summary['legacy93/right_closed_hand_main'],omission=summary['main144/omit_final_event'])
 write(N/'STAGE_RECORDS.json',allrecords);write(N/'COMPARISON.json',dict(models=results,primary='final_42',new_web_model_deployed=False,private_access=False));write(N/'CURRENT_EXECUTION.json',dict(status='EVALUATED_PENDING_FOLD_AND_VERIFICATION',training=None,evaluation=None))
if __name__=='__main__':
 try:main()
 except BaseException:
  (N/f'EVAL_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
