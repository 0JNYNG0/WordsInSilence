from unit import *
import traceback,statistics
follow=prior.load_module('_attempt_consistency',P/'followup_contract.py')

def assemble(r,phase,d0,d1,presence):
 out=prior.decide(r['word_id'],phase,d0,d1,r['p'],r['raw'],r['timestamp_us'],r['frame_id'])
 out=follow.apply_guard(out,r['p'],r['raw'],r['timestamp_us'])
 hand=np.stack([r['raw'][:,237:258].sum(1)>=5,r['raw'][:,258:279].sum(1)>=5],1)
 # Look for a long sensor gap after an observed action, before the observed end.
 active=np.flatnonzero(phase[:,1:].max(1)>.4);start=int(active[0]) if len(active) else 0
 for index,c in enumerate(out['cards']):
  groups=sorted({GROUPS[g] for group in c['groups'] for g in group});c['event_presence_probability']=[float(presence[g]) for g in groups]
  sides=[0 if h=='L' else 1 for h in c['hands']];avail=hand[:,sides].all(1)
  missing=prior.evidence.runs((~avail)&(np.arange(len(avail))>=start),r['timestamp_us'],300)
  c['event_observation_gap']=missing
  prior_event=any(a.get('query_indices') is not None for a in out['cards'][:index])
  absent=all(presence[g]<=.35 for g in groups);temporal_present=all(q['span'] is not None for q in c['phase_evidence'])
  if c.get('required',True) and absent:
   if not missing and prior_event and out['return_to_rest_evidence'] and not temporal_present:
    c.update(status='missing_candidate',display_status='uncertain',uncertainty_reason='learned_event_absence_with_other_event_and_rest',candidate_message='이 동작까지 이어서 해보세요. 빠졌을 가능성이 있어요.',reference_only=True,educational_wrong=False)
   elif c['display_status']=='similar':
    c.update(status='unknown',display_status='uncertain',uncertainty_reason='event_presence_and_localization_disagree',candidate_message='동작 구간의 근거가 서로 달라 아직 판단하기 어려워요.')
  c['presentation']=dict(hand=c['hands'],segments=[s for e in c['phase_evidence'] for s in e['available_runs']],difference_segments=c.get('difference_spans',[]),text=c.get('candidate_message',c['instruction']),color='orange' if c['display_status']=='difference' else 'green' if c['display_status']=='similar' else 'gray',claim='Checked features only; not whole-sign correctness')
 out['sequence_order']=[dict(card=c['id'],segments=[s for e in c['phase_evidence'] for s in e['available_runs']]) for c in out['cards']]
 return out

def stage_records(r,phase,name):
 records=[];seen=set()
 for i in np.flatnonzero(r['stage_gold']>=0):
  fid=int(r['frame_id'][i])
  if fid in seen:continue
  seen.add(fid);pred=int(phase[i].argmax());target=GROUPS[int(r['stage_gold'][i])]
  records.append(dict(model=name,clip_id=r['clip_id'],kind=r['kind'],frame=fid,index=int(i),target=target,pred=GROUPS.get(pred-1,-1),confident=bool(phase[i,pred]>=.65)))
 return records
def main():
 assert (U/'TRAINING_COMPLETE.json').exists();assert not (U/'EVALUATION_COMPLETE.json').exists();torch.set_num_threads(4);torch.set_num_interop_threads(1)
 rows=load_rows();names=['inner_normal_only_42','inner_attempt_supervised_42']+[f'final_{c}_{s}' for c in ['normal_only','attempt_supervised'] for s in [42,43,44]]
 write(U/'EVAL_FROZEN.json',dict(models=names,primary='final_attempt_supervised_42',threshold=.65,absence_threshold=.35,metrics='Same original reviewed frame identities per condition; no duplicated frames for slow condition. Same public normal/mutated inputs. Historical DEV only.',code_sha256=sha(U/'evaluate.py'),input_sha256=sha(U/'INPUTS.json'),no_model_or_threshold_selection=True))
 ck=torch.load(M/'runs/motion_only_42/model.pt',weights_only=True,map_location='cpu');motion=prior.motion.MotionNet();motion.load_state_dict(ck['model']);motion.eval();results={}
 for name in names:
  net=AttemptHead();net.load_state_dict(torch.load(U/'runs'/name/'model.pt',weights_only=True,map_location='cpu'));net.eval();selected=[r for r in rows if (7<=r['person']<=9 if name.startswith('inner') else r['role']=='development')];records=[];cases=[];direction=[];times=[]
  with torch.inference_mode():
   for r in selected:
    b=batch([r]);t=time.perf_counter();ph,d0,pr=net(b['h'],b['p'],b['g'],b['word'],b['valid'],torch.zeros_like(b['p']));_,d1,_=net(b['h'],b['p'],b['g'],b['word'],b['valid'],torch.ones_like(b['p']));elapsed=time.perf_counter()-t
    phase=ph[0].softmax(-1).numpy();d0=d0[0].sigmoid().numpy();d1=d1[0].sigmoid().numpy();presence=pr[0].sigmoid().numpy();out=assemble(r,phase,d0,d1,presence);records+=stage_records(r,phase,name)
    if r['kind']=='full_normal':times.append(elapsed)
    target=U/'decisions'/name/f"{r['clip_id']}_{r['kind']}.json";write(target,out)
    cases.append(dict(clip_id=r['clip_id'],word=r['word_id'],kind=r['kind'],display=[c['display_status'] for c in out['cards']],status=[c['status'] for c in out['cards']],required=[c.get('required',True) for c in out['cards']],phase_present=[all(e['span'] is not None for e in c['phase_evidence']) for c in out['cards']],presence=presence.tolist(),omitted_groups=r['removed_groups'],decision=str(target),sha256=sha(target)))
    if name.endswith('42'):
     path=U/'predictions'/name/f"{r['clip_id']}_{r['kind']}.npz";path.parent.mkdir(exist_ok=True,parents=True);np.savez_compressed(path,phase=phase,d0=d0,d1=d1,presence=presence)
     done=set()
     for c in out['cards']:
      m=c.get('motion_evidence');span=m.get('span') if m else None
      if not span or tuple(span) in done:continue
      done.add(tuple(span));features,eligible,info=prior.motion.features(r['raw'],r['timestamp_us'],*span,'along_other_forearm',m['moving_hand'],m['anchor_arm']);prob=motion((torch.tensor(features)-ck['mean'])/ck['std']).softmax(-1).numpy();pred=int(prob.argmax())
      learned='unknown' if not eligible or max(prob)<.65 or pred==1 else 'supported' if pred-1==m['expected_sign'] else 'contradicted'
      direction.append(dict(clip_id=r['clip_id'],kind=r['kind'],span=span,geometry=m['state'],learned=learned,eligible=eligible,confidence=float(max(prob)),used_for_feedback='existing geometric evidence; conflicting learned model cannot override'))
  summaries={}
  for kind in sorted({c['kind'] for c in cases}):
   cs=[c for c in cases if c['kind']==kind];rr=[r for r in records if r['kind']==kind]
   summaries[kind]=dict(clips=len(cs),stage_correct=sum(r['pred']==r['target'] for r in rr),stage_total=len(rr),stage_confident_correct=sum(r['confident'] and r['pred']==r['target'] for r in rr),required_present_clips=sum(all(s for s,req in zip(c['phase_present'],c['required']) if req) for c in cs),difference_clips=sum('difference' in c['display'] for c in cs),missing_candidate_clips=sum('missing_candidate' in c['status'] for c in cs),all_required_similar=sum(all(s=='similar' for s,req in zip(c['display'],c['required']) if req) for c in cs),uncertain_clips=sum('uncertain' in c['display'] for c in cs),presence_absence_below_threshold=sum(all(c['presence'][g]<=.35 for g in c['omitted_groups']) for c in cs if c['omitted_groups']))
  result=dict(by_kind=summaries,head_seconds=dict(median=statistics.median(times),p95=float(np.percentile(times,95)),scope='Cached observation latent+attributes+geometry to three heads only; no detector/DTW/UI timing'),cases=len(cases))
  for suffix,value in [('records',records),('cases',cases),('direction',direction)]:write(U/'scores'/f'{name}_{suffix}.json',value)
  write(U/'scores'/f'{name}.json',result);results[name]=result;event('evaluated',model=name,normal=summaries['full_normal'],omission=summaries['omit_final_event'])
 write(U/'EVALUATION_COMPLETE.json',dict(models=results,primary='final_attempt_supervised_42',no_DEV_tuning=True));write(U/'CURRENT_EXECUTION.json',dict(status='EVALUATED_PENDING_VERIFY',pid=None))
if __name__=='__main__':
 try:main()
 except BaseException:
  (U/f'EVAL_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
