from episode import *
import argparse,random,traceback,os
from collections import Counter

def labels():
 result=[];lookup={r['alias']:r for r in rows()}
 for role,file in [('fit','LABELS_FIT.json'),('development','LABELS_DEV.json')]:
  for clip in read(U/file)['clips']:
   assert lookup[clip['alias']]['role']==role
   for i,e in enumerate(clip['events']):
    if e['state'] not in CLASSES:continue
    result.append(dict(e,alias=clip['alias'],role=role,event_id=f"{clip['alias']}_{i:02}"))
 return result

def prepare():
 rs={r['alias']:r for r in rows()};data={a:arrays(r) for a,r in rs.items()};events=[];checks=[]
 for r in rs.values():
  for p,h in [(r['source'],r['source_sha256']),(r['input'],r['input_sha256']),(r['record'],r['record_sha256'])]:assert sha(p)==h,p
  a=data[r['alias']];assert len(a['x'])==r['frames'] and np.diff(a['frame_id']).min()==1 and (np.diff(a['timestamp_us'])>0).all()
  native=raw_features(r);assert np.array_equal(native['mask'],a['mask'])
  assert np.allclose(native['x'][a['mask']],a['x'][a['mask']],atol=1e-5),r['alias']
  transformed=raw_features(r,1.31,(123.,-51.))
  checks.append(dict(alias=r['alias'],test='raw_pixels_translation_uniform_scale',pass_=bool(np.array_equal(native['mask'],transformed['mask']) and np.allclose(native['x'][native['mask']],transformed['x'][native['mask']],atol=1e-5))))
 for e in labels():
  a=data[e['alias']];lo=e['core_start_frame'];hi=e['core_end_frame'];assert 0<=lo<=hi<len(a['x'])
  assert e['start_bracket'][0]<=lo<=e['start_bracket'][1] and e['end_bracket'][0]<=hi<=e['end_bracket'][1]
  w=interval(a,lo,hi);z,ok,cov=encode(w)
  events.append(dict(e,eligible=ok,input_coverage=cov,frames=len(z),class_index=CLASSES.index(e['state'])))
 counts=Counter(e['state'] for e in events if e['role']=='fit' and e['eligible'])
 assert all(counts[c]>=3 for c in CLASSES),counts
 for c in CLASSES:
  assert len({rs[e['alias']]['word_id'] for e in events if e['role']=='fit' and e['eligible'] and e['state']==c})>=2,c
 assert all(x['pass_'] for x in checks)
 # Verify masked padding does not change output and missing does not become other.
 torch.manual_seed(1);m=Head().eval();w=interval(next(iter(data.values())),20,35);z,_,_=encode(w)
 p=infer(m,[z]);p2=infer(m,[z,np.tile(z,(3,1))])[0]
 assert np.allclose(p[0],p2,atol=1e-6)
 empty=copy.deepcopy(w);empty['mask'][:]=False;assert not encode(empty)[1] and geometry(empty)==-1
 checks.extend([dict(test='padding_batch_invariance',pass_=True),dict(test='missing_is_unknown',pass_=True)])
 write(U/'PREPARED.json',dict(events=events,counts_by_role={role:dict(Counter(e['state'] for e in events if e['role']==role)) for role in ['fit','development']},fit_supported_counts=dict(counts),checks=checks,excluded_fit=[e['event_id'] for e in events if e['role']=='fit' and not e['eligible']],all_dev_events_remain_denominator=True,physical_rgb_draft_not_expert=True))
 print(json.dumps(dict(events=len(events),fit_supported_counts=counts),ensure_ascii=False))

def freeze():
 p=read(U/'PREPARED.json');pins={}
 files=[U/'episode.py',Path(__file__),U/'AUTHORIZATION.json',U/'LABELS_FIT.json',U/'LABELS_DEV.json',U/'PREPARED.json',U/'review/SOURCE_USAGE_AND_RANGE.json',OLD/'LEARNING_INPUTS.json',OLD/'runs/pose_capacity_matched_42/probabilities.npy',OPS/'word100_guide_video_feedback_20261028/motion_observer.py']
 for r in rows():files.extend(Path(r[k]) for k in ['source','input','record','observation'])
 for f in files:pins[str(f)]=sha(f)
 protected=read(OLD/'PROTOCOL.json')['protected']
 for f,h in protected.items():assert sha(f)==h
 protocol=dict(status='FROZEN_BEFORE_NEW_TRAINING_AND_SCORES',seeds=SEEDS,epochs=160,optimizer='AdamW lr0.003 weight_decay0.001',epoch_selection='fixed final160; no development epoch/candidate selection',class_weight='inverse eligible fit window class count, normalized mean1',exposure='one full eligible fit-event batch per epoch; exact window IDs logged',model='two masked 24-channel 1D convolutions k3, pooled physical episode classifier,3states,41features',parameters=sum(p.numel() for p in Head().parameters()),support=SUPPORT,geometry_vertical_displacement=.03,window=WINDOW,classes=CLASSES,target='image-plane right-hand motion, not complete gesture or linguistic correctness',old='reuse stored pose_capacity_matched_42 up/down indices5/6; known-window mean >=0.5 elseother; same evidence gate',automatic='centered13-original-frame window, no target word or source interval used. Unknown separated, no gap filling, short proposals retained and marked',event_evaluation='all annotated cores denominator; majority supported frame label per core. Missing counts as failure. Boundary offsets reported for overlapping same-state proposal against uncertainty brackets. Unannotated time is unknown truth, not false positive ground truth.',reference_range='For each expression fit3 LOPO nearest complete reference distances, threshold maximum of3. Dev compares all3 references; whole-reference best chosen for proximity only when all3 comparable. Reference-specific scopes and missing retained.',range_threshold_use='descriptive empirical proximity; never pedagogical correct/incorrect',development_gate='candidate mean macroF1 >=old+0.05 and >=geometry; annotated other correct count no lower; automatic core macroF1 no worse than old; missing controls must abstain. If fails: no adoption, still finish diagnostics/research UI.',controls=['raw coordinates uniform translation+scale','timestamps0.75/1.25 without dropping samples','all-hand-missing and central short missing','reverse and delete central short interval as synthetic physical changes, not linguistic error ground truth'],s5='empirical normal-range development proceeds; independent KSL correctness/unknown environment generalization remains later evaluation',test_history='16 previously development-used clips; no held/fresh or ID9 inference',scope='C/T3F/old shape and production unchanged; no new SignRep or RGB inference or user data',pins=pins,protected=protected)
 protocol['reference_range']='Per expression fit3 LOPO nearest sufficiently supported observed-only whole-reference distances; threshold max3. Each reference coverage>=.75 required. Path missing/reference-direction unknown and overall incompleteness are separate and always reported. No per-frame reference mixing.'
 protocol['automatic']='Centered13-original-frame window, no target word/source interval; actual center hand support mandatory for all conditions. Original frame/large PTS gaps abstain; short proposals retained. All other features unchanged.'
 protocol['boundary_evaluation']='Main up/down bracket offsets only. Other cores are not true event boundaries. No exhaustive full-clip GT; automatic boundary adoption remains false even if core content improves.'
 for name in ['LABELS_CROSS_REVIEW.json','TRAINING_CODE_REVIEW.json','EPISODE_CONTRACT_REVIEW.json']:
  path=U/'review'/name
  assert path.exists(),path
  protocol['pins'][str(path)]=sha(path)
 write(U/'FROZEN.json',protocol);print('FROZEN',protocol['parameters'])

def check_pins():
 f=read(U/'FROZEN.json')
 for group in ['pins','protected']:
  for p,h in f[group].items():assert sha(p)==h,p
 return f

def train():
 f=check_pins();torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
 data={r['alias']:arrays(r) for r in rows()};events=read(U/'PREPARED.json')['events'];fit=[e for e in events if e['role']=='fit' and e['eligible']]
 inp=[encode(interval(data[e['alias']],e['core_start_frame'],e['core_end_frame']))[0] for e in fit];x,v=batch(inp);y=torch.tensor([e['class_index'] for e in fit]);count=torch.bincount(y,minlength=3).float();weight=1/count;weight/=weight.mean()
 for seed in SEEDS:
  d=U/'runs'/str(seed)
  if (d/'COMPLETE.json').exists():
   done=read(d/'COMPLETE.json');assert sha(d/'head.pt')==done['checkpoint_sha256'];print('reuse complete seed',seed);continue
  d.mkdir(parents=True,exist_ok=False);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
  m=Head();opt=torch.optim.AdamW(m.parameters(),lr=.003,weight_decay=.001);lossfn=nn.CrossEntropyLoss(weight=weight);start=time.perf_counter();history=[]
  for epoch in range(1,161):
   m.train();opt.zero_grad();logits=m(x,v);loss=lossfn(logits,y);loss.backward();opt.step();history.append(dict(epoch=epoch,loss=float(loss.detach()),event_ids=[e['event_id'] for e in fit]))
  m.eval();torch.save(m.state_dict(),d/'head.pt');write(d/'TRAIN_TRACE.json',dict(seed=seed,history=history,fit_count=len(fit),development_examples_seen=0,optimizer_steps=160))
  write(d/'COMPLETE.json',dict(seed=seed,seconds=time.perf_counter()-start,checkpoint_sha256=sha(d/'head.pt'),trace_sha256=sha(d/'TRAIN_TRACE.json'),fixed_epoch=160,final_training_loss=history[-1]['loss']))
  print('trained',seed,history[-1]['loss'],flush=True)

def evaluate():
 assert not (U/'EVALUATION.json').exists()
 for s in SEEDS:
  d=U/'runs'/str(s)
  assert not (d/'window_predictions.npy').exists() and not (d/'window_probabilities.npy').exists() and not (d/'auto').exists(),'Partial or complete evaluation exists; inspect, do not overwrite'
 write(U/'EVALUATION_RUNNING.json',dict(pid=os.getpid(),start=time.time()))
 check_pins();torch.set_num_threads(4);data={r['alias']:arrays(r) for r in rows()};events=read(U/'PREPARED.json')['events'];oldinfo=read(OLD/'LEARNING_INPUTS.json');stored=np.load(OLD/'runs/pose_capacity_matched_42/probabilities.npy',mmap_mode='r');oldlookup={r['alias']:i for i,r in enumerate(oldinfo['rows'])};old={a:stored[oldlookup[a],:len(d['x'])].copy() for a,d in data.items()}
 window_items=[encode(interval(data[e['alias']],e['core_start_frame'],e['core_end_frame']))[0] for e in events];evaluations=[];autos={};eventrows=[]
 for seed in SEEDS:
  m=load_model(seed);p=infer(m,window_items);pred=p.argmax(1);pred[[not e['eligible'] for e in events]]=-1
  d=U/'runs'/str(seed);np.save(d/'window_probabilities.npy',p,allow_pickle=False);np.save(d/'window_predictions.npy',pred,allow_pickle=False)
  for a,arr in data.items():
   z=automatic(arr,m,old[a]);(d/'auto').mkdir(exist_ok=True);np.savez_compressed(d/'auto'/f'{a}.npz',**z)
   autos[(seed,a)]=z
  for role in ['fit','development']:
   ids=[i for i,e in enumerate(events) if e['role']==role];truth=[events[i]['class_index'] for i in ids]
   evaluations.append(dict(condition=f'candidate_{seed}',role=role,**score(truth,pred[ids])))
 for method in ['old','geometry']:
  pp=[]
  for e in events:
   a=data[e['alias']];w=interval(a,e['core_start_frame'],e['core_end_frame'])
   pp.append(geometry(w) if method=='geometry' else old_predict(old[e['alias']][e['core_start_frame']:e['core_end_frame']+1],w))
  for role in ['fit','development']:
   ids=[i for i,e in enumerate(events) if e['role']==role];evaluations.append(dict(condition=method,role=role,**score([events[i]['class_index'] for i in ids],[pp[i] for i in ids])))
  for i,e in enumerate(events):e[method]=int(pp[i])
 auto_scores=[]
 for method in ['old','geometry']+[f'candidate_{s}' for s in SEEDS]:
  pred=[];truth=[];boundary=[]
  for e in events:
   if e['role']!='development':continue
   seed=int(method.split('_')[-1]) if method.startswith('candidate') else 42
   zz=autos[(seed,e['alias'])];arr=data[e['alias']];states=zz['pred' if method.startswith('candidate') else method];ix=(arr['frame_id']>=e['core_start_frame'])&(arr['frame_id']<=e['core_end_frame']);part=states[ix];n=len(part)
   best=int(np.argmax(np.bincount(part[part>=0],minlength=3))) if (part>=0).sum()>=SUPPORT*n else -1
   pred.append(best);truth.append(e['class_index'])
   rr=runs(states,arr['frame_id'],arr['timestamp_us']);overlaps=[r for r in rr if r['state']==e['state'] and r['start_frame']<=e['core_end_frame'] and r['end_frame']>=e['core_start_frame']]
   match=max(overlaps,key=lambda r:min(r['end_frame'],e['core_end_frame'])-max(r['start_frame'],e['core_start_frame']),default=None)
   def off(v,b):return max(b[0]-v,0,v-b[1])
   if e['state']!='other':boundary.append(dict(event_id=e['event_id'],match=match,start_outside_bracket_frames=None if match is None else off(match['start_frame'],e['start_bracket']),end_outside_bracket_frames=None if match is None else off(match['end_frame'],e['end_bracket'])))
  auto_scores.append(dict(condition=method,**score(truth,pred),boundary=boundary))
 # Preserve old shape probabilities, native coordinates/masks and reference scope; new motion is a separate channel.
 dev=[x for x in evaluations if x['role']=='development'];c=[x for x in dev if x['condition'].startswith('candidate')];o=next(x for x in dev if x['condition']=='old');g=next(x for x in dev if x['condition']=='geometry')
 candidate_mean=float(np.mean([v['macro_f1'] for v in c]));auto_mean=float(np.mean([v['macro_f1'] for v in auto_scores if v['condition'].startswith('candidate')]))
 gate=candidate_mean>=o['macro_f1']+.05 and candidate_mean>=g['macro_f1'] and all(v['confusion'][2][2]>=o['confusion'][2][2] for v in c) and auto_mean>=auto_scores[0]['macro_f1']
 for i,e in enumerate(events):
  p={str(s):int(np.load(U/'runs'/str(s)/'window_predictions.npy')[i]) for s in SEEDS};eventrows.append(dict(e,candidate=p))
 write(U/'EVALUATION.json',dict(known_windows=evaluations,automatic_core=auto_scores,events=eventrows,provisional_quality_gate_pass=bool(gate),final_gate_requires_controls=True,boundary_adoption_pass=False,boundary_note='Main event bracket offsets only; no whole-clip exhaustive GT. Other windows are not true event boundaries. Core content gate is not boundary adoption.',adoption='research diagnostic only; never production adoption based on this core score',limitations=['provisional sampled RGB interval labels','only4expressions16consumed clips','partial cores not complete phase transcript','no educational error/acceptance labels','same-Codex review is not independent human ground truth']))
 print(json.dumps(dict(dev=dev,gate=bool(gate)),ensure_ascii=False))
 (U/'EVALUATION_RUNNING.json').unlink()

def empirical():
 assert not any((U/x).exists() for x in ['EMPIRICAL_RANGE.json','CONTROLS.json','FINAL_GATE.json','EMPIRICAL_RUNNING.json']),'Existing output/lock; do not repeat'
 write(U/'EMPIRICAL_RUNNING.json',dict(pid=os.getpid(),start=time.time()))
 check_pins();torch.set_num_threads(4);rs=rows();data={r['alias']:arrays(r) for r in rs};states={r['alias']:np.load(U/'runs/42/auto'/f"{r['alias']}.npz")['pred'] for r in rs};out=[];cal=[]
 for word in sorted({r['word_id'] for r in rs}):
  refs=[r for r in rs if r['word_id']==word and r['role']=='fit'];assert len(refs)==3;lo=[]
  for q in refs:
   pairs=[dict(reference=r['alias'],**compare(data[q['alias']],data[r['alias']],states[q['alias']],states[r['alias']])) for r in refs if r!=q]
   valid=all(p['status']=='comparable' for p in pairs);lo.append(dict(query=q['alias'],pairs=pairs,nearest=min(p['distance'] for p in pairs) if valid else None))
  vals=[x['nearest'] for x in lo];threshold=max(vals) if all(v is not None for v in vals) else None
  cal.append(dict(word_id=word,threshold=threshold,lopo=lo))
  q=next(r for r in rs if r['word_id']==word and r['role']=='development');pairs=[dict(reference=r['alias'],**compare(data[q['alias']],data[r['alias']],states[q['alias']],states[r['alias']])) for r in refs]
  valid=all(p['status']=='comparable' for p in pairs) and threshold is not None
  nearest=min(pairs,key=lambda p:p['distance'] if p.get('distance') is not None else float('inf'))
  out.append(dict(alias=q['alias'],label=q['label'],word_id=word,pairs=pairs,threshold=threshold,nearest_reference=nearest['reference'],nearest_distance=nearest.get('distance'),observed_proximity='insufficient' if not valid else 'within_empirical_range' if nearest['distance']<=threshold else 'outside_empirical_range',evidence_status='complete_relative_to_references' if all(p.get('evidence_status')=='complete_relative_to_reference' for p in pairs) else 'incomplete_observation',educational_verdict=None))
 write(U/'EMPIRICAL_RANGE.json',dict(calibration=cal,development=out,calibration_roles='fit only; three LOPO descriptive maxima, not statistical95% or KSL allowance',all_references_preserved=True,personal_or_held_data=False))
 # Fixed full16 controls, not selected successes. No observation recomputation in timing-only variants.
 controls=[];m=load_model(42)
 for r in rs:
  a=data[r['alias']];orig=states[r['alias']];base=compare(a,a,orig,orig)
  for factor in [.75,1.25]:
   b=copy.deepcopy(a);b['timestamp_us']=np.round(b['timestamp_us']/factor).astype(np.int64);b['x'][:,[61,62,65,66]]*=factor
   z=automatic(b,m)['pred'];p=compare(b,a,z,orig);controls.append(dict(alias=r['alias'],test=f'timestamps_only_speed_{factor}',pass_=bool(np.array_equal(z,orig) and p['distance']==base['distance']),reference_frames=p['reference_frames'],expected=p['expected']))
  b=raw_features(r,1.31,(123.,-51.));z=automatic(b,m)['pred'];p=compare(b,a,z,orig);controls.append(dict(alias=r['alias'],test='raw_translation_uniform_scale',pass_=bool(np.array_equal(z,orig) and abs(p['distance']-base['distance'])<1e-5)))
  for kind in ['all_missing','short_missing','reverse','delete_middle']:
   b=copy.deepcopy(a);n=len(b['x']);lo=n//2-2;hi=n//2+3
   if kind=='all_missing':b['mask'][:]=False
   elif kind=='short_missing':b['mask'][lo:hi,:]=False
   elif kind=='reverse':
    b['x']=b['x'][::-1].copy();b['mask']=b['mask'][::-1].copy()
   else:
    ids=np.r_[np.arange(lo),np.arange(hi,n)];b={k:v[ids].copy() for k,v in b.items() if k in ['x','mask','frame_id','timestamp_us']}
   z=automatic(b,m)['pred'];p=compare(b,a,z,orig)
   controls.append(dict(alias=r['alias'],test=kind,pass_=bool(p['reference_frames']==base['reference_frames'] and p['expected']==base['expected'] and (kind!='all_missing' or ((z<0).all() and p['status']=='insufficient')) and (kind!='short_missing' or (z[lo:hi]<0).all())),result=p,synthetic_difference_not_linguistic_error=kind in ['reverse','delete_middle']))
 write(U/'CONTROLS.json',dict(rows=controls,passed=sum(x['pass_'] for x in controls),total=len(controls),interpretation='Timestamp-only speed does not simulate motion blur or fewer camera frames. Similarity-cost response to synthetic deletion/reversal is reported, not called real KSL error sensitivity. Translation/scale keeps observation masks fixed and does not prove detector invariance.'))
 ev=read(U/'EVALUATION.json');write(U/'FINAL_GATE.json',dict(physical_core_quality=ev['provisional_quality_gate_pass'],all_contract_controls=all(x['pass_'] for x in controls),physical_core_development_pass=ev['provisional_quality_gate_pass'] and all(x['pass_'] for x in controls),automatic_boundary_adoption=False,educational_validation=False,production_replacement=False,decision='Complete bounded experiment and display diagnostics; no unvalidated correction deployment'))
 print('empirical+controls',len(controls),sum(x['pass_'] for x in controls))
 (U/'EMPIRICAL_RUNNING.json').unlink()

if __name__=='__main__':
 action=sys.argv[1]
 try:globals()[action]()
 except BaseException:
  out=U/'failures';out.mkdir(exist_ok=True);(out/f'{action}_{time.time_ns()}.txt').write_text(traceback.format_exc(),encoding='utf8');raise
