from pathlib import Path
import json,hashlib,sys,collections,time
import numpy as np,torch
from region_extent import RegionExtent,features
from body_core import face_geometry
from joint_core import joint_sequence,integrate
U=Path(__file__).resolve().parent;O=U.parent;B=O/'word100_action_evidence_block_20261114';S=O/'word100_goal_state_learning_20261114';T=O/'word100_terminal_relation_targets_20261114';G=O/'word100_web_completion_graph_20261111';RT=O/'word100_rtmw_joint_learning_20261108';V=U/'extent_augmented'
read=lambda p:json.loads(p.read_text('utf-8-sig'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,d):
 with(U/n).open('x',encoding='utf-8')as f:json.dump(d,f,ensure_ascii=False,indent=2)
assert read(V/'EVALUATION.json')['preliminary_pass']
saved=read(B/'ASSEMBLED_EVIDENCE.json');pm={(r['clip_id'],r['kind']):r for r in read(G/'typed_motion_recovery_20261113/null_candidate_learning/evaluation/COMPARISON.json')['predictions']if r['seed']==42}
ready={r['clip_id']:r for r in read(O/'word100_shared_motion_learning_20261109/READY.json')['rows']}
write('JOINT_FROZEN_V4.json',dict(gates=['component_preserved','combined_errors','missing_not_pass','original_time_evidence'],
 scope='Existing 246 automatic outputs plus 27 already reviewed prefix/speed/visibility controls. Same-action integration, not whole-sign educational correctness.',
 no_new_heldout_accuracy=True,required_checks=['normal and short sufficient joint shape/body sequence retained','partial or goal-only never completed','all old candidates/features/order retained','combined differences coexist','missing observations not corrections','native evidence address retained'],
 optional_body_candidates='First BodyHeads and unaugmented extent rejected; forearm posture ambiguity remains explicitly unsupported. Augmented face-extent region passed fixed27known-probe gate.',
 no_expected_order_selection=True,models={r['path']:r['sha256']for r in read(V/'TRAINED.json')['rows']},
 pins={str(p):sha(p)for p in[U/'joint_core.py',U/'region_extent.py',Path(__file__),B/'ASSEMBLED_EVIDENCE.json',B/'EXTENDED_COMPARISON.json',V/'EVALUATION.json',T/'EVALUATION.json']},
 fixed_controls='Existing source-frame maps and visibility masks; no new pose/region labels on probe.',web_changed=False,final8_opened=False))
models=[];torch.set_num_threads(2)
for r in read(V/'TRAINED.json')['rows']:
 m=RegionExtent();m.load_state_dict(torch.load(r['path'],weights_only=True));m.eval();models.append(m)
cache={};dense={};arrays={};audit=[];timing=[]
replay={(r['clip_id'],r['kind']):r for r in read(O/'word100_completion_fullpath_20261110/REPLAY_INPUTS.json')['rows']}
reverse={r['clip_id']:r['reversed_source_frame_span']for r in read(O/'word100_automatic_window_learning_20261110/FULLPATH_COMPLETE.json')['rows']if r['model']=='learned_core_42'and r['kind']=='reversed_core_auto_order'}
editplan={r['clip_id']:r for r in read(O/'word100_observation_guard_20261108/INPUT_PLAN.json')['rows']}
def source(cid):
 if cid not in cache:
  rec=read(RT/'records'/f'{cid}.json');rawpath=Path(rec['raw_path']);chpath=Path(rec['channel_path']);assert sha(rawpath)==rec['raw_sha256']and sha(chpath)==rec['channel_sha256']
  raw=np.load(rawpath);ch=np.load(chpath);face,ok=face_geometry(raw['xy_px'],ch['point_available']);r=ready[cid];assert sha(Path(r['path']))==r['sha256'];x=np.load(r['path'])['raw_x']
  assert len(x)==len(face);cache[cid]=(x,face,ok,raw['frame_id'])
 return cache[cid]
def predict(x,face,ok,hand):
 f,v=features(x,face,ok,hand)
 with torch.no_grad():pp=np.stack([m(torch.tensor(f)).softmax(-1).numpy()for m in models])
 votes=pp.argmax(-1);known=v&(votes==votes[:1]).all(0)&(pp.max(-1)>=.8).all(0)
 states=[['face_front','upper_body'][int(votes[0,i])]if known[i]else None for i in range(len(v))]
 return states,v,pp
for sid,r in pm.items():
 assert sha(Path(r['source']))==r['source_sha256'];z=np.load(r['source']);x=z['x'];original,fg,fo,fid=source(sid[0]);mapping=[]
 cropped_edit=sid[1]in ['main_reversed','closed_hand_and_reversed','right_closed_hand_main']
 if cropped_edit:
  sourcepath=Path(editplan[sid[0]].get('cache')or O/'word100_observation_guard_20261108/crops'/f'{sid[0]}.npz')
  zz=np.load(sourcepath);xy=zz['xy_px'];av=np.isfinite(xy).all(-1)&(xy[:,:,0]>=0)&(xy[:,:,0]<640)&(xy[:,:,1]>=0)&(xy[:,:,1]<480)
  fg,fo=face_geometry(xy,av);fid=zz['frame_id']
 # Follow recorded edit provenance, not ambiguous pose matching. Repeated body
 # poses do not identify a unique face frame. Check geometry only afterwards.
 if sid in replay:mapping=replay[sid]['source_indices']
 elif sid[1]=='dynamic_complete_blocks':mapping=np.load(G/'dynamic_blocks/inputs'/f'{sid[0]}.npz')['source_indices'].tolist()
 else:
  mapping=np.arange(len(original))
  if sid[1]=='reversed_core_auto_order':
   lo,hi=reverse[sid[0]];ii=np.flatnonzero((fid>=lo)&(fid<=hi));mapping[ii]=mapping[ii[::-1]]
  elif sid[1]in ['main_reversed','closed_hand_and_reversed']:
   lo,hi=editplan[sid[0]]['intervals'][0];t=zz['timestamp_us'];ii=np.flatnonzero((t>=lo*1e6)&(t<=hi*1e6));mapping[ii]=mapping[ii[::-1]]
  else:assert sid[1]=='right_closed_hand_main'
  mapping=mapping.tolist()
 assert len(mapping)==len(x)
 body=original[:,:18].reshape(-1,9,2);body=body/np.maximum(np.linalg.norm(body[:,2]-body[:,1],axis=1)[:,None,None],1e-8)
 check=x[:,:18].reshape(-1,9,2);check=check/np.maximum(np.linalg.norm(check[:,2]-check[:,1],axis=1)[:,None,None],1e-8)
 if cropped_edit:
  cb=xy[:,[0,5,6]];cb=(cb-(cb[:,1:2]+cb[:,2:3])/2)/np.maximum(np.linalg.norm(cb[:,2]-cb[:,1],axis=1)[:,None,None],1e-8)
  delta=float(np.max(np.abs(check[:,:3]-cb[mapping])));assert delta<=1e-5,(sid,'cropped body anchors disagree',delta)
 else:
  delta=float(np.max(np.abs(check-body[mapping])));assert delta<=1e-5,(sid,'recorded mapping disagrees',delta)
 assert len(mapping)==len(x);start=time.perf_counter();perhand={}
 for hand in ['L','R']:
  states,v,pp=predict(x,fg[mapping],fo[mapping],hand);perhand[hand]=dict(states=states,available=v.tolist())
  k='__'.join(sid)+'__'+hand;arrays[k+'__prob']=pp;arrays[k+'__available']=v
 dense[sid]=dict(hands=perhand,source_frames=fid[mapping].tolist(),times=z['ts'].tolist())
 audit.append(dict(clip_id=sid[0],kind=sid[1],mapping=mapping,source_frames=fid[mapping].tolist(),input_sha256=r['source_sha256'],body_match_tolerance=1e-6,coordinate_unit_normalized=True))
 timing.append(dict(clip_id=sid[0],kind=sid[1],frames=len(x),seconds=time.perf_counter()-start))
out=[]
for row in saved['rows']:
 sid=(row['clip_id'],row['kind']);d=dense[sid];extra=[]
 for card in row['original_candidates']:
  chosen=card.get('chosen');span=chosen.get('span')if chosen else None
  if span is None:continue
  a,b=span;regions={h:dict(states=v['states'][a:b+1],available=v['available'][a:b+1])for h,v in d['hands'].items()}
  ex=dict(card=card['card'],span=span,region=regions,source_frames=d['source_frames'][a:b+1],timestamps_us=d['times'][a:b+1],joint=None)
  # Declarative guide binding; numeric/classification functions are shared.
  if card['card']=='close_chest':
   path=next(p for p in row['qualified_shape_paths']if p['card']==card['card'])['hands'][0];shape=dict(path['raw'],content_status=path['content_status'])
   av=[bool(v and s is not None)for v,s in zip(regions['R']['available'],shape['original_states'])]
   ex['joint']=joint_sequence(shape,regions['R']['states'],av,ex['timestamps_us'],ex['source_frames'],shape['association_confirmed'])
  extra.append(ex)
 out.append(integrate(row,extra))
controls=[];arr=np.load(S/'source_arrays.npz');old=read(T/'EVALUATION.json')
for r in read(B/'EXTENDED_COMPARISON.json')['rows']:
 cid=r['clip_id'];ids=r['source_frames'];original,fg,fo,fid=source(cid);x=arr[cid+'_x'][ids].copy()
 # Preserve exactly the old controls' unavailable hand observations.
 visible=np.asarray(r['available'],bool);x[~visible,216:237]=0
 states,v,pp=predict(x,fg[ids],fo[ids],'R');match=next((v for v in old['rows']if v['clip_id']==cid and v['variant']==r['variant']and v['requirement']=='hand_close_to_upper_body'),None)
 ts=match['timestamps_us']if match else arr[cid+'_timestamp_us'][ids].tolist()
 ev=r['evidence'];result=joint_sequence(ev,states,(v&visible).tolist(),ts,ids,True)
 controls.append(dict(clip_id=cid,variant=r['variant'],joint=result,old_shape_evidence=ev))
np.savez_compressed(U/'JOINT_REGION_DENSE.npz',**arrays)
write('JOINT_COMPARISON.json',dict(rows=out,controls=controls,provenance=audit,timing=timing,source_scenarios=len(pm),source_videos=len({sid[0]for sid in pm}),old_neural_inference_calls=0,remaining=['forearm across/upright boundary','thumb/fingertip contact and fine-shape coverage','true learner error validation','sealed test and web connection']))
print(json.dumps(dict(rows=len(out),controls=[dict(clip=r['clip_id'],variant=r['variant'],status=r['joint']['status'],positive=r['joint']['observed_feature_sequence'])for r in controls],normal_joint=[dict(clip=r['clip_id'],card=c['card'],state=c['state'])for r in out if r['kind']=='full_normal'and r['seed']==42 for c in r['joint_cards']]),ensure_ascii=False))
