"""Diagnose direction stability within reviewed source boundary uncertainty bands."""
from boundary_direction_core import *
import itertools,time,traceback

def main():
 M.mkdir(exist_ok=True);assert not(M/'INPUTS_COMPLETE.json').exists();torch.set_num_threads(4);torch.set_num_interop_threads(1)
 rc=load('_bd_req',O/'word100_guide_requirement_learning_20261111/requirement_core.py');observer=rc.frozen_observer();head=DirectionHead(observer).eval();base=encoder()
 allready=read(O/'word100_shared_motion_learning_20261109/READY.json')['rows'];ready={r['clip_id']:r for r in allready if r['role']=='fit'}
 sources=read(U/'episode_membership/coverage_review/SOURCE_EPISODES_DENSE.json')['rows']+read(D/'SOURCE_EPISODES_ADDED.json')['rows'];sm={r['clip_id']:r for r in sources}
 crop={r['clip_id']:r for p in [R/'CROP_INPUTS_COMPLETE.json',D/'CROP_INPUTS_COMPLETE.json']for r in read(p)['rows']}
 motion={r['clip_id']:r['pairs']for r in read(O/'word100_motion_components_20261109/READY.json')['rows']if r['role']=='fit'};extra=read(O/'word100_broad_factorized_learning_20261110/ADDITIONAL_LABELS.json')['motion']
 selected={c:r for c,r in ready.items()if c in sm or(r['person']not in [8,9]and(motion.get(c)or extra.get(c)))}
 sys.path.insert(0,str(O/'word100_observation_guard_20261108'));guard=load('_bd_guard',O/'word100_observation_guard_20261108/guard.py');ad=load('_bd_adapter',O/'word100_rtmw_shared_input_20261108/adapter.py')
 cfg=dict(at=stamp(),scope='Copy and finetune movement head only, if source-boundary audit justifies it. No common100/TCN/shape changes.',hypothesis='Direction output is sensitive to choosing a different valid start/end within the same reviewed source action. Teach valid boundary variants with original motion-label replay.',label_rule='For reviewed monotonic families, enumerate only source onset/offset bands retaining all progression witnesses. No guessed motion outside reviewed interval. Reverse is opposite-direction contrast, not correct word. Holds from explicit source RGB hold intervals.',families={2:dict(axis=0,target=0),3:dict(axis=1,target=2),4:dict(axis=2,target=2),7:dict(axis=3,target=2)},training_people=[1,2,3,4,5,6,7],probe_people=[8,9],views=['full','upper','actual_crop_when_available'],seeds=[42,43,44],steps=800,lr=.0003,weight_decay=.001,batch='Balanced axis/target with half original-pair replay when available',selected='final step only; no probe checkpoint selection',frozen_common100=True,semantic_completion=None,web_changed=False,pins={str(p):sha(p)for p in[Path(__file__),W/'boundary_direction_core.py',W/'transfer_local_timefix_core.py',L/'VERIFICATION.json',L/'CONTRACT_AUDIT.json',U/'episode_membership/coverage_review/SOURCE_EPISODES_DENSE.json',D/'SOURCE_EPISODES_ADDED.json',O/'word100_broad_factorized_learning_20261110/runs/coupled_42/model.pt']});write(M/'FROZEN.json',cfg)
 rows=[];contexts=[];geoms=[];targets=[];baseline=[];pins={};skips=[]
 for ci,(cid,r)in enumerate(selected.items()):
  assert r['role']=='fit'and sha(r['path'])==r['sha256']and sha(r['channel'])==r['channel_sha256'];z=np.load(r['path']);ch=np.load(r['channel']);ts=z['timestamp_us'];pins[r['path']]=r['sha256'];pins[r['channel']]=r['channel_sha256']
  views={'full':ch['full'],'upper':ch['feedback']if'feedback'in ch.files else ch['upper_mask']}
  if cid in crop:
   cr=crop[cid];assert sha(cr['path'])==cr['sha256'];a=np.load(cr['path']);q=guard.quarantine(a['xy_px'],a['score'],a['timestamp_us'],640,480);views['actual_crop']=ad.convert(q['xy_px'],a['score'],a['timestamp_us'],a['frame_id'],640,480)['tcn280'];pins[cr['path']]=cr['sha256']
  requests=[]
  if r['person']not in [8,9]:
   for p in list(motion.get(cid,[]))+list(extra.get(cid,[])):
    requests.append(dict(a=p['a'],b=p['b'],axis=p['axis'],target=p['target'],moving=p.get('moving','R'),anchor=p.get('anchor','L'),origin='original_motion_replay'))
  if cid in sm:
   s=sm[cid]
   for group in s['groups']:
    family=group['family']
    if family not in [2,3,4,7]:continue
    spec=cfg['families'][family];pts={p for e in s['events']if e['event_id']in group['event_ids']for p in e['progression_witness_frames']}
    for a,b in itertools.product(range(group['onset'][0],group['onset'][1]+1),range(group['offset'][0],group['offset'][1]+1)):
     if a<b and all(a<=p<=b for p in pts):requests.append(dict(a=a,b=b,axis=spec['axis'],target=spec['target'],moving='R',anchor='L',origin='reviewed_boundary_variant',family=family))
    for a,b in s.get('holds',[]):
     if b-a>=3:requests.append(dict(a=a,b=b,axis=spec['axis'],target=1,moving='R',anchor='L',origin='reviewed_hold',family=family))
  # A variant remains source-labelled, not an independent new clip.
  for view,x in views.items():
   h=hidden(base,normalized_pose(x))
   for req in requests:
    for reverse in [False,True]:
     a,b=req['a'],req['b'];g,ok,info=rc.prior.geometry(x,ts,a,b,req['axis'],req['moving'],req['anchor'],reverse)
     if not ok:skips.append(dict(clip_id=cid,view=view,request=req,reverse=reverse,reason='insufficient_body_observation'));continue
     ha,hb=(h[b],h[a])if reverse else(h[a],h[b]);ctx=np.r_[ha,hb,hb-ha];target=2-req['target']if reverse else req['target']
     with torch.inference_mode():
      logits=head(torch.tensor(ctx[None]),torch.tensor(g[None]))[0].numpy()
      direct=observer.movement(torch.tensor(h[None]),torch.tensor(g[None]),torch.tensor([[0,b,a]if reverse else[0,a,b]]))[0].numpy()
     assert np.allclose(logits,direct,atol=1e-6)
     rows.append({**req,'clip_id':cid,'person':r['person'],'source_sha256':r['source_sha256'],'split':'probe'if r['person']in[8,9]else'train','view':view,'reverse':reverse,'target':target,'geometry':info});contexts.append(ctx);geoms.append(g);targets.append(target);baseline.append(logits)
  print('direction boundary source',ci+1,len(selected),len(rows),flush=True)
 data=M/'inputs.npz';np.savez_compressed(data,context=np.array(contexts),g=np.array(geoms),y=np.array(targets),baseline_logits=np.array(baseline));assert all(np.isfinite(a).all()for a in [contexts,geoms,baseline])
 write(M/'INPUTS_COMPLETE.json',dict(at=stamp(),path=str(data),sha256=sha(data),rows=rows,skipped=skips,pins=pins,head_copy_matches=True,observed_source_clips=len(selected),upstream_exposed_probe=True,labels_not_expert_gold=True));print('Direction boundary audit inputs ready')
if __name__=='__main__':
 try:main()
 except BaseException:M.mkdir(exist_ok=True);(M/f'PREP_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),'utf-8');raise
