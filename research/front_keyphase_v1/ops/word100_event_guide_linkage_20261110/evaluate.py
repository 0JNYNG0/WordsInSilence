from link_core import *
import copy,traceback

def infer(net,x,ts):
 g=old.old.geometry(x,ts);q=old.relation_features(x)
 with torch.inference_mode():p,_,_=net(torch.tensor(g[None]),torch.tensor(q[None]),torch.zeros((1,len(x),64)),torch.tensor([len(x)]))
 return p.softmax(-1)[0].numpy()

def warning(card,kind,message,sp):
 card.update(status='difference_candidate',display_status='difference',educational_wrong=False,message=message,candidate_message=message)
 card.setdefault('difference_components',[])
 if kind not in card['difference_components']:card['difference_components'].append(kind)
 card.setdefault('guide_link_difference_spans',[]).extend(sp)

def apply_links(baseline,prob,x,ts,fh,base,contracts):
 out=copy.deepcopy(baseline);ss=spans(prob,x,ts);bindings=[];directions=[];ordering=[]
 ev=baseline['phase_events']+baseline['phase_partials'];cache={}
 for c,spec in zip(out['cards'],contracts):
  assert c['id']==spec['id']
  # A failed expected handshape is not a reason to remove the attempted movement.
  matches=[s for s in ss if s['context'] in spec['contexts']];bound=[]
  if spec['axis'] is not None:
   for e in ev:
    if e['axis']!=spec['axis'] or e['moving']!='R' or e['b']-e['a']<3:continue
    overlap=sum(max(0,min(e['b'],s['b'])-max(e['a'],s['a'])+1) for s in matches)
    if overlap/(e['b']-e['a']+1)<.25:continue
    a,b=e['a'],e['b'];key=(a,b,spec['axis'])
    if key not in cache:
     geo,ok,info=prior.geometry(x,ts,a,b,spec['axis'],'R','L')
     with torch.inference_mode():mp=base.movement(torch.tensor(fh[None]),torch.tensor(geo[None]),torch.tensor([[0,a,b]])).softmax(-1)[0].numpy()
     pred=int(mp.argmax());delta=info.get('signed_change',0);agree=pred==1 or np.sign(delta)==pred-1
     cache[key]=dict(span=[a,b],axis=spec['axis'],moving='R',prediction=pred,confidence=float(mp.max()),eligible=bool(ok and agree),geometry=info,complete=e.get('complete',False))
    bound.append(cache[key])
  bid=dict(card=c['id'],contexts=spec['contexts'],context_spans=matches,movement_links=bound,required=spec['required'],parent=spec.get('parent'),original_window_used=False)
  bindings.append(bid)
  signs={e['prediction']-1 for e in bound if e['eligible'] and e['confidence']>=.65 and e['prediction']!=1}
  expected=spec['sign'];state='not_specified' if expected is None else 'unknown'
  if expected is not None and len(signs)==1:state='supported' if signs=={expected} else 'contradicted'
  if expected is not None and len(signs)>1:state='mixed'
  check=dict(card=c['id'],state=state,expected=expected,observed_signs=sorted(signs),evidence=bound)
  directions.append(check)
  if state=='contradicted' and not c.get('observation_quarantine_overlap'):
   warning(c,'guide_link_direction','표시된 동작에서 이동 방향이 참고와 반대로 보여요. 화살표 방향을 확인해 보세요.',[e['span'] for e in bound if e['eligible'] and e['confidence']>=.65])
  c['guide_link_observation']=bid;c['guide_link_direction']=check
  if not matches:
   c['guide_link_absence']=dict(state='unconfirmed',reason='No reliable independent body-context span; absence alone is not proof of omission.',completion_allowed=False)
 # Optional arrival states and same-card alternating subcontexts are not independent mandatory steps.
 required=[(i,c) for i,c in enumerate(contracts) if c['required'] and not c.get('parent')]
 for (ia,ca),(ib,cb) in zip(required,required[1:]):
  aa=bindings[ia]['context_spans'];bb=bindings[ib]['context_spans'];state=order_state(aa,bb)
  check=dict(pair=[ca['id'],cb['id']],state=state,first=aa,second=bb,optional_children_excluded=True);ordering.append(check)
  if state=='contradicted' and not any(out['cards'][i].get('observation_quarantine_overlap') for i in [ia,ib]):
   warning(out['cards'][ib],'guide_link_order','표시된 두 행동의 순서가 참고와 반대로 보여요. 먼저 해야 할 행동부터 이어 보세요.',[[s['a'],s['b']] for s in aa+bb])
 out.update(guide_link_version='independent_body_context_v1',guide_link_bindings=bindings,guide_link_order_checks=ordering,guide_link_direction_checks=directions,guide_link_probabilities=prob.tolist(),guide_link_absence_is_not_omission=True)
 return out

def compact(d):return {k:v for k,v in be.ev.compact(d).items() if k not in ['motion','shape']}

def main():
 assert not (U/'EVALUATION_COMPLETE.json').exists() and (U/'TRAINING_COMPLETE.json').exists()
 cfg=read(U/'FROZEN.json')
 for p,h in {**cfg['pins'],**cfg['protections'],**read(U/'EVAL_PINS.json')}.items():assert sha(p)==h,p
 torch.set_num_threads(4);torch.set_num_interop_threads(1)
 runtime=be.ev.prepare_runtime();saved={r['clip_id']+'_'+r['kind']:r for r in read(B/'FULLPATH_COMPLETE.json')['rows'] if r['seed']==42}
 contracts=read(U/'GUIDE_CONTRACTS.json')['words'];wordmap={r['word_id']:r['word'] for r in old.rows() if r['word']>=0}
 base=prior.Network('coupled',read(old.P/'NORMALIZATION.json'));base.load_state_dict(torch.load(old.P/'runs/coupled_42/model.pt',weights_only=True,map_location='cpu'));base.eval()
 for c in runtime:
  r=saved[c['key']];assert sha(r['result'])==r['sha256'];c['baseline']=read(r['result']);c['baseline_file']=r['result'];c['fh'],_=be.ev.frozen_forward(base,c['x'],wordmap[c['baseline']['word']])
 gold={r['clip_id']:r for r in read(U/'LINK_LABELS.json')['rows']};oldmap={0:0,1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:6,9:7,10:7,11:8,12:9,13:9}
 allrows=[];frames=[];event_scores=[]
 for seed in cfg['seeds']:
  net=LinkNet();net.load_state_dict(torch.load(U/'runs'/str(seed)/'model.pt',weights_only=True,map_location='cpu'));net.eval()
  for c in runtime:
   before=c['baseline'];t=time.perf_counter();prob=infer(net,c['x'],c['ts']);out=apply_links(before,prob,c['x'],c['ts'],c['fh'],base,contracts[before['word']]);seconds=time.perf_counter()-t
   for key in ['path','tcn_top3','frame_id','phase_events','phase_partials']:assert out[key]==before[key]
   for a,b in zip(out['cards'],before['cards']):
    assert a.get('shape_evidence')==b.get('shape_evidence') and a.get('motion_evidence')==b.get('motion_evidence')
   path=U/'fullpath'/str(seed)/c['key']/'after.json';write(path,out)
   allrows.append(dict(seed=seed,clip_id=c['row']['clip_id'],kind=c['row']['kind'],before=compact(before),after=compact(out),baseline=c['baseline_file'],baseline_sha256=sha(c['baseline_file']),result=str(path),sha256=sha(path),seconds=seconds,frames=len(c['x']),model_frames=len(prob)))
   if c['row']['kind']=='full_normal' and c['row']['clip_id'] in gold:
    for a in gold[c['row']['clip_id']]['anchors']:
     j=a['frame'];bp=np.asarray(before['phase_probability'][j]);oldcls=oldmap[int(bp.argmax())];newcls=int(prob[j].argmax());frames.append(dict(seed=seed,clip_id=c['row']['clip_id'],frame=j,target=a['label'],old=oldcls,new=newcls,confidence=float(prob[j].max())))
    # Existing visual movement-boundary annotations, linked to generic relation contexts.
    for ar in read(B/'OBSERVED_BOUNDARIES.json')['rows']:
     if ar['role']!='development' or ar['clip_id']!=c['row']['clip_id']:continue
     for ev in ar['events']:
      lo,hi=ev['onset'][0],ev['offset'][1];matches=[]
      for li in out['guide_link_bindings']:
       for e in li['movement_links']:
        if e['axis']==ar['axis']:matches.append((be.ev.iou([lo,hi],e['span']),li['card'],e['span']))
      best=max(matches,default=(0.,None,None));event_scores.append(dict(seed=seed,clip_id=ar['clip_id'],axis=ar['axis'],span=[lo,hi],linked=best[0]>=.5,best=best))
  subset=[r for r in allrows if r['seed']==seed];event('evaluated',seed=seed,changed=sum(r['before']!=r['after'] for r in subset),frame_correct=sum(r['new']==r['target'] for r in frames if r['seed']==seed),frame_total=sum(r['seed']==seed for r in frames))
 write(U/'FULLPATH_COMPLETE.json',dict(rows=allrows));write(U/'FRAME_CONTEXT_SCORES.json',frames);write(U/'EVENT_LINK_SCORES.json',event_scores);write(U/'EVALUATION_COMPLETE.json',dict(seeds=cfg['seeds'],conditions=78,outputs=len(allrows),historical_development=True,new_independent_test=False));write(U/'CURRENT_EXECUTION.json',dict(status='EVALUATED_PENDING_SAVED_VERIFICATION',pid=None))
if __name__=='__main__':
 try:main()
 except BaseException:(U/f'EVAL_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),'utf-8');raise
