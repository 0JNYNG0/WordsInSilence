from completion_model import *
from train_completion import intervals
import os,traceback

def tensor_ob(z):return dict(h=torch.tensor(z['h']),q=torch.tensor(z['q']),fh=torch.tensor(z['fh'][None]))
def old_quality(net,ob,events,frozen):
 ss=[(0,e['a'],e['b'],e['axis']*2+(e['moving']=='L')) for e in events]
 sf=rc.segment_features(frozen,ob,ss)
 with torch.inference_mode():return net.quality(torch.tensor(sf)).softmax(-1).numpy()

def common_binding(saved,pred,directions,mode,contracts):
 cards=copy.deepcopy(saved['cards']);ledger=saved['ledger'];bindings=[]
 for c,s in zip(cards,contracts):
  # The moving subject is declarative guide metadata. A second displayed hand
  # is not automatically a second movement (e.g. an arm used as the anchor).
  subjects=s['movement_subjects']
  if s['axis'] is None:continue
  es=[]
  for i,e in enumerate(ledger):
   if e['axis']!=s['axis'] or e['family'] not in s['families'] or e['moving'] not in subjects or e['role_coverage']<.65 or e['valid']<.8:continue
   q=pred[i];d=directions[i];complete=q['physical_event_complete'] if mode in ['head_only','combined'] else bool(e['accepted'])
   readable=d['eligible'] and d['confidence']>=.9 and d['prediction']!=1
   if mode=='head_only':readable=readable and complete
   # Retain all reliable direction attempts, including partial contrary motion.
   es.append(dict(span=[e['a'],e['b']],moving=e['moving'],anchor=opposite(e['moving']),direction=d['prediction']-1 if readable else None,direction_readable=bool(readable),physical_complete=bool(complete),semantic_complete=False,quality=q))
  signs={e['direction'] for e in es if e['direction'] is not None};expected=s['sign'];st='not_specified' if expected is None else 'unknown'
  if expected is not None and signs:st='contradicted' if signs=={-expected} else 'mixed' if len(signs)>1 else 'supported'
  completed=any(e['physical_complete'] for e in es);c['bilateral_completion']=dict(state=st,physical_event_complete=completed,semantic_complete=False,observed=es,expected_direction=expected,subjects=subjects)
  if not c.get('observation_quarantine_overlap'):
   if st=='contradicted':
    jc.integration.add_warning(c,'bilateral_direction','표시된 손의 이동 방향을 확인해 보세요. '+s['instruction'])
    c['highlight_spans']=[e['span'] for e in es if e['direction_readable']]
   elif (st=='mixed' or (es and not completed)) and c['display_status']=='similar':
    c.update(display_status='uncertain',message='이동 일부는 보이지만, 같은 행동을 끝까지 했는지는 아직 확인하지 못했어요.',educational_wrong=False)
  bindings.append(dict(card=c['id'],state=st,events=es))
 return cards,bindings

def main():
 assert (U/'TRAINING_COMPLETE.json').exists() and not (U/'EVALUATION_COMPLETE.json').exists()
 torch.set_num_threads(4);torch.set_num_interop_threads(1);cfg=read(U/'FROZEN.json')
 for p,h in cfg['pins'].items():assert sha(p)==h,p
 guide=jc.ep.family_contracts()
 for specs in guide.values():
  for spec in specs:
   spec['movement_subjects']=spec['hands'] if spec['axis'] in [3,4] else spec['hands'][:1]
   spec['anchors']={h:opposite(h) for h in spec['movement_subjects']};spec['shared_relation']=spec['axis']==3
 write(U/'GUIDE_SUBJECTS.json',guide)
 write(U/'EVAL_PINS.json',dict(pins={str(p):sha(p) for p in [U/'evaluate_completion.py',U/'completion_model.py',U/'LABELS.json',U/'FROZEN.json',U/'GUIDE_SUBJECTS.json']},gates=cfg['gate'],actual_learner_errors=False,post_score_tuning=False))
 sources={r['clip_id']:r for r in read(U/'PUBLIC_CACHE.json')['rows']};cached={r['key']:r for r in read(R/'EVALUATION_INPUTS.json')['rows']};savedrows=read(R/'safety_reconcile/FULLPATH_COMPLETE.json')['rows'];frozen=rc.frozen_model();models={};oldnets={}
 for seed in cfg['seeds']:
  n=CompletionNet(seed,cfg['norm']);n.load_state_dict(torch.load(U/'runs'/str(seed)/'model.pt',weights_only=True,map_location='cpu'));n.eval();models[seed]=n
  n=rc.RoleAdapter(seed);n.load_state_dict(torch.load(R/'runs'/str(seed)/'model.pt',weights_only=True,map_location='cpu'));n.eval();oldnets[seed]=n
 # Direct interval evaluation separates action understanding from localization.
 direct=[]
 for ann in read(U/'LABELS.json')['rows']:
  if ann['role']=='fit':continue
  clip=ann['clip_id'];z=np.load(cached[clip+'_full_normal']['path']);x,ts=z['x'],z['ts'];ob=tensor_ob(z);rows=intervals(ann);es=[dict(a=a,b=b,axis=ann['axis'],moving=ann['moving']) for a,b,*_ in rows]
  for seed in cfg['seeds']:
   op=old_quality(oldnets[seed],ob,es,frozen);new=infer_batch(models[seed],x,ts,es)
   for ix,(r,e,p,q) in enumerate(zip(rows,es,op,new)):
    direct.append(dict(clip_id=clip,person=ann['person'],seed=seed,event=e,kind=r[-1],target=r[2],boundary_target=r[3],direction_target=r[4],before=int(p.argmax()),before_complete=bool(p[2]>=.65 and p.argmax()==2),before_probability=p.tolist(),after=q))
 write(U/'DIRECT_SCORES.json',direct)
 # Final feedback comparison reuses all stored completed inference. No edit kind
 # or annotation is passed into common_binding or the learned network.
 outputs=[];records=[]
 motions={}
 for seed in cfg['seeds']:
  n=mc.MotionNet();n.load_state_dict(torch.load(M/'runs'/str(seed)/'model.pt',weights_only=True,map_location='cpu'),strict=False);n.eval();motions[seed]=n
 for rr in savedrows:
  seed=rr['seed'];key=rr['clip_id']+'_'+rr['kind'];src=cached[key];assert sha(src['path'])==src['sha256'];assert sha(rr['result'])==rr['sha256'];z=np.load(src['path']);x,ts=z['x'],z['ts'];saved=read(rr['result']);es=saved['ledger'];t=time.perf_counter();pred=infer_batch(models[seed],x,ts,es);ds=[mc.direction(motions[seed],x,ts,e['a'],e['b'],e['axis'],e['moving'],opposite(e['moving'])) for e in es];elapsed=time.perf_counter()-t;contracts=guide[rr['clip_id'].split('_')[2]]
  for mode in ['head_only','binding_only','combined']:
   cards,bindings=common_binding(saved,pred,ds,mode,contracts)
   for a,b in zip(cards,saved['cards']):
    assert a.get('shape_evidence')==b.get('shape_evidence') and a.get('motion_evidence')==b.get('motion_evidence')
    if b['display_status']=='difference':assert a['display_status']=='difference'
   p=U/'feedback'/str(seed)/mode/f'{key}.json';write(p,dict(cards=cards,bindings=bindings,previous=rr['result'],previous_sha256=rr['sha256'],predictions=pred,protected=saved['protected']))
   records.append(dict(seed=seed,clip_id=rr['clip_id'],kind=rr['kind'],mode=mode,before=rc.compact(saved['cards']),after=rc.compact(cards),result=str(p),sha256=sha(p),additional_seconds=elapsed))
  outputs.append(dict(seed=seed,key=key,candidates=len(es)))
  if len(outputs)%78==0:event('feedback_compared',seed=seed,conditions=78)
 write(U/'FULLPATH_COMPLETE.json',dict(rows=records));write(U/'EVALUATION_COMPLETE.json',dict(direct_rows=len(direct),conditions=78,seeds=cfg['seeds'],modes=3,outputs=len(records),base_TCN_DTW_reruns=0));write(U/'CURRENT_EXECUTION.json',dict(status='EVALUATED_PENDING_VERIFICATION',pid=None))
if __name__=='__main__':
 try:main()
 except BaseException:(U/f'EVAL_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),'utf-8');raise
