from relative_core import *
from collections import defaultdict
import traceback,os

def assign_ledger(ledger,prob):
 out=[]
 for e in ledger:
  a,b=e['a'],e['b'];p=prob[a:b+1].mean(0);fam=int(p.argmax());cov=float((prob[a:b+1,fam]>=.65).mean());q=copy.deepcopy(e)
  q.update(family=fam,family_probability=p.tolist(),role_coverage=cov,previous_family=e['family'],assignment_source='learned whole-sequence residual',accepted=False)
  # Keep all candidates in audit. No expected direction or completion score
  # participates in assigning an observed action to a guide family.
  q['assignment_eligible']=bool(fam in range(1,8) and cov>=.65 and e['valid']>=.8);out.append(q)
 return out

def bind(saved,pred,directions,contracts,newprob=None):
 ledger=assign_ledger(saved['ledger'],newprob) if newprob is not None else copy.deepcopy(saved['ledger']);cards=copy.deepcopy(saved['cards']);bindings=[]
 for c,s in zip(cards,contracts):
  es=[];reject=[]
  if s['axis'] is None:continue
  for i,e in enumerate(ledger):
   if e['axis']!=s['axis'] or e['moving'] not in s['movement_subjects']:continue
   if e['family'] not in s['families'] or e['role_coverage']<.65 or e['valid']<.8:
    reject.append(dict(event=e['event'],span=[e['a'],e['b']],family=e['family'],reason='role_or_observation_not_supported'));continue
   key=(e['axis'],e['moving'],e['a'],e['b']);d=directions.get(key)
   es.append(dict(event=e['event'],span=[e['a'],e['b']],moving=e['moving'],axis=e['axis'],direction=d['direction'] if d else None,direction_readable=bool(d and d['direction_readable']),physical_complete=pred[i]['physical_event_complete'],semantic_complete=False,quality=pred[i]))
  signs={e['direction'] for e in es if e['direction'] is not None};expected=s['sign'];state='not_specified' if expected is None else 'unknown'
  if expected is not None and signs:state='contradicted' if signs=={-expected} else 'mixed' if len(signs)>1 else 'supported'
  c['completion_evidence']=dict(physical_interval_complete=any(e['physical_complete'] for e in es),sign_completion_confirmed=False,state=state,observed=es);c['verdict_scope']='observed_features_only'
  c['completion_message']='움직임의 시작·끝 관측과 단어 전체의 완료 여부는 따로 확인해요.'
  if not c.get('observation_quarantine_overlap'):
   if state=='contradicted':jc.integration.add_warning(c,'bilateral_direction','표시된 손의 이동 방향을 확인해 보세요. '+s['instruction'])
   elif (state=='mixed' or (newprob is not None and not es)) and c['display_status']=='similar':c.update(display_status='uncertain',message='이 움직임이 설명의 어느 행동인지 아직 확실하게 연결하지 못했어요.',educational_wrong=False)
  bindings.append(dict(card=c['id'],state=state,events=es,rejected=reject))
 orders=[]
 if newprob is not None:
  spans={}
  for s in contracts:
   probs=newprob[:,s['families']].sum(-1);spans[s['id']]=[dict(a=a,b=b) for a,b in bcq.bc.groups(probs>=.65) if b-a>=2]
  req=[i for i,s in enumerate(contracts) if s['required'] and not s.get('parent')]
  for ia,ib in zip(req,req[1:]):
   aa,bb=spans[contracts[ia]['id']],spans[contracts[ib]['id']];state=rc.lc.order_state(aa,bb);orders.append(dict(first=contracts[ia]['id'],second=contracts[ib]['id'],state=state,first_spans=aa,second_spans=bb))
   if state=='contradicted' and not any(cards[i].get('observation_quarantine_overlap') for i in [ia,ib]):jc.integration.add_warning(cards[ib],'sequence_role_order','이 두 행동의 순서를 확인해 보세요. 먼저 '+contracts[ia]['instruction'])
 return cards,bindings,ledger,orders

def main():
 assert (U/'TRAINING_COMPLETE.json').exists() and (U/'PRECHECK.json').exists() and not (U/'EVALUATION_COMPLETE.json').exists();torch.set_num_threads(4);torch.set_num_interop_threads(1);cfg=read(U/'FROZEN.json')
 for p,h in cfg['pins'].items():assert sha(p)==h
 pins=[U/'evaluate_relative.py',U/'precheck.py',U/'TRAINING_COMPLETE.json',Q/'DIRECT_SCORES.json',Q/'FULLPATH_COMPLETE.json',Q/'CROSS_SCORES.json',Q/'EXTRA_LEFT_SCORES.json',R/'FRAME_ROLE_SCORES.json']
 write(U/'EVAL_FROZEN.json',dict(pins={str(p):sha(p) for p in pins},conditions=['control_only','position_only','binding_only','position_and_binding'],primary=cfg['primary'],scope='Same historical public development plus nonfocus probes. Not fresh independent or natural learner-error evaluation.',gates=cfg['gates'],additional_gate='New completion must improve or preserve BOTH complete recall per hand and partial false completion against previous Q and older P; no unseen evaluation tuning; role frame accuracy must not regress.',candidate_policy='All original candidates retained. Learned guide role is used without desired direction. Unknown direction after reassignment remains unknown; old completed direction inference not rerun.'))
 nets={};roles={}
 for r in read(U/'TRAINING_COMPLETE.json')['runs']:
  assert sha(r['path'])==r['sha256']
  n=SequenceBinding(r['seed']) if r['mode']=='binding' else RelativeNet(r['seed'],cfg['norm'],cfg['position_norm'],r['mode']);n.load_state_dict(torch.load(r['path'],weights_only=True,map_location='cpu'));n.eval()
  if r['mode']=='binding':roles[r['seed']]=n
  else:nets[r['seed'],r['mode']]=n
 cached={r['key']:r for r in read(R/'EVALUATION_INPUTS.json')['rows']};groups=defaultdict(list)
 for r in read(Q/'DIRECT_SCORES.json'):
  if r['mode']=='endpoints':groups[r['previous']['seed'],r['previous']['clip_id']].append(r)
 direct=[];write(U/'CURRENT_EXECUTION.json',dict(status='EVALUATING',pid=os.getpid()))
 for (seed,clip),rows in groups.items():
  src=cached[clip+'_full_normal'];assert sha(src['path'])==src['sha256'];z=np.load(src['path'])
  for mode in cfg['completion_modes']:
   pp=predict_completion(nets[seed,mode],z['x'],z['ts'],[r['previous']['event'] for r in rows])
   for r,p in zip(rows,pp):direct.append(dict(previous=r,mode=mode,after=p))
 write(U/'DIRECT_SCORES.json',direct);event('direct_evaluated',rows=len(direct))
 full=[];fr=[];frame_rows=read(R/'FRAME_ROLE_SCORES.json');guide=read(P/'GUIDE_SUBJECTS.json');qc={(r['seed'],r['clip_id'],r['kind']):r for r in read(Q/'FULLPATH_COMPLETE.json')['rows'] if r['condition']=='endpoints_joint'}
 for rr in read(R/'safety_reconcile/FULLPATH_COMPLETE.json')['rows']:
  seed=rr['seed'];key=rr['clip_id']+'_'+rr['kind'];src=cached[key];assert sha(src['path'])==src['sha256'];assert sha(rr['result'])==rr['sha256'];saved=read(rr['result']);z=np.load(src['path']);qr=qc[seed,rr['clip_id'],rr['kind']];assert sha(qr['result'])==qr['sha256'];before=read(qr['result']);directions={}
  for b in before['bindings']:
   for e in b['events']:directions[(e['axis'],e['moving'],*e['span'])]=e
  start=time.perf_counter();newprob=roles[seed].predict(z['z'],np.array(saved['role_probabilities']));role_seconds=time.perf_counter()-start
  if rr['kind']=='full_normal':
   for r in frame_rows:
    if r['seed']==seed and r['clip_id']==rr['clip_id']:fr.append(dict(r,previous=r['after'],after=int(newprob[r['frame']].argmax()),new_confidence=float(newprob[r['frame']].max())))
  pp={};times={}
  for mode in cfg['completion_modes']:
   start=time.perf_counter();pp[mode]=predict_completion(nets[seed,mode],z['x'],z['ts'],saved['ledger']);times[mode]=time.perf_counter()-start
  oldpred=[dict(p,physical_event_complete=p['complete_joint']) for p in before['predictions']]
  for condition in ['control_only','position_only','binding_only','position_and_binding']:
   cp=oldpred if condition=='binding_only' else pp['control'] if condition=='control_only' else pp['position'];rp=newprob if condition in ['binding_only','position_and_binding'] else None
   cards,bindings,ledger,orders=bind(saved,cp,directions,guide[rr['clip_id'].split('_')[2]],rp);path=U/'feedback'/str(seed)/condition/f'{key}.json'
   write(path,dict(cards=cards,bindings=bindings,ledger=ledger,orders=orders,predictions=cp,role_probability=newprob.tolist() if rp is not None else None,previous=qr['result'],previous_sha256=qr['sha256'],protected=saved['protected']))
   full.append(dict(seed=seed,clip_id=rr['clip_id'],kind=rr['kind'],condition=condition,before=rc.compact(before['cards']),older=rc.compact(saved['cards']),after=rc.compact(cards),result=str(path),sha256=sha(path),additional_seconds=(role_seconds if rp is not None else 0)+(0 if condition=='binding_only' else times['control' if condition=='control_only' else 'position'])))
 write(U/'FRAME_ROLE_SCORES.json',fr);write(U/'FULLPATH_COMPLETE.json',dict(rows=full));event('fullpath_evaluated',rows=len(full),frames=len(fr))
 sys.modules['motion_core']=mc;reader=load('_relative_public_convert',M/'evaluate.py');raws={r['clip_id']:r for r in read(A/'EXTRACTION_COMPLETE.json')['rows'] if r['person']==14};cg=defaultdict(list);eg=defaultdict(list)
 for r in read(Q/'CROSS_SCORES.json'):
  if r['mode']=='endpoints':cg[r['previous']['clip_id'],r['previous']['kind'],r['previous']['seed']].append(r)
 for r in read(Q/'EXTRA_LEFT_SCORES.json'):
  if r['mode']=='endpoints':eg[r['clip_id'],r['seed']].append(r)
 cross=[];extra=[]
 for clip,src in raws.items():
  assert sha(src['raw'])==src['raw_sha256']
  for kind in ['normal','whole_reverse','speed075','speed125']:
   x,ts=reader.convert(src['raw'],kind=='whole_reverse');ts=np.asarray(ts)/dict(speed075=.75,speed125=1.25).get(kind,1)
   for seed in cfg['seeds']:
    rows=cg[clip,kind,seed]
    for mode in cfg['completion_modes']:
     pp=predict_completion(nets[seed,mode],x,ts,[r['previous']['event'] for r in rows])
     cross.extend(dict(previous=r,mode=mode,after=p) for r,p in zip(rows,pp))
     if kind=='normal' and (clip,seed) in eg:
      er=eg[clip,seed];ep=predict_completion(nets[seed,mode],x,ts,[r['event'] for r in er]);extra.extend(dict(previous=r,mode=mode,after=p) for r,p in zip(er,ep))
 write(U/'CROSS_SCORES.json',cross);write(U/'EXTRA_LEFT_SCORES.json',extra);write(U/'EVALUATION_COMPLETE.json',dict(direct=len(direct),full=len(full),frame_roles=len(fr),cross=len(cross),extra=len(extra),old_TCN_DTW_RTMW_inference_repeated=0))
 write(U/'CURRENT_EXECUTION.json',dict(status='EVALUATED_PENDING_VERIFICATION',pid=None));event('evaluation_complete')

if __name__=='__main__':
 try:main()
 except BaseException:(U/f'EVAL_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),'utf-8');raise
