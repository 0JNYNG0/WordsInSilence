from core import *
import copy,traceback,statistics
sys.modules['phase_core']=old
ev=load('_ob_saved_evaluation',PREV/'evaluate.py')
def infer(net,x,ts,fh):
 g=old.old.geometry(x,ts);q=old.relation_features(x)
 with torch.inference_mode():ph,_,_=net(torch.tensor(g[None]),torch.tensor(q[None]),torch.tensor(fh[None]),torch.tensor([len(x)]))
 return ph.sigmoid()[0].numpy(),q
def propose(ph,q,stage):
 events=[];partials=[]
 for k in range(10):
  valid=q[:,k,-2:].max(-1)>.5;ee,pp=decode(ph[:,k],valid)
  for e in ee:
   a,b=e['a'],e['b'];p=np.array([stage[a:b+1,g].sum(-1).mean() for g in ev.groups]);e.update(axis=k//2,moving='L' if k%2 else 'R',semantic_group=int(p.argmax()),semantic_confidence=float(p.max()),observed=float(valid[a:b+1].mean()),phases_source='derived from predicted onset/offset, not perframe3phaseclassifier');events.append(e)
  partials.extend(dict(p,axis=k//2,moving='L' if k%2 else 'R') for p in pp)
 return events,partials
def match_events(events,truth):
 pairs=[]
 for i,t in enumerate(truth):
  for j,e in enumerate(events):
   if e['axis']==t['axis'] and e['moving']==t['moving']:pairs.append((ev.iou(t['span'],[e['a'],e['b']]),i,j))
 matched={};used=set()
 for overlap,i,j in sorted(pairs,reverse=True):
  if overlap<.5 or i in matched or j in used:continue
  matched[i]=(j,overlap);used.add(j)
 return matched
def component_eval(net,base,focus):
 result=[]
 for variant in read(U/'FROZEN.json')['conditions']:
  for r in focus:
   x,ids,ts=ev.transforms.transform(r,variant);fh,stage=ev.frozen_forward(base,x,r['word']);ph,q=infer(net,x,ts,fh);events,parts=propose(ph,q,stage)
   for linked in [False,True]:
    es=events if linked else events+[dict(p) for p in parts if p['b']-p['a']>=3];truth=[]
    for pi,p in enumerate(r['pairs']):
     a=int(np.argmin(abs(ids-p['a'])));b=int(np.argmin(abs(ids-p['b'])));lo,hi=sorted([a,b])
     if p['target']!=1:truth.append(dict(pair=pi,span=[lo,hi],axis=p['axis'],moving=p.get('moving','R'),target=2-p['target'] if a>b else p['target']))
     else:
      k=p['axis']*2+(p.get('moving','R')=='L');known=q[lo:hi+1,k,-2:].max(-1)>.5;ok=known.mean()>=.8 and known.sum()>=4;pr=4 if ph[lo:hi+1,k,0].mean()<.5 else 2;result.append(dict(variant=variant,linked=linked,task='hold',clip_id=r['clip_id'],pair=pi,target=4,pred=pr if ok else -1,eligible=bool(ok)))
    matches=match_events(es,truth)
    for i,t in enumerate(truth):
     mm=matches.get(i);e=es[mm[0]] if mm else None;pred=-1;conf=0.;geook=False
     if e:
      p=r['pairs'][t['pair']];g,geook,_=prior.geometry(x,ts,e['a'],e['b'],p['axis'],p.get('moving','R'),p.get('anchor','L'))
      with torch.inference_mode():pr=base.movement(torch.tensor(fh[None]),torch.tensor(g[None]),torch.tensor([[0,e['a'],e['b']]])).softmax(-1)[0].numpy()
      pred=int(pr.argmax()) if geook else -1;conf=float(pr.max())
     result.append(dict(variant=variant,linked=linked,task='movement_interval',clip_id=r['clip_id'],pair=t['pair'],target=t['target'],pred=pred,matched=bool(mm),overlap=float(mm[1]) if mm else 0.,boundary_error_frames=float((abs(e['a']-t['span'][0])+abs(e['b']-t['span'][1]))/2) if e else None,predicted_span=[e['a'],e['b']] if e else None,confidence=conf,eligible=bool(geook),all_proposals=len(es),partial_count=len(parts)))
 return result
def local_directions(out,x,ts,fh,base):
 contracts=ev.rt.consumer.refine.__globals__['CONTRACTS'];allspans=out['phase_events']+out['phase_partials'];cache={};added=[]
 for c in out['cards']:
  m=c.get('motion_evidence');contract=contracts.get(c['id'])
  if not m or not m.get('span') or not contract:continue
  lo,hi=m['span'];observations=[]
  for s in allspans:
   if s['axis']!=2 or s['moving']!=contract['moving_hand']:continue
   a,b=max(lo,s['a']),min(hi,s['b'])
   if b-a+1<4:continue
   key=(a,b,contract['moving_hand'],contract['anchor_arm'])
   if key not in cache:
    g,ok,info=prior.geometry(x,ts,a,b,2,key[2],key[3])
    with torch.inference_mode():p=base.movement(torch.tensor(fh[None]),torch.tensor(g[None]),torch.tensor([[0,a,b]])).softmax(-1)[0].numpy()
    cache[key]=dict(span=[a,b],class_index=int(p.argmax()),confidence=float(p.max()),eligible=bool(ok),observation=info)
   observations.append(cache[key])
  signs={v['class_index']-1 for v in observations if v['eligible'] and v['confidence']>=.65 and v['class_index']!=1};expected=contract['expected_sign'];state='unknown' if len(signs)!=1 else 'supported' if signs=={expected} else 'contradicted';c['observed_boundary_direction']=dict(state=state,observations=observations,whole_event_complete=False)
  if state=='contradicted' and m.get('old_geometry_state')=='contradicted' and not c.get('observation_quarantine_overlap'):
   c.update(status='difference_candidate',display_status='difference',message='표시된 이동 구간에서 참고와 반대 방향이 보여요. 이동 방향을 확인해 보세요.',educational_wrong=False);c['candidate_message']=c['message'];c.setdefault('difference_components',[])
   if 'observed_boundary_direction' not in c['difference_components']:c['difference_components'].append('observed_boundary_direction')
   added.append(c['id'])
 return added
def boundary_metrics(outputs,annotations):
 results=[]
 for clip,events in outputs.items():
  truth=[]
  for a in annotations:
   if a['clip_id']!=clip:continue
   truth.extend(dict(axis=a['axis'],moving=a['moving'],span=[e['onset'][0],e['offset'][1]],onset=e['onset'],offset=e['offset']) for e in a['events'])
  matches=match_events(events,truth)
  for i,t in enumerate(truth):
   mm=matches.get(i);e=events[mm[0]] if mm else None
   def dist(v,band):return max(band[0]-v,0,v-band[1])
   de=[dist(e['a'],t['onset']),dist(e['b'],t['offset'])] if e else None
   results.append(dict(clip_id=clip,axis=t['axis'],onset=t['onset'],offset=t['offset'],matched=bool(mm),predicted_span=[e['a'],e['b']] if e else None,overlap=float(mm[1]) if mm else 0.,boundary_distance_to_reviewed_ranges=de,both_in_reviewed_ranges=de==[0,0]))
 return results
def main():
 assert not (U/'EVALUATION_COMPLETE.json').exists();cfg=read(U/'FROZEN.json');assert (U/'TRAINING_COMPLETE.json').exists()
 for p,h in read(U/'EVAL_PINS.json').items():assert sha(p)==h,p
 torch.set_num_threads(4);torch.set_num_interop_threads(1);base=prior.Network('coupled',read(old.P/'NORMALIZATION.json'));base.load_state_dict(torch.load(old.P/'runs/coupled_42/model.pt',weights_only=True,map_location='cpu'));base.eval();focus=[r for r in old.rows() if r['role']=='development' and r['word']>=0];runtime=ev.prepare_runtime();latest={r['clip_id']+'_'+r['kind']:r for r in read(PREV/'PARTIAL_COMPLETE.json')['rows']};wordmap={r['word_id']:r['word'] for r in focus}
 for c in runtime:
  r=latest[c['key']];assert sha(r['result'])==r['sha256'];c['baseline']=read(r['result']);c['fh'],c['stage']=ev.frozen_forward(base,c['x'],wordmap[c['baseline']['word']])
 annotations=[r for r in read(U/'OBSERVED_BOUNDARIES.json')['rows'] if r['role']=='development'];devclips={r['clip_id'] for r in annotations};oldnormal={r['clip_id']:read(r['result'])['phase_events'] for r in read(PREV/'FULLPATH_COMPLETE.json')['rows'] if r['seed']==42 and r['kind']=='full_normal' and r['clip_id'] in devclips};write(U/'BASELINE_NEW_BOUNDARY_SCORE.json',boundary_metrics(oldnormal,annotations))
 allrows=[];summaries={}
 for seed in cfg['seeds']:
  net=BoundaryNet();net.load_state_dict(torch.load(U/'runs'/str(seed)/'model.pt',weights_only=True,map_location='cpu'));net.eval();scores=component_eval(net,base,focus);write(U/'scores'/f'{seed}.json',scores);normal={}
  for c in runtime:
   t=time.perf_counter();ph,q=infer(net,c['x'],c['ts'],c['fh']);events,parts=propose(ph,q,c['stage']);out=ev.supplement(c['baseline'],events,parts,c['x']);added=local_directions(out,c['x'],c['ts'],c['fh'],base);elapsed=time.perf_counter()-t;out['phase_scope']='Observed RGB boundary-range supervised auxiliary; no hidden/semantic completion guarantee.';p=U/'fullpath'/str(seed)/c['key']/'after.json';write(p,out)
   before=ev.compact(c['baseline']);after=ev.compact(out);assert before['motion']==after['motion'] and before['shape']==after['shape'];assert out['path']==c['baseline']['path'] and out['tcn_top3']==c['baseline']['tcn_top3']
   allrows.append(dict(seed=seed,clip_id=c['row']['clip_id'],kind=c['row']['kind'],before={k:v for k,v in before.items() if k not in ['motion','shape']},after={k:v for k,v in after.items() if k not in ['motion','shape']},added_direction_cues=added,result=str(p),sha256=sha(p),seconds=elapsed,frames=len(c['x']),model_frames=len(ph),shape_direction_preserved=True,tcn_dtw_preserved=True,events=len(events),partials=len(parts)))
   if c['row']['kind']=='full_normal' and c['row']['clip_id'] in devclips:normal[c['row']['clip_id']]=events
  boundaries=boundary_metrics(normal,annotations);write(U/'scores'/f'{seed}_boundaries.json',boundaries);su={}
  for variant in cfg['conditions']:
   su[variant]={}
   for linked in [False,True]:
    m=[r for r in scores if r['variant']==variant and r['linked']==linked and r['task']=='movement_interval'];h=[r for r in scores if r['variant']==variant and r['linked']==linked and r['task']=='hold'];su[variant][str(linked)]=dict(moving=21,matched=sum(r['matched'] for r in m),direction_correct=sum(r['pred']==r['target'] for r in m),hold=11,hold_correct=sum(r['pred']==r['target'] for r in h))
  summaries[str(seed)]=dict(component=su,reviewed_boundaries=dict(total=len(boundaries),matched=sum(r['matched'] for r in boundaries),both_boundaries_in_range=sum(r['both_in_reviewed_ranges'] for r in boundaries)));event('evaluated',seed=seed,normal=su['normal'],boundaries=summaries[str(seed)]['reviewed_boundaries'])
 write(U/'FULLPATH_COMPLETE.json',dict(rows=allrows));write(U/'EVALUATION_COMPLETE.json',dict(models=summaries,historicalDEV=True,no_private=True,no_default_change=True));write(U/'CURRENT_EXECUTION.json',dict(status='EVALUATED_AWAITING_SAVED_VERIFICATION',pid=None))
if __name__=='__main__':
 try:main()
 except BaseException:(U/f'EVAL_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),'utf-8');raise
