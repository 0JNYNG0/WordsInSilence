"""Separate saved-output verification; no model execution or training."""
from pathlib import Path
import json,hashlib,math,statistics
U=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,r):Path(p).write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n','utf-8')

def main():
 assert (U/'EVALUATION_COMPLETE.json').exists();checks=[]
 def check(n,v):
  assert v,n
  checks.append(n)
 for f in ['PROTECTIONS.json','TRAIN_CODE_PIN.json']:
  for p,h in read(U/f).items():check('preserved '+p,sha(p)==h)
 for f in ['FROZEN.json','EVAL_FROZEN.json']:
  for p,h in read(U/f)['pins'].items():check('pin '+p,sha(p)==h)
 for r in read(U/'TRAINING_COMPLETE.json')['runs']:check('model '+r['path'],sha(r['path'])==r['sha256'])
 direct=read(U/'DIRECT_SCORES.json');cross=read(U/'CROSS_SCORES.json');extra=read(U/'EXTRA_LEFT_SCORES.json');frames=read(U/'FRAME_ROLE_COMPARABLE.json');full=read(U/'FULLPATH_COMPLETE.json')['rows']
 correction=read(U/'METRIC_CORRECTION.json');check('raw frame scores preserved',sha(U/'FRAME_ROLE_SCORES.json')==correction['first_frame_scores_sha256']);check('comparable scores hash',sha(U/'FRAME_ROLE_COMPARABLE.json')==correction['corrected_sha256'])
 for raw,new in zip(read(U/'FRAME_ROLE_SCORES.json'),frames):check('same legacy class mapping',new['after']==(raw['after'] if raw['after']<8 else 0) and new['target']==raw['target'] and new['new_raw_after']==raw['after'])
 check('direct600',len(direct)==600);check('cross1956',len(cross)==1956);check('extra132',len(extra)==132);check('full936',len(full)==936);check('frames201',len(frames)==201)
 for r in direct+cross+extra:
  p=r['after'];check('finite normalized',all(math.isfinite(v) for v in p['quality_probability']) and abs(sum(p['quality_probability'])-1)<1e-5)
  check('no semantic success',not p['semantic_complete'])
  if not p['eligible']:check('missing not promoted',not p['physical_event_complete'])
 ds={};cs={};xs={};rs={};fs={}
 def summarize(rows,extra=False):
  s=dict(n=len(rows),whole=0,partial=0,whole_previous=0,whole_older=0,whole_after=0,partial_false_previous=0,partial_false_older=0,partial_false_after=0,nonwhole_false_previous=0,nonwhole_false_older=0,nonwhole_false_after=0,quality_previous=0,quality_after=0)
  for r in rows:
   q=r['previous'];p=r['after'];t=q['label']['quality'] if extra else q['previous']['target'];older=q['before'] if extra else q['previous']['after'];pr=q['after'];s['whole']+=t==2;s['partial']+=t==1;s['quality_previous']+=pr['quality']==t;s['quality_after']+=p['quality']==t
   for name,complete in [('previous',pr['complete_joint']),('older',older['physical_event_complete']),('after',p['physical_event_complete'])]:
    s['whole_'+name]+=t==2 and complete;s['partial_false_'+name]+=t==1 and complete;s['nonwhole_false_'+name]+=t!=2 and complete
  return s
 for seed in [42,43,44]:
  rr=[r for r in frames if r['seed']==seed];rs[str(seed)]=dict(n=len(rr),before=sum(r['previous']==r['target'] for r in rr),after=sum(r['after']==r['target'] for r in rr),regressed=[r for r in rr if r['previous']==r['target'] and r['after']!=r['target']],improved=[r for r in rr if r['previous']!=r['target'] and r['after']==r['target']])
  for mode in ['control','position']:
   for side in ['R','L']:
    rows=[r for r in direct if r['mode']==mode and r['previous']['previous']['seed']==seed and r['previous']['previous']['event']['moving']==side];ds[f'{seed}/{mode}/{side}']=summarize(rows)
   rows=[r for r in cross if r['mode']==mode and r['previous']['previous']['seed']==seed and r['previous']['previous']['known']];s=summarize(rows);s['unknown']=sum(r['mode']==mode and r['previous']['previous']['seed']==seed and not r['previous']['previous']['known'] for r in cross);cs[f'{seed}/{mode}']=s
   rows=[r for r in extra if r['mode']==mode and r['previous']['seed']==seed];xs[f'{seed}/{mode}']=summarize(rows,True)
 for r in full:
  check('result sha',sha(r['result'])==r['sha256']);o=read(r['result']);check('old sha',sha(o['previous'])==o['previous_sha256']);b=read(o['previous']);check('protected TCN DTW',o['protected']==b['protected'])
  # No candidate is erased even when assigned to preparation/return/unknown.
  base=read(b['previous']);check('ledger retained',len(o['ledger'])==len(base['ledger']))
  for x,y in zip(o['ledger'],base['ledger']):check('candidate identity retained',all(x[k]==y[k] for k in ['a','b','axis','moving','event']))
  for x,y in zip(o['cards'],b['cards']):
   check('shape direction evidence intact',x.get('shape_evidence')==y.get('shape_evidence') and x.get('motion_evidence')==y.get('motion_evidence'))
   if 'completion_evidence' in x:check('separate whole verdict',not x['completion_evidence']['sign_completion_confirmed'])
  key=f"{r['seed']}/{r['condition']}/{r['kind']}";s=fs.setdefault(key,dict(n=0,warning_before=0,warning_after=0,similar_before=0,similar_older=0,similar_after=0,new_warnings=0,lost_warnings=0));s['n']+=1
  for label in ['before','after']:s['warning_'+label]+=r[label]['warning'];s['similar_'+label]+=r[label]['similar_no_warning']
  s['similar_older']+=r['older']['similar_no_warning'];s['new_warnings']+=r['after']['warning'] and not r['before']['warning'];s['lost_warnings']+=r['before']['warning'] and not r['after']['warning']
 gates={}
 for cond in ['control_only','position_only','binding_only','position_and_binding']:
  mode='control' if cond=='control_only' else 'position';dd=[v for k,v in ds.items() if '/'+mode+'/' in k];cc=[v for k,v in cs.items() if k.endswith('/'+mode)];xx=[v for k,v in xs.items() if k.endswith('/'+mode)];ff=[(k,v) for k,v in fs.items() if '/'+cond+'/' in k];uses_new_completion=cond!='binding_only';uses_new_binding=cond in ['binding_only','position_and_binding']
  g=dict(whole_per_side_preserved=(not uses_new_completion or all(v['whole_after']>=max(v['whole_previous'],v['whole_older']) for v in dd)),partial_false_not_increased=(not uses_new_completion or all(v['partial_false_after']<=min(v['partial_false_previous'],v['partial_false_older']) for v in dd)),other_false_not_increased=(not uses_new_completion or all(v['nonwhole_false_after']<=v['nonwhole_false_previous'] for v in dd)),cross_whole_preserved=(not uses_new_completion or all(v['whole_after']>=v['whole_previous'] for v in cc)),cross_false_not_increased=(not uses_new_completion or all(v['nonwhole_false_after']<=v['nonwhole_false_previous'] for v in cc)),extra_left_preserved=(not uses_new_completion or all(v['whole_after']>=v['whole_previous'] and v['partial_false_after']<=v['partial_false_previous'] for v in xx)),role_accuracy_preserved=(not uses_new_binding or all(v['after']>=v['before'] for v in rs.values())),normal_speed_no_new_warning=all(v['new_warnings']==0 for k,v in ff if k.split('/')[-1] in ['full_normal','speed075','speed125']),normal_similarity_preserved=all(v['similar_after']>=max(v['similar_before'],v['similar_older']) for k,v in ff if k.split('/')[-1] in ['full_normal','speed075','speed125']),existing_warnings_retained=all(v['lost_warnings']==0 for _,v in ff),missing_observation_not_success=all(v['similar_after']==0 for k,v in ff if k.split('/')[-1] in ['all_hand_missing','mask_final_event']))
  g['pass']=all(g.values());gates[cond]=g
 timing=sorted(r['additional_seconds'] for r in full if r['condition']=='position_and_binding')
 final=dict(status='PASS_REVIEW_REQUIRED' if gates['position_and_binding']['pass'] else 'COMPLETE_NOT_ADOPTED',gates=gates,direct=ds,cross=cs,extra_left=xs,role=rs,feedback=fs,timing=dict(median_seconds=statistics.median(timing),p95_seconds=timing[int((len(timing)-1)*.95)],scope='New completion features/inference and sequence-role residual only; not camera/RTMW/TCN/DTW/video latency.'),training=read(U/'TRAINING_COMPLETE.json'),source_scope=read(U/'LABEL_AUDIT.json'),private_data=0,web_changed=False,common100_frozen=True,independent_human=False,natural_learner_error_accuracy=None)
 write(U/'FINAL_RESULT.json',final);write(U/'VERIFICATION.json',dict(status='PASS',checks=len(checks),check_names=checks,implementation='Codex',verifier='Separate saved-output script by same Codex, not independent human',no_inference=True))
 print(json.dumps(dict(status=final['status'],gates=gates,direct={k:v for k,v in ds.items() if '/position/' in k},role={k:{n:v[n] for n in ['n','before','after']} for k,v in rs.items()},checks=len(checks)),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
