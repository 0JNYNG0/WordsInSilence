"""Saved-output-only integrity and arithmetic checks; no models or inference."""
from pathlib import Path
import json,hashlib,math,statistics
from collections import Counter,defaultdict
U=Path(__file__).resolve().parent;O=U.parent
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,r):Path(p).write_text(json.dumps(r,ensure_ascii=False,indent=2),'utf-8')

def main():
 assert (U/'EVALUATION_COMPLETE.json').exists()
 checks=[]
 def check(name,v):
  assert v,name
  checks.append(name)
 for p,h in read(U/'PROTECTIONS.json').items():check('preserved '+p,sha(p)==h)
 for f in ['FROZEN.json','EVAL_FROZEN.json']:
  for p,h in read(U/f)['pins'].items():check('pin '+p,sha(p)==h)
 for r in read(U/'TRAINING_COMPLETE.json')['runs']:check('model '+r['path'],sha(r['path'])==r['sha256'])
 direct=read(U/'DIRECT_SCORES.json');cross=read(U/'CROSS_SCORES.json');extra=read(U/'EXTRA_LEFT_SCORES.json');full=read(U/'FULLPATH_COMPLETE.json')['rows']
 check('direct inputs same count',len(direct)==600);check('cross inputs same count',len(cross)==1956);check('full conditions same count',len(full)==936)
 for row in direct+cross+extra:
  p=row['after'];check('finite normalized quality',all(math.isfinite(v) for v in p['quality_probability']) and abs(sum(p['quality_probability'])-1)<1e-5)
  check('no semantic completion claim',p['semantic_complete']==False)
  check('strict subset joint',not p['complete_strict'] or p['complete_joint'])
  if not p['eligible']:check('unobserved not complete',not p['complete_joint'])
 labels=read(U/'LABELS.json')['rows'];fit=[r for r in labels if r['role']=='fit'];dev=[r for r in labels if r['role']!='fit']+read(U/'EXTRA_EVAL_LABELS.json')['rows']
 check('source separation',not {r['source_sha256'] for r in fit}&{r['source_sha256'] for r in dev})
 check('ID separation',not {r['person'] for r in fit}&{r['person'] for r in dev})
 ds={};cs={};xs={};fs={}
 def newstats(rr,variant,source='previous'):
  out=dict(n=len(rr),whole=0,partial=0,quality_before=0,quality_previous=0,quality_after=0,whole_before=0,whole_previous=0,whole_after=0,partial_false_before=0,partial_false_previous=0,partial_false_after=0,nonwhole_false_before=0,nonwhole_false_previous=0,nonwhole_false_after=0)
  for r in rr:
   b=r[source];p=r['after']
   if source=='previous':t=b['target'];old=b['before'];oc=b['before_complete'];pc=b['after']['physical_event_complete'];pq=b['after']['quality']
   else:t=r['label']['quality'];old=b['quality'];oc=b['physical_event_complete'];pc=oc;pq=old
   nc=p['complete_'+variant];out['whole']+=t==2;out['partial']+=t==1
   out['quality_before']+=old==t;out['quality_previous']+=pq==t;out['quality_after']+=p['quality']==t
   for label,complete in [('before',oc),('previous',pc),('after',nc)]:
    out['whole_'+label]+=t==2 and complete;out['partial_false_'+label]+=t==1 and complete;out['nonwhole_false_'+label]+=t!=2 and complete
  return out
 for seed in [42,43,44]:
  for mode in ['data_repaired','endpoints']:
   for v in ['strict','joint']:
    c=mode+'_'+v
    for side in ['R','L']:
     rr=[r for r in direct if r['previous']['seed']==seed and r['mode']==mode and r['previous']['event']['moving']==side]
     ds[f'{seed}/{c}/{side}']=newstats(rr,v)
    rr=[r for r in cross if r['previous']['seed']==seed and r['mode']==mode and r['previous']['known']]
    s=newstats(rr,v);s['unknown']=sum(r['previous']['seed']==seed and r['mode']==mode and not r['previous']['known'] for r in cross);cs[f'{seed}/{c}']=s
    rr=[r for r in extra if r['seed']==seed and r['mode']==mode];xs[f'{seed}/{c}']=newstats(rr,v,'before')
 for r in full:
  check('result hash '+r['result'],sha(r['result'])==r['sha256']);out=read(r['result']);prev=read(out['previous'])
  check('prior result hash',sha(out['previous'])==out['previous_sha256']);check('TCN DTW unchanged',out['protected']==prev['protected'])
  for a,b in zip(out['cards'],prev['cards']):
   check('shape direction intact',a.get('shape_evidence')==b.get('shape_evidence') and a.get('motion_evidence')==b.get('motion_evidence'))
   if b['display_status']=='difference':check('old warning retained',a['display_status']=='difference')
   if 'completion_evidence' in a:check('feature vs whole distinction',not a['completion_evidence']['sign_completion_confirmed'] and a['verdict_scope']=='observed_features_only')
  key=f"{r['seed']}/{r['condition']}/{r['kind']}";s=fs.setdefault(key,dict(n=0,warning_before=0,warning_after=0,similar_before=0,similar_previous=0,similar_after=0,new_warnings=0,lost_warnings=0))
  s['n']+=1
  for key2 in ['before','after']:s['warning_'+key2]+=r[key2]['warning'];s['similar_'+key2]+=r[key2]['similar_no_warning']
  s['similar_previous']+=r['previous_candidate']['similar_no_warning'];s['new_warnings']+=r['after']['warning'] and not r['before']['warning'];s['lost_warnings']+=r['before']['warning'] and not r['after']['warning']
 gates={}
 for c in ['data_repaired_strict','data_repaired_joint','endpoints_strict','endpoints_joint']:
  dd=[v for k,v in ds.items() if '/'+c+'/' in k];cc=[v for k,v in cs.items() if k.endswith('/'+c)];xx=[v for k,v in xs.items() if k.endswith('/'+c)];ff=[(k,v) for k,v in fs.items() if '/'+c+'/' in k]
  g=dict(direct_whole_vs_deployed=all(v['whole_after']>=v['whole_before'] for v in dd),direct_whole_vs_previous_candidate=all(v['whole_after']>=v['whole_previous'] for v in dd),partial_false_vs_deployed=all(v['partial_false_after']<=v['partial_false_before'] for v in dd),partial_false_vs_previous_candidate=all(v['partial_false_after']<=v['partial_false_previous'] for v in dd),other_false_not_increased=all(v['nonwhole_false_after']<=v['nonwhole_false_before'] for v in dd),cross_whole_preserved=all(v['whole_after']>=v['whole_before'] for v in cc),cross_false_not_increased=all(v['nonwhole_false_after']<=v['nonwhole_false_before'] for v in cc),extra_left_whole_preserved=all(v['whole_after']>=v['whole_before'] for v in xx),extra_left_partial_false_not_increased=all(v['partial_false_after']<=v['partial_false_before'] for v in xx),normal_speed_warnings_preserved=all(v['new_warnings']==0 for k,v in ff if k.split('/')[-1] in ['full_normal','speed075','speed125']),normal_feature_similarity_preserved=all(v['similar_after']>=v['similar_before'] for k,v in ff if k.split('/')[-1] in ['full_normal','speed075','speed125']),prior_warnings_preserved=all(v['lost_warnings']==0 for _,v in ff),missing_observation_not_success=all(v['similar_after']==0 for k,v in ff if k.split('/')[-1] in ['all_hand_missing','mask_final_event']))
  g['pass']=all(g.values());gates[c]=g
 timing=[r['additional_seconds'] for r in full if r['condition']=='endpoints_joint'];timing.sort()
 result=dict(status='PASS_FOR_ADOPTION_REVIEW' if gates['endpoints_joint']['pass'] else 'COMPLETE_MODEL_NOT_ADOPTED',primary='endpoints_joint',gates=gates,direct=ds,cross=cs,extra_left=xs,feedback=fs,timing=dict(median_seconds=statistics.median(timing),p95_seconds=timing[int(.95*(len(timing)-1))],scope='Only candidate geometry + new completion neural inference; no RTMW/camera/TCN/DTW/rendering.'),training=read(U/'TRAINING_COMPLETE.json'),preflight=read(U/'PREFLIGHT.json'),independent_human=False,fresh_evaluation=False,natural_learner_error_accuracy=None,web_model_changed=False)
 write(U/'FINAL_RESULT.json',result);write(U/'VERIFICATION.json',dict(status='PASS',checks=len(checks),check_names=checks,verifier='Separate saved-output script by the same Codex; not independent human or sign-language expert',inference_rerun=False))
 print(json.dumps(dict(status=result['status'],gates=gates,direct={k:v for k,v in ds.items() if 'endpoints_joint' in k},cross={k:v for k,v in cs.items() if 'endpoints_joint' in k},extra={k:v for k,v in xs.items() if 'endpoints_joint' in k},checks=len(checks)),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
