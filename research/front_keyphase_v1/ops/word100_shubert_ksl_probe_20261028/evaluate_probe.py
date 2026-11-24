"""Agreement with two blinded AI drafts. No independent-human accuracy claim."""
from pathlib import Path
from collections import Counter,defaultdict
import json,hashlib
import numpy as np
U=Path(__file__).resolve().parent;P=U.parent/'word100_shubert_feasibility_20261028'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,x):(U/n).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def counts(records,method):
 c=Counter(z['outcomes'][method] for z in records)
 a=c['agree'];d=c['different'];u=c['unknown']
 return {'total':len(records),'agree':a,'different':d,'unknown':u,'known':a+d,'known_agreement':a/(a+d) if a+d else None,'agreement_over_all':a/len(records) if records else None,'coverage':(a+d)/len(records) if records else None}
def summarize(records,methods):
 common=[z for z in records if all(z['outcomes'][m]!='unknown' for m in methods)]
 return {'all_eligible':{m:counts(records,m) for m in methods},'common_known_count':len(common),'common_known':{m:counts(common,m) for m in methods}}
def main():
 if (U/'EVALUATION.json').exists():raise RuntimeError('Completed evaluation exists')
 data=read(U/'CORRESPONDENCE.json');frozen=read(U/'FROZEN.json');rubric=read(U/'RUBRIC.json');fields=rubric['fields']
 samples=read(U/'SAMPLES.json')['samples'];ids={x['sample_id'] for x in samples};meta={x['sample_id']:x for x in data['sample_metadata']}
 reviews=[]
 for name in ['REVIEW_A.json','REVIEW_B.json']:
  r=read(U/name);by={z['sample_id']:z['values'] for z in r['rows']}
  assert len(by)==len(r['rows'])==108 and set(by)==ids
  assert len(set(r['images_viewed']))==36
  for sid,values in by.items():
   assert len(values)==10
   assert all(v in [0,1,None] for v in values[:7]),(sid,values)
   assert all(v in ['up','down','still',None] for v in values[7:9])
   assert values[9] in ['closer','farther','still',None]
  reviews.append(by)
 consensus={};review_counts=Counter()
 for sid in sorted(ids):
  values=[]
  for a,b in zip(reviews[0][sid],reviews[1][sid]):
   if a is not None and b is not None and a==b:values.append(a);review_counts['agreed_known']+=1
   else:
    values.append(None)
    review_counts['known_disagreement' if a is not None and b is not None else 'one_or_both_unknown']+=1
  consensus[sid]=values
 save('CONSENSUS.json',{'status':'PROVISIONAL_TWO_BLIND_CODEX_RGB_DRAFTS_NOT_HUMAN_TRUTH','fields':fields,'values':consensus,'counts':dict(review_counts),'review_pins':{n:sha(U/n) for n in ['REVIEW_A.json','REVIEW_B.json']}})
 provenance={r['alias']:np.load(P/'inputs'/r['alias']/'provenance.npz') for r in frozen['rows']}
 sample_by={s['sample_id']:s for s in samples}
 def evidence(sid,fi):
  s=sample_by[sid];p=provenance[s['alias']];slots=[1] if fi in [0,2,4,7] else ([0] if fi in [1,3,5,8] else [0,1])
  indices=s['context_indices'] if fi>=7 else [s['frame_index']]
  return bool((p['source_kind'][np.ix_(indices,slots)]==1).all())
 # Shape: same-hand other-word 1NN, physically current evidence only.
 shape_records=[];shape_exclusion=Counter();shape_class_support=defaultdict(Counter)
 for q in data['shape_queries']:
  sid=q['query_sample_id'];indices=[0,2] if q['side']=='R' else [1,3]
  for fi in indices:
   target=consensus[sid][fi];key=fields[fi]
   if target is None:shape_exclusion['query_visual_unknown']+=1;continue
   if not q['query_current'] or not q['query_angles_valid'] or not q['matches']:shape_exclusion['query_or_reference_sensor_unavailable']+=1;continue
   supported=any(consensus[c][fi]==target and evidence(c,fi) for c in q['candidate_ids'])
   if not supported:shape_exclusion['target_class_absent_from_other_words']+=1;continue
   shape_class_support[key][str(target)]+=1
   outcomes={};refs={}
   for method in ['geometry','dino','shubert']:
    ref=q['matches'][method]['reference_sample_id'];pred=consensus[ref][fi];refs[method]=ref
    outcomes[method]='unknown' if pred is None or not evidence(ref,fi) else ('agree' if pred==target else 'different')
   shape_records.append({'query_sample_id':sid,'query':q['query'],'field':key,'target':target,'annotation_active':meta[sid]['annotation_active'],'reference_ids':refs,'outcomes':outcomes})
 # Stage correspondence proxy: same-word, each reference independently.
 grouped=defaultdict(dict)
 for p in data['stage_pairs']:grouped[(p['query'],p['reference'])][p['method']]=p
 stage_records=[];stage_exclusion=Counter()
 for (qa,ra),methods in grouped.items():
  assert set(methods)=={'geometry','shubert','time_fraction'}
  qids=methods['geometry']['query_sample_ids']
  for k,sid in enumerate(qids):
   for fi in range(10):
    target=consensus[sid][fi]
    if target is None:stage_exclusion['query_visual_unknown']+=1;continue
    if not evidence(sid,fi):stage_exclusion['query_current_evidence_unavailable']+=1;continue
    refs={};outcomes={}
    for method,p in methods.items():
     rid=p['reference_sample_ids'][p['mapping'][k]];refs[method]=rid;pred=consensus[rid][fi]
     outcomes[method]='unknown' if pred is None or not evidence(rid,fi) else ('agree' if pred==target else 'different')
    stage_records.append({'query_sample_id':sid,'query':qa,'reference':ra,'field':fields[fi],'target':target,'group':'motion' if fi>=7 else 'static','annotation_active':meta[sid]['annotation_active'],'reference_ids':refs,'outcomes':outcomes})
 shape_methods=['geometry','dino','shubert'];stage_methods=['geometry','shubert','time_fraction']
 output={'status':'COMPLETED_EXPLORATORY_AI_DRAFT_AGREEMENT_NOT_KSL_ACCURACY',
  'frozen_sha256':sha(U/'FROZEN.json'),'correspondence_sha256':sha(U/'CORRESPONDENCE.json'),'consensus_sha256':sha(U/'CONSENSUS.json'),
  'review_counts':dict(review_counts),'total_review_cells':1080,'hand_query_feature_slots':144,'stage_query_feature_slots':720,
  'shape_exclusions':dict(shape_exclusion),'shape_class_support':{k:dict(v) for k,v in shape_class_support.items()},
  'shape':{'all':summarize(shape_records,shape_methods),'annotation_active':summarize([z for z in shape_records if z['annotation_active']],shape_methods),'by_feature':{field:summarize([z for z in shape_records if z['field']==field],shape_methods) for field in fields[:4]},'records':shape_records},
  'stage_exclusions':dict(stage_exclusion),
  'stage':{'all':summarize(stage_records,stage_methods),'annotation_active':summarize([z for z in stage_records if z['annotation_active']],stage_methods),
     'active_static':summarize([z for z in stage_records if z['annotation_active'] and z['group']=='static'],stage_methods),
     'active_motion':summarize([z for z in stage_records if z['annotation_active'] and z['group']=='motion'],stage_methods),
     'by_query':{alias:summarize([z for z in stage_records if z['query']==alias and z['annotation_active']],stage_methods) for alias in sorted(set(z['query'] for z in stage_records))},'records':stage_records},
  'limitations':['Same-model-family AI drafts, no independent human ground truth','Only four query clips/one recorded query person, not independent hundreds of examples','Nine anchors cannot validate exact phase boundaries/short omissions','Per-hand angle/DINO vs joint contextual SHuBERT have intentionally different information access; not a matched-capacity hand classifier comparison','Current evidence is detector observation, not guaranteed landmark truth','Native test, not masked real webcam test','No training, thresholds or candidate selection tuned from results'],
  'training_runs':0,'operating_changes':0,'private_access':False,'script_sha256':sha(__file__)}
 assert len(shape_records)+sum(shape_exclusion.values())==144
 assert len(stage_records)+sum(stage_exclusion.values())==720
 save('EVALUATION.json',output)
 print(json.dumps({k:output[k] for k in ['status','review_counts','shape_exclusions','stage_exclusions']},ensure_ascii=False))
 print('shape all',json.dumps(output['shape']['all']));print('stage active',json.dumps(output['stage']['annotation_active']))
if __name__=='__main__':main()
