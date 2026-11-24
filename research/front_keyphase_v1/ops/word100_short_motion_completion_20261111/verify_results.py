"""Saved-output arithmetic/provenance verification. No training or inference."""
import json,hashlib,collections
from pathlib import Path
U=Path(__file__).resolve().parent;O=U.parent;J=O/'word100_episode_joint_linkage_20261111';V=J/'proposal_adapter'
def r(p):return json.loads(Path(p).read_text('utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
checks=[]
def ck(name,v):
 if not v:raise AssertionError(name)
 checks.append(name)
def correct(v,label,conf=False):
 if v is None:return 0
 return int(v['eligible'] and v['prediction']==label and (not conf or v['confidence']>=.65))
def fusecheck(v):
 before,after,new=v['before'],v['after'],v['raw_new'];use=before['eligible'] and before['prediction']==1 and new['eligible'] and new['prediction'] in [0,2] and new['confidence']>=.90
 ck('fuse_trigger',after['rescue_used']==use);ck('fuse_prediction',after['prediction']==(new['prediction'] if use else before['prediction']));ck('fuse_eligibility',after['eligible']==before['eligible'])
for p,h in {**r(U/'FROZEN.json')['protections'],**r(U/'FROZEN.json')['pins'],**r(U/'CODE_PINS.json'),**r(U/'EVAL_PINS.json')}.items():ck('pinned_'+p,sha(p)==h)
train=r(U/'TRAIN_SAMPLES.json')['rows'];ck('train664',len(train)==664);ck('fit_ids',all(1<=v['person']<=9 for v in train));ck('common899',r(U/'TRAIN_SAMPLES.json')['common100_videos_preserved']==899)
ck('no_eval_source_in_train',not ({v['source_sha256'] for v in train}&{v['source_sha256'] for v in r(U/'EVALUATION_LABELS.json')['rows']}))
ck('train_labels_motion39hold22',collections.Counter(v['kind'] for v in r(U/'FIT_LABEL_AUDIT.json')['rows'])=={'motion':39,'hold':22})
for a in r(U/'EVALUATION_LABELS.json')['rows']:ck('rgb_sheet',sha(a['sheet'])==a['sheet_sha256'])
result=r(U/'RESULT.json');ds=r(U/'DIRECTION_RESULTS.json');hs=r(U/'HOLD_RESULTS.json');fr=r(U/'FULLPATH_COMPLETE.json')['rows'];cs=r(U/'COMPLETION_RESULTS.json')
ck('fullpath468',len(fr)==78*2*3);ck('directions264',len(ds)==(16*2+10*2+9*4)*3);ck('holds162',len(hs)==(9*2+6*2+6*4)*3)
for key,stored in result['direction'].items():
 seed,group,kind=key.split('/');rows=[v for v in ds if v['seed']==int(seed) and v['group']==group and v['kind']==kind]
 for mode,met in stored.items():
  ck('denominator',met['total']==len(rows));views=[v['manual'] if mode=='manual' else v['automatic'][mode] for v in rows]
  ck('matched',met['matched']==sum(mode=='manual' or v['matched'] for v in views))
  for which in ['before','raw_new','after']:
   ck('arithmetic',met[which]==sum(correct(v.get(which),z['truth']['direction']) for z,v in zip(rows,views)))
   ck('confidence_arithmetic',met[which+'_confident']==sum(correct(v.get(which),z['truth']['direction'],True) for z,v in zip(rows,views)))
  for view in views:
   if 'after' in view:fusecheck(view)
for h in hs:fusecheck(h)
for key,stored in result['holds'].items():
 seed,group,kind=key.split('/');rows=[v for v in hs if v['seed']==int(seed) and v['group']==group and v['kind']==kind];ck('hold_denominator',len(rows)==stored['total'])
 for which in ['before','raw_new','after']:ck('hold_count',stored[which]==sum(correct(v[which],1) for v in rows))
 ck('false_rescues',len(stored['false_rescues'])==sum(v['after']['rescue_used'] for v in rows))
for rr in fr:
 ck('feedback_hash',sha(rr['result'])==rr['sha256']);p=r(rr['result']);before=r(p['baseline']);ck('baseline_hash',sha(p['baseline'])==p['baseline_sha256'])
 for key,h in p['protected'].items():ck('field_'+key,hashlib.sha256(json.dumps(before[key],sort_keys=True,ensure_ascii=False).encode()).hexdigest()==h)
 for a,b in zip(p['cards'],before['cards']):ck('shape_preserved',a.get('shape_evidence')==b.get('shape_evidence'));ck('prior_motion_evidence_preserved',a.get('motion_evidence')==b.get('motion_evidence'))
 display=[c.get('display_status',c['status']) for c in p['cards']];ck('compact_display',display==rr['after']['display']);ck('warning_arithmetic',rr['after']['warning']==any(s=='difference' for s in display));ck('similar_arithmetic',rr['after']['similar_no_warning']==all(s=='similar' for s in display))
 if rr['kind'] in ['full_normal','speed075','speed125']:ck('normal_warning_preserved',not rr['after']['warning']);ck('normal_summary_preserved',rr['before']==rr['after'])
 if rr['kind'] in ['all_hand_missing','mask_final_event','omit_final_event']:ck('unknown_not_normal',not rr['after']['similar_no_warning'])
for c in cs:ck('semantic_completion_not_invented',c['guard']['semantic_complete'] is False)
for seed in [42,43,44]:
 for group in ['additional_replay','reserved_id12','reserved_id13']:
  for kind in {v['kind'] for v in ds if v['group']==group}:
   rows=[v for v in ds if v['seed']==seed and v['group']==group and v['kind']==kind]
   for mode in ['baseline','combined']:
    byclip=collections.defaultdict(set)
    for v in rows:
     e=v['automatic'][mode]['event']
     if e:
      key=e['a'],e['b'],e['axis'],e['moving'];ck('one_to_one_event',key not in byclip[v['clip_id']]);byclip[v['clip_id']].add(key)
ck('all_repeats_completed',r(U/'EVALUATION_COMPLETE.json')['seeds']==[42,43,44]);ck('shared_encoder_frozen',r(U/'TRAINING_COMPLETE.json')['common100_encoder_changed'] is False)
summary=dict(status='PASS',checks=len(checks),failed=0,verification='Separate stdlib output verification by same Codex; not independent human labels.',read_only_old_results=True,inference_repeated=False,normal_speed_conditions_per_seed=36,original_movement_cases=35,semantic_completion_proven=False)
(U/'VERIFICATION.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),'utf-8');print(json.dumps(summary,ensure_ascii=False))
