"""Separate saved-output arithmetic audit. Does not load a model or rerun inference."""
from pathlib import Path
import json,hashlib,math,statistics
from collections import Counter
U=Path(__file__).resolve().parent;O=U.parent;P=O/'word100_broad_factorized_learning_20261110'
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf-8')
checks=[]
def ck(name,test):checks.append(dict(name=name,passed=bool(test)))
def total(rr):return dict(n=len(rr),correct=sum(r['pred']==r['target'] for r in rr),eligible=sum(r['eligible'] for r in rr),accepted=sum(r['eligible'] and r['confident'] for r in rr),accepted_correct=sum(r['eligible'] and r['confident'] and r['pred']==r['target'] for r in rr))
def aggregate(ds):
 return dict(n=len(ds),all_required_similar=sum(d['all_required_similar'] for d in ds),any_difference=sum('difference' in d['display'] for d in ds),any_uncertain=sum('uncertain' in d['display'] for d in ds),direction_supported=sum('supported' in d.get('motion_states',[]) for d in ds),direction_contradicted=sum('contradicted' in d.get('motion_states',[]) for d in ds),direction_unknown=sum('unknown' in d.get('motion_states',[]) for d in ds))
cfg=read(U/'FROZEN.json');ef=read(U/'EVAL_FROZEN.json');evaluation=read(U/'EVALUATION_COMPLETE.json');full=read(U/'FULLPATH_COMPLETE.json')['rows'];names=[f'{c}_{s}' for c in cfg['conditions'] for s in cfg['seeds']]
for p,h in {**cfg['pins'],**cfg['code_pins'],**cfg['protections'],**ef['code_pins']}.items():ck('unchanged:'+p,sha(p)==h)
ck('468automatic_conditions',len(full)==468);ck('no_automatic_errors',all(r['status']=='OK' for r in full));summary={}
for name in names:
 train=read(U/'runs'/name/'COMPLETE.json');ck('all899each_epoch:'+name,len(train['seen'])==899 and len(train['history'])==12 and all(e['all100_clips_seen']==899 for e in train['history']));ck('trained_not_only_frozen:'+name,train['encoder_delta']>0);ck('weights:'+name,sha(U/'runs'/name/'model.pt')==train['sha256']);ck('all_motion_classes_trained:'+name,all(train['counts'].get('motion_class_'+str(i),0)>0 for i in range(4)))
 rr=read(U/'scores'/f'{name}.json');words=read(U/'scores'/f'{name}_words.json');ck('300aux:'+name,len(words)==300 and total(words)==evaluation['models'][name]['aux_word100'])
 for variant in ef['variants']:
  for pad in ef['paddings']:
   for task,n in [('movement',32),('order',12)]:
    subset=[r for r in rr if r['variant']==variant and r['padding']==pad and r['task']==task];ck('denominator:'+name+variant+pad+task,len(subset)==n);ck('arithmetic:'+name+variant+pad+task,total(subset)==evaluation['models'][name]['by_variant'][variant][pad][task])
 for r in rr:
  probs=r['probability'] if isinstance(r['probability'],list) else [r['probability']];ck('finite_probability',all(math.isfinite(p) and 0<=p<=1 for p in probs))
  if r['task']=='movement':ck('attention_keeps_window',len(r['attention'])==r['b']-r['a']+1 and abs(sum(r['attention'])-1)<1e-5)
 results=[r for r in full if r['model']==name];ck('78paths:'+name,len(results)==78)
 groups={};order_groups={}
 for kind in sorted(set(r['kind'] for r in results)):
  sel=[r for r in results if r['kind']==kind and r['status']=='OK'];groups[kind]={key:aggregate([r[key] for r in sel if key in r]) for key in ['baseline','previous','candidate_raw','before_order','candidate']}
  os=[]
  for r in sel:
   ck('resultsha',sha(r['result'])==r['sha256']);d=read(r['result']);os.extend(d['learned_order_checks']);ck('no_promoted_completion',not r['candidate']['all_required_similar'] or r['candidate_raw']['all_required_similar'])
   if 'tcn_identical' in r:ck('fixedTCN_DTW_frames',r['tcn_identical'] and r['dtw_identical'] and r['frames_preserved'])
   for card in d['cards']:
    ck('not_expert_error_claim',not card.get('educational_wrong',False));m=card.get('motion_evidence')
    if m and m.get('learned_class')==3:ck('mixed_not_supported',m['state']=='unknown')
  order_groups[kind]=dict(n=len(os),states=dict(Counter(o['state'] for o in os)),high_conf_wrong_order=sum(o['state']=='contradicted' for o in os))
 main=[r for r in results if r['kind']=='full_normal'];normal_safe=all(r['candidate']['display']==r['previous']['display'] for r in main)
 speed=[r for r in results if r['kind'] in ['speed075','speed125']];speed_safe=all(r['candidate']['display']==r['previous']['display'] for r in speed)
 missing=[r for r in results if r['kind'] in ['all_hand_missing','mask_final_event','omit_final_event']];missing_safe=all(not r['candidate']['all_required_similar'] for r in missing)
 controls=[r for r in results if r['kind'] in ['right_closed_hand_main','main_reversed','closed_hand_and_reversed']];controls_safe=all('difference' in r['candidate']['display'] for r in controls)
 motion_improved=any('contradicted' in r['candidate']['motion_states'] and 'contradicted' not in r['previous']['motion_states'] for r in controls if r['kind'] in ['main_reversed','closed_hand_and_reversed'])
 shape_only=[r for r in controls if r['kind']=='right_closed_hand_main'];shape_specific=all('contradicted' not in r['candidate']['motion_states'] for r in shape_only) and order_groups['right_closed_hand_main']['states'].get('contradicted',0)==0
 order_reversal_safe=order_groups['reversed_core_auto_order']['states'].get('supported',0)==0
 times=sorted(r['seconds'] for r in results[1:]);summary[name]=dict(component=evaluation['models'][name],automatic=groups,automatic_order=order_groups,training_seconds=train['seconds'],cached_full_ms_median=statistics.median(times)*1000,cached_full_ms_p95=times[math.ceil(.95*len(times))-1]*1000,gates=dict(normal_unchanged=normal_safe,speed_unchanged=speed_safe,missing_not_complete=missing_safe,all_existing_error_alerts_kept=controls_safe,new_direction_detection=motion_improved,shape_only_not_false_direction_or_order=shape_specific,automatic_reversal_not_supported=order_reversal_safe),deployment_approved=False)
write(U/'VERIFICATION.json',dict(passed=all(r['passed'] for r in checks),count=len(checks),failed=[r for r in checks if not r['passed']],checks=checks,reviewer='Same Codex, separate saved-output arithmetic/code checks; not independent human ground truth. No model inference.'))
write(U/'SUMMARY.json',dict(models=summary,primary='learned_core_42',comparison='uniform_42',old_shape='Pcoupled42 frozen; previous227/228 preserved checkpoint, not newly expanded annotation coverage.',historical_development=True,default_web_changed=False))
print(json.dumps(dict(passed=all(r['passed'] for r in checks),checks=len(checks),failed=[r for r in checks if not r['passed']][:8],gates={k:v['gates'] for k,v in summary.items()})))
