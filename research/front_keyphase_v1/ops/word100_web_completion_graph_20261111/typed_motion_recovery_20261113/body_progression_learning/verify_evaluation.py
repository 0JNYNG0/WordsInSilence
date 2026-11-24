"""Compact saved full/local/static audit. No model inference or result reruns."""
from pathlib import Path
from collections import Counter,defaultdict
import json,hashlib,datetime,itertools,importlib.util
import numpy as np
B=Path(__file__).resolve().parent;H=B.parent;G=H.parent;P=H/'progression_presence_learning';S=H/'paired_static_progression_learning';N=H/'null_candidate_learning'
spec=importlib.util.spec_from_file_location('_body_eval_arithmetic',P/'verify.py');u=importlib.util.module_from_spec(spec);spec.loader.exec_module(u)
rd,sha,ck,pins,key,orders=u.rd,u.sha,u.ck,u.pins,u.key,u.orders
F=[2,3,4,5,7];AX={2:0,3:1,4:2,5:4,7:3};ANCHORS=['start','middle','end']
def aggregate(rows,keys,metrics,countname='rows'):
 groups=defaultdict(list)
 for r in rows:groups[tuple(r[k] for k in keys)].append(r)
 return [dict(**dict(zip(keys,k)),**{countname:len(rr)},**{name:sum(int(fn(r)) for r in rr) for name,fn in metrics.items()}) for k,rr in sorted(groups.items())]
def grouped(rows,keys):
 d=defaultdict(list)
 for r in rows:d[tuple(r[k] for k in keys)].append(r)
 return d
def bucket(n):return 0 if n<=6 else 1 if n<=10 else 2 if n<=15 else 3 if n<=21 else 4
def main():
 assert not(B/'EVALUATION_REVIEW.json').exists()
 priorreview=rd(B/'INPUT_TRAINING_REVIEW.json');ck('prior input/training review PASS',priorreview['status']=='PASS')
 tc=rd(B/'TRAINING_COMPLETE.json');meta=rd(B/'INPUTS_COMPLETE.json');idx=rd(N/'TRAINING_INPUT_INDEX.json')['rows'];masks=rd(N/'local_bag_evaluation/RULES_FROZEN.json')['partial_masks'];matched=rd(B/'FIT_LABEL_AUDIT.json')['matched_buckets']
 loc=rd(B/'local_evaluation/COMPARISON.json');stat=rd(B/'static_evaluation/COMPARISON.json');full=rd(B/'evaluation/COMPARISON.json')
 for folder,data in [('local_evaluation',loc),('static_evaluation',stat)]:
  rules=rd(B/folder/'RULES_FROZEN.json');ck(folder+' rule hash',sha(B/folder/'RULES_FROZEN.json')==data['rules_sha256']);pins(rules['pins'],folder+' contract');pins(data['prediction_pins'],folder+' predictions')
 ef=rd(B/'evaluation/FROZEN.json');pins(ef['pins'],'full contract')
 oldloc=rd(N/'local_bag_evaluation/COMPARISON.json');oldbags={(r['seed'],r['bag_index']):r for r in oldloc['rows']};oldpartial={(r['seed'],r['bag_index'],r['candidate_index']):r for r in oldloc['partial_rows']}
 cg=grouped(loc['candidate_rows'],['seed','bag_index']);bg={(r['seed'],r['bag_index']):r for r in loc['bag_rows']};ug={(r['seed'],r['bag_index']):r for r in loc['unknown_rows']};pg=grouped(loc['partial_rows'],['seed','bag_index']);sg=grouped(stat['rows'],['seed','bag_index'])
 staticmeta={r['bag_index']:r for r in meta['static_bags']};labeldata=[]
 ck('local858 bags',len(bg)==len(loc['bag_rows'])==858 and set(bg)==set(oldbags)==set(ug))
 for i,r in enumerate(idx):
  z=np.load(meta['body_bags'][i]['path']);fullmask=z['y'];partial=np.zeros(len(fullmask),bool)
  for c in masks[i]['candidates']:partial[c['candidate_index']]=True
  hold=r['origin']=='reviewed_local_hold';known=np.ones(len(fullmask),bool) if hold else fullmask|partial;y=np.zeros(len(fullmask),bool) if hold else fullmask|partial
  labeldata.append(dict(full=fullmask,partial=partial,known=known,y=y,spans=z['spans']))
 oldstatic={(r['seed'],r['bag_index'],r['candidate_index'],r['anchor']):r for r in rd(P/'static_contract_audit/RESULTS.json')['rows']}
 matchedstatic={(r['seed'],r['bag_index'],r['candidate_index'],r['anchor']):r for r in stat['matched_prior_rows']}
 for run in tc['rows']:
  seed=run['seed'];pred=np.load(run['predictions']);spred=np.load(run['static_predictions'])
  for i,r in enumerate(idx):
   d=labeldata[i];score=pred[f'logits_{i}'][:,F.index(r['family'])];positive=score>=0;rr=cg.get((seed,i),[]);expectedix=np.flatnonzero(d['known'])
   ck('all known candidate rows:'+str((seed,i)),[c['candidate_index'] for c in rr]==expectedix.tolist())
   valid=True
   for c,k in zip(rr,expectedix):
    a,b=map(int,d['spans'][k]);kind='full' if d['full'][k] else 'partial' if d['partial'][k] else 'hold'
    valid&=all(c[name]==r[name] for name in ['clip_id','person','split','family','view','reverse']) and c['label_kind']==kind and c['expected_progression']==bool(d['y'][k]) and c['predicted_progression']==bool(positive[k]) and c['score']==float(score[k]) and c['correct']==(bool(d['y'][k])==bool(positive[k])) and c['span']==[a,b] and c['frames']==b-a+1 and c['duration_bucket']==bucket(b-a+1) and c['duration_bucket_in_fit_matched']==(bucket(b-a+1) in matched[str(r['family'])]) and c['semantic_completion'] is None
   ck('known targets/scores/denominators:'+str((seed,i)),valid)
   row=bg[seed,i];previous=oldbags[seed,i];ch=previous['baseline'];ci=ch['index'];support=None if ci is None else bool(positive[ci]);unknown=~d['known'];knownpos=d['known']&d['y'];anyraw=bool(positive.any());anyknown=bool((positive&knownpos).any())
   expected=dict(origin=r['origin'],candidate_count=len(score),known_candidates=int(d['known'].sum()),unknown_candidates=int(unknown.sum()),any_progression_raw=anyraw,any_known_positive_supported=anyknown,any_unknown_progression_prediction=bool((positive&unknown).any()),only_unknown_progression=bool(anyraw and not(positive&d['known']).any()),prior_body_gate_admitted_count=previous['body_gate_admitted'],old_chosen_index=ci,old_chosen_span=ch['span'],old_chosen_known_full=ch['known_positive'],old_chosen_supported=support,old_extent_unchanged=True,known_old_full_extent_supported=bool(ch['known_positive'] and support),known_old_full_extent_unsupported=bool(ch['known_positive'] and support is False),old_hold_false_selection=previous['baseline_false_hold_selection'],previous_null_hold_false_selection=previous['new_false_hold_selection'],hold_false_positive_any_raw=bool(r['origin']=='reviewed_local_hold' and anyraw),hold_false_positive_old_chosen=bool(r['origin']=='reviewed_local_hold' and support),semantic_completion=None)
   ck('old choice and bag arithmetic:'+str((seed,i)),all(row[k]==v for k,v in expected.items()))
   unknownrow=ug[seed,i];ck('unknown excluded:'+str((seed,i)),unknownrow['unknown_candidates']==int(unknown.sum()) and unknownrow['predicted_progression_on_unknown']==int((positive&unknown).sum()) and unknownrow['excluded_from_performance_denominator'] is True)
   pr=pg.get((seed,i),[]);ck('same partial candidate denominator:'+str((seed,i)),[q['candidate_index'] for q in pr]==[q['candidate_index'] for q in masks[i]['candidates']])
   valid=True
   for c in pr:
    k=c['candidate_index'];old=oldpartial[seed,i,k];new=bool(positive[k]);ad=old['old_body_gate_admitted']
    valid&=c['predicted_progression']==new and c['score']==float(score[k]) and c['previous_body_gate_admitted']==ad and c['previous_null_state']==old['state'] and c['previous_above_null']==old['above_null_ungated'] and c['after_same_partial_body_gate']==bool(ad and new) and c['old_null_suppressed']==(old['state']=='learned_null_suppressed') and c['recovered_from_null_suppression']==bool(old['state']=='learned_null_suppressed' and new) and c['lost_previous_above_null']==bool(old['state']=='candidate_above_null' and not new) and c['semantic_completion'] is None
   ck('partial same gate before-after:'+str((seed,i)),valid)
   if i not in staticmeta:continue
   z=np.load(staticmeta[i]['path']);values=spred[f'logits_{i}'][:,F.index(r['family'])];sr=sg[seed,i];ck('all static rows:'+str((seed,i)),len(sr)==len(values)==len(z['mapping']))
   valid=True
   for j,(c,(k,anchor,frame)) in enumerate(zip(sr,z['mapping'])):
    a,b=map(int,d['spans'][k]);eligible=bool(z['eligible'][j]);v=float(values[j]);predicted=v>=0;kind='full' if d['full'][k] else 'partial'
    valid&=all(c[name]==r[name] for name in ['clip_id','person','split','family','view','reverse','source_sha256']) and c['static_index']==j and c['candidate_index']==int(k) and c['anchor']==ANCHORS[anchor] and c['anchor_frame_transformed']==int(frame) and c['source_candidate_kind']==kind and c['transformed_span']==[a,b] and c['frames']==b-a+1 and c['duration_bucket']==bucket(b-a+1) and c['observation_eligible']==eligible and c['observation_mask']==float(eligible) and c['expected_observed_progression']==(False if eligible else None) and c['score']==v and c['predicted_progression']==predicted and c['semantic_completion'] is None
    kk=(seed,i,int(k),ANCHORS[anchor])
    if kk in oldstatic:
     old=oldstatic[kk];ma=matchedstatic[kk];valid&=ma['previous_score']==old['static_logit'] and ma['previous_false_progression']==old['static_false_progression'] and ma['score']==v and ma['new_false_progression']==predicted and ma['in_new_training_augmentation_pool'] is True
   ck('static provenance visibility score:'+str((seed,i)),valid)
 ck('all matched216',len(matchedstatic)==len(stat['matched_prior_rows'])==216 and set(matchedstatic)==set(oldstatic))
 cm=dict(positive_labels=lambda r:r['expected_progression'],negative_labels=lambda r:not r['expected_progression'],true_positive=lambda r:r['expected_progression'] and r['predicted_progression'],false_negative=lambda r:r['expected_progression'] and not r['predicted_progression'],false_positive=lambda r:not r['expected_progression'] and r['predicted_progression'],true_negative=lambda r:not r['expected_progression'] and not r['predicted_progression'])
 for name,keys in [('candidate_summary',['seed','split','family','label_kind','duration_bucket']),('candidate_all_durations',['seed','split','family','label_kind']),('candidate_all_families',['seed','split','label_kind']),('matched_duration_scope',['seed','split','family','duration_bucket_in_fit_matched','label_kind'])]:ck('local aggregate '+name,loc[name]==aggregate(loc['candidate_rows'],keys,cm))
 bm={name:(lambda r,k=name:r[k]) for name in ['any_progression_raw','any_known_positive_supported','only_unknown_progression','old_chosen_known_full','known_old_full_extent_supported','known_old_full_extent_unsupported','old_hold_false_selection','previous_null_hold_false_selection','hold_false_positive_any_raw','hold_false_positive_old_chosen']};bm.update(old_chosen_present=lambda r:r['old_chosen_index'] is not None,old_chosen_supported=lambda r:r['old_chosen_supported'] is True)
 for name,keys in [('bag_summary',['seed','split','origin','family']),('bag_all_families',['seed','split','origin'])]:ck('bag aggregate '+name,loc[name]==aggregate(loc['bag_rows'],keys,bm))
 pm=dict(body_gate_admitted=lambda r:r['previous_body_gate_admitted'],old_null_suppressed=lambda r:r['old_null_suppressed'],new_progression_positive=lambda r:r['predicted_progression'],new_positive_after_saved_body_gate=lambda r:r['after_same_partial_body_gate'],recovered_from_null_suppression=lambda r:r['recovered_from_null_suppression'],lost_previous_above_null=lambda r:r['lost_previous_above_null'])
 for name,keys in [('partial_comparison',['seed','split','family','duration_bucket']),('partial_all_families',['seed','split'])]:ck('partial aggregate '+name,loc[name]==aggregate(loc['partial_rows'],keys,pm))
 sm=dict(observed_eligible=lambda r:r['observation_eligible'],excluded_unobserved=lambda r:not r['observation_eligible'],false_progression_observed=lambda r:r['observation_eligible'] and r['predicted_progression'],rejected_static_observed=lambda r:r['observation_eligible'] and not r['predicted_progression'],predicted_progression_on_unobserved=lambda r:not r['observation_eligible'] and r['predicted_progression'])
 for name,keys in [('summary',['seed','split','family','duration_bucket','anchor']),('by_family',['seed','split','family']),('by_duration',['seed','split','duration_bucket']),('by_anchor',['seed','split','anchor']),('overall',['seed','split'])]:ck('static aggregate '+name,stat[name]==aggregate(stat['rows'],keys,sm,'generated'))
 oldfull=rd(N/'evaluation/COMPARISON.json');before={key(r):r for r in oldfull['rows']};op={key(r):r for r in oldfull['predictions']};pred={key(r):r for r in full['predictions']};new={key(r):r for r in full['rows']};runtime=rd(B/'RUNTIME_FEATURES_COMPLETE.json');rt={(r['clip_id'],r['kind']):r for r in runtime['rows']}
 ck('246fullrows82runtime',len(new)==len(full['rows'])==len(pred)==246 and set(new)==set(before) and len(rt)==82)
 pins(rd(B/'RUNTIME_FEATURES_FROZEN.json')['pins'],'runtime contract');bodyids=np.array([part*290+64+k*10+j for part in range(4) for k in range(10) for j in [0,1,2,3,8]])
 for k,r in rt.items():
  pins({r['path']:r['sha256'],r['original_feature']:r['original_feature_sha256']},'runtime features');z=np.load(r['path']);o=np.load(r['original_feature']);x=z['x'];parts=x.reshape(-1,4,290);q=parts[:,:,64:164].reshape(-1,4,10,10)
  ck('runtime spans and explicitbody unchanged:'+str(k),np.array_equal(z['spans'],o['spans']) and np.array_equal(x[:,bodyids],o['x'][:,bodyids]))
  ck('runtime no explicit hand:'+str(k),np.isfinite(x).all() and not np.count_nonzero(q[:,:,:,4:8]) and not np.count_nonzero(q[:,:,:,9]) and not np.count_nonzero(parts[:,:,164:]))
 diags=[];f2=0;sources={}
 for kk,row in new.items():
  pr=pred[kk];oldpr=op[kk];r=rt[kk[1:]];pins({pr['path']:pr['sha256']},'full predictions')
  ck('full source reused bodyfeature swapped only:'+str(kk),pr['source']==oldpr['source']==r['source'] and pr['source_sha256']==oldpr['source_sha256']==r['source_sha256'] and pr['feature_path']==r['path'] and pr['feature_sha256']==r['sha256'])
  if pr['source'] not in sources:pins({pr['source']:pr['source_sha256']},'public runtime source');sources[pr['source']]=np.load(pr['source'])['ts']
  ts=sources[pr['source']];ar=np.load(pr['path']);oa=np.load(oldpr['path']);cc=row['conditions'];old=before[kk]['conditions'];ck('three full conditions:'+str(kk),set(cc)==set(ef['conditions']))
  for mode in ['baseline','local_full']:ck('oldcard/raworder/positiveoverlap:'+str((kk,mode)),cc[mode]['cards']==old[mode]['cards'] and cc[mode]['original_order']==old[mode]['order'] and cc[mode]['order']==orders(cc[mode]['cards'],ts))
  cur=cc['progression_separate'];expected=[]
  for i,(c,oc) in enumerate(zip(cur['cards'],old['local_full']['cards'])):
   if c['family']==2:ck('family2 exact:'+str((kk,i)),c==oc);f2+=1;continue
   spans=ar[f'spans_{i}'];gate=ar[f'presence_{i}'];lo=ar[f'progression_logits_{i}'];eligible=gate>=0;yes=eligible&(lo>=0);ci=None
   ck('candidate spans/gate unchanged:'+str((kk,i)),np.array_equal(spans,oa[f'spans_{i}']) and np.array_equal(gate,oa[f'presence_{i}']))
   if oc['chosen']:
    match=np.flatnonzero((spans==oc['chosen']['span']).all(1));assert len(match)==1;ci=int(match[0]);ck('oldselected eligibility:'+str((kk,i)),eligible[ci])
   support=bool(ci is not None and yes[ci]);state='no_body_gate_admission' if not eligible.any() else 'no_progression_support' if not yes.any() else 'original_extent_unavailable' if ci is None else 'selected_extent_supported' if support else 'progression_observed_extent_uncertain'
   ck('presence state/rawextent preserved:'+str((kk,i)),c['progression_state']==state and c['progression_observed']==bool(yes.any()) and c['original_extent']==oc['chosen'] and c['chosen']==(oc['chosen'] if support else None) and c['semantic_completion'] is None)
   ignored={'truth','chosen','original_extent','progression_state','progression_observed'};ck('allother card fields unchanged:'+str((kk,i)),{k:v for k,v in c.items() if k not in ignored}=={k:v for k,v in oc.items() if k not in ignored})
   if 'truth' in c:
    t=c['truth'];ot=oc['truth'];seen=set(c['chosen']['body_observed']) if c['chosen'] else set();ck('truth invariant and arithmetic:'+str((kk,i)),{k:v for k,v in t.items() if k not in ['all_witness','wrong_points']}=={k:v for k,v in ot.items() if k not in ['all_witness','wrong_points']} and t['all_witness']==(set(t['witness'])<=seen) and t['wrong_points']==len(set(t['wrong_witness'])&seen))
   d=dict(card=c['card'],family=c['family'],state=state,eligible_candidates=int(eligible.sum()),progression_candidates=int(yes.sum()),original_index=ci,original_extent_supported=support,original_logit=float(lo[ci]) if ci is not None else None);expected.append(d);diags.append(dict(seed=kk[0],clip_id=kk[1],kind=kk[2],**d))
  ck('full diagnostic/order arithmetic:'+str(kk),cur['diagnostics']==expected and cur['order']==orders(cur['cards'],ts) and cur['original_order']==cc['local_full']['order'])
 ck('family2 preserved72',f2==72)
 summary=[]
 for seed,mode,kind in itertools.product([42,43,44],ef['conditions'],sorted({r['kind'] for r in full['rows']})):
  rr=[r for r in full['rows'] if r['seed']==seed and r['kind']==kind];cards=[c for r in rr for c in r['conditions'][mode]['cards']];truth=[c['truth'] for c in cards if 'truth'in c]
  summary.append(dict(seed=seed,condition=mode,kind=kind,cases=len(rr),cards=len(cards),unsupported_extent=sum(c['chosen'] is None for c in cards),known_groups=len(truth),all_witness=sum(t['all_witness'] for t in truth),wrong_points=sum(t['wrong_points'] for t in truth),order=dict(Counter(o['state'] for r in rr for o in r['conditions'][mode]['order'])),progression_states=dict(Counter(c['progression_state'] for c in cards if 'progression_state' in c))))
 ck('all full denominators',summary==full['summary']);tradeoff=[];contracts=[]
 for name,path in [('P',P),('S',S),('B',B)]:
  l=loc if name=='B' else rd(path/'local_evaluation/COMPARISON.json');f=full if name=='B' else rd(path/'evaluation/COMPARISON.json')
  for r in l['candidate_all_families']:tradeoff.append(dict(model=name,**r))
  for kind in ['full_normal','speed075','speed125','right_closed_hand_main','closed_hand_and_reversed','all_hand_missing','omit_final_event','dynamic_complete_blocks']:
   rr=[c for row in f['rows'] if row['kind']==kind for c in row['conditions']['progression_separate']['cards'] if c['family']!=2]
   contracts.append(dict(model=name,kind=kind,cards=len(rr),states=dict(Counter(c['progression_state'] for c in rr))))
 protections=rd(G/'PROTECTIONS.json');pins(protections,'protected');ck('protected477',len(protections)==477)
 output=dict(at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(),status='PASS' if all(c['ok'] for c in u.checks) else 'FAIL',checks=len(u.checks),failed=[c for c in u.checks if not c['ok']],local_bag_rows=len(bg),local_candidate_rows=len(loc['candidate_rows']),local_partial_rows=len(loc['partial_rows']),static_rows=len(stat['rows']),runtime_scenarios=len(rt),full_rows=len(new),family2_preserved=f2,protected_files=len(protections),local_tradeoffs=tradeoff,full_contracts=contracts,static_overall=stat['overall'],matched_static_training_diagnostic=stat['matched_prior_summary'],full_summary=summary,diagnostics=diags,no_model_forward=True,no_training_or_evaluation_rerun=True,full_G2_pass=False,script_sha256=sha(__file__),reviewer='Codex motion_contract_review_1113 saved arithmetic independent from root execution; B input/training author is same agent, root performed separate code review. Not human semantic truth.')
 (B/'EVALUATION_REVIEW.json').write_text(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf-8')
 print(json.dumps({k:output[k] for k in ['status','checks','failed','local_bag_rows','local_candidate_rows','local_partial_rows','static_rows','runtime_scenarios','full_rows','family2_preserved','protected_files']},ensure_ascii=False))
 for r in contracts:
  if r['kind'] in ['full_normal','speed075','speed125','right_closed_hand_main','closed_hand_and_reversed']:print(json.dumps(r))
 if output['failed']:raise SystemExit(1)
if __name__=='__main__':main()
