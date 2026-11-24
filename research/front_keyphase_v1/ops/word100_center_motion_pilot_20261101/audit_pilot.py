"""Prespecified raw paired review metrics; no consensus-only pseudo-pass."""
from pathlib import Path
from collections import Counter
import json,hashlib
V=Path(__file__).resolve().parent
STATES={'hold','up','down','horizontal','compound'}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
 with Path(p).open('x',encoding='utf8') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False)
def overlap(a,b):return max(a['first_frame'],b['first_frame'])<=min(a['last_frame'],b['last_frame'])
def hand_match(a,b):return a==b or a=='both' or b=='both'
def main():
 frozen=read(V/'PILOT_FROZEN.json')
 for p,h in frozen['pins'].items():assert sha(p)==h,p
 manifest=read(V/'PILOT_MANIFEST.json')['rows'];by={r['sample_id']:r for r in manifest}
 gates=read(V/'PILOT_GATES.json');annotations=[];sourcepins={};contract_failures=[]
 for name in ('PILOT_A.json','PILOT_B.json'):
  p=V/'review'/name;d=read(p);sourcepins[str(p)]=sha(p)
  assert d['status']=='COMPLETE' and len(d['rows'])==24
  rr={r['sample_id']:r for r in d['rows']};assert set(rr)==set(by) and len(rr)==24
  for sid,r in rr.items():
   src=by[sid];assert r['alias']==src['alias']
   assert r['image_sha256']==sha(src['image'])==src['image_sha256']
   assert r['actual_image_viewed'] is True
   for h in 'RL':
    assert r['state'][h] is None or r['state'][h] in STATES
    assert r['centre_roi'][h] is None or type(r['centre_roi'][h]) is bool
   for e in r['events']:
    assert e['hand'] in ('R','L','both') and e['kind'] in ('onset','offset','reversal','rotation','shape_change','hand_crossing')
    assert e['evidence'] in ('direct','uncertain') and type(e['bounded_transition']) is bool
    assert src['frame_ids'][0]<=e['first_frame']<=e['last_frame']<=src['frame_ids'][-1]
    if e['bounded_transition']:
     assert e['evidence']=='direct' and e['first_frame']>src['frame_ids'][0] and e['last_frame']<src['frame_ids'][-1]
    if e['kind'] in ('onset','offset','reversal') and max(e['first_frame'],src['support_frames'][0])<=min(e['last_frame'],src['support_frames'][-1]):
     for h in ('R','L') if e['hand']=='both' else (e['hand'],):
      if r['state'][h] is not None:contract_failures.append(dict(reviewer=name,sample_id=sid,hand=h,state=r['state'][h],event=e,reason='centre_support_crosses_translation_boundary'))
  annotations.append(rr)
 a,b=annotations;counts=Counter();hands={h:Counter() for h in 'RL'};conflicts=[];rows=[]
 agreed={'hold':set(),'translation':set()};events_count={k:set() for k in ('crossing','boundary','rotation','short_any','short_translation','short_shape')}
 people={k:set() for k in agreed}
 for sid,src in by.items():
  aa,bb=a[sid],b[sid];row={'sample_id':sid,'alias':src['alias'],'role':'fit','hands':{}}
  for h in 'RL':
   av,bv=aa['state'][h],bb['state'][h];ak,bk=av is not None,bv is not None
   both=ak and bk;same=both and av==bv;roi=aa['centre_roi'][h] is True and bb['centre_roi'][h] is True
   m=hands[h];m['all']+=1;m['both_known']+=both;m['known_unknown_mismatch']+=ak!=bk;m['both_known_conflicts']+=both and not same
   m['both_roi']+=roi;m['both_roi_known']+=roi and both;m['both_roi_agreed']+=roi and same
   m['roi_mismatch']+=aa['centre_roi'][h]!=bb['centre_roi'][h]
   if src['strata']==['hash_ordinary']:
    m['ordinary_all']+=1;m['ordinary_both_known']+=both
   if same:
    family='hold' if av=='hold' else 'translation';agreed[family].add(sid);people[family].add(src['person']);m['agreed_'+family]+=1
   if not (av==bv):conflicts.append(dict(sample_id=sid,hand=h,a=av,b=bv,kind='known_category_conflict' if both else 'known_unknown_mismatch'))
   row['hands'][h]=dict(a=av,b=bv,agreed_state=av if same else None,both_roi=roi,
     supervised_state=av if same and roi else None,unknown_is_not_negative=True)
  paired=[]
  for ea in aa['events']:
   for eb in bb['events']:
    if ea['evidence']!='direct' or eb['evidence']!='direct' or ea['kind']!=eb['kind'] or not hand_match(ea['hand'],eb['hand']) or not overlap(ea,eb):continue
    kind=ea['kind'];paired.append(dict(kind=kind,a=ea,b=eb))
    if kind=='hand_crossing':events_count['crossing'].add(sid)
    if kind in ('onset','offset','reversal'):events_count['boundary'].add(sid)
    if kind=='rotation':events_count['rotation'].add(sid)
    short=ea['bounded_transition'] and eb['bounded_transition'] and max(ea['last_frame']-ea['first_frame']+1,eb['last_frame']-eb['first_frame']+1)<=gates['short_transition_max_native_frames']
    if short and kind in ('onset','offset','reversal','shape_change'):
     events_count['short_any'].add(sid);events_count['short_shape' if kind=='shape_change' else 'short_translation'].add(sid)
  row['paired_events']=paired;rows.append(row)
 total={k:sum(hands[h][k] for h in 'RL') for k in set().union(*(set(v) for v in hands.values()))}
 checks={
  'annotation_scope_contract':not contract_failures,
  'raw_motion_category_reliability':total['both_known_conflicts']/max(total['both_known'],1)<=gates['both_known_motion_category_disagreement_max'],
  'known_unknown_mismatch':total['known_unknown_mismatch']/48<=gates['known_unknown_state_mismatch_max_all_hand_windows'],
 }
 for h in 'RL':
  m=hands[h]
  checks[h+'_physical_coverage']=m['both_known']>=gates['both_known_physical_state_min_per_hand']
  checks[h+'_ordinary_coverage']=m['ordinary_both_known']>=gates['both_known_physical_state_min_per_hand_ordinary12']
  checks[h+'_roi_coverage']=m['both_roi']>0 and m['both_roi_known']/m['both_roi']>=gates['both_roi_confirmed_state_known_fraction_min_per_hand']
  checks[h+'_hold_and_movement_present']=m['agreed_hold']>0 and m['agreed_translation']>0
 for kind in ('hold','translation'):
  checks[kind+'_support']=len(agreed[kind])>=gates['agreed_'+kind+'_min_distinct_windows'] and len(people[kind])>=gates['each_hold_translation_min_people']
 for kind,key in [('crossing','agreed_visible_crossing_min_windows'),('boundary','agreed_visible_boundary_min_windows'),('rotation','agreed_visible_rotation_min_windows'),('short_any','agreed_direct_bounded_short_transition_min_windows'),('short_translation','agreed_direct_bounded_short_translation_transition_min_windows'),('short_shape','agreed_direct_bounded_short_shape_transition_min_windows')]:
  checks[kind+'_support']=len(events_count[kind])>=gates[key]
 result=dict(status='PASS_PILOT_PROTOCOL_ONLY_BROAD_LABELS_NEXT' if all(checks.values()) else 'REPAIR_PILOT_BEFORE_BROAD_LABELS_OR_TRAINING',checks=checks,
  total=total,by_hand={h:dict(v) for h,v in hands.items()},agreement_support={k:dict(windows=sorted(v),people=sorted(people[k])) for k,v in agreed.items()},
  event_support={k:sorted(v) for k,v in events_count.items()},conflicts=conflicts,contract_failures=contract_failures,rows=rows,
  raw_motion_category_conflict_rate=total['both_known_conflicts']/max(total['both_known'],1),known_unknown_mismatch_rate=total['known_unknown_mismatch']/48,
  all_hand_windows=48,all_windows=24,source_pins={**frozen['pins'],**sourcepins,str(V/'audit_pilot.py'):sha(__file__)},
  ground_truth='Two separate fresh-context Codex image readings; not independent human/KSL semantic truth',training_started=False,
  original_failed_protocol_unchanged=True,performance_improvement_claim=None)
 write(V/'PILOT_AUDIT.json',result)
 print(json.dumps({k:v for k,v in result.items() if k in ('status','checks','total','by_hand','raw_motion_category_conflict_rate','known_unknown_mismatch_rate')},ensure_ascii=False))
if __name__=='__main__':main()
