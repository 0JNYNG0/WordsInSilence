"""Independent synthetic tests + read-only raw structure checks; no 162-case policy execution."""
from pathlib import Path
import copy, hashlib, importlib.util, json, math, sys
from collections import Counter

U=Path(__file__).resolve().parents[1]; O=U.parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
spec=importlib.util.spec_from_file_location('reviewed_flow_guard',U/'flow_guard.py')
fg=importlib.util.module_from_spec(spec);spec.loader.exec_module(fg)
checks=0;failures=[];categories=Counter();source_hashes={}
def check(ok,name,detail=None):
    global checks
    checks+=1;categories[name.split('/')[0]]+=1
    if not ok:failures.append(dict(test=name,detail=detail))
def reject(fn,name):
    try:fn()
    except (AssertionError,KeyError,ValueError,TypeError):check(True,name);return
    check(False,name,'Malformed input was accepted')
def track(p):
    p=Path(p);source_hashes[str(p)]=sha(p);return read(p)

GROUPS=['L_shape','R_shape','L_location','R_location','L_axis','R_axis','L_arm','R_arm','hands_relation','L_to_R_forearm','R_to_L_forearm','L_direction','R_direction']
def anchor(sid='a',core=(2,4),span=(2,4),state='supported_core_candidate',supported=True):
    groups={g:dict(expected=6,sufficient=supported) for g in GROUPS}
    return dict(segment_id=sid,reference_core=list(core),state=state,
        selected=None if span is None else dict(query_core=list(span),groups=groups))

def paths(n,m,q=0,r=0,p=None):
    p=[] if p is None else p;p=p+[[q,r]]
    if q==n-1 and r==m-1:yield p;return
    for dq,dr in [(1,0),(0,1),(1,1)]:
        if q+dq<n and r+dr<m:yield from paths(n,m,q+dq,r+dr,p)

# Exhaustive small monotonic paths, independently enumerate every reference link.
path_count=0
for n in range(1,6):
 for m in range(1,6):
  for p in paths(n,m):
   path_count+=1;before=copy.deepcopy(p)
   anchors=[anchor(f's{lo}_{hi}',(lo,hi),(0,n-1)) for lo in range(m) for hi in range(lo,m)]
   abefore=copy.deepcopy(anchors);out=fg.guard(p,n,m,anchors)
   check(p==before and anchors==abefore,'synthetic/input_unchanged')
   check(out['path']==before,'synthetic/full_path_exact')
   check(out['query_scope_retained']==n and out['reference_scope_retained']==m,'synthetic/scopes_retained')
   check(len(out['mapping'])==m and len(out['anchors'])==len(anchors),'synthetic/denominators')
   for r,x in enumerate(out['mapping']):
    qs=[q for q,rr in before if rr==r]
    check(x['reference_frame']==r and x['query_frames']==qs,'synthetic/all_links_per_reference')
    check(x['query_range']==[min(qs),max(qs)] and x['display_representative']==qs[(len(qs)-1)//2] and x['representative_is_display_only'] is True,'synthetic/display_only_representative')
   for a in out['anchors']:
    lo,hi=a['reference_core'];links=[p for p in before if lo<=p[1]<=hi]
    check(a['selected_path']==links and a['core_reference_opportunities']==hi-lo+1,'synthetic/core_all_links_and_denominator')
    check(a['global_query_range']==[min(q for q,r in links),max(q for q,r in links)],'synthetic/core_full_range')
    check(all(a[k] is None for k in ['sign_correct','sign_error','correction_instruction']),'synthetic/no_sign_verdict')

p=[[i,i] for i in range(10)]
for span,relation in [((2,4),'contained'),((1,3),'overlap_extends'),((1,5),'overlap_extends'),((6,8),'disjoint'),(None,'local_unavailable')]:
 a=fg.guard(p,10,10,[anchor(span=span)])['anchors'][0]
 check(a['range_relation']==relation,'ranges/'+relation)
 check(a['status']==('window_compatible_not_exact_phase_proof' if relation=='contained' else 'local_corroboration_unresolved'),'ranges/status_'+relation)
for mode in ['all_zero_expected','one_expected_unsupported','ambiguous','missing_local']:
 a=anchor()
 if mode=='all_zero_expected':
  for g in a['selected']['groups'].values():g['expected']=0
 elif mode=='one_expected_unsupported':a['selected']['groups']['R_shape']['sufficient']=False
 elif mode=='ambiguous':a['state']='ambiguous_occurrence'
 else:a['selected']=None
 b=fg.guard(p,10,10,[a])['anchors'][0]
 check(b['status']=='local_corroboration_unresolved','support/'+mode)
a=anchor();a['selected']['groups']['L_shape']=dict(expected=0,sufficient=False)
check(fg.guard(p,10,10,[a])['anchors'][0]['local_observation_supported'] is True,'support/reference_unobserved_excluded')
for spans,expected in [(((1,2),(5,6)),'forward_separated'),(((5,6),(1,2)),'reverse_separated'),(((1,5),(4,6)),'overlap')]:
 out=fg.guard(p,10,10,[anchor('a',(1,2),spans[0]),anchor('b',(5,6),spans[1])])
 check(out['local_relationships'][0]['state']==expected,'order/'+expected)
 if expected!='forward_separated':check(all('local_order_unresolved' in x['flags'] for x in out['anchors']),'order/unresolved_both')

# Labels/aliases cannot change numeric evidence or decisions; IDs are opaque identifiers.
a=anchor();original=fg.guard(p,10,10,[a]);mut=copy.deepcopy(a);mut['segment_id']='other_word_other_person';mut['word_id']='WORD_FAKE';mut['physical_label']='fake_error';mut['score']=999
changed=fg.guard(p,10,10,[mut]);changed['anchors'][0]['segment_id']='a'
check(changed==original,'invariance/alias_label_metadata')

# Invalid topology, ranges, non-integer coordinates and duplicate IDs must be rejected.
for bad in [[],[[0,0],[2,2]],[[0,0],[1,1],[1,1]],[[0,0],[1,1],[0,2]],[[0,0],[1,1],[2,3]],[[0,0],[True,1],[2,2]],[[0,0],[1.0,1],[2,2]]]:
 reject(lambda bad=bad:fg.guard(bad,3,3,[]),'validation/invalid_path')
reject(lambda:fg.guard(p,10,10,[anchor(),anchor()]),'validation/duplicate_stage')
reject(lambda:fg.guard(p,10,10,[anchor(core=(4,2))]),'validation/reversed_core')
reject(lambda:fg.guard(p,10,10,[anchor(span=(0,10))]),'validation/out_of_range_local')
for state in ['observation_insufficient','no_candidate','unrecognized_state',None]:
 a=anchor(state=state)
 try:
  b=fg.guard(p,10,10,[a])['anchors'][0]
  check(b['status']=='local_corroboration_unresolved','validation/state_fail_closed_'+str(state))
 except (AssertionError,KeyError,ValueError,TypeError):check(True,'validation/state_fail_closed_'+str(state))
for value in ['false',1,None]:
 a=anchor();a['selected']['groups']['R_shape']['sufficient']=value
 reject(lambda a=a:fg.guard(p,10,10,[a]),'validation/sufficient_strict_bool_'+str(value))
for value in [-1,True,1.5,float('nan')]:
 a=anchor();a['selected']['groups']['R_shape']['expected']=value
 reject(lambda a=a:fg.guard(p,10,10,[a]),'validation/expected_nonnegative_integer_'+str(value))

# Raw input inspection only: do NOT call guard, search, DTW, model or metric on real cases.
split=O/'word100_core_context_split_20261101';rel=O/'word100_relational_sequence_compare_20261031'
manifest=track(split/'RESULT_MANIFEST.json');counts=Counter();stagecounts=Counter();states=Counter();origin_pairs={}
emp=track(O/'word100_semantic_episode_pilot_20261031/EMPIRICAL_RANGE.json')
for w in emp['calibration']:
 for q in w['lopo']:
  for pp in q['pairs']:origin_pairs[(q['query'],pp['reference'])]=pp['path_original_frames']
for q in emp['development']:
 for pp in q['pairs']:origin_pairs[(q['alias'],pp['reference'])]=pp['path_original_frames']
expected13=set(GROUPS)
for row in manifest['results']:
 counts[row['kind']]+=1;local=track(row['file']);check(sha(row['file'])==row['sha256'],'raw/hash_manifest')
 base=track(rel/('pairs' if row['kind']=='person_pair' else 'controls')/(row['name']+'.json'))
 n,m=base['query_frames'],base['reference_frames'];path=base['path']
 check(path[0]==[0,0] and path[-1]==[n-1,m-1],'raw/full_endpoints')
 check({q for q,r in path}==set(range(n)) and {r for q,r in path}==set(range(m)),'raw/full_scopes')
 check(all((b[0]-a[0],b[1]-a[1]) in [(1,0),(0,1),(1,1)] for a,b in zip(path,path[1:])),'raw/monotonicity')
 if row['kind']=='person_pair':check(path==origin_pairs[(local['query'],local['reference'])],'raw/semantic_episode_origin_exact')
 stages={s['segment_id']:s for s in base['stages']};check(len(stages)==len(base['stages']),'raw/unique_global_stage_ids')
 for a in local['anchors']:
  stagecounts[row['kind']]+=1;states[a['state']]+=1
  check(a['segment_id'] in stages,'raw/stage_join_exists')
  g=stages[a['segment_id']];check(g['reference_core']==a['reference_core'],'raw/reference_core_exact')
  check(set(g['groups'])==expected13,'raw/global_13_groups')
  if a['selected']:
   check(set(a['selected']['groups'])==expected13,'raw/local_13_groups')
   for v in a['selected']['groups'].values():
    check(type(v['expected']) is int and v['expected']>=0 and type(v['sufficient']) is bool,'raw/local_group_schema')
   vals=[v for v in a['selected']['groups'].values() if v['expected']>0]
   supports=bool(vals) and all(v['sufficient'] for v in vals)
   check(supports==(a['state'] in ['supported_core_candidate','ambiguous_occurrence']),'raw/state_support_consistency')
check(dict(counts)==dict(control=126,person_pair=36),'raw/162_case_denominator',dict(counts))
check(stagecounts['person_pair']==117,'raw/117_person_stage_denominator',dict(stagecounts))
for path,h in source_hashes.items():check(sha(path)==h,'preservation/raw_file_sha_unchanged')

review=dict(status='PASS' if not failures else 'FAIL_REVIEW_COUNTEREXAMPLES',reviewer='same Codex B; independent code/synthetic arithmetic review, not human KSL validation',
 checks=checks,failures=failures,check_categories=dict(categories),synthetic_monotonic_paths=path_count,
 real_raw_structure=dict(cases=dict(counts),stage_opportunities=dict(stagecounts),states=dict(states),policy_executed_on_real_cases=False),
 reviewed_file_hashes={str(U/n):sha(U/n) for n in ['flow_guard.py','run.py','POLICY.json','PLAN_KO.txt']},
 raw_input_pins=source_hashes,
 contract_findings=[
  'This is an evidence withholding policy, not a new alignment or measured alignment improvement.',
  'Person pairs use B_same_path, verified byte-for-value against semantic_episode path_original_frames. Numerical controls use separately generated relational-feature full DTW.',
  'Every full-path link and all original q/r indices remain; median display points are never the computational path.',
  'Broad labels and prior diagnostic audits are attached only after guard output; they do not enter guard arguments. Metadata mutation synthetic verifies invariance.',
  'No new search/DTW/model inference/162-case policy run was executed by this reviewer.',
  'Synthetic pass counts and raw structure checks are not signing accuracy. Normal false-withholding cost requires later saved-result reporting.',
  'Only adjacent reference-core relationships are assessed by the policy; compatible range is not full stage, occurrence, motion or linguistic proof.'
 ])
dest=U/'review/CODE_REVIEW.json'
if dest.exists():
 old=read(dest);preserved=U/'review'/('CODE_REVIEW_PREVIOUS_'+sha(dest)[:12]+'.json')
 if not preserved.exists():preserved.write_bytes(dest.read_bytes())
dest.write_text(json.dumps(review,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
print(json.dumps({k:review[k] for k in ['status','checks','failures','synthetic_monotonic_paths','real_raw_structure']},ensure_ascii=False))
