"""Read-only exhaustive audit of prior global-DTW physical label links.
No imports/calls to any matcher, detector or model. Only saved masks and paths.
"""
from pathlib import Path
from collections import Counter,defaultdict
from fractions import Fraction
import hashlib,json,math
import numpy as np
U=Path(__file__).resolve().parents[1];O=U.parent;S=O/'word100_crossperson_stage_audit_20261101'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fraction(v):return dict(numerator=v.numerator,denominator=v.denominator,value=float(v))
a=read(S/'ASSESSMENT.json');labels=read(S/'PHYSICAL_CORRESPONDENCE.json');causes=read(S/'CAUSE_REVIEW.json')
auth=read(S/'AUTHORIZATION.json');rows={r['alias']:r for r in auth['rows']}
inputs=[Path(r['input']) for r in rows.values()];sources=[S/n for n in ['ASSESSMENT.json','PHYSICAL_CORRESPONDENCE.json','CAUSE_REVIEW.json','AUTHORIZATION.json']]+inputs
pins={str(p):sha(p) for p in sources}
for r in rows.values():assert pins[str(Path(r['input']))]==r['input_sha256']
tables={};known_by_kind={};source_time={};masks={}
for c in labels['clips']:
 t=[None]*c['native_frame_count'];k=defaultdict(list)
 for reg in c['regions']:
  for i in range(reg['outer'][0],reg['outer'][1]+1):
   assert t[i] is None;t[i]=reg
   if reg['core']:k[reg['kind']].append(i)
 tables[c['alias']]=t;known_by_kind[c['alias']]=k
 # Group masks reproduce saved 69-feature validity semantics, without computing
 # costs, similarities, new alignment or labels. Numeric x is used only for
 # finite/nondegenerate geometric mask requirements in the frozen groups.
 with np.load(rows[c['alias']]['input'],allow_pickle=False) as z:
  x=z['x'].astype(float);m=z['mask'].astype(bool)&np.isfinite(x)
  frame=z['frame_id'];assert np.array_equal(frame,np.arange(len(frame)))
  source_time[c['alias']]=z['timestamp_us'].astype(int).tolist()
 gm={}
 for side,o in [('L',0),('R',28)]:
  gm[side+'_shape']=m[:,o+11:o+25];gm[side+'_location']=m[:,[o,o+1,o+8,o+9]];gm[side+'_axis']=m[:,o+25:o+27]
  v=x[:,o+2:o+4]-x[:,o+4:o+6];ok=m[:,o+2:o+6].all(1)&(np.linalg.norm(v,axis=1)>1e-8)
  gm[side+'_arm']=np.column_stack([m[:,[o+4,o+5,o+10]],ok,ok,ok])
  other=28 if o==0 else 0;direction=x[:,other+2:other+4]-x[:,other+4:other+6]
  good=m[:,o:o+2].all(1)&m[:,other+2:other+6].all(1)&((direction*direction).sum(1)>1e-8)
  gm[side+'_to_'+('R' if side=='L' else 'L')+'_forearm']=np.column_stack([good,good])
  gm[side+'_direction']=m[:,[63,64] if side=='L' else [67,68]]
 gm['hands_relation']=m[:,56:59];masks[c['alias']]=gm

def physical_reg(alias,frame):
 r=tables[alias][frame]
 return r if r and r['core'] and r['kind'] else None
def state(q,r,qi,ri):
 qr=physical_reg(q,qi);rr=physical_reg(r,ri)
 if rr is None:return 'reference_uncertain'
 if qr is not None:return 'same_kind_core' if qr['kind']==rr['kind'] else 'different_kind_core'
 x=tables[q][qi]
 return 'boundary' if x and not x['core'] else 'unreviewed_or_uncertain'
FAMILIES={
 'right_flat_hand_moves_along_left_forearm':('right_flat_over_left_forearm','move'),
 'right_flat_hand_holds_over_left_distal_forearm':('right_flat_over_left_forearm','hold'),
 'left_thumb_up_hand_rises_above_right':('left_thumb_up_above_right','move'),
 'left_thumb_up_hand_high_hold':('left_thumb_up_above_right','hold'),
 'bilateral_open_hands_converge_near_face:move':('bilateral_open_central_convergence','move'),
 'bilateral_open_hands_converge_near_face:hold':('bilateral_open_central_convergence','hold'),
 'right_closed_hand_reorients_toward_shoulder:move':('right_closed_near_shoulder','move'),
 'right_closed_hand_reorients_toward_shoulder:hold':('right_closed_near_shoulder','hold')}
def nearest(alias,frame,kind):
 ids=known_by_kind[alias].get(kind,[])
 if not ids:return None
 d=min(abs(j-frame) for j in ids);return dict(distance_frames=d,nearest_frames=[j for j in ids if abs(j-frame)==d])

different=[];ordinal=[];summary={};per_event=[];mask_summary_checks=0
for role in ['fit','dev']:
 summary[role]=dict(event_count=0,reference_frame_denominator=0,path_link_count=0,states=defaultdict(Fraction),different_kind_links=0,different_ordinal_links=0,
  different_kind_reference_weight=Fraction(),different_ordinal_reference_weight=Fraction(),stage_insufficient_overlap_links=0,stage_insufficient_overlap_weight=Fraction(),
  exact_link_missing_overlap_links=0,exact_link_missing_overlap_weight=Fraction(),exact_link_reference_unobserved_overlap_links=0,all_groups_query_shared_on_difference_links=0,
  family_reference_weight=defaultdict(Fraction),center_states=Counter(),center_occurrences=Counter())
for e in a['events']:
 q,r=e['query'],e['reference'];role='fit' if e['role']=='fit' else 'dev';m=e['methods']['global_dtw'];ss=summary[role]
 paths=m['path'];counts=Counter(ri for qi,ri in paths);core=e['reference_core'];ss['event_count']+=1;ss['reference_frame_denominator']+=core[1]-core[0]+1;ss['path_link_count']+=len(paths)
 # Check these independently reconstructed masks against the frozen stage
 # summaries before attributing an exact-link support fact to them.
 ids=np.arange(core[0],core[1]+1)
 for g,qm in masks[q].items():
  rm=masks[r][g];comp=np.zeros_like(rm);missing=0;shared_query=set()
  for qi,ri in paths:
   joint=qm[qi]&rm[ri];comp[ri]|=joint;missing+=int((rm[ri]&~qm[qi]).sum())
   if joint.any():shared_query.add(qi)
  expected=int(rm[ids].sum());compared=int(comp[ids].sum());saved=m['groups'][g]
  values={'expected':expected,'compared':compared,'path_missing_cells':missing,'reference_possible':len(ids)*rm.shape[1],'reference_unobserved':len(ids)*rm.shape[1]-expected,'distinct_shared_query_frames':len(shared_query)}
  for field,value in values.items():assert saved[field]==value;mask_summary_checks+=1
 ss['center_states'][m['state']]+=1;ss['center_occurrences'][m['occurrence']]+=1;local=[];state_counts=defaultdict(Fraction)
 for qi,ri in paths:
  assert core[0]<=ri<=core[1];s=state(q,r,qi,ri);w=Fraction(1,counts[ri]);ss['states'][s]+=w;state_counts[s]+=w
  qr=physical_reg(q,qi);rr=physical_reg(r,ri)
  wrongordinal=bool(s=='same_kind_core' and qr['occurrence'] is not None and rr['occurrence'] is not None and qr['occurrence']!=rr['occurrence'])
  if s!='different_kind_core' and not wrongordinal:continue
  category='same_kind_different_documented_ordinal' if wrongordinal else 'other_physical_kind'
  qfamily=FAMILIES.get(qr['kind']);rfamily=FAMILIES.get(rr['kind'])
  if not wrongordinal and qfamily and rfamily and qfamily[0]==rfamily[0] and qfamily[1]!=rfamily[1]:category='same_activity_move_hold_transition'
  group_evidence={}
  for g,qm in masks[q].items():
   rm=masks[r][g][ri];qm=qm[qi];group_evidence[g]=dict(reference_expected_cells=int(rm.sum()),shared_cells=int((rm&qm).sum()),query_missing_expected_cells=int((rm&~qm).sum()),reference_unobserved_cells=int((~rm).sum()))
  missing=[g for g,v in group_evidence.items() if v['query_missing_expected_cells']>0];unobserved=[g for g,v in group_evidence.items() if v['reference_expected_cells']==0]
  row=dict(case=e['case'],query=q,reference=r,segment_id=e['segment_id'],role=role,query_frame=qi,reference_frame=ri,query_timestamp_us=source_time[q][qi],reference_timestamp_us=source_time[r][ri],reference_equal_weight=fraction(w),
    category=category,query_kind=qr['kind'],reference_kind=rr['kind'],query_region=qr['id'],reference_region=rr['id'],query_core=qr['core'],reference_physical_core=rr['core'],unchanged_search_reference_core=core,
    query_occurrence=qr['occurrence'],reference_occurrence=rr['occurrence'],query_interior_edge_distance=min(qi-qr['core'][0],qr['core'][1]-qi),reference_interior_edge_distance=min(ri-rr['core'][0],rr['core'][1]-ri),
    query_distance_to_reference_kind=nearest(q,qi,rr['kind']),reference_distance_to_query_kind=nearest(r,ri,qr['kind']),
    stage_insufficient_groups=m['insufficient_groups'],stage_reference_unobserved_groups=m['reference_unobserved_groups'],exact_link_query_missing_groups=missing,exact_link_reference_unobserved_groups=unobserved,exact_link_group_masks=group_evidence,
    cause_scope='label transition proximity and stored observation overlap; not a proven causal feature failure or sign error')
  if wrongordinal:ordinal.append(row);ss['different_ordinal_links']+=1;ss['different_ordinal_reference_weight']+=w
  else:
   different.append(row);local.append(row);ss['different_kind_links']+=1;ss['different_kind_reference_weight']+=w
   ss['stage_insufficient_overlap_links']+=bool(m['insufficient_groups']);ss['stage_insufficient_overlap_weight']+=w*bool(m['insufficient_groups'])
   ss['exact_link_missing_overlap_links']+=bool(missing);ss['exact_link_missing_overlap_weight']+=w*bool(missing)
   ss['exact_link_reference_unobserved_overlap_links']+=bool(unobserved);ss['all_groups_query_shared_on_difference_links']+=not missing
   ss['family_reference_weight'][qfamily[0] if category=='same_activity_move_hold_transition' else category]+=w
 assert sum(state_counts.values())==core[1]-core[0]+1
 for s,v in state_counts.items():assert math.isclose(float(v),m['trace']['reference_equal_weight_counts'].get(s,0),abs_tol=1e-10)
 if local:per_event.append(dict(case=e['case'],query=q,reference=r,segment_id=e['segment_id'],role=role,query_frame_span=[min(x['query_frame'] for x in local),max(x['query_frame'] for x in local)],reference_frame_span=[min(x['reference_frame'] for x in local),max(x['reference_frame'] for x in local)],links=len(local),reference_weight=sum(x['reference_equal_weight']['value'] for x in local),stage_insufficient_groups=m['insufficient_groups'],exact_link_missing_group_union=sorted({g for x in local for g in x['exact_link_query_missing_groups']})))

for role,s in summary.items():
 original=a['summary'][role]['global_dtw'];assert s['event_count']==original['total'];assert s['reference_frame_denominator']==original['path']['expected_reference_frames']
 for k,v in s['states'].items():assert math.isclose(float(v),original['path']['reference_equal_weight_counts'].get(k,0),abs_tol=1e-10)
 s['different_kind_percent']=100*float(s['different_kind_reference_weight'])/s['reference_frame_denominator']
 s['different_ordinal_percent']=100*float(s['different_ordinal_reference_weight'])/s['reference_frame_denominator']
 s['affected_events']=sum(x['role']==role for x in per_event)
 rs=[x for x in different if x['role']==role]
 s['cause_classes']={'same_activity_move_hold_transition':sum(x['category']=='same_activity_move_hold_transition' for x in rs),'opposite_hand_or_other_activity_kind':sum(x['category']=='other_physical_kind' for x in rs),'same_kind_different_ordinal':sum(x['role']==role for x in ordinal)}
 s['at_least_one_exact_interior_edge_links']=sum(min(x['query_interior_edge_distance'],x['reference_interior_edge_distance'])==0 for x in rs)
 s['nearest_counterpart_kind_distance_range_frames']=[min(x[z]['distance_frames'] for x in rs for z in ['query_distance_to_reference_kind','reference_distance_to_query_kind']),max(x[z]['distance_frames'] for x in rs for z in ['query_distance_to_reference_kind','reference_distance_to_query_kind'])]
assert len(different)==20 and len(ordinal)==0
assert all(x['category']=='same_activity_move_hold_transition' for x in different)
assert all(min(x['query_interior_edge_distance'],x['reference_interior_edge_distance'])==0 for x in different)
previous_global=[x for x in causes['rows'] if x['method']=='global_dtw']
previous_links={(x['case'],x['segment'],p['query'],p['reference']) for x in previous_global for p in x['different_kind_links']}
new_links={(x['case'],x['segment_id'],x['query_frame'],x['reference_frame']) for x in different}
assert previous_links<=new_links
for p,h in pins.items():assert sha(p)==h
def serial(v):
 if isinstance(v,Fraction):return fraction(v)
 if isinstance(v,dict):return {k:serial(x) for k,x in v.items()}
 if isinstance(v,list):return [serial(x) for x in v]
 return v
out=dict(status='COMPLETE_EXHAUSTIVE_SAVED_BASELINE_CAUSE_AUDIT',reviewer='same Codex A; reused prior provisional physical labels, no new ground truth',
 scope='global_dtw saved core-scope links of all36 pairs117 opportunities; consumed DEV diagnostic, not fresh generalization',
 denominator_contract='Every original reference frame in each unchanged reference core has total weight1 divided equally over all its saved query links. fit880/dev440/all1320. Raw20 links are a separate micro count.',
 summary=summary,total_different_kind_links=len(different),total_different_ordinal_links=len(ordinal),frozen_mask_summary_checks=mask_summary_checks,affected_events=per_event,different_kind_links=different,same_kind_different_ordinal_links=ordinal,
 old_cause_review_coverage=dict(global_rows=len(previous_global),global_links=len(previous_links),new_exhaustive_links=len(new_links),additional_links=len(new_links-previous_links)),
 priorities_ko=[
  '1순위: 남은 불일치는 모두 동일 활동의 이동↔유지 경계이다. 전체 순서를 이미 보존하는 baseline을 지키면서 움직임/유지 시점의 근거를 먼저 점검한다. 반복회차 오류가 남아 있다고 가정해서는 안 된다.',
  '20개 링크 모두 적어도 한쪽이 확실 interior의 첫/끝 프레임이다. 경계와 가까운 현상이라는 근거이며, 주석을 넓히거나 자동으로 같은 단계로 바꾸는 근거는 아니다. 양쪽에서 상대 kind까지의 정확한 프레임 거리도 보존했다.',
  '2순위: 단계 전체 support 부족과 해당 링크의 expected-cell 누락을 분리한다. 누락 중복은 인과 증명이 아니며, 움직임을 잘못했다고 판단할 수 없다.',
  '단계 support 부족과 겹치는 불일치 질량은 fit4.5/11.5, dev2/8이다. 그러나 불일치20링크 자체는 각각13그룹63셀 모두 reference-expected/query-shared이며 누락0이다. 따라서 이20연결을 해당프레임 손검출실패나ROI밖 때문이라고 설명할 근거가 없다. 인접프레임 support는 별개다.',
  '3순위: 반대손/다른 활동 종류/같은 종류의 다른 회차는 이 baseline의 확실한 경로 비교에서 0건이다. 새 후보가 정상 순서·회차를 해치지 않는 비회귀 조건으로 유지한다.',
  '개발 1.8%는 8/440=1.818%이고 fit은11.5/880=1.307%다. 전체19.5/1320=1.477%를 개발1.8%나20/1320으로 바꿔 말하지 않는다.',
  '물리 라벨은 사후 감사에만 사용한다. 단어별 예외, 후보입력 라벨 주입, DEV 임계값조정, 가림=오류, 필수반복수 정답 추가는 하지 않는다.'],
 caveats=['All20 mismatches are within previously reviewed definite physical interiors, not frames labeled boundary. Their proximity to annotation edges is descriptive only; some movement/hold boundaries are gradual.',
  'Same-kind different ordinal search includes every saved core-scope link; it is not restricted to the117 representative centers. No mismatch was observed in the reviewed comparable portions; unknowns cannot prove occurrence identity.',
  'No new RGB inspection in this audit. The prior 16-video provisional physical review is reused unchanged. No independent human/linguistic verdict.',
  'Exact-link mask calculations use frozen finite69-feature and geometric-degeneracy rules. They distinguish reference-expected/query-missing cells from reference-unobserved cells; missing storage is not a fist/rest label.',
  'This identifies overlap and transition localization, not a counterfactual causal explanation of the aligner.'],
 activity=dict(new_alignment_searches=0,new_DP=0,new_model_inference=0,new_training=0,new_RGB_views=0,old_writes=0),source_pins=pins)
(U/'review/BASELINE_CAUSE_AUDIT.json').write_text(json.dumps(serial(out),ensure_ascii=False,indent=2),encoding='utf-8')
print('complete',len(different),'different-kind links',len(ordinal),'different-ordinal links;',mask_summary_checks,'saved mask checks')
print('old cause coverage',out['old_cause_review_coverage'])
