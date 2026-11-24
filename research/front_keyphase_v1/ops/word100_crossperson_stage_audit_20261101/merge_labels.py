from pathlib import Path
from collections import Counter
import json,hashlib
U=Path(__file__).resolve().parent;O=U.parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
 with Path(p).open('x',encoding='utf8') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False)
def runs(ids):
 out=[]
 for i in sorted(ids):
  if out and i==out[-1][1]+1:out[-1][1]=i
  else:out.append([i,i])
 return out
def canonical(k,phase):
 # Common annotation vocabulary only; neither matcher nor features are changed.
 if k=='bilateral_open_hands_lift_and_approach_center':return 'bilateral_open_hands_converge_near_face:move'
 if k=='right_closed_hand_reorients_toward_shoulder':return k+(':hold' if phase in ['hold','post_reorientation_hold'] else ':move')
 if k=='bilateral_open_hands_converge_near_face':return k+(':move' if phase=='converging' else ':hold')
 return k
def main():
 fit=read(U/'review/FIT_PHYSICAL_STAGES.json');dev=read(U/'review/DEV_PHYSICAL_STAGES.json');ph={c['alias']:c for c in read(O/'word100_core_stage_evidence_20261031/PHASE_DATA.json')['clips']};kinds={};clips=[];mapping=[]
 for origin,data in [('fit',fit),('development',dev)]:
  for c in data['clips']:
   regions=[];outer=set();known=set();frames=c['native_frame_count'];reviewed=set(c['reviewed_native_frame_ids'])
   assert sha(c['source_webm'])==c['source_webm_sha256'] and frames==ph[c['alias']]['frames']
   if origin=='fit':
    expected={s['segment_id']:s['core'] for s in ph[c['alias']]['segments'] if s['role']=='core_candidate'}
    assert len(c['segments'])==len(expected) and {s['segment_id']:s['core'] for s in c['segments']}==expected
   for s in c['segments']:
    items=s['parts'] if origin=='fit' else [s]
    for p in items:
     core=p['core'] if origin=='fit' else p['certain_interior'];out=p['outer'] if origin=='fit' else [p['start_bracket'][0],p['end_bracket'][1]]
     assert all(type(i) is int for i in core+out) and 0<=out[0]<=core[0]<=core[1]<=out[1]<frames
     ids=set(range(core[0],core[1]+1));assert ids<=reviewed and not ids&known;known|=ids;outer.update(range(out[0],out[1]+1))
     broad=p['kind_key'] if origin=='fit' else p['proposed_kind_key'];kind=canonical(broad,p['motion_subphase']);desc=p['description_ko'];rid=p['part_id'] if origin=='fit' else p['stage_id'];occ=s['occurrence_index_candidates'] if origin=='fit' else s['cycle_or_unknown'];occ=(occ[0] if len(occ)==1 else None) if isinstance(occ,list) else occ
     kinds.setdefault(kind,dict(label_ko=desc,physical_kind=broad,motion_subphase=p['motion_subphase']))
     if kind.endswith(':move'):kinds[kind]['label_ko']={'right_closed_hand_reorients_toward_shoulder:move':'닫힌 오른손을 어깨 쪽으로 옮기는 과정','bilateral_open_hands_converge_near_face:move':'양손을 얼굴 아래 중앙 쪽으로 모으는 과정'}[kind]
     if kind.endswith(':hold'):kinds[kind]['label_ko']={'right_closed_hand_reorients_toward_shoulder:hold':'닫힌 오른손을 어깨 옆에 두고 유지','bilateral_open_hands_converge_near_face:hold':'양손을 얼굴 아래 중앙에 모아 유지'}[kind]
     regions.append(dict(id=rid,region_type='observed_physical_core',core=core,outer=core,kind=kind,occurrence=occ,description_ko=desc,source_segment_id=s.get('segment_id'),original_outer=out,source_kind=broad,motion_subphase=p['motion_subphase'],fine_finger_3d_or_exact_contact_validated=False))
     mapping.append(dict(alias=c['alias'],region=rid,source_kind=broad,source_phase=p['motion_subphase'],canonical=kind))
   # Coarse outers describe the same transition envelope. Confirmed interiors from
   # the same continuous RGB review take precedence; gaps stay explicit, not normal.
   for n,(a,b) in enumerate(runs(outer-known)):
    regions.append(dict(id=f"{c['alias']}_boundary{n:02d}",region_type='boundary',core=None,outer=[a,b],kind=None,occurrence=None,description_ko='검토 범위 안의 경계·전환 또는 세부 단계 미확정'))
   regions.sort(key=lambda r:(r['core'] or r['outer'])[0]);assert set().union(*(set(range(r['outer'][0],r['outer'][1]+1)) for r in regions))==outer
   clips.append(dict(alias=c['alias'],role=origin,native_frame_count=frames,regions=regions,reviewed_native_frames=sorted(reviewed),reviewed_known_frames=len(known),uncertain_transition_frames=len(outer-known),outside_reviewed_scope_frames=frames-len(outer)))
 scope=['공개 4표현, 기록상 수행자 4명, 16영상의 개발 사후 감사입니다. 새로운 미사용 평가 집단의 성능이 아닙니다.','참고 12영상의 핵심·경계 연속 552프레임, 개발 4영상의 전체 624프레임을 에이전트가 직접 보았습니다. 정밀한 손가락 3D 정답이나 모든 프레임의 언어적 정답은 아닙니다.','움직임과 유지가 섞인 이전 범위를 실제 관측으로 나눴습니다. 검색에 쓴 원래 핵심 범위와 저장 예측은 바꾸지 않았습니다.','같은 동작 종류와 반복 회차를 따로 봅니다. 반복 횟수의 언어적 필수성이나 정상 변이 허용치를 확정하지 않습니다.','단계 경계와 미확인은 모두 분모에 남겼습니다. 가림이나 단계 일치를 수어 오류/정답으로 바꾸지 않습니다.','기존 자료와 과거 결과를 본 Codex의 검토이며 엄격한 이중맹검이나 독립 사람의 수어 정답이 아닙니다. 이번 주석 통합 동안 개인쌍 예측을 열지 않았습니다.']
 root_images=[f'evidence/{a}/{n}.jpg' for a,n in [('P036','continuous_072_095'),('P036','continuous_096_119'),('P048','continuous_072_095'),('P066','continuous_048_071'),('P066','continuous_072_095'),('P066','continuous_096_119'),('P072','continuous_072_095'),('P072','continuous_096_119')]]+[f'evidence/{a}/core_contact.jpg' for a in ['P033','P045','P063','P069']]
 write(U/'PHYSICAL_CORRESPONDENCE.json',dict(status='RGB_PHYSICAL_DRAFT_FROZEN_FOR_SAVED_OUTPUT_AUDIT',scope=scope,kinds=kinds,clips=clips,source_hashes={n:sha(U/'review'/n) for n in ['FIT_PHYSICAL_STAGES.json','DEV_PHYSICAL_STAGES.json']},neural_training_target=False,linguistic_truth=False))
 write(U/'review/LABEL_RECONCILIATION.json',dict(status='PASS_COMMON_VOCABULARY_AND_BOUNDARY_PREFLIGHT',clips=len(clips),source_core_anchors_preserved=39,dev_new_core_regions=14,canonical_mapping=mapping,root_direct_crosscheck_images={n:sha(U/n) for n in root_images},root_scope='12 contact sheets / 8 clips crosscheck; agents cover all16; not independent human truth',rules=['movement/hold split shared across both annotation sets','short rise+peak belongs to same visible rise episode; final sustained hold separate','coarse overlapping outers merged then remove all confirmed interiors; remainder uncertainty, not different-kind','fit rest/prep/return not densely labeled and therefore remains unreviewed; never normal by default'],prediction_files_opened=False))
 print('LABELS',len(clips),len(kinds),sum(len(c['regions']) for c in clips),'regions')
if __name__=='__main__':main()
